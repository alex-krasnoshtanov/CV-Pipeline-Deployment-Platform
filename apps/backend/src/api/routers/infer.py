"""POST /infer - accept an uploaded plant image, return segmentation
+ landmarks + confidence scores.

Auth: optional. Anonymous callers and authenticated callers share
the same 20/min ceiling, but the rate-limit bucket is keyed by
session id / API key when present, by IP otherwise. So:
- Anon users on the same NAT share one bucket (20/min total).
- Each authed user has their own bucket (20/min each).

This is a deliberate trade-off over the dynamic-limit-string
pattern, which slowapi does not support cleanly: the limit
callable is invoked with no arguments and cannot see the request.
Per-key bucketing gives you 90% of the benefit (a logged-in user
cannot be locked out by anonymous scrapers on the same IP) with
none of the complexity.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cv_pipeline.schema import Metadata
from cv_pipeline.validation import ValidationError
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import SESSION_COOKIE_NAME, optional_user
from api.db import get_db
from api.db.models import User
from api.metrics import inference_confidence
from api.schemas.infer import InferenceResponse
from api.services import inference_service
from api.services.image_persistence import persist_inference_image
from api.services.prediction_service import save_prediction

logger = logging.getLogger(__name__)

__all__ = ["router", "run_inference"]

router = APIRouter(prefix="/infer", tags=["infer"])


def _infer_rate_key(request: Request) -> str:
    """Bucket per-user when authed, per-IP otherwise.

    slowapi calls this with the request on every request to /infer
    to decide which counter to increment. We do NOT validate the
    credential here (that costs a DB hit, on the hot path, and the
    auth dep already does it downstream). We only use it as a
    bucket label - a forged cookie ends up in its own bucket and
    still hits 401 at the auth dep.

    We use a prefix of the credential so different sessions / keys
    do not collide, but truncate so the full token never appears
    in slowapi's storage keys (memory backend, but still hygienic).
    """
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid:
        return f"sid:{sid[:16]}"
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key[:16]}"
    return f"ip:{get_remote_address(request)}"


_limiter = Limiter(
    key_func=_infer_rate_key,
    storage_uri="memory://",
    strategy="fixed-window",
)

# Map cv_pipeline ValidationError.error_code -> HTTP status.
_ERROR_STATUS: dict[str, int] = {
    "UNSUPPORTED_FILE_TYPE": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "UNSUPPORTED_COLOR_MODE": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "FILE_TOO_LARGE": status.HTTP_413_CONTENT_TOO_LARGE,
    "IMAGE_TOO_SMALL": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "IMAGE_TOO_LARGE": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "CORRUPT_FILE": status.HTTP_422_UNPROCESSABLE_CONTENT,
}


@router.post(
    "",
    response_model=InferenceResponse,
    summary="Run segmentation + landmark detection on a single image.",
)
@_limiter.limit("20/minute")
async def run_inference(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="The plant image to segment."),
    plate_id: str | None = Form(None),
    experiment_id: str | None = Form(None),
    timestamp: str | None = Form(None),
    current_user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Accept a plant image and return the inference result.

    Args:
        request: FastAPI request (used to reach the shared model on
            ``app.state`` and to expose ``request.state`` for slowapi).
        background_tasks: Queue for post-response work; used to persist
            the input image off the hot path.
        image: Uploaded image via multipart/form-data.
        plate_id: Optional Petri dish identifier (pass-through).
        experiment_id: Optional experiment identifier (pass-through).
        timestamp: Optional ISO 8601 capture timestamp (pass-through).
        current_user: Authenticated user or None for anonymous calls.
        db: Async database session injected by ``get_db``.

    Returns:
        A ``dict`` that FastAPI serialises through ``InferenceResponse``.

    Raises:
        HTTPException: 413/422 on cv_pipeline validation failure,
            503 if the model is not loaded, 429 on rate-limit
            breach (raised by slowapi, handled in main.py).
    """
    state = getattr(request.app.state, "service_state", None)
    if state is None or not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "MODEL_NOT_READY",
                "message": "Model is not loaded yet.",
            },
        )
    is_cloud = state.serving_mode == "azure_ml"
    model = getattr(request.app.state, "model", None)
    if not is_cloud and model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "MODEL_NOT_READY",
                "message": "Local model is not loaded.",
            },
        )

    tmp_path = await inference_service.save_upload_to_tempfile(
        upload_file=image.file,
        filename=image.filename,
    )
    # When the image is handed to the persistence background task, that
    # task owns the tempfile and deletes it. Otherwise we clean up here.
    handed_off = False

    try:
        metadata = Metadata(
            plate_id=plate_id,
            experiment_id=experiment_id,
            timestamp=timestamp,
        )
        if is_cloud:
            from api.services.endpoint_client import run_endpoint_inference

            result = await run_endpoint_inference(
                image_path=tmp_path,
                metadata=metadata,
            )
        else:
            result = await inference_service.run_pipeline_inference(
                image_path=tmp_path,
                model=model,
                metadata=metadata,
            )
        logger.info(
            "Inference completed for '%s': %d landmark(s), confidence=%.3f, user=%s.",
            image.filename,
            result.landmark_count,
            result.mask_confidence,
            current_user.id if current_user else "anonymous",
        )
        # Drift signal: record every successful prediction's mask
        # confidence so the /metrics scrape exposes its distribution.
        inference_confidence.observe(result.mask_confidence)

        result.image_filename = image.filename or result.image_filename

        # Anonymous calls save with user_id=NULL. The predictions
        # table allows it. Anon users cannot retrieve their own
        # history (no row binds them), which is the intended UX:
        # log in to get persistence.
        prediction_id = None
        user_id = current_user.id if current_user else None
        try:
            prediction_id = await save_prediction(db, result, user_id=user_id)
        except Exception:
            logger.exception("Unexpected error saving prediction to database.")

        # Persist the input image only for authenticated predictions:
        # the per-user folder needs a user_id, and feedback (the only
        # consumer) requires auth. Runs after the response, never blocks.
        if prediction_id is not None and user_id is not None:
            suffix = Path(image.filename or "").suffix or ".png"
            background_tasks.add_task(
                persist_inference_image,
                prediction_id=prediction_id,
                user_id=user_id,
                src_path=tmp_path,
                suffix=suffix,
            )
            handed_off = True

        response = result.to_dict()
        if prediction_id is not None:
            response["prediction_id"] = str(prediction_id)
        return response
    except ValidationError as exc:
        logger.info(
            "Validation rejected '%s': [%s] %s",
            image.filename,
            exc.error_code,
            exc.message,
        )
        raise HTTPException(
            status_code=_ERROR_STATUS.get(
                exc.error_code, status.HTTP_422_UNPROCESSABLE_CONTENT
            ),
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc
    finally:
        if not handed_off:
            inference_service.cleanup_tempfile(tmp_path)
