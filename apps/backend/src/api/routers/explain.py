"""POST /explain - return a Seg-Grad-CAM heatmap for an uploaded image.

Companion to /infer. Where /infer returns the mask + landmarks, this returns
a heatmap of which regions drove the root classification, for the frontend's
explainability tab.

Auth + rate limit: mirrors /infer (anonymous allowed for an easy demo) but
with a tighter 10/min ceiling because Grad-CAM is heavier than inference
(forward + backward per patch, and in cloud mode it runs on CPU).

Serving-mode handling lives in the explain service, not here: this router does
not care whether the model is the in-memory one (local/on-prem) or a lazily
loaded CPU one (cloud). It just hands the request's ``app`` to the service.
"""

from __future__ import annotations

import logging

from cv_pipeline.schema import Metadata
from cv_pipeline.validation import ValidationError
from fastapi import (
    APIRouter,
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

from api.auth.dependencies import SESSION_COOKIE_NAME, optional_user
from api.db.models import User
from api.schemas.explain import ExplainResponse
from api.services import inference_service
from api.services.explain_service import run_explanation

logger = logging.getLogger(__name__)

__all__ = ["router", "explain_image"]

router = APIRouter(prefix="/explain", tags=["explain"])


def _explain_rate_key(request: Request) -> str:
    """Bucket per-user when authed, per-IP otherwise (same scheme as /infer)."""
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid:
        return f"sid:{sid[:16]}"
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key[:16]}"
    return f"ip:{get_remote_address(request)}"


_limiter = Limiter(
    key_func=_explain_rate_key,
    storage_uri="memory://",
    strategy="fixed-window",
)

# Same cv_pipeline error_code -> HTTP status mapping as /infer.
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
    response_model=ExplainResponse,
    summary="Compute a Seg-Grad-CAM explanation heatmap for a single image.",
)
@_limiter.limit("10/minute")
async def explain_image(
    request: Request,
    image: UploadFile = File(..., description="The plant image to explain."),
    plate_id: str | None = Form(None),
    experiment_id: str | None = Form(None),
    timestamp: str | None = Form(None),
    current_user: User | None = Depends(optional_user),
) -> dict:
    """Accept a plant image and return a Grad-CAM heatmap.

    Args:
        request: FastAPI request (used to reach ``app`` for model resolution
            and to expose ``request.state`` for slowapi).
        image: Uploaded image via multipart/form-data.
        plate_id: Optional Petri dish identifier (pass-through).
        experiment_id: Optional experiment identifier (pass-through).
        timestamp: Optional ISO 8601 capture timestamp (pass-through).
        current_user: Authenticated user or None for anonymous calls.

    Returns:
        A ``dict`` serialised through ``ExplainResponse``.

    Raises:
        HTTPException: 413/422 on cv_pipeline validation failure, 503 if no
            model is available, 504 if the explanation times out, 500 on any
            other explanation failure, 429 on rate-limit breach (slowapi).
    """
    state = getattr(request.app.state, "service_state", None)
    if state is None or not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "MODEL_NOT_READY",
                "message": "Model is not ready yet.",
            },
        )

    tmp_path = await inference_service.save_upload_to_tempfile(
        upload_file=image.file,
        filename=image.filename,
    )

    try:
        metadata = Metadata(
            plate_id=plate_id,
            experiment_id=experiment_id,
            timestamp=timestamp,
        )
        result = await run_explanation(
            app=request.app,
            image_path=tmp_path,
            metadata=metadata,
        )
        result.image_filename = image.filename or result.image_filename
        logger.info(
            "Explanation completed for '%s': layer=%s, peak=%.4f, user=%s.",
            image.filename,
            result.target_layer,
            result.heatmap_peak,
            current_user.id if current_user else "anonymous",
        )
        return result.to_dict()
    except ValidationError as exc:
        logger.info(
            "Explain validation rejected '%s': [%s] %s",
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
    except TimeoutError as exc:
        logger.warning("Explanation timed out for '%s'.", image.filename)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error_code": "EXPLAIN_TIMEOUT",
                "message": (
                    "Explanation timed out. The image may be too large for "
                    "CPU-based explanation; try a smaller crop."
                ),
            },
        ) from exc
    except HTTPException:
        # Re-raise structured HTTP errors unchanged so the broad handler below
        # never masks them as a generic 500.
        raise
    except Exception as exc:
        # Grad-CAM-specific failures (hook never fired, shape mismatch in
        # reconstruction, CUDA/CPU OOM) would otherwise surface as an opaque
        # 500 with no body. Return a structured error instead. /infer is
        # unaffected; this degrades only the opt-in explain path.
        logger.exception("Explanation failed unexpectedly for '%s'.", image.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "EXPLAIN_FAILED",
                "message": "Failed to compute the explanation.",
            },
        ) from exc
    finally:
        inference_service.cleanup_tempfile(tmp_path)
