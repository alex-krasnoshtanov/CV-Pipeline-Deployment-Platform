"""/feedback router.

- ``POST /feedback`` — record a researcher's assessment of a
  prediction. Requires authentication: anonymous users cannot leave
  feedback, because feedback binds to a specific user_id for the
  training-data review workflow.
- ``GET /feedback/review-queue`` — admin-only list of predictions
  awaiting correction.
- ``POST /feedback/relabel`` — admin-only submission of a corrected
  mask and/or resolved verdict for a flagged prediction.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin, require_user
from api.db import get_db
from api.db.models import User
from api.schemas.feedback import (
    FeedbackRequest,
    FeedbackResponse,
    RelabelRequest,
    ReviewQueueItem,
)
from api.services.feedback_service import (
    PredictionNotFoundError,
    get_review_queue,
    save_correction,
    save_feedback,
)
from api.services.mask_validation import MaskValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Flag a prediction as good, bad, or uncertain.",
)
async def submit_feedback(
    body: FeedbackRequest,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Record feedback on a prediction.

    Args:
        body: Feedback payload with prediction_id, flag, and optional
            notes. A corrected mask is not accepted here; use the
            admin-only POST /feedback/relabel endpoint.
        current_user: Authenticated user (cookie OR X-API-Key).
        db: Async database session.

    Returns:
        A dict serialised through ``FeedbackResponse``.

    Raises:
        HTTPException: 401 if not authenticated, 404 if prediction
            not found, 422 if the prediction_id format is invalid or
            an unexpected field (e.g. corrected_mask_b64) is sent.
    """
    try:
        row = await save_feedback(
            db=db,
            prediction_id=body.prediction_id,
            user_id=current_user.id,
            flag=body.flag,
            notes=body.notes,
        )
    except PredictionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "NOT_FOUND",
                "message": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "INVALID_PREDICTION_ID",
                "message": str(exc),
            },
        ) from exc

    return {
        "feedback_id": str(row.id),
        "prediction_id": str(row.prediction_id),
        "flag": row.flag,
        "created_at": row.created_at,
    }


@router.get(
    "/review-queue",
    response_model=list[ReviewQueueItem],
    summary="List predictions awaiting reviewer correction (admin).",
)
async def review_queue(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return predictions flagged bad/uncertain and not yet resolved.

    Args:
        limit: Maximum number of predictions to return (1-100).
        offset: Number of predictions to skip (pagination).
        current_user: Authenticated admin (cookie OR X-API-Key).
        db: Async database session.

    Returns:
        A list of dicts serialised through ``ReviewQueueItem``.
    """
    return await get_review_queue(db, limit=limit, offset=offset)


@router.post(
    "/relabel",
    response_model=FeedbackResponse,
    summary="Submit a corrected mask or resolved verdict (admin).",
)
async def relabel(
    body: RelabelRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Record an admin correction as a new feedback row.

    Args:
        body: Relabel payload with prediction_id and at least one of
            corrected_mask_b64 or flag, plus optional notes.
        current_user: Authenticated admin (cookie OR X-API-Key).
        db: Async database session.

    Returns:
        A dict serialised through ``FeedbackResponse``.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not an admin,
            404 if the prediction is not found, 422 if the
            prediction_id is invalid or the mask fails validation.
    """
    try:
        row = await save_correction(
            db=db,
            prediction_id=body.prediction_id,
            admin_id=current_user.id,
            corrected_mask_b64=body.corrected_mask_b64,
            flag=body.flag,
            notes=body.notes,
        )
    except PredictionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "NOT_FOUND",
                "message": str(exc),
            },
        ) from exc
    except MaskValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": exc.error_code,
                "message": exc.message,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "INVALID_PREDICTION_ID",
                "message": str(exc),
            },
        ) from exc

    return {
        "feedback_id": str(row.id),
        "prediction_id": str(row.prediction_id),
        "flag": row.flag,
        "created_at": row.created_at,
    }
