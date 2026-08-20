"""Prediction persistence service"""

from __future__ import annotations

import logging
import uuid

from cv_pipeline.schema import InferenceResult
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Prediction

logger = logging.getLogger(__name__)


async def save_prediction(
    db: AsyncSession,
    result: InferenceResult,
    user_id: uuid.UUID | None = None,
) -> uuid.UUID | None:
    """Persist an inference result to the predictions table.

    Args:
        db: Active async database session.
        result: Completed inference result from the cv-pipeline.
        user_id: UUID of the authenticated user who made the request.

    Returns:
        The generated prediction UUID on success, or ``None`` if the
        write failed.
    """
    row = Prediction(
        image_filename=result.image_filename,
        image_width_px=result.image_width_px,
        image_height_px=result.image_height_px,
        plate_id=result.metadata.plate_id if result.metadata else None,
        experiment_id=result.metadata.experiment_id if result.metadata else None,
        pipeline_version=result.pipeline_version,
        model_version=result.model_version,
        mask_b64=result.mask_b64,
        mask_confidence=result.mask_confidence,
        landmark_count=result.landmark_count,
        landmarks=[lm.to_dict() for lm in result.landmarks],
        user_id=user_id,
    )
    try:
        db.add(row)
        await db.commit()
        await db.refresh(row)
        logger.info("Prediction saved: id=%s, user_id=%s", row.id, user_id)
        return row.id
    except (IntegrityError, OperationalError) as exc:
        logger.error(
            "Failed to save prediction for '%s': %s",
            result.image_filename,
            exc,
        )
        await db.rollback()
        return None
