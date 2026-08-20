"""Feedback persistence service."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Feedback, Prediction
from api.services.mask_validation import validate_corrected_mask

logger = logging.getLogger(__name__)


class PredictionNotFoundError(Exception):
    """Raised when a prediction_id does not match any row in the DB."""


async def save_feedback(
    db: AsyncSession,
    prediction_id: str,
    user_id: uuid.UUID,
    flag: str,
    notes: str | None,
) -> Feedback:
    """Validate the prediction exists and persist the feedback row.

    Public feedback records a verdict only. A corrected mask is never
    accepted here: it is a training label and must go through the
    admin-only ``save_correction`` path, which validates it. The
    stored row therefore always has a null ``corrected_mask_b64``,
    which is what keeps the prediction in the review queue until an
    admin resolves it.

    Args:
        db: Active async database session.
        prediction_id: UUID string of the prediction being flagged.
        user_id: UUID of the authenticated user submitting feedback.
        flag: One of ``'good'``, ``'bad'``, ``'uncertain'``.
        notes: Optional free-text notes (max 2000 characters).

    Returns:
        The created ``Feedback`` row.

    Raises:
        ValueError: If ``prediction_id`` is not a valid UUID string.
        PredictionNotFoundError: If the UUID does not match any
            existing prediction.
    """
    try:
        pred_uuid = uuid.UUID(prediction_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"'{prediction_id}' is not a valid UUID.",
        ) from exc

    result = await db.execute(
        select(Prediction).where(Prediction.id == pred_uuid),
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise PredictionNotFoundError(
            f"Prediction '{prediction_id}' not found.",
        )

    row = Feedback(
        prediction_id=pred_uuid,
        user_id=user_id,
        flag=flag,
        notes=notes,
        corrected_mask_b64=None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info(
        "Feedback saved: id=%s, prediction=%s, flag=%s",
        row.id,
        prediction_id,
        flag,
    )
    return row


async def get_review_queue(
    db: AsyncSession,
    limit: int,
    offset: int,
) -> list[dict]:
    """Return predictions awaiting reviewer correction.

    A prediction needs review when it has at least one ``bad`` or
    ``uncertain`` feedback row and no resolving row yet — that is, no
    feedback row that is ``good`` or carries a corrected mask. Once a
    reviewer relabels it (adds such a row), it drops out of the queue.

    Args:
        db: Active async database session.
        limit: Maximum number of predictions to return.
        offset: Number of predictions to skip (pagination).

    Returns:
        A list of dicts, oldest prediction first, each carrying the
        prediction fields plus the latest flag and notes for context.
    """
    flagged = select(Feedback.prediction_id).where(
        Feedback.flag.in_(["bad", "uncertain"]),
    )
    resolved = select(Feedback.prediction_id).where(
        or_(
            Feedback.flag == "good",
            Feedback.corrected_mask_b64.is_not(None),
        ),
    )
    stmt = (
        select(Prediction)
        .where(Prediction.id.in_(flagged))
        .where(Prediction.id.not_in(resolved))
        .order_by(Prediction.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    predictions = (await db.execute(stmt)).scalars().all()
    if not predictions:
        return []

    pred_ids = [p.id for p in predictions]
    feedback_rows = (
        (
            await db.execute(
                select(Feedback)
                .where(Feedback.prediction_id.in_(pred_ids))
                .order_by(
                    Feedback.prediction_id,
                    Feedback.created_at.desc(),
                ),
            )
        )
        .scalars()
        .all()
    )

    # First row seen per prediction is the latest (desc order above).
    latest: dict[uuid.UUID, Feedback] = {}
    for row in feedback_rows:
        latest.setdefault(row.prediction_id, row)

    items: list[dict] = []
    for prediction in predictions:
        fb = latest.get(prediction.id)
        items.append(
            {
                "prediction_id": str(prediction.id),
                "image_filename": prediction.image_filename,
                "image_width_px": prediction.image_width_px,
                "image_height_px": prediction.image_height_px,
                "image_uri": prediction.image_uri,
                "mask_b64": prediction.mask_b64,
                "mask_confidence": prediction.mask_confidence,
                "flag": fb.flag if fb else None,
                "notes": fb.notes if fb else None,
                "created_at": prediction.created_at,
            }
        )
    return items


async def save_correction(
    db: AsyncSession,
    prediction_id: str,
    admin_id: uuid.UUID,
    corrected_mask_b64: str | None,
    flag: str | None,
    notes: str | None,
) -> Feedback:
    """Persist a reviewer's correction as a new feedback row.

    The correction is recorded as its own ``Feedback`` row (not an
    update of the original flag) so provenance is preserved: who
    flagged the prediction and who later corrected it remain
    distinct, auditable events.

    Args:
        db: Active async database session.
        prediction_id: UUID string of the prediction being corrected.
        admin_id: UUID of the admin submitting the correction.
        corrected_mask_b64: Optional corrected mask as base64 PNG. If
            present it is validated against the prediction's image
            dimensions and normalised before storage.
        flag: Optional resolved verdict (``good``/``bad``/
            ``uncertain``). Defaults to ``bad`` when a corrected mask
            is supplied without an explicit flag.
        notes: Optional free-text notes (max 2000 characters).

    Returns:
        The created ``Feedback`` row.

    Raises:
        ValueError: If ``prediction_id`` is not a valid UUID string.
        PredictionNotFoundError: If the UUID does not match any
            existing prediction.
        MaskValidationError: If the corrected mask fails validation.
    """
    try:
        pred_uuid = uuid.UUID(prediction_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"'{prediction_id}' is not a valid UUID.",
        ) from exc

    result = await db.execute(
        select(Prediction).where(Prediction.id == pred_uuid),
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise PredictionNotFoundError(
            f"Prediction '{prediction_id}' not found.",
        )

    normalised_mask: str | None = None
    if corrected_mask_b64 is not None:
        normalised_mask = validate_corrected_mask(
            corrected_mask_b64,
            prediction.image_width_px,
            prediction.image_height_px,
        )

    resolved_flag = flag if flag is not None else "bad"

    row = Feedback(
        prediction_id=pred_uuid,
        user_id=admin_id,
        flag=resolved_flag,
        notes=notes,
        corrected_mask_b64=normalised_mask,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info(
        "Correction saved: id=%s, prediction=%s, flag=%s, mask=%s",
        row.id,
        prediction_id,
        resolved_flag,
        normalised_mask is not None,
    )
    return row
