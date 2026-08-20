"""POST /monitoring/check — admin-triggered monitoring health check.

Runs the rolling-confidence drift detector in one call.  Intended to be
invoked by a cron job, the Airflow schedule, or an admin manually.

Auth: admin role required (403 for non-admin authenticated users,
401 for unauthenticated).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.config import get_settings
from api.db import get_db
from api.db.models import User
from api.services.drift_detector import check_confidence_drift

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post(
    "/check",
    summary="Run rolling-confidence drift detection.",
)
async def monitoring_check(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Run rolling-confidence drift detection.

    The check reads the database; it does not modify production data.
    It samples the most recent predictions, computes the low-confidence
    fraction, updates the Prometheus gauges, and (optionally) raises a
    Teams alert when the configured threshold is crossed.

    Args:
        current_user: Authenticated admin user.
        db: Async database session.

    Returns:
        Dict with a ``drift`` sub-dict from the drift check.

    Raises:
        HTTPException: 401 if not authenticated, 403 if not admin.
    """
    settings = get_settings()
    logger.info("Monitoring check requested by admin %s", current_user.id)

    drift = await check_confidence_drift(db, settings)

    return {"drift": drift}
