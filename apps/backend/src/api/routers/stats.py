"""GET /stats — monitoring dashboard aggregation endpoint.

Returns business-insight and operational metrics derived from the
predictions, feedback, and model_versions tables.  Auth-protected:
requires a valid session cookie or X-API-Key header.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_user
from api.db import get_db
from api.db.models import User
from api.schemas.stats import StatsResponse
from api.services.stats_service import get_stats

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get(
    "",
    response_model=StatsResponse,
    summary="Monitoring dashboard statistics.",
)
async def read_stats(
    request: Request,
    current_user: User = Depends(require_user),
    db: AsyncSession = Depends(get_db),
) -> StatsResponse:
    """Return aggregated business and operational statistics.

    Aggregates prediction volume, confidence distribution, feedback rates,
    and per-model-version metrics from the database.  All heavy computation
    is pushed to SQL; the handler is a thin delegation layer.

    Args:
        request: FastAPI request, used to read app state (serving version)
            and detect whether the /metrics route is registered.
        current_user: Authenticated user (cookie or X-API-Key).
        db: Async database session.

    Returns:
        :class:`~api.schemas.stats.StatsResponse` with ``business`` and
        ``operational`` sub-objects.

    Raises:
        HTTPException: 401 if not authenticated.
    """
    logger.info("Stats requested by user %s", current_user.id)

    service_state = getattr(request.app.state, "service_state", None)
    serving_version: str | None = (
        getattr(service_state, "model_version", None) if service_state else None
    )

    # Detect whether Prometheus /metrics is actually registered — avoids
    # hardcoding True when the instrumentator might not be configured.
    prometheus_available = any(
        getattr(r, "path", None) == "/metrics" for r in request.app.routes
    )

    return await get_stats(
        db,
        serving_version=serving_version,
        prometheus_available=prometheus_available,
    )
