"""Health-check router that reports service readiness.

Contract:
- Returns 200 with ``{"status": "ok", ...}`` when the model is loaded
  and the service is ready to accept inference traffic.
- Returns 503 with ``{"status": "loading", ...}`` during the startup
  window before the lifespan loads the checkpoint.

The endpoint is exempt from authentication (R9 allows /health as the
only unauthenticated route — needed so Docker Compose and Portainer
healthchecks can probe without a shared secret).
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Request, Response, status

from api.schemas.health import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def healthcheck(request: Request, response: Response) -> HealthResponse:
    """Return backend readiness state.

    Args:
        request: FastAPI request, used to reach the shared service
            state published by the lifespan in ``api.main``.
        response: FastAPI response, used to override the status code
            to 503 when the model has not yet loaded.

    Returns:
        A ``HealthResponse`` with ``status="ok"`` when ready,
        ``status="loading"`` with HTTP 503 otherwise.
    """
    state = request.app.state.service_state
    timestamp = datetime.now(UTC)

    if state.model_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=state.model_version,
            pipeline_version=state.pipeline_version,
            serving_mode=state.serving_mode,
            timestamp=timestamp,
        )

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="loading",
        model_loaded=False,
        model_version=None,
        pipeline_version=state.pipeline_version,
        serving_mode=None,
        timestamp=timestamp,
    )
