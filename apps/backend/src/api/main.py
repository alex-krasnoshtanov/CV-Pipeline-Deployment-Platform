"""FastAPI application entrypoint for the backend service."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from cv_pipeline import __version__ as _pipeline_version
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware

from api.db.session import init_db
from api.middleware import exception_handlers
from api.middleware.request_id import (
    RequestIDLogFilter,
    RequestIDMiddleware,
    request_id_ctx,
)
from api.routers.auth import router as auth_router
from api.routers.explain import router as explain_router
from api.routers.feedback import router as feedback_router
from api.routers.health import router as health_router
from api.routers.infer import router as infer_router
from api.routers.monitoring import router as monitoring_router
from api.routers.stats import router as stats_router
from api.routers.users import router as users_router
from api.services.model_loader import load_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ServiceState:
    """Runtime service status exposed through the health endpoint."""

    model_loaded: bool
    model_version: str
    pipeline_version: str
    serving_mode: str


def _serving_mode() -> str:
    """Resolve serving mode from environment configuration.

    Returns:
        ``"azure_ml"`` when ``MODEL_ENDPOINT_URL`` is set, otherwise
        ``"local"``.
    """
    if os.getenv("MODEL_ENDPOINT_URL"):
        return "azure_ml"
    return "local"


def _build_initial_state() -> ServiceState:
    """Create the default startup state before model loading completes."""
    return ServiceState(
        model_loaded=False,
        model_version=os.getenv("SERVING_MODEL_VERSION")
        or os.getenv("MODEL_VERSION", "unknown"),
        pipeline_version=_pipeline_version,
        serving_mode=_serving_mode(),
    )


def _configure_logging() -> None:
    """Install a logging config that includes the request ID on every record.

    The filter is attached to the handler (not the logger) so records
    propagating from child loggers - httpx, uvicorn, etc. - also get
    the ``request_id`` attribute set before the formatter touches them.
    Attaching to the logger only filters records produced BY that
    logger, which leaves child-logger records without the attribute
    and crashes the formatter.

    Idempotent - TestClient re-uses the app across tests.
    """
    root_logger = logging.getLogger()
    # Only install if we have not already done so.
    already_configured = any(
        any(isinstance(f, RequestIDLogFilter) for f in h.filters)
        for h in root_logger.handlers
    )
    if already_configured:
        return

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] [req=%(request_id)s] "
                "%(name)s: %(message)s",
            )
        )
        handler.addFilter(RequestIDLogFilter())
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
    else:
        # Respect existing handlers (pytest, uvicorn) - just attach
        # the filter so %(request_id)s is always populated.
        for h in root_logger.handlers:
            if not any(isinstance(f, RequestIDLogFilter) for f in h.filters):
                h.addFilter(RequestIDLogFilter())


def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return JSON instead of slowapi's default plaintext response.

    Matches our existing error envelope so the frontend can show it
    the same way as other 4xx responses.
    """
    _ = request
    return JSONResponse(
        status_code=429,
        content={
            "error_code": "RATE_LIMITED",
            "message": f"Rate limit exceeded: {exc.detail}",
            "pipeline_version": _pipeline_version,
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id_ctx.get(),
        },
    )


def _cors_origins() -> list[str]:
    """Parse the CORS_ORIGINS env var into a list.

    Returns:
        Empty list (no origins allowed) by default - explicit is
        better than the wildcard surprise. Set CORS_ORIGINS to a
        comma-separated list of allowed origins. With cookie auth
        enabled below, "*" is silently rejected by browsers, so
        you MUST list specific origins (e.g. http://localhost:3000).
    """
    raw = os.getenv("CORS_ORIGINS", "")
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise and publish backend model readiness for health checks.

    Failure to load the model is logged but does NOT crash the process
    - /health will keep returning 503 so a compose healthcheck or
    Portainer watchdog can restart the container.
    """
    _configure_logging()

    try:
        init_db(os.getenv("DB_URL", ""))
    except ValueError as exc:
        logger.exception("Database initialisation failed: %s", exc)

    state = _build_initial_state()
    app.state.service_state = state

    if state.serving_mode == "azure_ml":
        state.model_loaded = True
        logger.info(
            "Azure ML serving mode detected; model loading delegated to endpoint"
        )
        yield
        return

    try:
        model = load_model()
        app.state.model = model
        state.model_loaded = True
        state.model_version = model.model_version
        logger.info("Model loaded - version %s.", model.model_version)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        logger.exception("Model load failed: %s", exc)

    yield


app = FastAPI(
    title="CV Pipeline API",
    version=_pipeline_version,
    description=(
        "NPEC plant organ segmentation and root tip detection. "
        "Implements the CV Pipeline Specification."
    ),
    lifespan=lifespan,
)

# Per-IP rate limit: protects /infer from hammer-loops if a key
# leaks. 20 req/min keeps the app and the /infer route aligned while
# still allowing a modest burst for manual experimentation.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["20/minute"],
    storage_uri="memory://",
    strategy="fixed-window",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: _rate_limit_handler(r, e))
app.add_middleware(SlowAPIMiddleware)

# Order matters: request ID first so downstream middleware and handlers
# can read it. CORS after so preflight responses also carry the ID.
# SessionMiddleware is required by Authlib to persist the OAuth state
# nonce across the /github/login -> /github/callback redirect. The
# secret is independent from JWT_SIGNING_KEY: this protects the state
# cookie; JWT_SIGNING_KEY signs the issued tokens.
_session_secret = os.environ.get("SESSION_SECRET", "")
if not _session_secret:
    logger.warning(
        "SESSION_SECRET is not set; using an insecure fallback. "
        "OAuth flow will work in dev but MUST be set in production."
    )
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret or "dev-only-insecure-fallback",
    # Rename from Starlette's default "session" to avoid collision with
    # our own session_id cookie that the email+password flow sets.
    session_cookie="oauth_state",
    same_site="lax",
    https_only=os.getenv("COOKIE_SECURE", "false").lower() == "true",
)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    # Credentials ON: required for the session-cookie auth flow.
    # The browser only attaches the session cookie on cross-origin
    # requests when (a) the frontend uses `credentials: 'include'`
    # AND (b) the backend reflects the exact Origin back in
    # Access-Control-Allow-Origin. The latter rules out "*" in
    # CORS_ORIGINS - set it to the frontend origin explicitly
    # (e.g. http://localhost:3000).
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    expose_headers=["X-Request-ID"],
)

exception_handlers.install(app)

Instrumentator(
    excluded_handlers=["/health", "/metrics", "/stats"],
).instrument(app).expose(app)

app.include_router(health_router)
app.include_router(infer_router)
app.include_router(explain_router)
app.include_router(feedback_router)
app.include_router(stats_router)
app.include_router(monitoring_router)
app.include_router(users_router)
app.include_router(auth_router)
