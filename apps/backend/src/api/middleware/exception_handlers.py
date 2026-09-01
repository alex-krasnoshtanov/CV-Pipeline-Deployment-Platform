"""Global exception handlers for the backend API.

Ensures every unhandled exception reaches the client as a structured
error body matching specification section 6, rather than FastAPI's default
``{"detail": "Internal Server Error"}``.

Copilot instruction compliance: "Endpoints must handle missing or
malformed input gracefully with appropriate HTTP status codes."
A bare ``except`` inside the route handler is brittle; a global
handler catches anything that slips through and produces a consistent
wire format.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from cv_pipeline import __version__ as _pipeline_version
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.middleware.request_id import request_id_ctx

logger = logging.getLogger(__name__)


def _error_body(error_code: str, message: str) -> dict:
    """Build an error body matching CV Pipeline Spec section 6."""
    return {
        "error_code": error_code,
        "message": message,
        "pipeline_version": _pipeline_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id_ctx.get(),
    }


async def http_exception_handler(
    _request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Format raised ``HTTPException`` uniformly.

    If the raised exception already carries a structured detail
    (``dict`` with ``error_code`` / ``message``), preserve it under the
    ``detail`` key and mirror the key fields at the top level for easier
    client handling. Otherwise wrap the string detail in the standard
    envelope.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "error_code" in detail:
        error_code = detail.get("error_code", "HTTP_ERROR")
        message = detail.get("message", str(detail))
        body = {
            "detail": detail,
            "error_code": error_code,
            "message": message,
            "pipeline_version": _pipeline_version,
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id_ctx.get(),
        }
    else:
        message = str(detail)
        body = {
            "detail": message,
            "error_code": "HTTP_ERROR",
            "message": message,
            "pipeline_version": _pipeline_version,
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id_ctx.get(),
        }

    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=exc.headers or {},
    )


async def validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Convert Pydantic / form validation errors to the standard envelope."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body(
            error_code="VALIDATION_ERROR",
            message=f"Request failed validation: {exc.errors()}",
        ),
    )


async def unhandled_exception_handler(
    _request: Request,
    exc: Exception,
) -> JSONResponse:
    """Final backstop — log the traceback, return a structured 500."""
    logger.exception("Unhandled exception reached the backstop handler: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. See server logs for details.",
        ),
    )


def install(app: FastAPI) -> None:
    """Register all handlers on the application instance.

    Args:
        app: The FastAPI app to install handlers onto.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
