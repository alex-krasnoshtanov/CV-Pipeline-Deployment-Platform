"""Request ID middleware.

Ensures every incoming request carries a correlation ID that appears
in both the response headers and every log line emitted while
handling that request. This is the foundation for the distributed
tracing requirement (R15 — tracing must be implemented to identify
bottlenecks in the deployment pipeline).

If the client sets ``X-Request-ID``, the middleware reuses it;
otherwise a fresh UUID4 is generated.

Design rationale:
- Uses a contextvar so ``logger.info(...)`` can include the request ID
  without every log call having to pass it explicitly.
- The filter attached to the root logger injects ``request_id`` into
  the log record so log formatters can display it.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar

from fastapi import Request as _Request
from starlette.middleware.base import BaseHTTPMiddleware as _BaseHTTPMiddleware
from starlette.responses import Response as _Response

_REQUEST_ID_HEADER = "X-Request-ID"

# Context variable holding the current request ID for the active task.
# Set by the middleware on each incoming request; read by the logging
# filter so every log line is automatically annotated.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

__all__ = ["request_id_ctx", "RequestIDMiddleware", "RequestIDLogFilter"]


class RequestIDMiddleware(_BaseHTTPMiddleware):
    """Attach a correlation ID to every request and response."""

    async def dispatch(
        self,
        request: _Request,
        call_next,
    ) -> _Response:
        """Assign or reuse a request ID, then forward the call.

        Args:
            request: The incoming Starlette request.
            call_next: Handler for the next middleware / route.

        Returns:
            The downstream response with the ``X-Request-ID`` header set.
        """
        incoming = request.headers.get(_REQUEST_ID_HEADER)
        req_id = incoming or str(uuid.uuid4())
        token = request_id_ctx.set(req_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers[_REQUEST_ID_HEADER] = req_id
        return response


class RequestIDLogFilter(logging.Filter):
    """Logging filter that adds ``request_id`` to every log record.

    Attach this to the root logger so any module can use
    ``%(request_id)s`` in its log format without manual plumbing.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        """Inject the current request ID into the log record."""
        record.request_id = request_id_ctx.get()
        return True
