"""ASGI middleware — request ID injection and timing logging."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds a request ID and logs request/response timing.

    Injects X-Request-ID header if not present.
    Logs: method, path, status_code, duration_ms, request_id.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or propagate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind request context for structured logging
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start = time.monotonic()

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error(
                "Request failed",
                duration_ms=round(duration_ms, 2),
                client_ip=request.client.host if request.client else "unknown",
            )
            structlog.contextvars.clear_contextvars()
            raise

        duration_ms = (time.monotonic() - start) * 1000

        # Inject request ID into response headers
        response.headers["X-Request-ID"] = request_id

        # Log at appropriate level based on status
        log_fn = logger.info
        if response.status_code >= 500:
            log_fn = logger.error
        elif response.status_code >= 400:
            log_fn = logger.warning

        log_fn(
            "Request completed",
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            client_ip=request.client.host if request.client else "unknown",
        )

        structlog.contextvars.clear_contextvars()
        return response
