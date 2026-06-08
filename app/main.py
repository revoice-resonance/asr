"""Whisper ASR API — FastAPI application entry point.

Usage:
    python -m app.main
    uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.middleware import RequestLoggingMiddleware
from app.routes.health import router as health_router
from app.routes.transcription import limiter, router as transcription_router
from app.services.transcriber import TranscriptionWorker


# --- Logging Setup ---


def setup_logging() -> None:
    """Configure structlog for structured JSON logging in production
    or colored console output in development.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    if settings.log_format == "console":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            timestamper,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Silence noisy libraries
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("ctranslate2").setLevel(logging.WARNING)


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model on startup, cleanup on shutdown."""
    logger = structlog.get_logger(__name__)

    logger.info("Starting Whisper ASR API server")

    # Verify model path exists
    model_path = settings.model_path_resolved
    if not model_path.exists():
        logger.error(
            "Model path not found",
            path=str(model_path),
            hint="Set MODEL_PATH env var or run deploy.sh to download the model",
        )
        sys.exit(1)

    # Initialize and start transcription worker
    worker = TranscriptionWorker()
    try:
        await worker.start()
    except Exception:
        logger.exception("Failed to load model")
        sys.exit(1)

    # Store worker on app state for route access
    app.state.worker = worker

    logger.info(
        "Server ready",
        host=settings.host,
        port=settings.port,
        model_path=str(model_path),
        device=settings.model_device,
        auth_enabled=settings.auth_enabled,
        rate_limit=f"{settings.rate_limit_rpm}/min" if settings.rate_limit_rpm > 0 else "disabled",
    )

    yield

    # Shutdown
    logger.info("Shutting down server")
    await worker.stop()
    logger.info("Server stopped")


# --- App Factory ---


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    setup_logging()

    app = FastAPI(
        title="Whisper ASR API",
        description="Production-grade speech-to-text API powered by faster-whisper",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.log_format == "console" else None,
        redoc_url=None,
    )

    # --- Middleware ---
    # Order matters: last added = outermost (executed first)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging (innermost — wraps the actual handler)
    app.add_middleware(RequestLoggingMiddleware)

    # --- Rate Limiter ---
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # --- Routes ---
    app.include_router(health_router)
    app.include_router(transcription_router)

    # --- Global Exception Handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all exception handler for unhandled errors."""
        logger = structlog.get_logger(__name__)
        logger.exception(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal server error",
                    "type": "internal_error",
                    "code": None,
                    "param": None,
                }
            },
        )

    return app


# --- Module-level app instance (for uvicorn) ---

app = create_app()


# --- Direct run ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
        workers=settings.workers,
    )
