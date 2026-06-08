"""FastAPI dependencies — auth, settings injection."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, settings

# --- Security ---

_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Verify the Bearer token against configured API keys.

    If no API keys are configured, authentication is skipped.
    """
    if not settings.auth_enabled:
        return

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Expected: Bearer <api_key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if token not in settings.api_keys_set:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Settings ---


def get_settings() -> Settings:
    """Dependency that returns the application settings singleton."""
    return settings
