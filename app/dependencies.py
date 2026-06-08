"""FastAPI dependencies — settings injection."""

from __future__ import annotations

from app.config import Settings, settings


# --- Settings ---


def get_settings() -> Settings:
    """Dependency that returns the application settings singleton."""
    return settings
