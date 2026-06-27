"""FastAPI dependencies — settings and database session injection."""

from __future__ import annotations

from app.config import Settings, settings
from app.database import get_db

__all__ = ["get_settings", "get_db", "settings"]


def get_settings() -> Settings:
    """Dependency that returns the application settings singleton."""
    return settings
