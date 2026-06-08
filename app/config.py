"""Application configuration via pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Whisper ASR API settings.

    Loaded from .env file and environment variables (env vars take precedence).
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1

    # --- Model ---
    model_path: str = "models/whisper-large-v3-turbo-ct2"
    model_compute_type: str = "float16"
    model_device: str = "cuda"
    model_device_index: int = 0

    # --- Model Download ---
    model_download_url: str = ""
    hf_model_id: str = ""

    # --- Audio Limits ---
    max_upload_bytes: int = 524_288_000  # 500 MB
    max_audio_duration: int = 600  # 10 minutes
    default_language: str = "zh"

    # --- VAD ---
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_duration_ms: int = 500

    # --- Rate Limiting ---
    rate_limit_rpm: int = 60  # requests per minute, 0 = disabled
    rate_limit_burst: int = 10

    # --- Authentication ---
    api_keys: str = ""  # comma-separated

    # --- Logging ---
    log_level: str = "info"
    log_format: str = "json"  # "json" or "console"

    # --- Temp Files ---
    temp_dir: str = "/tmp/whisper_api"

    # --- Proxy ---
    http_proxy: str = ""
    https_proxy: str = ""

    @property
    def api_keys_set(self) -> set[str]:
        """Parse comma-separated API keys into a set."""
        if not self.api_keys.strip():
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        """Whether API key authentication is enabled."""
        return len(self.api_keys_set) > 0

    @property
    def model_path_resolved(self) -> Path:
        """Resolve model path (expand ~ and make absolute)."""
        return Path(self.model_path).expanduser().resolve()

    @property
    def temp_dir_resolved(self) -> Path:
        """Resolve temp directory path."""
        return Path(self.temp_dir).expanduser().resolve()


# Singleton
settings = Settings()
