"""Application configuration via pydantic-settings.

All values can be overridden via environment variables or a .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Whisper ASR API settings.

    Loaded from .env file and environment variables (env vars take precedence).
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: str = ""  # comma-separated, e.g. "https://app.example.com,https://admin.example.com"

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

    # --- Logging ---
    log_level: str = "info"
    log_format: str = "json"  # "json" or "console"

    # --- Temp Files ---
    temp_dir: str = "/tmp/whisper_api"

    # --- Database (empty = stateless mode for backward compatibility) ---
    database_url: str = ""
    database_pool_size: int = 5
    database_pool_overflow: int = 10

    # --- File Storage ---
    storage_path: str = "/data/asr_storage"

    # --- Task Scheduler ---
    task_poll_interval: int = 5  # seconds between DB polls for PENDING tasks
    task_max_retries: int = 3    # max retries for failed tasks before giving up
    max_concurrent_tasks: int = 2  # max in-flight DB tasks processed concurrently

    # --- Callback ---
    main_backend_callback_url: str = ""  # e.g. http://main:8080/api/v1/callback/asr
    callback_max_retries: int = 5
    callback_retry_base_delay: float = 1.0  # exponential backoff base (seconds)

    # --- Proxy ---
    http_proxy: str = ""
    https_proxy: str = ""

    # --- Validators ---

    @field_validator("model_device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid = {"cpu", "cuda", "auto"}
        if v not in valid:
            raise ValueError(f"model_device must be one of {valid}, got '{v}'")
        return v

    @field_validator("model_compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        valid = {
            "default", "auto", "int8", "int8_float16", "int8_float32",
            "int8_bfloat16", "int16", "float16", "float32", "bfloat16",
        }
        if v not in valid:
            raise ValueError(f"model_compute_type must be one of {valid}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"log_level must be one of {valid}, got '{v}'")
        return v.lower()

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        valid = {"json", "console"}
        if v not in valid:
            raise ValueError(f"log_format must be one of {valid}, got '{v}'")
        return v

    # --- Properties ---

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list.

        Returns ["*"] if no origins are configured (allow all, credentials disabled).
        """
        if not self.cors_origins.strip():
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def model_path_resolved(self) -> Path:
        """Resolve model path (expand ~ and make absolute)."""
        return Path(self.model_path).expanduser().resolve()

    @property
    def temp_dir_resolved(self) -> Path:
        """Resolve temp directory path."""
        return Path(self.temp_dir).expanduser().resolve()

    @property
    def storage_path_resolved(self) -> Path:
        """Resolve permanent storage directory path."""
        return Path(self.storage_path).expanduser().resolve()

    @property
    def database_enabled(self) -> bool:
        """Whether database integration is configured."""
        return bool(self.database_url.strip())


# Singleton
settings = Settings()
