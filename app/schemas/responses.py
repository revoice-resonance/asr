"""Pydantic response models — OpenAI-compatible where applicable."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# --- Health ---


class HealthResponse(BaseModel):
    """Basic health check response."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """Readiness check response."""

    status: str
    model_loaded: bool
    gpu_available: bool
    queue_depth: int


class GPUInfoResponse(BaseModel):
    """GPU memory and utilization info."""

    device_name: str
    device_index: int
    total_memory_mb: int
    used_memory_mb: int
    free_memory_mb: int
    utilization_pct: float


# --- Transcription (OpenAI-compatible) ---


class TranscriptionSegment(BaseModel):
    """A single transcribed segment with timestamps."""

    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscriptionResponse(BaseModel):
    """Simple text-only transcription (response_format=json)."""

    text: str


class VerboseTranscriptionResponse(BaseModel):
    """Verbose transcription with segments and metadata (response_format=verbose_json)."""

    text: str
    language: str
    duration: float
    segments: list[TranscriptionSegment] = Field(default_factory=list)


# --- Error ---


class ErrorDetail(BaseModel):
    """Detailed error information."""

    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""

    error: ErrorDetail
