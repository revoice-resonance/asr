"""Health check endpoints — liveness, readiness, GPU metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.schemas.responses import GPUInfoResponse, HealthResponse, ReadyResponse

if TYPE_CHECKING:
    from app.services.transcriber import TranscriptionWorker

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Kubernetes liveness probe — always returns 200 if the process is alive."""
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness(request: Request) -> ReadyResponse:
    """Kubernetes readiness probe — checks model, GPU, and queue health.

    Returns 503 if the service is not ready to accept traffic.
    """
    worker: "TranscriptionWorker" = request.app.state.worker

    model_loaded = worker.is_ready
    gpu_available = (
        settings.model_device == "cpu" or
        (torch.cuda.is_available() and torch.cuda.device_count() > 0)
    )
    queue_depth = worker.queue_depth

    ready = model_loaded and gpu_available

    if not ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_loaded": model_loaded,
                "gpu_available": gpu_available,
                "queue_depth": queue_depth,
            },
        )

    return ReadyResponse(
        status="ready",
        model_loaded=model_loaded,
        gpu_available=gpu_available,
        queue_depth=queue_depth,
    )


@router.get("/health/gpu", response_model=GPUInfoResponse)
async def gpu_info() -> GPUInfoResponse:
    """GPU memory and utilization metrics."""
    if not torch.cuda.is_available():
        raise HTTPException(status_code=404, detail="No CUDA device available")

    device_index = settings.model_device_index
    if device_index >= torch.cuda.device_count():
        raise HTTPException(
            status_code=400,
            detail=f"Device index {device_index} out of range "
                    f"(available: {torch.cuda.device_count()})",
        )

    props = torch.cuda.get_device_properties(device_index)
    total_mem = props.total_memory // (1024 * 1024)
    used_mem = torch.cuda.memory_allocated(device_index) // (1024 * 1024)
    free_mem = total_mem - used_mem

    return GPUInfoResponse(
        device_name=props.name,
        device_index=device_index,
        total_memory_mb=total_mem,
        used_memory_mb=used_mem,
        free_memory_mb=free_mem,
        utilization_pct=round(used_mem / total_mem * 100, 1) if total_mem > 0 else 0.0,
    )

