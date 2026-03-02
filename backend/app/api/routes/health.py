"""Health and system metrics endpoints."""

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.loader import ModelManager
from app.api.routes._shared import monitor
from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health status."""
    settings = get_settings()
    mm = ModelManager.get_instance()
    metrics = monitor.get_metrics()

    return HealthResponse(
        status=monitor.get_health_status(mm.loaded),
        model_loaded=mm.loaded,
        device=mm.device_name if mm.loaded else settings.device,
        memory={
            "used_mb": metrics.get("ram_used_mb", 0),
            "available_mb": metrics.get("ram_available_mb", 0),
            "model_size_mb": mm.memory_mb if mm.loaded else 0,
        },
        uptime_seconds=monitor.uptime_seconds,
        version=settings.app_version,
    )


@router.get("/api/v1/metrics", tags=["System"])
async def get_metrics():
    """Current system performance metrics."""
    return monitor.get_metrics()
