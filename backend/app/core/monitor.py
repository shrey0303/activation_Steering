"""
Performance monitor.

Tracks system metrics: memory, GPU utilisation, latencies.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import psutil
import torch
from loguru import logger


class PerformanceMonitor:
    """Lightweight system metrics collector."""

    def __init__(self) -> None:
        self._start_time = time.time()
        self._last_latency: float = 0.0
        self._last_tokens_per_sec: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def update_generation_metrics(
        self, latency_ms: float, tokens_per_sec: float
    ) -> None:
        self._last_latency = latency_ms
        self._last_tokens_per_sec = tokens_per_sec

    def get_metrics(self) -> Dict[str, Any]:
        """Return current system metrics."""
        memory = psutil.virtual_memory()

        metrics: Dict[str, Any] = {
            "uptime_seconds": round(self.uptime_seconds, 1),
            "cpu_percent": psutil.cpu_percent(interval=0),
            "ram_used_mb": round(memory.used / (1024 ** 2), 1),
            "ram_available_mb": round(memory.available / (1024 ** 2), 1),
            "ram_percent": memory.percent,
            "last_latency_ms": round(self._last_latency, 1),
            "last_tokens_per_sec": round(self._last_tokens_per_sec, 1),
        }

        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.mem_get_info()
                metrics["gpu_free_mb"] = round(gpu_mem[0] / (1024 ** 2), 1)
                metrics["gpu_total_mb"] = round(gpu_mem[1] / (1024 ** 2), 1)
                metrics["gpu_used_mb"] = round(
                    (gpu_mem[1] - gpu_mem[0]) / (1024 ** 2), 1
                )
            except Exception:
                pass

        return metrics

    def get_health_status(self, model_loaded: bool) -> str:
        """Return 'healthy', 'degraded', or 'unhealthy'.
        
        'healthy' = service is running and can accept requests.
        'degraded' = service is under memory pressure.
        'unhealthy' = critical memory exhaustion.
        
        Note: model_loaded is NOT a factor — the service itself is
        healthy even without a model. Use /ready for model readiness.
        """
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            return "unhealthy"
        if memory.percent > 85:
            return "degraded"
        return "healthy"
