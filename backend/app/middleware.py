"""
Production concurrency guard for SteerOps.

Two middleware layers:
1. SessionLock — single-user GPU guard (lightweight endpoints exempt)
2. RateLimiter — per-IP request throttling

Makes single-tenant SteerOps safe for public demos without
requiring a full multi-tenant rewrite.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from loguru import logger


# Read-only/stateless endpoints that bypass session lock
EXEMPT_PATHS: Set[str] = {
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/health",
    "/api/v1/metrics",
    "/api/v1/models",
    "/api/v1/models/load-status",
    "/api/v1/patches",
    "/api/v1/vectors",
    "/api/v1/features",
}


# --- Session Lock ---

class SessionLockMiddleware(BaseHTTPMiddleware):
    """
    Single-user GPU guard. First user to hit a GPU endpoint acquires the lock;
    subsequent users get 503. Lock auto-expires after `timeout` seconds of inactivity.
    No Redis, no queues — simplest possible concurrency guard for single-GPU demos.
    """

    def __init__(self, app, timeout: int = 300):
        super().__init__(app)
        self._lock = asyncio.Lock()
        self._current_session: str | None = None
        self._last_activity: float = 0.0
        self._timeout = timeout

    def _get_session_id(self, request: Request) -> str:
        ip = request.client.host if request.client else "unknown"
        ua = request.headers.get("user-agent", "")[:64]
        return f"{ip}|{ua}"

    def _is_exempt(self, path: str) -> bool:
        if path in EXEMPT_PATHS:
            return True
        for exempt in EXEMPT_PATHS:
            if path.startswith(exempt + "/"):
                return True
        return False

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        if self._is_exempt(path):
            return await call_next(request)

        session_id = self._get_session_id(request)
        now = time.time()

        async with self._lock:
            if (
                self._current_session is not None
                and self._current_session != session_id
                and (now - self._last_activity) > self._timeout
            ):
                logger.info(
                    f"Session expired (idle {now - self._last_activity:.0f}s). "
                    f"Releasing lock."
                )
                self._current_session = None

            if (
                self._current_session is not None
                and self._current_session != session_id
            ):
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "SteerOps is currently in use by another session. "
                            "This is a single-GPU research tool — please try again "
                            "in a few minutes, or clone the repo to run locally."
                        ),
                        "retry_after_seconds": 60,
                    },
                    headers={"Retry-After": "60"},
                )

            self._current_session = session_id
            self._last_activity = now

        response = await call_next(request)

        async with self._lock:
            if self._current_session == session_id:
                self._last_activity = time.time()

        return response


# --- Rate Limiter ---

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    In-memory per-IP rate limiter using a sliding window.
    No external dependencies — suitable for single-instance deployments.
    """

    def __init__(
        self,
        app,
        gpu_limit: int = 30,
        light_limit: int = 120,
        window_seconds: int = 60,
    ):
        super().__init__(app)
        self._gpu_limit = gpu_limit
        self._light_limit = light_limit
        self._window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup(self, ip: str, now: float):
        cutoff = now - self._window
        self._requests[ip] = [
            t for t in self._requests[ip] if t > cutoff
        ]

    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        path = request.url.path
        now = time.time()

        self._cleanup(ip, now)

        is_light = path in EXEMPT_PATHS
        limit = self._light_limit if is_light else self._gpu_limit
        count = len(self._requests[ip])

        if count >= limit:
            logger.warning(f"Rate limit hit: {ip} ({count}/{limit})")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded ({limit} requests per "
                        f"{self._window}s). Please slow down."
                    ),
                },
                headers={"Retry-After": str(self._window)},
            )

        self._requests[ip].append(now)
        return await call_next(request)
