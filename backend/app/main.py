"""FastAPI application entry point."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.storage.database import DatabaseManager

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _start_time
    _start_time = time.time()
    settings = get_settings()

    logger.info("SteerOps starting up...")
    logger.info(f"   Model: {settings.model_name}")
    logger.info(f"   Device: {settings.device}")
    logger.info(f"   Debug: {settings.debug}")

    db = DatabaseManager(settings.database_path)
    await db.initialize()
    app.state.db = db
    logger.info("Database initialised")

    app.state.model_manager = None
    app.state.start_time = _start_time

    logger.info("SteerOps ready")

    yield

    logger.info("SteerOps shutting down...")
    await db.close()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-Grade Activation-Level LLM Debugger — "
            "Mathematically detect and fix problematic model layers."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if settings.deploy_mode == "production":
        from app.middleware import SessionLockMiddleware, RateLimitMiddleware

        app.add_middleware(RateLimitMiddleware, gpu_limit=30, light_limit=120)
        app.add_middleware(SessionLockMiddleware, timeout=300)
        logger.info("Production mode: session lock + rate limiting enabled")

    app.include_router(api_router)
    app.include_router(ws_router)

    return app


app = create_app()
