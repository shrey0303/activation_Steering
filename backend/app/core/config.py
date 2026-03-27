"""
Application configuration via Pydantic Settings.
Loads from env vars and .env file with dev defaults.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Global application settings, loaded once at startup."""

    app_name: str = "SteerOps"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "info"

    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "auto"
    quantization_enabled: bool = True
    quantization_bits: int = 4
    cache_dir: str = str(Path.home() / ".cache" / "huggingface")
    memory_limit_gb: float = 8.0

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:80",
        "http://localhost:3000",
        "http://localhost",
        "*",  # Override via STEEROPS_CORS_ORIGINS for production
    ]

    database_url: str = "sqlite+aiosqlite:///./steerops.db"
    database_path: str = str(
        Path(__file__).resolve().parent.parent / "steerops.db"
    )

    max_steering_strength: float = 10.0
    min_steering_strength: float = -10.0
    default_max_tokens: int = 200
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    patches_dir: str = str(
        Path(__file__).resolve().parent.parent / "patches"
    )
    logs_dir: str = str(
        Path(__file__).resolve().parent.parent / "logs"
    )

    ws_ping_interval: int = 30
    ws_max_connections: int = 10

    # "local" = no concurrency guards (dev use)
    # "production" = session lock + rate limiting
    deploy_mode: str = "local"

    model_config = {
        "env_prefix": "STEEROPS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
