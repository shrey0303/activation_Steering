"""
REST API endpoints.

All routes are prefixed with the router's prefix set here.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from typing import Any, Dict
from uuid import uuid4

# Timeout constants for blocking operations (seconds)
_SCAN_TIMEOUT = 600
_GENERATE_TIMEOUT = 120
_ANALYZE_TIMEOUT = 120
_COMPUTE_TIMEOUT = 120

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import get_settings
from app.core.engine import SteeringEngine
from app.core.interpreter import ResponseInterpreter
from app.core.loader import ModelManager
from app.core.monitor import PerformanceMonitor
from app.core.resolver import LayerResolver
from app.core.scanner import LayerScanner
from app.core.vector_calculator import VectorCalculator
from app.core.feature_extractor import FeatureExtractor, FeatureDictionary
from app.schemas import (
    ActivationsRequest,
    ActivationsResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LayerAnalysis,
    LayerProfile,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    ModelsResponse,
    PatchExportRequest,
    PatchExportResponse,
    PatchListResponse,
    PatchMetadata,
    ScanRequest,
    ScanResponse,
)

router = APIRouter()
_monitor = PerformanceMonitor()
_interpreter = ResponseInterpreter()
_vector_calc = VectorCalculator(max_prompts=20)
_feature_dict: FeatureDictionary | None = None

# â”€â”€ Analysis cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stores the last behavior description from analyze so export can
# pick the semantically closest CAA concept instead of guessing.
_last_analysis: Dict[str, Any] = {}
_best_concept_cache: Dict[str, str] = {}


def _match_behavior_to_concept(behavior: str) -> str:
    """
    Find the closest CAA concept for a behavior description.
    Uses semantic keyword matching against concept descriptions
    from contrastive_pairs.json.

    Falls back to 'politeness' if nothing matches.
    """
    if behavior in _best_concept_cache:
        return _best_concept_cache[behavior]

    behavior_lower = behavior.lower()

    # Direct keyword mapping â€” fast path
    KEYWORD_MAP = {
        "polite": "politeness", "rude": "politeness", "courteous": "politeness",
        "honest": "politeness", "kind": "politeness", "respectful": "politeness",
        "helpful": "politeness", "friendly": "politeness",
        "toxic": "toxicity", "harmful": "toxicity", "offensive": "toxicity",
        "hateful": "toxicity", "angry": "toxicity", "aggressive": "toxicity",
        "safe": "toxicity", "unsafe": "toxicity",
        "creative": "creativity", "imaginative": "creativity",
        "artistic": "creativity", "inventive": "creativity", "original": "creativity",
        "verbose": "verbosity", "concise": "verbosity", "brief": "verbosity",
        "detailed": "verbosity", "short": "verbosity", "long": "verbosity",
        "refuse": "refusal", "reject": "refusal", "decline": "refusal",
        "comply": "refusal", "obey": "refusal", "obedient": "refusal",
    }

    for keyword, concept in KEYWORD_MAP.items():
        if keyword in behavior_lower:
            _best_concept_cache[behavior] = concept
            logger.info(f"Matched behavior '{behavior}' â†’ concept '{concept}' (keyword: {keyword})")
            return concept

    # Fallback: try available concepts from VectorCalculator
    try:
        concepts = _vector_calc.available_concepts
        for c in concepts:
            if c["id"] in behavior_lower or behavior_lower in c.get("description", "").lower():
                _best_concept_cache[behavior] = c["id"]
                return c["id"]
    except Exception:
        pass

    # Ultimate fallback
    _best_concept_cache[behavior] = "politeness"
    return "politeness"



def _get_model_manager(request: Request) -> ModelManager:
    """Return the model manager instance. Does NOT auto-load any model."""
    mm = ModelManager.get_instance()
    return mm


def _require_model_loaded() -> ModelManager:
    """Return the model manager, raising 400 if no model is loaded."""
    mm = ModelManager.get_instance()
    if not mm.loaded:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Load a model first via the Control Panel.",
        )
    return mm



@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """System health status."""
    settings = get_settings()
    mm = ModelManager.get_instance()
    metrics = _monitor.get_metrics()

    return HealthResponse(
        status=_monitor.get_health_status(mm.loaded),
        model_loaded=mm.loaded,
        device=mm.device_name if mm.loaded else settings.device,
        memory={
            "used_mb": metrics.get("ram_used_mb", 0),
            "available_mb": metrics.get("ram_available_mb", 0),
            "model_size_mb": mm.memory_mb if mm.loaded else 0,
        },
        uptime_seconds=_monitor.uptime_seconds,
        version=settings.app_version,
    )



@router.get("/api/v1/models", response_model=ModelsResponse, tags=["Models"])
async def list_models(request: Request):
    """List available / loaded models."""
    mm = ModelManager.get_instance()
    settings = get_settings()

    # Check if model has been scanned
    db = request.app.state.db
    scanned = False
    if mm.loaded:
        profile = await db.get_model_profile(mm.model_name)
        scanned = profile is not None

    models = []
    if mm.loaded:
        models.append(
            ModelInfo(
                id=mm.model_name.split("/")[-1].lower(),
                name=mm.model_name,
                loaded=True,
                parameters=f"{mm.memory_mb / 500:.1f}B (est.)",
                quantized=mm.quantized,
                quantization_bits=mm.quantization_bits,
                memory_mb=mm.memory_mb,
                device=mm.device_name,
                num_layers=mm.num_layers,
                hidden_dim=mm.hidden_dim,
                scanned=scanned,
            )
        )
    else:
        models.append(
            ModelInfo(
                id=settings.model_name.split("/")[-1].lower(),
                name=settings.model_name,
                loaded=False,
            )
        )

    return ModelsResponse(
        models=models,
        active_model=mm.model_name if mm.loaded else None,
    )




import threading

# Module-level loading state
_load_state = {
    "status": "idle",   # idle | loading | done | error
    "message": "",
    "model_name": "",
    "progress": "",
    "error": None,
}
_load_lock = threading.Lock()


def _background_load(model_name: str, device: str, quantize: bool, quantization_bits: int, app_state):
    """Run model loading in a background thread to avoid HTTP timeout."""
    global _load_state
    try:
        with _load_lock:
            _load_state["status"] = "loading"
            _load_state["model_name"] = model_name
            _load_state["progress"] = "Downloading model from HuggingFace..."
            _load_state["error"] = None

        mm = ModelManager.get_instance()
        mm.load(
            model_name=model_name,
            device_preference=device,
            quantize=quantize,
            quantization_bits=quantization_bits,
        )
        app_state.model_manager = mm

        with _load_lock:
            _load_state["status"] = "done"
            _load_state["progress"] = "Model loaded successfully"
            _load_state["message"] = f"Model {model_name} loaded"

    except Exception as e:
        logger.error(f"Background model load failed: {e}")
        with _load_lock:
            _load_state["status"] = "error"
            _load_state["error"] = str(e)
            _load_state["progress"] = ""


