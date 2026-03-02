"""Model loading, listing, and unloading endpoints."""

from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.core.config import get_settings
from app.core.engine import SteeringEngine
from app.core.loader import ModelManager
from app.api.routes._shared import require_model_loaded
from app.schemas import (
    LoadModelRequest,
    ModelInfo,
    ModelsResponse,
)

router = APIRouter()

# --- Background Model Loading State ---

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


@router.post("/api/v1/models/load", tags=["Models"])
async def load_model_endpoint(body: LoadModelRequest, request: Request):
    """Kick off model loading in background. Returns immediately."""
    global _load_state

    with _load_lock:
        if _load_state["status"] == "loading":
            return {
                "status": "loading",
                "message": f"Already loading {_load_state['model_name']}...",
            }

    thread = threading.Thread(
        target=_background_load,
        args=(body.model_name, body.device, body.quantize, body.quantization_bits, request.app.state),
        daemon=True,
    )
    thread.start()

    return {
        "status": "loading",
        "message": f"Loading {body.model_name} in background...",
    }


@router.get("/api/v1/models/load-status", tags=["Models"])
async def load_model_status():
    """Poll this endpoint to check model loading progress."""
    with _load_lock:
        state = dict(_load_state)

    if state["status"] == "done":
        mm = ModelManager.get_instance()
        state["model"] = {
            "id": mm.model_name.split("/")[-1].lower(),
            "name": mm.model_name,
            "loaded": True,
            "parameters": f"{mm.memory_mb / 500:.1f}B (est.)",
            "quantized": mm.quantized,
            "quantization_bits": mm.quantization_bits,
            "memory_mb": mm.memory_mb,
            "device": mm.device_name,
            "num_layers": mm.num_layers,
            "hidden_dim": mm.hidden_dim,
        }
        _load_state["status"] = "idle"

    return state


@router.post("/api/v1/models/unload", tags=["Models"])
async def unload_model():
    """Unload the currently loaded model to free memory."""
    mm = ModelManager.get_instance()
    if not mm.loaded:
        raise HTTPException(status_code=400, detail="No model is currently loaded")
    name = mm.model_name
    # Clean up steering hooks before unloading the model
    SteeringEngine.reset_instance()
    await asyncio.to_thread(mm.unload)
    return {"success": True, "message": f"Model {name} unloaded"}
