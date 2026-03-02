"""Remote model endpoints (HuggingFace Hub inference without download)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.core.constants import GENERATE_TIMEOUT
from app.core.remote_model import RemoteModelManager

router = APIRouter()


@router.post("/api/v1/models/remote-connect", tags=["Remote"])
async def remote_connect_model(request: Request):
    """Connect to a remote model via HuggingFace Hub (no download)."""
    body = await request.json()
    model_name = body.get("model_name", "").strip()
    hf_token = body.get("hf_token", None)

    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    rmm = RemoteModelManager.get_instance()
    try:
        info = await asyncio.to_thread(rmm.connect, model_name, hf_token)
        return {"status": "connected", "model": info}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/models/remote-disconnect", tags=["Remote"])
async def remote_disconnect():
    """Disconnect from remote model."""
    rmm = RemoteModelManager.get_instance()
    rmm.disconnect()
    return {"status": "disconnected"}


@router.post("/api/v1/remote/scan", tags=["Remote"])
async def remote_scan():
    """Scan a remote model using config metadata (no weights)."""
    rmm = RemoteModelManager.get_instance()
    if not rmm.loaded:
        raise HTTPException(status_code=400, detail="No remote model connected")

    try:
        result = await asyncio.to_thread(rmm.remote_scan)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/remote/generate", tags=["Remote"])
async def remote_generate(request: Request):
    """Generate text using HuggingFace Inference API."""
    rmm = RemoteModelManager.get_instance()
    if not rmm.loaded:
        raise HTTPException(status_code=400, detail="No remote model connected")

    body = await request.json()
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 200)
    temperature = body.get("temperature", 0.7)

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(rmm.generate, prompt, max_tokens, temperature),
            timeout=GENERATE_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Remote generation timed out")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/remote/activations", tags=["Remote"])
async def remote_activations(request: Request):
    """Simulated activations for a prompt (heuristic-based)."""
    rmm = RemoteModelManager.get_instance()
    if not rmm.loaded:
        raise HTTPException(status_code=400, detail="No remote model connected")

    body = await request.json()
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    activations = await asyncio.to_thread(rmm.simulate_activations, prompt)
    return {"activations": activations, "remote": True, "estimated": True}
