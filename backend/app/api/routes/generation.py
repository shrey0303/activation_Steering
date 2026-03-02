"""Text generation and activation capture endpoints."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.core.constants import GENERATE_TIMEOUT
from app.core.engine import SteeringEngine
from app.core.loader import ModelManager
from app.api.routes._shared import require_model_loaded, monitor
from app.schemas import (
    ActivationsRequest,
    ActivationsResponse,
    GenerateRequest,
    GenerateResponse,
)

router = APIRouter()


@router.post("/api/v1/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(body: GenerateRequest, request: Request):
    """Generate text with optional steering intervention."""
    mm = require_model_loaded()
    engine = SteeringEngine.get_instance(mm)

    # Apply steering if configured
    if body.steering:
        import torch as _torch
        direction_vec = None
        if body.steering.direction_vector:
            direction_vec = _torch.tensor(
                body.steering.direction_vector, dtype=_torch.float32
            )
        engine.add_intervention(
            layer_idx=body.steering.layer,
            strength=body.steering.strength,
            direction_vector=direction_vec,
        )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                engine.generate,
                prompt=body.prompt,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
                top_p=body.top_p,
            ),
            timeout=GENERATE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        engine.clear_interventions()
        raise HTTPException(status_code=504, detail=f"Generation timed out after {GENERATE_TIMEOUT}s")
    finally:
        # NOTE: Save active vectors to shared state so export can use them,
        # then clear interventions to prevent leakage into the next request.
        from app.api.routes._shared import last_analysis
        last_analysis["hook_vectors"] = {}
        for hook_id, hook in engine._hooks.items():
            if hook.direction_vector is not None:
                dv = hook.direction_vector
                if hasattr(dv, "cpu"):
                    dv = dv.cpu().float().tolist()
                # Use layer index as key for patch export
                last_analysis["hook_vectors"][hook.layer_idx] = dv
        engine.clear_interventions()

    monitor.update_generation_metrics(
        result["latency_ms"], result["tokens_per_sec"]
    )

    return GenerateResponse(
        text=result["text"],
        prompt=result["prompt"],
        tokens_generated=result["tokens_generated"],
        latency_ms=result["latency_ms"],
        tokens_per_sec=result["tokens_per_sec"],
        steering_applied=result["steering_applied"],
        steering_overhead_ms=result["steering_overhead_ms"],
        metrics={"interventions": result.get("interventions", [])},
    )


@router.post(
    "/api/v1/activations",
    response_model=ActivationsResponse,
    tags=["Visualization"],
)
async def capture_activations(body: ActivationsRequest, request: Request):
    """Capture per-layer activation values for the heatmap display."""
    mm = require_model_loaded()
    engine = SteeringEngine.get_instance(mm)

    t0 = time.perf_counter()
    activations = await asyncio.to_thread(
        engine.capture_activations,
        prompt=body.prompt,
        layers=body.layers,
        aggregation=body.aggregation,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ActivationsResponse(
        activations=activations,
        prompt=body.prompt,
        num_layers=mm.num_layers,
        capture_time_ms=round(elapsed_ms, 1),
    )
