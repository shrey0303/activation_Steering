"""Concept vector computation and evaluation endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.core.constants import GENERATE_TIMEOUT, COMPUTE_TIMEOUT
from app.core.engine import SteeringEngine
from app.core.loader import ModelManager
from app.api.routes._shared import vector_calc

router = APIRouter()


@router.get("/api/v1/concepts", tags=["Concepts"])
async def list_concepts():
    """List available concepts for CAA vector computation."""
    return {"concepts": vector_calc.available_concepts}


@router.post("/api/v1/concepts/compute", tags=["Concepts"])
async def compute_concept_vector(request: Request):
    """
    Compute a CAA steering vector for a concept at a specific layer.

    Body: {"concept": "politeness", "layer": 14}
    """
    mm = ModelManager.get_instance()
    if not mm.loaded or mm.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    body = await request.json()
    concept = body.get("concept")
    layer_idx = body.get("layer")

    if not concept or layer_idx is None:
        raise HTTPException(
            status_code=422,
            detail="Both 'concept' and 'layer' are required.",
        )

    cached = vector_calc.load_cached_vector(
        concept, layer_idx, mm.model_name
    )
    if cached:
        cached["from_cache"] = True
        return cached

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                vector_calc.compute_vector,
                model=mm.model,
                tokenizer=mm.tokenizer,
                concept=concept,
                layer_idx=layer_idx,
            ),
            timeout=COMPUTE_TIMEOUT,
        )
        vector_calc.cache_vector(
            concept, layer_idx, mm.model_name, result
        )
        result["from_cache"] = False
        return result
    except Exception as e:
        logger.error(f"CAA computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/evaluate", tags=["Evaluation"])
async def evaluate_steering(request: Request):
    """
    Compare model outputs with and without steering.

    Body: {
        "test_prompts": ["Tell me about climate", "How to cook pasta"],
        "steering": [{"layer": 14, "strength": 3.0, "direction_vector": [...]}],
        "max_tokens": 100
    }
    """
    from app.core.evaluator import Evaluator

    mm = ModelManager.get_instance()
    if not mm.loaded or mm.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    body = await request.json()
    test_prompts = body.get("test_prompts", [])
    steering = body.get("steering", [])
    max_tokens = body.get("max_tokens", 100)

    if not test_prompts:
        raise HTTPException(
            status_code=422, detail="'test_prompts' list is required."
        )
    if not steering:
        raise HTTPException(
            status_code=422, detail="'steering' configs are required."
        )

    evaluator = Evaluator()
    engine = SteeringEngine.get_instance(mm)

    target_concept = body.get("target_concept")  # Optional: "politeness", "creativity", etc.

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                evaluator.evaluate,
                model=mm.model,
                tokenizer=mm.tokenizer,
                engine=engine,
                test_prompts=test_prompts[:5],  # cap at 5 for speed
                steering_configs=steering,
                max_tokens=max_tokens,
                target_concept=target_concept,
            ),
            timeout=GENERATE_TIMEOUT * 2,  # evaluations run multiple generations
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Evaluation timed out")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
