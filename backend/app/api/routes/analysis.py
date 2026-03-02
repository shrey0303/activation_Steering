"""Scan and analyze endpoints."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.core.constants import SCAN_TIMEOUT, ANALYZE_TIMEOUT
from app.core.loader import ModelManager
from app.core.scanner import LayerScanner
from app.core.resolver import LayerResolver
from app.api.routes._shared import (
    require_model_loaded,
    interpreter,
    vector_calc,
    last_analysis,
    match_behavior_to_concept,
)
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    LayerAnalysis,
    LayerProfile,
    ScanRequest,
    ScanResponse,
)

router = APIRouter()


@router.post("/api/v1/scan", response_model=ScanResponse, tags=["Analysis"])
async def scan_model(body: ScanRequest, request: Request):
    """
    Mathematically profile all layers of the loaded model.
    Results are cached in SQLite – subsequent calls return instantly.
    """
    mm = require_model_loaded()
    db = request.app.state.db

    scanner = LayerScanner(mm)
    current_hash = scanner.get_scan_hash()

    if not body.force_rescan:
        profile = await db.get_model_profile(mm.model_name)
        if profile and profile.get("scan_hash") == current_hash:
            logger.info("Returning cached scan results")
            mappings = await db.get_layer_mappings(mm.model_name)
            return ScanResponse(
                model_name=mm.model_name,
                num_layers=mm.num_layers,
                hidden_dim=mm.hidden_dim,
                architecture=mm.architecture,
                layer_profiles=[
                    LayerProfile(
                        layer_index=m["layer_index"],
                        category=m["category"],
                        confidence=m["confidence"],
                        behavioral_role=m.get("behavioral_role", ""),
                        weight_stats=m.get("weight_stats", {}),
                        description=m.get("description", ""),
                    )
                    for m in mappings
                ],
                scan_time_ms=0.0,
                from_cache=True,
            )

    # Perform scan (CPU-bound → thread pool with timeout)
    t0 = time.perf_counter()
    try:
        profiles = await asyncio.wait_for(
            asyncio.to_thread(scanner.scan),
            timeout=SCAN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Scan timed out after {SCAN_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    profile_id = await db.save_model_profile(
        model_name=mm.model_name,
        architecture=mm.architecture,
        num_layers=mm.num_layers,
        hidden_dim=mm.hidden_dim,
        scan_hash=current_hash,
    )
    await db.save_layer_mappings(profile_id, profiles)

    return ScanResponse(
        model_name=mm.model_name,
        num_layers=mm.num_layers,
        hidden_dim=mm.hidden_dim,
        architecture=mm.architecture,
        layer_profiles=[
            LayerProfile(**{k: v for k, v in p.items()})
            for p in profiles
        ],
        scan_time_ms=round(elapsed_ms, 1),
        from_cache=False,
    )


@router.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(body: AnalyzeRequest, request: Request):
    """
    User provides a prompt and expected response, OR a behaviour description.

    Three modes:
    1. Response mode: prompt + expected_response (prompt required)
    2. Behavior-only mode: behavior_description alone (no prompt needed)
    3. Prompt + Behavior mode: both prompt and behavior_description
    """
    import traceback as tb

    try:
        mm = require_model_loaded()
        db = request.app.state.db

        # Validate: at least one analysis input is required
        has_response = bool(body.expected_response and body.expected_response.strip())
        has_behavior = bool(body.behavior_description and body.behavior_description.strip())
        has_prompt = bool(body.prompt and body.prompt.strip())

        if not has_response and not has_behavior:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'expected_response' or 'behavior_description'.",
            )

        # For behavior-only mode without prompt, use a generic context
        effective_prompt = body.prompt if has_prompt else ""

        # Ensure model is scanned
        mappings = await db.get_layer_mappings(mm.model_name)
        if not mappings:
            raise HTTPException(
                status_code=400,
                detail="Model not scanned yet. Call POST /api/v1/scan first.",
            )

        t0 = time.perf_counter()

        # Step 1: Interpret expected response or behavior
        # Use behavior_description if provided; otherwise treat expected_response as behavior text
        analysis_text = (
            body.behavior_description if has_behavior else body.expected_response
        )
        interpretation = await asyncio.wait_for(
            asyncio.to_thread(
                interpreter.interpret,
                text=analysis_text,
                prompt=effective_prompt,
            ),
            timeout=ANALYZE_TIMEOUT,
        )

        # Step 2: Resolve to layers
        resolver = LayerResolver(layer_mappings=mappings, model_name=mm.model_name)
        resolved = resolver.resolve(interpretation)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Cache the analysis behavior AND pre-compute direction vectors
        # so export can retrieve them instantly without re-running CAA.
        behavior_text = body.behavior_description or body.expected_response or ""
        last_analysis.clear()
        last_analysis["behavior"] = behavior_text
        last_analysis["layers"] = {r.layer_index: r.category for r in resolved}
        last_analysis["timestamp"] = time.time()
        last_analysis["vectors"] = {}  # layer_index → direction_vector (list)

        # Pre-compute direction vectors for each detected layer
        concept = match_behavior_to_concept(behavior_text)
        logger.info(
            f"Pre-computing direction vectors for {len(resolved)} layers "
            f"(behavior='{behavior_text}', concept='{concept}')"
        )
        for r in resolved:
            try:
                vec_result = vector_calc.compute_vector(
                    mm.model, mm.tokenizer, concept, r.layer_index
                )
                dv = vec_result.get("direction_vector")
                if dv is not None:
                    last_analysis["vectors"][r.layer_index] = dv
                    logger.info(
                        f"  Layer {r.layer_index}: direction vector computed "
                        f"(dim={len(dv) if isinstance(dv, list) else 'tensor'})"
                    )
            except Exception as e:
                logger.warning(f"  Layer {r.layer_index}: vector computation failed: {e}")

        logger.info(
            f"Cached analysis: '{behavior_text}' → {len(last_analysis['vectors'])} "
            f"vectors pre-computed"
        )

        # Build response
        detected_layers = [r.layer_index for r in resolved]
        detailed = {}
        for r in resolved:
            # Find corresponding mapping
            mapping = next(
                (m for m in mappings if m["layer_index"] == r.layer_index), {}
            )
            detailed[r.layer_index] = LayerAnalysis(
                layer_index=r.layer_index,
                anomaly_score=abs(r.recommended_strength) / 10.0,
                confidence=r.confidence,
                category=r.category,
                behavioral_role=mapping.get("behavioral_role", ""),
                explanation=r.reason,
                recommended_intervention={
                    "strength": r.recommended_strength,
                    "direction": r.direction,
                    "expected_outcome": (
                        f"{'Enhance' if r.direction > 0 else 'Suppress'} "
                        f"{r.category} behaviour"
                    ),
                },
                statistics=mapping.get("weight_stats", {}),
            )

        overall_conf = (
            sum(r.confidence for r in resolved) / len(resolved)
            if resolved else 0.0
        )

        return AnalyzeResponse(
            prompt=body.prompt,
            expected_response=body.expected_response,
            detected_layers=detected_layers,
            detailed_analysis=detailed,
            overall_confidence=round(overall_conf, 3),
            interpretation=interpretation.to_dict(),
            processing_time_ms=round(elapsed_ms, 1),
        )
    except HTTPException:
        raise  # Let FastAPI handle HTTP exceptions normally
    except Exception as e:
        logger.error(f"Analyze endpoint error: {e}\n{tb.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
