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

    # Start background thread
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
        # Reset state for next load
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




@router.post("/api/v1/scan", response_model=ScanResponse, tags=["Analysis"])
async def scan_model(body: ScanRequest, request: Request):
    """
    Mathematically profile all layers of the loaded model.
    Results are cached in SQLite â€“ subsequent calls return instantly.
    """
    mm = _require_model_loaded()
    db = request.app.state.db

    scanner = LayerScanner(mm)
    current_hash = scanner.get_scan_hash()

    # Check cache
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

    # Perform scan (CPU-bound â†’ thread pool with timeout)
    t0 = time.perf_counter()
    try:
        profiles = await asyncio.wait_for(
            asyncio.to_thread(scanner.scan),
            timeout=_SCAN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Scan timed out after {_SCAN_TIMEOUT}s")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Save to SQLite
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


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  4. ANALYZE (Prompt + Expected Response â†’ Layers)           â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
        mm = _require_model_loaded()
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
                _interpreter.interpret,
                text=analysis_text,
                prompt=effective_prompt,
            ),
            timeout=_ANALYZE_TIMEOUT,
        )

        # Step 2: Resolve to layers
        resolver = LayerResolver(layer_mappings=mappings, model_name=mm.model_name)
        resolved = resolver.resolve(interpretation)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Cache the analysis behavior AND pre-compute direction vectors
        # so export can retrieve them instantly without re-running CAA.
        behavior_text = body.behavior_description or body.expected_response or ""
        _last_analysis.clear()
        _last_analysis["behavior"] = behavior_text
        _last_analysis["layers"] = {r.layer_index: r.category for r in resolved}
        _last_analysis["timestamp"] = time.time()
        _last_analysis["vectors"] = {}  # layer_index â†’ direction_vector (list)

        # Pre-compute direction vectors for each detected layer
        concept = _match_behavior_to_concept(behavior_text)
        logger.info(
            f"Pre-computing direction vectors for {len(resolved)} layers "
            f"(behavior='{behavior_text}', concept='{concept}')"
        )
        for r in resolved:
            try:
                vec_result = _vector_calc.compute_vector(
                    mm.model, mm.tokenizer, concept, r.layer_index
                )
                dv = vec_result.get("direction_vector")
                if dv is not None:
                    _last_analysis["vectors"][r.layer_index] = dv
                    logger.info(
                        f"  âœ… Layer {r.layer_index}: direction vector computed "
                        f"(dim={len(dv) if isinstance(dv, list) else 'tensor'})"
                    )
            except Exception as e:
                logger.warning(f"  âš  Layer {r.layer_index}: vector computation failed: {e}")

        logger.info(
            f"Cached analysis: '{behavior_text}' â†’ {len(_last_analysis['vectors'])} "
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




@router.post("/api/v1/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(body: GenerateRequest, request: Request):
    """Generate text with optional steering intervention."""
    mm = _require_model_loaded()
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
            timeout=_GENERATE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        engine.clear_interventions()
        raise HTTPException(status_code=504, detail=f"Generation timed out after {_GENERATE_TIMEOUT}s")
    # NOTE: Do NOT clear interventions here â€” they must persist
    # so the export endpoint can pull direction vectors from active hooks.

    _monitor.update_generation_metrics(
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


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  6. CAPTURE ACTIVATIONS (for heatmap)                       â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@router.post(
    "/api/v1/activations",
    response_model=ActivationsResponse,
    tags=["Visualization"],
)
async def capture_activations(body: ActivationsRequest, request: Request):
    """Capture per-layer activation values for the heatmap display."""
    mm = _require_model_loaded()
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


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  7. PATCH EXPORT                                            â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@router.post(
    "/api/v1/patches/export",
    response_model=PatchExportResponse,
    tags=["Patches"],
)
async def export_patch(body: PatchExportRequest, request: Request):
    """Export a validated intervention as a deployable JSON configuration."""
    mm = ModelManager.get_instance()
    db = request.app.state.db
    settings = get_settings()

    patch_id = str(uuid4())

    # â”€â”€ Auto-compute direction vectors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Map layer categories â†’ best CAA concept for vector computation
    CATEGORY_TO_CONCEPT = {
        "style_personality": "politeness",
        "safety_alignment": "toxicity",
        "reasoning_planning": "refusal",
        "entity_semantic": "creativity",
        "information_integration": "verbosity",
        "knowledge_retrieval": "refusal",
        "output_distribution": "verbosity",
        "syntactic_processing": "creativity",
        "positional_morphological": "creativity",
        "token_embedding": "verbosity",
    }

    # Get cached layer mappings to lookup each layer's category
    layer_category_map = {}
    if mm.loaded:
        mappings = await db.get_layer_mappings(mm.model_name)
        layer_category_map = {m["layer_index"]: m["category"] for m in mappings}

    interventions = []
    for i in body.interventions:
        direction_vector = i.direction_vector

        if direction_vector is None:
            # â”€â”€ Strategy 0: Pull from analyze cache (INSTANT) â”€â”€
            # The analyze endpoint pre-computes vectors for all
            # recommended layers. This is the primary path.
            cached_vectors = _last_analysis.get("vectors", {})
            if i.layer in cached_vectors:
                direction_vector = cached_vectors[i.layer]
                logger.info(
                    f"Direction vector for layer {i.layer} "
                    f"retrieved from analyze cache"
                )

        if direction_vector is None:
            # â”€â”€ Strategy 1: Pull from active steering hooks â”€â”€â”€â”€
            # If the user steered with a custom vector, grab it.
            try:
                engine = SteeringEngine.get_instance(mm)
                for hook in engine._hooks.values():
                    if (
                        hook.layer_idx == i.layer
                        and hook.direction_vector is not None
                    ):
                        import torch as _torch
                        dv = hook.direction_vector
                        if isinstance(dv, _torch.Tensor):
                            dv = dv.cpu().float().tolist()
                        direction_vector = dv
                        logger.info(
                            f"Direction vector for layer {i.layer} "
                            f"pulled from active hook"
                        )
                        break
            except Exception as e:
                logger.debug(f"Could not pull vector from hooks: {e}")

        # â”€â”€ Strategy 2: CAA auto-compute (LAST RESORT) â”€â”€â”€â”€â”€â”€â”€â”€
        # Only runs if analyze was never called and no hooks exist.
        if direction_vector is None and mm.loaded and mm.model is not None:
            # Use cached behavior from last analyze call for concept matching
            behavior = _last_analysis.get("behavior", "")
            if behavior:
                concept = _match_behavior_to_concept(behavior)
            else:
                # Fallback to category-based mapping if no analyze was done
                category = layer_category_map.get(i.layer, "unknown")
                concept = CATEGORY_TO_CONCEPT.get(category, "politeness")
            try:
                result = _vector_calc.compute_vector(
                    mm.model, mm.tokenizer, concept, i.layer
                )
                direction_vector = result.get("direction_vector")
                logger.info(
                    f"Auto-computed CAA direction vector for layer {i.layer} "
                    f"(concept={concept}, behavior='{behavior}')"
                )
            except Exception as e:
                logger.warning(
                    f"Could not auto-compute vector for layer {i.layer}: {e}. "
                    f"Patch will be saved without direction vector for this layer."
                )

        interventions.append({
            "layer": i.layer,
            "strength": i.strength,
            "direction_vector": direction_vector,
            "notes": i.notes,
        })

    patch_data = {
        "metadata": {
            "name": body.patch_name,
            "version": "1.0",
            "model": mm.model_name if mm.loaded else "",
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "creator": "SteerOps",
            "description": body.description,
        },
        "interventions": interventions,
        "validation": body.validation_data,
        "deployment_instructions": (
            "Load this JSON at model startup and register forward hooks "
            "for each intervention layer."
        ),
    }

    # Save to SQLite
    await db.save_patch(
        patch_id=patch_id,
        name=body.patch_name,
        model_name=mm.model_name if mm.loaded else "",
        description=body.description,
        patch_data=patch_data,
    )

    # Also write to filesystem
    os.makedirs(settings.patches_dir, exist_ok=True)
    patch_path = os.path.join(
        settings.patches_dir, f"patch_{body.patch_name}.json"
    )
    with open(patch_path, "w") as f:
        json.dump(patch_data, f, indent=2)

    file_size_kb = os.path.getsize(patch_path) / 1024

    return PatchExportResponse(
        patch_id=patch_id,
        download_url=f"/api/v1/patches/download/{patch_id}",
        file_size_kb=round(file_size_kb, 2),
        patch=patch_data,
    )


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  7b. CONCEPT STEERING VECTORS (CAA)                         â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

