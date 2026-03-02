"""Patch export, listing, download endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from loguru import logger

from app.core.config import get_settings
from app.core.constants import CATEGORY_TO_CONCEPT
from app.core.engine import SteeringEngine
from app.core.loader import ModelManager
from app.api.routes._shared import (
    last_analysis,
    match_behavior_to_concept,
    vector_calc,
)
from app.schemas import (
    PatchExportRequest,
    PatchExportResponse,
    PatchListResponse,
    PatchMetadata,
)

router = APIRouter()


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

    # Get cached layer mappings to lookup each layer's category
    layer_category_map = {}
    if mm.loaded:
        mappings = await db.get_layer_mappings(mm.model_name)
        layer_category_map = {m["layer_index"]: m["category"] for m in mappings}

    interventions = []
    for i in body.interventions:
        direction_vector = i.direction_vector

        if direction_vector is None:
            # --- Strategy 0: Pull from analyze cache ---
            # The analyze endpoint pre-computes vectors for all
            # recommended layers. This is the primary path.
            cached_vectors = last_analysis.get("vectors", {})
            if i.layer in cached_vectors:
                direction_vector = cached_vectors[i.layer]
                logger.info(
                    f"Direction vector for layer {i.layer} "
                    f"retrieved from analyze cache"
                )

        if direction_vector is None:
            # --- Strategy 1a: Pull from cached hook vectors ---
            # generation.py saves vectors here before clearing hooks.
            cached_hook_vectors = last_analysis.get("hook_vectors", {})
            if i.layer in cached_hook_vectors:
                direction_vector = cached_hook_vectors[i.layer]
                logger.info(
                    f"Direction vector for layer {i.layer} "
                    f"pulled from generation hook cache"
                )

        if direction_vector is None:
            # --- Strategy 1b: Fallback to live hooks ---
            # Covers edge cases where hooks are registered outside /generate.
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

        # --- Strategy 2: CAA auto-compute ---
        # Only runs if analyze was never called and no hooks exist.
        if direction_vector is None and mm.loaded and mm.model is not None:
            # Use cached behavior from last analyze call for concept matching
            behavior = last_analysis.get("behavior", "")
            if behavior:
                concept = match_behavior_to_concept(behavior)
            else:
                # Fallback to category-based mapping if no analyze was done
                category = layer_category_map.get(i.layer, "unknown")
                concept = CATEGORY_TO_CONCEPT.get(category, "politeness")
            try:
                result = vector_calc.compute_vector(
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


@router.get("/api/v1/patches", response_model=PatchListResponse, tags=["Patches"])
async def list_patches(request: Request):
    """List all saved patches."""
    db = request.app.state.db
    patches = await db.get_patches()
    return PatchListResponse(
        patches=[
            PatchMetadata(
                id=p["id"],
                name=p["name"],
                model=p.get("model_name", ""),
                description=p.get("description", ""),
                created=p.get("created_at", ""),
                file_size_kb=p.get("file_size_kb", 0),
            )
            for p in patches
        ],
        total=len(patches),
    )


@router.get("/api/v1/patches/{patch_id}", tags=["Patches"])
async def get_patch(patch_id: str, request: Request):
    """Get a specific patch as JSON."""
    db = request.app.state.db
    patch = await db.get_patch(patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found")
    return patch.get("patch_data", {})


@router.get("/api/v1/patches/download/{patch_id}", tags=["Patches"])
async def download_patch(patch_id: str, request: Request):
    """
    Download a patch as a .json file (browser will trigger Save As).
    """
    db = request.app.state.db
    patch = await db.get_patch(patch_id)
    if not patch:
        raise HTTPException(status_code=404, detail="Patch not found")

    patch_data = patch.get("patch_data", {})
    patch_name = patch.get("name", patch_id[:8])
    filename = f"steerops_patch_{patch_name}.json"

    # Write to temp file for download — use background cleanup
    from starlette.background import BackgroundTask
    import os
    import tempfile

    tmp_path = os.path.join(tempfile.gettempdir(), f"steerops_{patch_id}.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(patch_data, f, indent=2)

    def cleanup(path: str):
        try:
            os.remove(path)
        except OSError:
            pass

    return FileResponse(
        path=tmp_path,
        filename=filename,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        background=BackgroundTask(cleanup, tmp_path),
    )
