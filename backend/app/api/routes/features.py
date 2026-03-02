"""Feature extraction, listing, and management endpoints."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from app.core.feature_extractor import FeatureExtractor, FeatureDictionary, _FEATURES_DB
from app.core.loader import ModelManager
from app.api.routes._shared import require_model_loaded, feature_dict

import app.api.routes._shared as _shared

_FEATURE_EXTRACTION_TIMEOUT = 300  # seconds

router = APIRouter()


@router.post("/api/v1/features/extract", tags=["Features"])
async def extract_features(request: Request):
    """
    Run offline PCA feature extraction on the loaded model.

    This is a heavy operation (~30-60s on CPU, ~5s on GPU).
    Results are cached in SQLite — subsequent calls are instant.
    """
    mm = require_model_loaded()
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}

    top_k = body.get("top_k", 20)
    label_top = body.get("label_top_n", 5)
    no_label = body.get("no_label", False)

    extractor = FeatureExtractor(
        top_k=top_k,
        auto_label_top_n=0 if no_label else label_top,
    )

    try:
        t0 = time.time()
        _shared.feature_dict = await asyncio.wait_for(
            asyncio.to_thread(
                extractor.extract,
                model=mm.model,
                tokenizer=mm.tokenizer,
                layer_modules=mm.get_layer_modules(),
                model_name=mm.model_name,
                device=mm.device,
            ),
            timeout=_FEATURE_EXTRACTION_TIMEOUT,
        )
        elapsed = time.time() - t0

        return {
            "status": "success",
            "total_features": len(_shared.feature_dict.all_features),
            "labeled_features": len(_shared.feature_dict.get_labeled()),
            "layers": _shared.feature_dict.layer_count,
            "extraction_time_seconds": round(elapsed, 1),
        }
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/features", tags=["Features"])
async def list_features(request: Request):
    """List all extracted features for the loaded model."""
    mm = require_model_loaded()

    # Load dict if not already in memory
    if _shared.feature_dict is None:
        _shared.feature_dict = FeatureDictionary.load(mm.model_name)

    layer_filter = request.query_params.get("layer")
    label_filter = request.query_params.get("label")
    search_query = request.query_params.get("search")

    features = _shared.feature_dict.all_features

    if layer_filter is not None:
        features = _shared.feature_dict.get_by_layer(int(layer_filter))
    elif label_filter:
        features = _shared.feature_dict.get_by_label(label_filter)
    elif search_query:
        features = _shared.feature_dict.search(search_query)

    return {
        "features": [
            {
                "feature_id": f.feature_id,
                "layer_idx": f.layer_idx,
                "component_idx": f.component_idx,
                "label": f.label,
                "variance_explained": round(f.variance_explained, 6),
            }
            for f in features
        ],
        "total": len(features),
        "available_labels": _shared.feature_dict.all_labels,
    }


@router.get("/api/v1/features/{feature_id}", tags=["Features"])
async def get_feature(feature_id: str, request: Request):
    """Get a specific feature with its vector."""
    mm = require_model_loaded()

    if _shared.feature_dict is None:
        _shared.feature_dict = FeatureDictionary.load(mm.model_name)

    feature = _shared.feature_dict.get(feature_id)
    if feature is None:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

    return {
        "feature_id": feature.feature_id,
        "layer_idx": feature.layer_idx,
        "component_idx": feature.component_idx,
        "label": feature.label,
        "variance_explained": round(feature.variance_explained, 6),
        "vector": feature.vector.tolist(),
        "vector_dim": len(feature.vector),
    }


@router.put("/api/v1/features/{feature_id}/label", tags=["Features"])
async def update_feature_label(feature_id: str, request: Request):
    """Update a feature's label (Discovery Dashboard)."""
    mm = require_model_loaded()

    if _shared.feature_dict is None:
        _shared.feature_dict = FeatureDictionary.load(mm.model_name)

    feature = _shared.feature_dict.get(feature_id)
    if feature is None:
        raise HTTPException(status_code=404, detail=f"Feature {feature_id} not found")

    body = await request.json()
    new_label = body.get("label", "")
    if not new_label:
        raise HTTPException(status_code=400, detail="label is required")

    # Update in-memory
    feature.label = new_label

    # Update in DB
    import sqlite3 as _sqlite3

    def _persist_label():
        if _FEATURES_DB.exists():
            conn = _sqlite3.connect(str(_FEATURES_DB))
            conn.execute(
                "UPDATE features SET label = ? WHERE feature_id = ? AND model_name = ?",
                (new_label, feature_id, mm.model_name),
            )
            conn.commit()
            conn.close()

    await asyncio.to_thread(_persist_label)

    return {"status": "updated", "feature_id": feature_id, "label": new_label}
