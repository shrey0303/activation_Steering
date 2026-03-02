"""
Contrastive pairs loader and vector cache management.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_PAIRS_DIR = _DATA_DIR / "contrastive_pairs"
_VECTORS_DIR = _DATA_DIR / "cached_vectors"


def load_pairs(cache: Dict | None = None) -> Dict[str, Dict[str, List[str]]]:
    """Load contrastive prompt pairs from per-concept JSON files."""
    if cache is not None and cache:
        return cache

    if not _PAIRS_DIR.exists():
        raise FileNotFoundError(
            f"Contrastive pairs directory not found: {_PAIRS_DIR}"
        )

    pairs: Dict[str, Dict[str, List[str]]] = {}
    for json_file in sorted(_PAIRS_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            pairs.update(json.load(f))

    if not pairs:
        raise FileNotFoundError(
            f"No concept files found in {_PAIRS_DIR}"
        )

    logger.info(
        f"Loaded contrastive pairs: "
        f"{list(pairs.keys())}"
    )
    return pairs


def cache_vector(
    concept: str,
    layer_idx: int,
    model_name: str,
    vector_data: Dict[str, Any],
) -> Path:
    """Cache a computed vector to disk for reuse."""
    _VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}_{concept}_layer{layer_idx}.json"
    path = _VECTORS_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vector_data, f, indent=2)

    logger.info(f"Cached vector: {path}")
    return path


def load_cached_vector(
    concept: str,
    layer_idx: int,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """Load a previously cached vector if available."""
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    filename = f"{safe_name}_{concept}_layer{layer_idx}.json"
    path = _VECTORS_DIR / filename

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
