"""
Feature Dictionary — O(1) runtime lookup for extracted features.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from app.core.feature_extractor.models import Feature, _FEATURES_DB


class FeatureDictionary:
    """O(1) runtime lookup for extracted features."""

    def __init__(self) -> None:
        self._features: Dict[str, Feature] = {}
        self._by_layer: Dict[int, List[Feature]] = {}
        self._by_label: Dict[str, List[Feature]] = {}

    @classmethod
    def load(cls, model_name: str, db_path: Path = _FEATURES_DB) -> "FeatureDictionary":
        """Load all features for a model from SQLite."""
        fd = cls()

        if not db_path.exists():
            logger.warning(f"Feature database not found: {db_path}")
            return fd

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT feature_id, layer_idx, component_idx, label, "
            "variance_explained, vector_path FROM features WHERE model_name = ?",
            (model_name,),
        )

        for row in cursor:
            fid, layer, comp, label, var_exp, vec_path = row
            vec_file = Path(vec_path)
            if not vec_file.exists():
                logger.warning(f"Missing vector file: {vec_file}")
                continue

            vector = np.load(str(vec_file))
            feature = Feature(
                feature_id=fid,
                layer_idx=layer,
                component_idx=comp,
                vector=vector,
                label=label,
                variance_explained=var_exp,
                model_name=model_name,
            )
            fd._features[fid] = feature
            fd._by_layer.setdefault(layer, []).append(feature)
            if label:
                fd._by_label.setdefault(label.lower(), []).append(feature)

        conn.close()
        logger.info(
            f"Loaded {len(fd._features)} features for {model_name} "
            f"across {len(fd._by_layer)} layers"
        )
        return fd

    def get(self, feature_id: str) -> Optional[Feature]:
        """O(1) lookup by feature ID."""
        return self._features.get(feature_id)

    def get_by_layer(self, layer_idx: int) -> List[Feature]:
        """Get all features for a specific layer."""
        return self._by_layer.get(layer_idx, [])

    def get_labeled(self) -> List[Feature]:
        """Get all features that have labels (not just L{n}_PC{m})."""
        return [
            f for f in self._features.values()
            if f.label and not f.label.startswith("L")
        ]

    def get_by_label(self, label: str) -> List[Feature]:
        """Get features matching a label."""
        return self._by_label.get(label.lower(), [])

    def search(self, query: str) -> List[Feature]:
        """Simple substring search across labels."""
        query_lower = query.lower()
        return [
            f for f in self._features.values()
            if query_lower in f.label.lower()
        ]

    @property
    def all_features(self) -> List[Feature]:
        return list(self._features.values())

    @property
    def all_labels(self) -> List[str]:
        return list(self._by_label.keys())

    @property
    def layer_count(self) -> int:
        return len(self._by_layer)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serializable list of all features (without vectors)."""
        return [
            {
                "feature_id": f.feature_id,
                "layer_idx": f.layer_idx,
                "component_idx": f.component_idx,
                "label": f.label,
                "variance_explained": round(f.variance_explained, 6),
            }
            for f in self._features.values()
        ]
