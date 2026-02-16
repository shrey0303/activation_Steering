"""
Layer Resolver v4 â€” Absolute Resolution via Feature Dictionary.

Replaces heuristic-based K-Means categorization with direct
feature dictionary lookup. If a feature was extracted from Layer 14,
the resolver targets Layer 14. No guessing, no heuristics.

Supports two resolution modes:
  1. Feature ID mode: user selects "L14_PC3" â†’ direct lookup
  2. Legacy mode: category-based (backward compat with old interpreter)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from loguru import logger


class ResolvedLayer:
    """One layer selected for intervention."""

    def __init__(
        self,
        layer_index: int,
        category: str,
        direction: float,
        recommended_strength: float,
        confidence: float,
        reason: str = "",
        feature_id: str = "",
        vector: Optional[Any] = None,
    ) -> None:
        self.layer_index = layer_index
        self.category = category
        self.direction = direction
        self.recommended_strength = recommended_strength
        self.confidence = confidence
        self.reason = reason
        self.feature_id = feature_id
        self.vector = vector  # numpy or torch tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "category": self.category,
            "direction": self.direction,
            "recommended_strength": round(self.recommended_strength, 2),
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "feature_id": self.feature_id,
        }


class LayerResolver:
    """
    Resolve features to concrete layer interventions.

    v4: Direct feature dictionary lookup. No heuristics.

    Feature ID mode (primary):
        resolver.resolve_features(["L14_PC0", "L22_PC2"])
        â†’ [ResolvedLayer(layer=14, ...), ResolvedLayer(layer=22, ...)]

    Legacy mode (backward compat):
        resolver.resolve(interpretation, ...)
        â†’ Works with old InterpretationResult format
    """

    def __init__(
        self,
        feature_dict=None,
        layer_mappings: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "",
    ) -> None:
        self.feature_dict = feature_dict
        self.model_name = model_name

        # Legacy support: category â†’ layers index
        self._cat_index: Dict[str, List[Dict[str, Any]]] = {}
        if layer_mappings:
            for lm in layer_mappings:
                cat = lm.get("category", "unknown")
                self._cat_index.setdefault(cat, []).append(lm)

    # â”€â”€ Primary Mode: Feature Dictionary Lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def resolve_features(
        self,
        feature_ids: List[str],
        strengths: Optional[Dict[str, float]] = None,
        directions: Optional[Dict[str, float]] = None,
    ) -> List[ResolvedLayer]:
        """
        Resolve feature IDs directly to layer interventions.

        O(1) per feature. No heuristics.

        Parameters
