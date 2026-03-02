"""
Resolver data model — represents a single layer selected for intervention.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


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
        self.vector = vector

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
