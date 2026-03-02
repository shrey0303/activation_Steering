"""
Layer Resolver v4 — Absolute Resolution via Feature Dictionary.

Replaces heuristic-based K-Means categorization with direct
feature dictionary lookup. If a feature was extracted from Layer 14,
the resolver targets Layer 14. No guessing, no heuristics.

Supports two resolution modes:
  1. Feature ID mode: user selects "L14_PC3" → direct lookup
  2. Legacy mode: category-based (backward compat with old interpreter)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.resolver.models import ResolvedLayer


class LayerResolver:
    """
    Resolve features to concrete layer interventions.

    v4: Direct feature dictionary lookup. No heuristics.

    Feature ID mode (primary):
        resolver.resolve_features(["L14_PC0", "L22_PC2"])
        → [ResolvedLayer(layer=14, ...), ResolvedLayer(layer=22, ...)]

    Legacy mode (backward compat):
        resolver.resolve(interpretation, ...)
        → Works with old InterpretationResult format
    """

    def __init__(
        self,
        feature_dict=None,
        layer_mappings: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "",
    ) -> None:
        self.feature_dict = feature_dict
        self.model_name = model_name

        # Legacy support: category → layers index
        self._cat_index: Dict[str, List[Dict[str, Any]]] = {}
        if layer_mappings:
            for lm in layer_mappings:
                cat = lm.get("category", "unknown")
                self._cat_index.setdefault(cat, []).append(lm)

    # --- Feature Dictionary Lookup ---

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
        ----------
        feature_ids : list of feature IDs (e.g., ["L14_PC0", "L22_PC2"])
        strengths : optional per-feature strength overrides
        directions : optional per-feature direction overrides (+1 or -1)
        """
        if self.feature_dict is None:
            raise RuntimeError(
                "No feature dictionary loaded. Run feature extraction first."
            )

        strengths = strengths or {}
        directions = directions or {}
        resolved = []

        # Detect total layers for bell-curve strength scaling
        total_layers = max(
            (f.layer_idx for f in self.feature_dict.all_features), default=1
        ) + 1

        for fid in feature_ids:
            feature = self.feature_dict.get(fid)
            if feature is None:
                logger.warning(f"Feature {fid} not found in dictionary")
                continue

            direction = directions.get(fid, 1.0)
            base_strength = strengths.get(fid, 2.0)

            # Bell-curve layer-aware scaling: peak at 60% depth
            # Based on Zou et al. "Representation Engineering" (2023):
            # middle layers (50-70% depth) are optimal for steering
            relative_pos = feature.layer_idx / total_layers
            bell_multiplier = math.exp(-((relative_pos - 0.6) ** 2) / 0.1)
            strength = base_strength * bell_multiplier

            resolved.append(
                ResolvedLayer(
                    layer_index=feature.layer_idx,
                    category=feature.label,
                    direction=direction,
                    recommended_strength=strength * direction,
                    confidence=min(feature.variance_explained * 10, 1.0),
                    reason=(
                        f"Feature {fid} ('{feature.label}') at layer "
                        f"{feature.layer_idx}, variance={feature.variance_explained:.4f}"
                    ),
                    feature_id=fid,
                    vector=feature.vector,
                )
            )

        logger.info(f"Resolved {len(resolved)} features to layers")
        return resolved

    # --- Legacy Mode ---

    def resolve(
        self,
        interpretation,
        top_k: int = 3,
    ) -> List[ResolvedLayer]:
        """
        Legacy resolver for old InterpretationResult format.

        Kept for backward compatibility with routes.py analyze endpoint.
        """
        candidates: List[ResolvedLayer] = []

        for intent_cat, score in interpretation.intent_scores.items():
            if abs(score) < 0.05:
                continue

            direction = 1.0 if score > 0 else -1.0
            group_layers = self._cat_index.get(intent_cat, [])

            if not group_layers:
                continue

            best = max(group_layers, key=lambda l: l.get("confidence", 0.0))
            strength = direction * min(abs(score) * 8.0, 10.0)
            sim = interpretation.layer_similarities.get(intent_cat, abs(score))
            conf = best.get("confidence", 0.5) * min(abs(score), 1.0)

            display_cat = intent_cat.replace("_", " ").title()
            candidates.append(
                ResolvedLayer(
                    layer_index=best["layer_index"],
                    category=intent_cat,
                    direction=direction,
                    recommended_strength=strength,
                    confidence=conf,
                    reason=(
                        f"Behaviour matched '{display_cat}' "
                        f"(similarity={sim:.2f}) -> layer {best['layer_index']}"
                    ),
                )
            )

        seen: Dict[int, ResolvedLayer] = {}
        for c in candidates:
            if c.layer_index not in seen or c.confidence > seen[c.layer_index].confidence:
                seen[c.layer_index] = c

        resolved = sorted(seen.values(), key=lambda r: r.confidence, reverse=True)
        return resolved[:top_k]

    def resolve_to_dict(
        self,
        interpretation=None,
        feature_ids: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Convenience: resolve and return serializable dict."""
        if feature_ids:
            resolved = self.resolve_features(feature_ids)
        elif interpretation:
            resolved = self.resolve(interpretation, top_k)
        else:
            resolved = []

        return {
            "resolved_layers": [r.to_dict() for r in resolved],
            "total_candidates": len(resolved),
            "model_name": self.model_name,
        }
