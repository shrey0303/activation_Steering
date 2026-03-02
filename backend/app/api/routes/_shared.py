"""Shared state, helpers, and imports used across all route modules."""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from app.core.config import get_settings
from app.core.constants import KEYWORD_MAP
from app.core.engine import SteeringEngine
from app.core.interpreter import ResponseInterpreter
from app.core.loader import ModelManager
from app.core.monitor import PerformanceMonitor
from app.core.resolver import LayerResolver
from app.core.scanner import LayerScanner
from app.core.vector_calculator import VectorCalculator
from app.core.feature_extractor import FeatureExtractor, FeatureDictionary

from fastapi import HTTPException


# --- Shared Singletons ---

monitor = PerformanceMonitor()
interpreter = ResponseInterpreter()
vector_calc = VectorCalculator(max_prompts=20)
feature_dict: FeatureDictionary | None = None

# --- Analysis Cache ---
# Stores the last behavior description from analyze so export can
# pick the semantically closest CAA concept instead of guessing.
last_analysis: Dict[str, Any] = {}
best_concept_cache: Dict[str, str] = {}


def require_model_loaded() -> ModelManager:
    """Return the model manager, raising 400 if no model is loaded."""
    mm = ModelManager.get_instance()
    if not mm.loaded:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Load a model first via the Control Panel.",
        )
    return mm


def match_behavior_to_concept(behavior: str) -> str:
    """
    Find the closest CAA concept for a behavior description.
    Uses semantic keyword matching against concept descriptions
    from contrastive_pairs.json.

    Falls back to 'politeness' if nothing matches.
    """
    if behavior in best_concept_cache:
        return best_concept_cache[behavior]

    behavior_lower = behavior.lower()

    # Direct keyword mapping — fast path
    for keyword, concept in KEYWORD_MAP.items():
        if keyword in behavior_lower:
            best_concept_cache[behavior] = concept
            logger.info(f"Matched behavior '{behavior}' → concept '{concept}' (keyword: {keyword})")
            return concept

    # Fallback: try available concepts from VectorCalculator
    try:
        concepts = vector_calc.available_concepts
        for c in concepts:
            if c["id"] in behavior_lower or behavior_lower in c.get("description", "").lower():
                best_concept_cache[behavior] = c["id"]
                return c["id"]
    except Exception:
        pass

    # Ultimate fallback
    best_concept_cache[behavior] = "politeness"
    return "politeness"
