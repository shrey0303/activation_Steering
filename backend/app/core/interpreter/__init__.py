"""
Intent Router + Response Interpreter.

Split into submodules:
  - models.py: Data classes, constants, utilities
  - intent_router.py: IntentRouter (bi-encoder + NLI direction)
  - response_interpreter.py: ResponseInterpreter (legacy analyze compat)
"""

from app.core.interpreter.models import (
    FeatureMatch,
    IntentRouterResult,
    InterpretationResult,
    LAYER_CATEGORIES,
    CATEGORY_LABELS,
    CATEGORY_HYPOTHESES,
    split_clauses,
)
from app.core.interpreter.intent_router import IntentRouter
from app.core.interpreter.response_interpreter import ResponseInterpreter

__all__ = [
    "FeatureMatch",
    "IntentRouterResult",
    "InterpretationResult",
    "IntentRouter",
    "ResponseInterpreter",
    "LAYER_CATEGORIES",
    "CATEGORY_LABELS",
    "CATEGORY_HYPOTHESES",
    "split_clauses",
]
