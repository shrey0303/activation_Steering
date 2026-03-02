"""
Contrastive Activation Addition (CAA) Vector Calculator.

Split into submodules:
  - layer_utils.py: Architecture-agnostic layer module discovery
  - loader.py: Contrastive pairs loading and vector caching
  - calculator.py: VectorCalculator class (activation capture + computation)
"""

from app.core.vector_calculator.calculator import VectorCalculator
from app.core.vector_calculator.layer_utils import get_layer_module
from app.core.vector_calculator.loader import (
    load_pairs,
    cache_vector,
    load_cached_vector,
    _PAIRS_DIR,
    _VECTORS_DIR,
)

__all__ = [
    "VectorCalculator",
    "get_layer_module",
    "load_pairs",
    "cache_vector",
    "load_cached_vector",
    "_PAIRS_DIR",
    "_VECTORS_DIR",
]
