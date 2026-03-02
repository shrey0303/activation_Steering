"""
Layer Resolver — maps features/intents to concrete layer interventions.

Split into submodules:
  - models.py: ResolvedLayer data class
  - resolver.py: LayerResolver logic
"""

from app.core.resolver.models import ResolvedLayer
from app.core.resolver.resolver import LayerResolver

__all__ = [
    "ResolvedLayer",
    "LayerResolver",
]
