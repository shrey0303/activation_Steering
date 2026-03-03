"""
Evaluation Pipeline — Production-Grade Before/After Comparison.

Split into submodules:
  - anchors.py: Concept anchor sentences for alignment scoring
  - metrics.py: Metric computation, aggregation, overall scoring
  - evaluator.py: Evaluator orchestrator class
"""

from app.core.evaluator.evaluator import Evaluator
from app.core.evaluator.anchors import get_concept_anchors, CONCEPT_ANCHORS
from app.core.evaluator.metrics import (
    compute_metrics,
    compute_perplexity,
    simple_sentiment,
    aggregate_metrics,
    compute_overall_score,
)

__all__ = [
    "Evaluator",
    "CONCEPT_ANCHORS",
    "get_concept_anchors",
    "compute_metrics",
    "compute_perplexity",
    "simple_sentiment",
    "aggregate_metrics",
    "compute_overall_score",
]
