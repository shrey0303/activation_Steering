"""
Mathematical Layer Scanner.

Split into submodules:
  - feature_analysis.py: Weight matrix SVD, attention entropy, FFN norms
  - categorizer.py: K-Means clustering, category assignment, descriptions
  - scanner.py: LayerScanner orchestrator
"""

from app.core.scanner.scanner import LayerScanner

__all__ = [
    "LayerScanner",
]
