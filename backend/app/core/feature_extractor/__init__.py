"""
PCA Feature Extraction Pipeline — Offline Feature Dictionary.

Split into submodules:
  - models.py: Feature dataclass, path constants
  - dictionary.py: FeatureDictionary (O(1) lookup)
  - extractor.py: FeatureExtractor pipeline + CLI
"""

from app.core.feature_extractor.models import Feature, _DATA_DIR, _FEATURES_DB, _VECTORS_DIR
from app.core.feature_extractor.dictionary import FeatureDictionary
from app.core.feature_extractor.extractor import FeatureExtractor, main

__all__ = [
    "Feature",
    "FeatureDictionary",
    "FeatureExtractor",
    "_DATA_DIR",
    "_FEATURES_DB",
    "_VECTORS_DIR",
    "main",
]
