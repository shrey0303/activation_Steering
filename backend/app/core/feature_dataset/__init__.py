"""
Diverse sentence dataset for PCA feature extraction.

Split into submodules:
  - prompts.py: DIVERSE_PROMPTS (activation collection)
  - labeling.py: LABELING_PROMPTS, BEHAVIORAL_KEYWORDS (auto-labeling)
"""

from typing import List

from app.core.feature_dataset.prompts import DIVERSE_PROMPTS
from app.core.feature_dataset.labeling import LABELING_PROMPTS, BEHAVIORAL_KEYWORDS


def get_diverse_prompts() -> List[str]:
    """Return all diverse prompts for activation collection."""
    return DIVERSE_PROMPTS.copy()


def get_labeling_prompts() -> List[str]:
    """Return probing prompts for auto-labeling."""
    return LABELING_PROMPTS.copy()


def get_behavioral_keywords() -> List[str]:
    """Return the vocabulary of behavioral labels."""
    return BEHAVIORAL_KEYWORDS.copy()


__all__ = [
    "DIVERSE_PROMPTS",
    "LABELING_PROMPTS",
    "BEHAVIORAL_KEYWORDS",
    "get_diverse_prompts",
    "get_labeling_prompts",
    "get_behavioral_keywords",
]
