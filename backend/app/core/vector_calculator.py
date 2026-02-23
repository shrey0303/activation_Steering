"""
Contrastive Activation Addition (CAA) â€” Vector Calculator.

Computes real steering vectors by:
1. Running positive prompts through the model and capturing activations
2. Running negative prompts through the model and capturing activations
3. direction = mean(positive_activations) - mean(negative_activations)
4. Normalise to unit vector

These vectors represent the semantic direction of a concept
(e.g., "politeness") in the model's latent space.

References:
- Turner et al. 2023: "Activation Addition: Steering Language Models
  Without Optimization"
- Rimsky et al. 2023: "Steering Llama 2 via Contrastive Activation
  Addition"
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger


# â”€â”€ Path to contrastive pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_DATA_DIR = Path(__file__).parent.parent / "data"
_PAIRS_FILE = _DATA_DIR / "contrastive_pairs.json"

# â”€â”€ Cached vectors directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_VECTORS_DIR = _DATA_DIR / "cached_vectors"


class VectorCalculator:
    """
    Compute Contrastive Activation Addition (CAA) steering vectors.

    For a given concept (e.g., "politeness") and a target layer:
      1. Encode N positive prompts â†’ capture hidden states at layer L
      2. Encode N negative prompts â†’ capture hidden states at layer L
      3. direction = mean(hâº) - mean(hâ»)
      4. Normalise to unit vector

    The resulting vector can be added to activations during inference
    to steer the model's behaviour.
    """

    def __init__(self, max_prompts: int = 20) -> None:
        """
        Parameters
        ----------
        max_prompts
            Maximum number of prompt pairs to use per concept.
            More = more accurate but slower. 20 is a good balance.
        """
        self._max_prompts = max_prompts
        self._contrastive_pairs: Optional[Dict] = None

    def _load_pairs(self) -> Dict[str, Dict[str, List[str]]]:
        """Load contrastive prompt pairs from JSON file."""
        if self._contrastive_pairs is not None:
            return self._contrastive_pairs

        if not _PAIRS_FILE.exists():
            raise FileNotFoundError(
                f"Contrastive pairs file not found: {_PAIRS_FILE}"
            )

        with open(_PAIRS_FILE, "r", encoding="utf-8") as f:
            self._contrastive_pairs = json.load(f)

        logger.info(
            f"Loaded contrastive pairs: "
            f"{list(self._contrastive_pairs.keys())}"
        )
        return self._contrastive_pairs

    @property
    def available_concepts(self) -> List[Dict[str, str]]:
        """List available concepts with their descriptions."""
        pairs = self._load_pairs()
        return [
            {
                "id": concept_id,
                "description": data.get("description", ""),
                "num_pairs": min(
                    len(data.get("positive", [])),
                    len(data.get("negative", [])),
                ),
            }
            for concept_id, data in pairs.items()
        ]

