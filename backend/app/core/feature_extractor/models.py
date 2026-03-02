"""
Feature data models and path constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_FEATURES_DB = _DATA_DIR / "features.db"
_VECTORS_DIR = _DATA_DIR / "feature_vectors"


@dataclass
class Feature:
    """A single extracted feature (PCA component)."""
    feature_id: str          # e.g., "L14_PC3"
    layer_idx: int           # transformer layer index
    component_idx: int       # PCA rank (0 = highest variance)
    vector: np.ndarray       # shape: (hidden_dim,)
    label: str               # auto-generated or user-provided label
    variance_explained: float  # fraction of variance this PC captures
    model_name: str          # which model this was extracted from

    def to_torch(self, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
        """Convert vector to PyTorch tensor."""
        return torch.from_numpy(self.vector).to(device=device, dtype=dtype)
