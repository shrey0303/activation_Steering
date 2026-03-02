"""
LayerScanner — slim orchestrator for the scan pipeline.

Delegates to:
  - feature_analysis: weight matrix analysis (SVD, attention, FFN, etc.)
  - categorizer: K-Means clustering + category assignment
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List

from loguru import logger

from app.core.loader import ModelManager
from app.core.scanner.feature_analysis import extract_layer_features
from app.core.scanner.categorizer import (
    categorise_layers,
    compute_layer_similarities,
)


class LayerScanner:
    """
    Mathematically profile every transformer layer by analysing weight
    matrices only — no forward pass required.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self.mm = model_manager

    def scan(self) -> List[Dict[str, Any]]:
        """
        Analyse all layers and return a list of LayerProfile dicts.

        Returns
        -------
        list of dict
            Each dict contains:
            - layer_index, category, confidence, behavioral_role,
              weight_stats, description
        """
        if not self.mm.loaded:
            raise RuntimeError("No model loaded – call ModelManager.load() first")

        t0 = time.perf_counter()
        layers = self.mm.get_layer_modules()
        n_layers = len(layers)

        if n_layers == 0:
            raise RuntimeError(
                "Could not detect transformer layers in this model architecture"
            )

        logger.info(f"🔬 Scanning {n_layers} layers of {self.mm.model_name}...")

        # --- Step 1: Extract per-layer features---
        features: List[Dict[str, float]] = []
        for idx, layer_module in enumerate(layers):
            feat = extract_layer_features(idx, layer_module, n_layers)
            features.append(feat)
            if (idx + 1) % 8 == 0 or idx == n_layers - 1:
                logger.debug(f"   Scanned layer {idx + 1}/{n_layers}")

        # --- Step 2: Compute inter-layer similarity---
        similarities = compute_layer_similarities(features)

        # --- Step 3: Categorise layers---
        profiles = categorise_layers(features, similarities, n_layers)

        elapsed = time.perf_counter() - t0
        logger.info(f"Scan complete in {elapsed:.1f}s")
        return profiles

    def get_scan_hash(self) -> str:
        """
        Quick hash of model identity + parameter count so we know
        when a re-scan is needed.
        """
        info = (
            f"{self.mm.model_name}|{self.mm.num_layers}|"
            f"{self.mm.hidden_dim}|{self.mm.architecture}"
        )
        return hashlib.sha256(info.encode()).hexdigest()[:16]
