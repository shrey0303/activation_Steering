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

    def _capture_activations(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prompts: List[str],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Run prompts through the model and capture hidden states at
        the specified layer.

        Returns tensor of shape (num_prompts, hidden_dim).
        """
        activations = []
        hook_handle = None
        captured = {}

        # Find the target layer module
        layer_module = self._get_layer_module(model, layer_idx)
        if layer_module is None:
            raise ValueError(
                f"Could not find layer {layer_idx} in model. "
                f"Check model architecture."
            )

        def hook_fn(module, input, output):
            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Take the mean across the sequence dimension â†’ (hidden_dim,)
            captured["activation"] = hidden.detach().mean(dim=1).squeeze(0)

        hook_handle = layer_module.register_forward_hook(hook_fn)

        try:
            model.eval()
            with torch.no_grad():
                for prompt in prompts:
                    # Apply chat template for instruction-tuned models
                    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                        try:
                            messages = [{"role": "user", "content": prompt}]
                            formatted = tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        except Exception:
                            formatted = prompt
                    else:
                        formatted = prompt

                    inputs = tokenizer(
                        formatted,
                        return_tensors="pt",
                        truncation=True,
                        max_length=128,
                        padding=True,
                    )
                    # Move to model's device
                    device = next(model.parameters()).device
                    inputs = {
                        k: v.to(device) for k, v in inputs.items()
                    }

                    model(**inputs)

                    if "activation" in captured:
                        activations.append(
                            captured["activation"].cpu().float()
                        )
                        captured.clear()
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        if not activations:
            raise RuntimeError(
                f"Failed to capture activations at layer {layer_idx}"
            )

        return torch.stack(activations)

    def compute_vector(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        concept: str,
        layer_idx: int,
    ) -> Dict[str, Any]:
        """
        Compute a CAA steering vector for a concept at a specific layer.

        Parameters
        ----------
        model
            The loaded transformer model.
        tokenizer
            The model's tokenizer.
        concept
            Concept ID from contrastive_pairs.json (e.g., "politeness").
        layer_idx
            Target layer index.

        Returns
        -------
        Dict with the computed vector and metadata.
        """
        t0 = time.perf_counter()

        pairs = self._load_pairs()
        if concept not in pairs:
            raise ValueError(
                f"Unknown concept '{concept}'. "
                f"Available: {list(pairs.keys())}"
            )

        concept_data = pairs[concept]
        pos_prompts = concept_data["positive"][: self._max_prompts]
        neg_prompts = concept_data["negative"][: self._max_prompts]

        logger.info(
            f"Computing CAA vector for '{concept}' at layer {layer_idx} "
            f"({len(pos_prompts)}+ / {len(neg_prompts)}- prompts)"
        )

        # Capture activations for positive and negative prompts
        pos_acts = self._capture_activations(
            model, tokenizer, pos_prompts, layer_idx
        )
        neg_acts = self._capture_activations(
            model, tokenizer, neg_prompts, layer_idx
        )

        # Compute contrastive direction
        mean_pos = pos_acts.mean(dim=0)
        mean_neg = neg_acts.mean(dim=0)
        direction = mean_pos - mean_neg

        # Normalise to unit vector
        norm = torch.norm(direction)
        if norm > 1e-8:
            direction = direction / norm

        elapsed = time.perf_counter() - t0

        logger.info(
            f"âœ… CAA vector computed for '{concept}' @ layer {layer_idx} "
            f"in {elapsed:.2f}s (dim={direction.shape[0]}, norm={float(norm):.4f})"
        )

        return {
            "concept": concept,
            "layer": layer_idx,
            "direction_vector": direction.tolist(),
            "dimension": int(direction.shape[0]),
            "magnitude": float(norm),
            "num_positive_prompts": len(pos_prompts),
            "num_negative_prompts": len(neg_prompts),
            "compute_time_ms": round(elapsed * 1000, 1),
        }

    def compute_for_patch(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        concept: str,
        layer_indices: List[int],
    ) -> Dict[int, List[float]]:
        """
        Compute CAA vectors for multiple layers (used during patch export).

        Returns dict mapping layer_idx â†’ direction_vector.
        """
        vectors = {}
        for layer_idx in layer_indices:
            try:
                result = self.compute_vector(
                    model, tokenizer, concept, layer_idx
                )
                vectors[layer_idx] = result["direction_vector"]
            except Exception as e:
                logger.warning(
                    f"Failed to compute vector for layer {layer_idx}: {e}"
                )
                vectors[layer_idx] = None
        return vectors

