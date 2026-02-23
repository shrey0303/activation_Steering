"""
Contrastive Activation Addition (CAA) — Vector Calculator.

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


# ── Path to contrastive pairs ─────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"
_PAIRS_FILE = _DATA_DIR / "contrastive_pairs.json"

# ── Cached vectors directory ──────────────────────────────────
_VECTORS_DIR = _DATA_DIR / "cached_vectors"


class VectorCalculator:
    """
    Compute Contrastive Activation Addition (CAA) steering vectors.

    For a given concept (e.g., "politeness") and a target layer:
      1. Encode N positive prompts → capture hidden states at layer L
      2. Encode N negative prompts → capture hidden states at layer L
      3. direction = mean(h⁺) - mean(h⁻)
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
            # Take the mean across the sequence dimension → (hidden_dim,)
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
            f"✅ CAA vector computed for '{concept}' @ layer {layer_idx} "
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

        Returns dict mapping layer_idx → direction_vector.
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

    def cache_vector(
        self,
        concept: str,
        layer_idx: int,
        model_name: str,
        vector_data: Dict[str, Any],
    ) -> Path:
        """Cache a computed vector to disk for reuse."""
        _VECTORS_DIR.mkdir(parents=True, exist_ok=True)

        safe_name = model_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_name}_{concept}_layer{layer_idx}.json"
        path = _VECTORS_DIR / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(vector_data, f, indent=2)

        logger.info(f"Cached vector: {path}")
        return path

    def load_cached_vector(
        self,
        concept: str,
        layer_idx: int,
        model_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a previously cached vector if available."""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_name}_{concept}_layer{layer_idx}.json"
        path = _VECTORS_DIR / filename

        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_layer_module(
        model: torch.nn.Module, layer_idx: int
    ) -> Optional[torch.nn.Module]:
        """
        Find the transformer layer module by index.
        Supports common architectures:
        - GPT-2: model.transformer.h[idx]
        - LLaMA/Mistral: model.model.layers[idx]
        - BERT: model.encoder.layer[idx]
        - GPT-Neo: model.gpt_neox.layers[idx] or model.transformer.h[idx]
        """
        # Try common paths
        paths_to_try = [
            ("model", "layers"),           # LLaMA, Mistral, Qwen
            ("transformer", "h"),          # GPT-2, GPT-J
            ("gpt_neox", "layers"),        # GPT-NeoX
            ("encoder", "layer"),          # BERT, RoBERTa
            ("decoder", "layers"),         # T5 decoder
        ]

        for parent_attr, layers_attr in paths_to_try:
            parent = getattr(model, parent_attr, None)
            if parent is not None:
                layers = getattr(parent, layers_attr, None)
                if layers is not None and layer_idx < len(layers):
                    return layers[layer_idx]

        # Fallback: search recursively for a list of modules
        for name, module in model.named_modules():
            if hasattr(module, "__len__") and not isinstance(module, str):
                try:
                    if layer_idx < len(module):
                        return module[layer_idx]
                except (TypeError, IndexError):
                    continue

        return None
