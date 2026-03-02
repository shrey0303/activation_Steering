"""
Layer module discovery — find transformer layer by index.

Architecture-agnostic utility supporting GPT-2, LLaMA, Mistral,
BERT, GPT-NeoX, T5, and others.
"""

from __future__ import annotations

from typing import Optional

import torch


def get_layer_module(
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

    # Fallback: search recursively for a ModuleList whose children
    # look like transformer layers (have self_attn/attention/mlp submodules).
    _LAYER_MARKERS = {"self_attn", "attention", "self_attention", "mlp", "feed_forward"}

    for name, module in model.named_modules():
        if hasattr(module, "__len__") and not isinstance(module, str):
            try:
                if layer_idx < len(module):
                    candidate = module[layer_idx]
                    # Validate: candidate should have at least one known
                    # transformer submodule to avoid matching embeddings etc.
                    child_names = {n for n, _ in candidate.named_children()}
                    if child_names & _LAYER_MARKERS:
                        return candidate
            except (TypeError, IndexError):
                continue

    return None
