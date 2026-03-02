"""
Weight-matrix feature extraction for the layer scanner.

Standalone functions that analyse transformer weight matrices
without running any forward passes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from loguru import logger


def dequantize_param(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """
    Safely dequantize a parameter to FP32 for analysis.

    Handles:
    - bitsandbytes Linear4bit / Linear8bitLt  (packed quantised)
    - Standard FP16 / BF16 params              (simple cast)
    - Regular FP32 params                      (no-op)

    Returns None if the parameter cannot be dequantised.
    """
    try:
        # --- bitsandbytes 4-bit---
        module_type = type(param).__module__ or ""
        cls_name = type(param).__qualname__ or ""

        # Check if this is a bnb Params4bit
        if "bitsandbytes" in module_type or "Params4bit" in cls_name:
            try:
                import bitsandbytes as bnb
                # bnb stores weights in .data as packed uint8
                # dequantize() returns the FP32 equivalent
                return bnb.functional.dequantize_4bit(
                    param.data, param.quant_state
                ).float().cpu()
            except Exception:
                logger.debug(f"bnb 4-bit dequantize failed for {cls_name}")

        # Check if param belongs to a bnb Linear4bit layer
        if hasattr(param, "quant_state") and param.quant_state is not None:
            try:
                import bitsandbytes as bnb
                return bnb.functional.dequantize_4bit(
                    param.data, param.quant_state
                ).float().cpu()
            except Exception:
                logger.debug("bnb 4-bit dequantize via quant_state failed")

        # --- bitsandbytes 8-bit---
        if hasattr(param, "SCB") or hasattr(param, "CB"):
            try:
                import bitsandbytes as bnb
                return param.data.float().cpu()
            except Exception:
                logger.debug("bnb 8-bit dequantize failed")

        # --- Standard param: cast to float32---
        return param.detach().float().cpu()

    except Exception:
        return None


def svd_analysis(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Singular value decomposition of weight matrices.

    The distribution of singular values reveals:
    - High effective rank  →  complex learned transformations
    - Rapid decay         →  low-rank, simpler representation
    - Entropy of spectrum →  diversity of learned features

    Uses randomized SVD (scipy) with aggressive downsampling
    for speed — 256 rows × 512 cols × k=32 per matrix.
    """
    from scipy.sparse.linalg import svds
    all_singular: List[np.ndarray] = []

    for name, w in weights.items():
        try:
            m, n = w.shape[0], w.shape[-1]
            if m < 2 or n < 2:
                continue

            # Aggressive downsampling for speed:
            # cap at 256 rows × 512 cols (was 1024 × unlimited)
            w_2d = w.reshape(-1, w.shape[-1])[:256, :512].numpy()
            k = min(32, min(w_2d.shape) - 1)  # scipy svds needs k < min(m,n)
            if k < 1:
                continue

            # Randomized SVD: O(m*n*k) instead of O(m*n*min(m,n))
            _, s, _ = svds(w_2d.astype(np.float64), k=k)
            all_singular.append(s[::-1])  # svds returns ascending
        except Exception as e:
            logger.debug(f"SVD failed for weight '{name}': {e}")
            continue

    if not all_singular:
        return {"svd_rank": 0.0, "svd_entropy": 0.0, "svd_decay": 0.0}

    combined = np.concatenate(all_singular)
    combined = combined / (combined.max() + 1e-10)  # normalise

    # Effective rank: entropy-based (Roy & Vetterli, 2007)
    p = combined / (combined.sum() + 1e-10)
    p = p[p > 1e-10]
    svd_entropy = float(-np.sum(p * np.log(p + 1e-10)))
    effective_rank = float(np.exp(svd_entropy))

    # Decay rate: ratio of 10th to 1st singular value
    sorted_sv = np.sort(combined)[::-1]
    idx_10pct = max(1, len(sorted_sv) // 10)
    decay = float(sorted_sv[idx_10pct] / (sorted_sv[0] + 1e-10))

    return {
        "svd_rank": effective_rank,
        "svd_entropy": svd_entropy,
        "svd_decay": decay,
    }


def attention_entropy(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Analyse attention projection weights.

    Higher entropy in Q/K projections → global attention (semantic).
    Lower entropy → local attention (syntactic).
    """
    attn_entropy_values: List[float] = []

    for name, w in weights.items():
        # Heuristic: attention projections usually have "q_proj", "k_proj",
        # "query", "key", "attn" in their name
        lower = name.lower()
        is_attn = any(
            kw in lower
            for kw in ["q_proj", "k_proj", "query", "key", "attn", "attention"]
        )
        if not is_attn:
            continue

        try:
            # Row-wise L2 norms as a proxy for head importance
            w_2d = w.reshape(-1, w.shape[-1])
            norms = torch.norm(w_2d, dim=1).numpy()
            norms = norms / (norms.sum() + 1e-10)
            norms = norms[norms > 1e-10]
            ent = float(-np.sum(norms * np.log(norms + 1e-10)))
            attn_entropy_values.append(ent)
        except Exception as e:
            logger.debug(f"Attention entropy failed for '{name}': {e}")
            continue

    if not attn_entropy_values:
        return {"attn_entropy": 0.0}

    return {"attn_entropy": float(np.mean(attn_entropy_values))}


def ffn_norm_analysis(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Analyse feed-forward network weight magnitudes.

    Large norms → strong non-linear transformation (reasoning).
    Small norms → light processing (embedding / output).
    """
    ffn_norms: List[float] = []

    for name, w in weights.items():
        lower = name.lower()
        is_ffn = any(
            kw in lower
            for kw in [
                "mlp", "ffn", "fc", "gate", "up_proj",
                "down_proj", "dense", "intermediate",
            ]
        )
        if not is_ffn:
            continue

        try:
            norm_val = float(torch.norm(w.float()).item())
            ffn_norms.append(norm_val)
        except Exception as e:
            logger.debug(f"FFN norm failed for '{name}': {e}")
            continue

    if not ffn_norms:
        return {"ffn_norm_mean": 0.0, "ffn_norm_max": 0.0}

    return {
        "ffn_norm_mean": float(np.mean(ffn_norms)),
        "ffn_norm_max": float(np.max(ffn_norms)),
    }


def general_weight_stats(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Overall weight statistics: sparsity, kurtosis, skewness."""
    all_vals: List[float] = []
    total_params = 0
    near_zero = 0

    for w in weights.values():
        try:
            flat = w.float().flatten().numpy()
            all_vals.append(float(np.std(flat)))
            total_params += len(flat)
            near_zero += int(np.sum(np.abs(flat) < 1e-4))
        except Exception as e:
            logger.debug(f"Weight stats failed: {e}")
            continue

    sparsity = near_zero / max(total_params, 1)

    return {
        "weight_std_mean": float(np.mean(all_vals)) if all_vals else 0.0,
        "sparsity": sparsity,
        "total_params": float(total_params),
    }


def position_heuristic(
    layer_idx: int, total_layers: int,
) -> Dict[str, float]:
    """Fallback when weight extraction fails — use relative position."""
    return {
        "svd_rank": 0.0,
        "svd_entropy": 0.0,
        "svd_decay": 0.0,
        "attn_entropy": 0.0,
        "ffn_norm_mean": 0.0,
        "ffn_norm_max": 0.0,
        "weight_std_mean": 0.0,
        "sparsity": 0.0,
        "total_params": 0.0,
    }


@torch.no_grad()
def extract_layer_features(
    layer_idx: int,
    layer_module: torch.nn.Module,
    total_layers: int,
) -> Dict[str, float]:
    """
    Extract mathematical features from a single layer's weight
    matrices.  All computation on CPU to avoid GPU contention.

    For quantised models, weights are dequantised to FP32 before
    analysis so scan quality is identical to full-precision models.
    """
    features: Dict[str, float] = {
        "layer_index": float(layer_idx),
        "relative_position": layer_idx / max(total_layers - 1, 1),
    }

    weight_tensors: Dict[str, torch.Tensor] = {}
    for name, param in layer_module.named_parameters():
        w = dequantize_param(param)
        if w is not None and w.ndim >= 2:
            weight_tensors[name] = w

    if not weight_tensors:
        # Fallback: position-based heuristic
        features.update(position_heuristic(layer_idx, total_layers))
        return features

    features.update(svd_analysis(weight_tensors))
    features.update(attention_entropy(weight_tensors))
    features.update(ffn_norm_analysis(weight_tensors))
    features.update(general_weight_stats(weight_tensors))

    return features
