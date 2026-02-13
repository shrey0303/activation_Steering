"""
Activation Steering Engine v2 â€” Production-Grade.

Replaces blind additive steering with a 5-step production pipeline:

  1. KV Cache Preservation  â€” never strip tuple elements from hook output
  2. Cooldown Check         â€” skip injection if circuit breaker fired
  3. Gating                 â€” skip if model already aligned with target
  4. Orthogonal Projection  â€” steer without destroying token meaning
  5. L2 Norm Preservation   â€” clamp activation magnitude within tolerance

Additional features:
  - Adaptive strength decay over token count
  - Logit entropy circuit breaker (cooldown-based, no model re-run)
  - Per-token diagnostics for real-time monitoring
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

from app.core.loader import ModelManager


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEERING HOOK â€” Production Pipeline                        â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@dataclass
class SteeringDiagnostics:
    """Per-token diagnostic information."""
    gated: bool = False
    cooldown_active: bool = False
    norm_deviation: float = 0.0
    effective_strength: float = 0.0
    cosine_similarity: float = 0.0
    token_index: int = 0
    entropy: float = 0.0


class SteeringHook:
    """
    Production-grade forward hook with orthogonal projection.

    5-step pipeline per forward pass:
      1. Preserve KV cache (never strip tuple elements)
      2. Check cooldown (circuit breaker recovery)
      3. Gate (skip if already aligned)
      4. Orthogonal projection + adaptive decay
      5. L2 norm preservation
    """

    def __init__(
        self,
        layer_idx: int,
        strength: float,
        direction_vector: Optional[torch.Tensor] = None,
        gate_threshold: Optional[float] = None,
        norm_tolerance: float = 0.05,
        decay_rate: float = 0.006,
        min_decay: float = 0.4,
        mode: str = "steer",  # "steer" or "erase" (LEACE null-space)
    ) -> None:
        self.layer_idx = layer_idx
        self.strength = strength
        self.direction_vector = direction_vector
        self.norm_tolerance = norm_tolerance
        self.decay_rate = decay_rate
        self.min_decay = min_decay
        self.mode = mode

        # Gating: adaptive threshold based on hidden dim
        # In high-D spaces, cosine similarity shrinks toward 0
        # Default is set when we first see the hidden dimension
        self._gate_threshold_override = gate_threshold
        self.gate_threshold: float = gate_threshold or 0.15

        # Runtime state
        self.handle: Optional[torch.utils.hooks.RemovableHook] = None
        self.token_count: int = 0
        self.cooldown_remaining: int = 0
        self.fired: bool = False
        self.overhead_ms: float = 0.0
        self.last_diagnostics = SteeringDiagnostics()

        # Cached direction vector (moved to device on first use)
        self._v_cached: Optional[torch.Tensor] = None
        self._threshold_calibrated: bool = False

    def _calibrate_threshold(self, hidden_dim: int) -> None:
        """
        Auto-calibrate gate threshold based on hidden dimension.

        In D-dimensional space, two random unit vectors have expected
        cosine similarity ~ 0. The std is ~1/sqrt(D).
        We set threshold = 3 * 1/sqrt(D) so gating only triggers
        when alignment is statistically significant.
        """
        if self._threshold_calibrated:
            return
        if self._gate_threshold_override is not None:
            self._threshold_calibrated = True
            return

        # 3-sigma above random baseline for small models
        # For large models (hidden_dim > 2048): disable gating entirely.
        # 7B+ models have strong concept specialization â€” activations are
        # naturally aligned with steering vectors, causing gate to fire
        # on every call and return output unmodified.
        if hidden_dim > 2048:
            self.gate_threshold = 999.0  # effectively disabled
            self._threshold_calibrated = True
            logger.debug(
                f"Layer {self.layer_idx}: gating DISABLED "
                f"(hidden_dim={hidden_dim} > 2048)"
            )
        else:
            self.gate_threshold = 3.0 / math.sqrt(hidden_dim)
            self._threshold_calibrated = True
            logger.debug(
                f"Layer {self.layer_idx}: gate threshold "
                f"{self.gate_threshold:.4f} (hidden_dim={hidden_dim})"
            )

    def reset_token_count(self) -> None:
        """Reset for a new generation."""
        self.token_count = 0
        self.cooldown_remaining = 0
        self.fired = False
        self.overhead_ms = 0.0
        self._v_cached = None
        self.last_diagnostics = SteeringDiagnostics()

    def get_hook_fn(self) -> Callable:
        """Return the hook function for PyTorch's register_forward_hook."""

        def _hook(
            module: torch.nn.Module,
            input: Any,
            output: Any,
        ) -> Any:
            t0 = time.perf_counter()
            diag = SteeringDiagnostics(token_index=self.token_count)

            try:
                # â”€â”€ STEP 0: Extract hidden states, preserve KV cache â”€â”€
                if isinstance(output, tuple):
                    x = output[0]           # (batch, seq, hidden_dim)
                    kv_cache = output[1:]   # past_key_values, attn, etc.
                else:
                    x = output
                    kv_cache = None

                # No direction vector â†’ mean-shift fallback (CAUTION: unpredictable)
                # This is a heuristic without mathematical basis in steering research.
                # Prefer always providing a direction_vector from feature extraction.
                if self.direction_vector is None:
                    logger.warning(
                        f"Layer {self.layer_idx}: no direction_vector â€” using mean-shift "
                        f"fallback (results will be unpredictable)"
                    )
                    mean_act = x.mean(dim=-1, keepdim=True)
                    x_shifted = x + self.strength * mean_act * 0.1
                    self.fired = True
                    if kv_cache is not None:
                        return (x_shifted,) + kv_cache
                    return x_shifted

                # Calibrate gate on first call
                hidden_dim = x.shape[-1]
                self._calibrate_threshold(hidden_dim)

                # Cache direction vector on correct device/dtype
                if self._v_cached is None or self._v_cached.device != x.device:
                    v = self.direction_vector.to(x.device, dtype=x.dtype)
                    # L2-normalize for stable cosine computations
                    self._v_cached = F.normalize(v, dim=-1)
                v = self._v_cached

                # â”€â”€ STEP 1: Cooldown check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if self.cooldown_remaining > 0:
                    self.cooldown_remaining -= 1
                    diag.cooldown_active = True
                    self.last_diagnostics = diag
                    return output  # pass through unmodified

                # â”€â”€ STEP 2: Gating â€” skip if already aligned â”€â”€â”€â”€â”€â”€
                # Use last token position for gating decision
                x_last = x[:, -1, :]  # (batch, hidden_dim)
                x_last_norm = F.normalize(x_last, dim=-1)
                cos_sim = (x_last_norm * v).sum(dim=-1).mean().item()
                diag.cosine_similarity = cos_sim

                if cos_sim > self.gate_threshold:
                    diag.gated = True
                    self.last_diagnostics = diag
                    return output  # model already heading there

                # â”€â”€ STEP 3: Steering or Erasure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                if self.mode == "erase":
                    # LEACE null-space: remove ALL linear info about v
                    # x_erased = x - (xÂ·vÌ‚)vÌ‚  (project onto null-space of v)
                    proj_coeff = (x * v).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
                    x_new = x - proj_coeff * v  # erase concept direction
                    diag.effective_strength = 0.0  # erasure is binary
                else:
                    # Orthogonal projection + adaptive decay (default)
                    # v_orth = v - proj_x(v) = v - (vÂ·xÌ‚)xÌ‚
                    x_norm = F.normalize(x, dim=-1)
                    proj_coeff = (x_norm * v).sum(dim=-1, keepdim=True)
                    proj = proj_coeff * x_norm
                    v_orth = v - proj  # orthogonal component

                    # Adaptive decay: full strength early, decay over time
                    decay = max(self.min_decay, 1.0 - self.token_count * self.decay_rate)
                    effective_strength = self.strength * decay

                    # Auto-scale by activation norm for model-size invariance
                    # This makes strength=2.5 produce ~same relative effect
                    # on 7B as it does on 0.5B
                    act_norm = x.norm(dim=-1, keepdim=True).mean().item()
                    scale_factor = max(act_norm / 10.0, 1.0)  # normalize: ~5-10 on 0.5B â†’ scale ~1x
                    effective_strength = effective_strength * scale_factor
                    diag.effective_strength = effective_strength

                    # Apply steering
                    x_new = x + effective_strength * v_orth

                # â”€â”€ STEP 4: L2 Norm Preservation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                orig_norm = x.norm(dim=-1, keepdim=True)
                new_norm = x_new.norm(dim=-1, keepdim=True)
                # Scale factor: keep within tolerance of original norm
