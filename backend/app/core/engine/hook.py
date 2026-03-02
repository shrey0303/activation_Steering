"""
Forward hook implementing orthogonal projection steering
with gating, norm preservation, and adaptive decay.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from app.core.engine.diagnostics import SteeringDiagnostics


class SteeringHook:
    """Forward hook implementing orthogonal projection steering with gating and norm preservation."""

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

        # Default threshold is calibrated on first forward pass based on hidden dim
        self._gate_threshold_override = gate_threshold
        self.gate_threshold: float = gate_threshold or 0.15
        self._gating_disabled: bool = False


        self.handle: Optional[torch.utils.hooks.RemovableHook] = None
        self.token_count: int = 0
        self.cooldown_remaining: int = 0
        self.fired: bool = False
        self.overhead_ms: float = 0.0
        self.last_diagnostics = SteeringDiagnostics()

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

        # 3-sigma above random baseline for small models.
        # For large models (hidden_dim >= 2048): disable gating entirely.
        # Models at 2B+ scale (e.g., Sarvam-2B with dim=2048, Qwen-7B with dim=3584)
        # have strong concept specialization — activations are naturally aligned
        # with steering vectors, causing gate to fire on every call and
        # return output unmodified.
        if hidden_dim >= 2048:
            self._gating_disabled = True
            self._threshold_calibrated = True
            logger.debug(
                f"Layer {self.layer_idx}: gating DISABLED "
                f"(hidden_dim={hidden_dim} >= 2048)"
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
                # Preserve KV cache tuple structure
                if isinstance(output, tuple):
                    x = output[0]           # (batch, seq, hidden_dim)
                    kv_cache = output[1:]   # past_key_values, attn, etc.
                else:
                    x = output
                    kv_cache = None

                # No direction vector → mean-shift fallback (CAUTION: unpredictable)
                # This is a heuristic without mathematical basis in steering research.
                # Prefer always providing a direction_vector from feature extraction.
                if self.direction_vector is None:
                    logger.warning(
                        f"Layer {self.layer_idx}: no direction_vector — using mean-shift "
                        f"fallback (results will be unpredictable)"
                    )
                    mean_act = x.mean(dim=-1, keepdim=True)
                    x_shifted = x + self.strength * mean_act * 0.1
                    self.fired = True
                    if kv_cache is not None:
                        return (x_shifted,) + kv_cache
                    return x_shifted

                hidden_dim = x.shape[-1]
                self._calibrate_threshold(hidden_dim)

                if self._v_cached is None or self._v_cached.device != x.device:
                    v = self.direction_vector.to(x.device, dtype=x.dtype)

                    self._v_cached = F.normalize(v, dim=-1)
                v = self._v_cached

                if self.cooldown_remaining > 0:
                    self.cooldown_remaining -= 1
                    diag.cooldown_active = True
                    self.last_diagnostics = diag
                    return output  # pass through unmodified

                x_last = x[:, -1, :]
                x_last_norm = F.normalize(x_last, dim=-1)
                cos_sim = (x_last_norm * v).sum(dim=-1).mean().item()
                diag.cosine_similarity = cos_sim

                if not self._gating_disabled and cos_sim > self.gate_threshold:
                    diag.gated = True
                    self.last_diagnostics = diag
                    return output  # model already heading there

                if self.mode == "erase":
                    # LEACE null-space projection: x_erased = x - (x·v̂)v̂
                    proj_coeff = (x * v).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
                    x_new = x - proj_coeff * v  # erase concept direction
                    diag.effective_strength = 0.0  # erasure is binary
                else:
                    # v_orth = v - proj_x(v): steer along orthogonal component only
                    x_norm = F.normalize(x, dim=-1)
                    proj_coeff = (x_norm * v).sum(dim=-1, keepdim=True)
                    proj = proj_coeff * x_norm
                    v_orth = v - proj

                    decay = max(self.min_decay, 1.0 - self.token_count * self.decay_rate)
                    effective_strength = self.strength * decay

                    # Auto-scale by activation norm for model-size invariance
                    # This makes strength=2.5 produce ~same relative effect
                    # on 7B as it does on 0.5B
                    act_norm = x.norm(dim=-1, keepdim=True).mean().item()
                    scale_factor = max(act_norm / 10.0, 1.0)  # normalize: ~5-10 on 0.5B → scale ~1x
                    effective_strength = effective_strength * scale_factor
                    diag.effective_strength = effective_strength

                    if self.token_count == 0:
                        logger.debug(
                            f"Layer {self.layer_idx}: effective_strength={effective_strength:.2f}, "
                            f"scale_factor={scale_factor:.2f}, act_norm={act_norm:.1f}, "
                            f"decay={decay:.3f}"
                        )

                    x_new = x + effective_strength * v_orth

                orig_norm = x.norm(dim=-1, keepdim=True)
                new_norm = x_new.norm(dim=-1, keepdim=True)
                # Wider tolerance for large models where perturbation needs to be proportionally bigger
                effective_tolerance = self.norm_tolerance
                if hidden_dim > 2048:
                    effective_tolerance = max(self.norm_tolerance, 0.25)
                scale = orig_norm / (new_norm + 1e-8)
                min_scale = 1.0 - effective_tolerance
                max_scale = 1.0 + effective_tolerance
                scale = scale.clamp(min=min_scale, max=max_scale)
                x_steered = x_new * scale

                actual_norm = x_steered.norm(dim=-1).mean().item()
                orig_norm_val = orig_norm.mean().item()
                diag.norm_deviation = abs(actual_norm - orig_norm_val) / (orig_norm_val + 1e-8)


                if torch.isnan(x_steered).any() or torch.isinf(x_steered).any():
                    logger.warning(
                        f"NaN/Inf at layer {self.layer_idx} – reverting"
                    )
                    self.cooldown_remaining = 10
                    return output

                self.fired = True
                self.token_count += 1
                self.last_diagnostics = diag


                if kv_cache is not None:
                    return (x_steered,) + kv_cache
                return x_steered

            finally:
                self.overhead_ms = (time.perf_counter() - t0) * 1000

        return _hook
