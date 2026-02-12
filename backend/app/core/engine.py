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

