"""
PyTorch forward-hook management for activation steering.

Each hook modifies the hidden state at a specific layer by adding
a scaled direction vector (or a mean-shift if no vector is provided).

Thread-safe: uses a threading lock to guard hook registration and
removal operations.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("steerops")


@dataclass
class HookHandle:
    """Tracks a registered forward hook."""
    layer_idx: int
    handle: torch.utils.hooks.RemovableHook
    strength: float
    description: str = ""


class HookManager:
    """
    Registers and manages forward hooks on transformer layer modules.

    Thread-safe: all hook operations are guarded by a lock.

    Usage:
        mgr = HookManager()
        mgr.add_hook(layer_module, layer_idx=5, strength=3.0)
        # ... run forward pass ...
        mgr.remove_all()
    """

    def __init__(self) -> None:
        self._hooks: List[HookHandle] = []
        self._lock = threading.Lock()

    @property
    def active_hooks(self) -> int:
        with self._lock:
            return len(self._hooks)

    def add_hook(
        self,
        layer_module: nn.Module,
        layer_idx: int,
        strength: float,
        direction_vector: Optional[torch.Tensor] = None,
    ) -> HookHandle:
        """
        Register a forward hook that steers the hidden state.

        Parameters
        ----------
        layer_module : nn.Module
            The transformer layer to hook into.
        layer_idx : int
            Layer index (for bookkeeping).
        strength : float
            Scaling factor for the intervention.
            Positive = amplify, negative = suppress.
        direction_vector : torch.Tensor, optional
            Unit vector defining the steering direction.
            If None, uses mean-shift (adds scaled mean activation).
        """
        def hook_fn(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
