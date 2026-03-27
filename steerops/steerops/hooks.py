"""
PyTorch forward-hook management for activation steering.

Hooks modify hidden states at a specific layer by adding a scaled
direction vector. Thread-safe via threading lock on registration/removal.
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
    """Registers and manages forward hooks on transformer layer modules. Thread-safe."""

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
        """Register a forward hook that adds strength * direction to the hidden state at this layer."""

        def hook_fn(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
            output,
        ):
            if isinstance(output, tuple):
                hidden = output[0]
            elif isinstance(output, torch.Tensor):
                hidden = output
            else:
                return output

            with torch.no_grad():
                if direction_vector is not None:
                    vec = direction_vector.to(hidden.device, hidden.dtype)
                    if vec.dim() == 1:
                        vec = vec.unsqueeze(0).unsqueeze(0)
                    hidden = hidden + strength * vec
                else:
                    mean_act = hidden.mean(dim=-1, keepdim=True)
                    hidden = hidden + strength * 0.1 * mean_act

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        with self._lock:
            handle = layer_module.register_forward_hook(hook_fn)
            hook = HookHandle(
                layer_idx=layer_idx,
                handle=handle,
                strength=strength,
                description=f"Steer L{layer_idx} @ {strength:+.1f}",
            )
            self._hooks.append(hook)
            logger.debug(f"Added hook: {hook.description}")
            return hook

    def remove_all(self) -> int:
        """Remove all registered hooks. Returns count removed."""
        with self._lock:
            count = len(self._hooks)
            for h in self._hooks:
                h.handle.remove()
            self._hooks.clear()
            if count:
                logger.debug(f"Removed {count} hooks")
            return count

    def remove_layer(self, layer_idx: int) -> bool:
        """Remove hooks for a specific layer."""
        with self._lock:
            remaining = []
            removed = False
            for h in self._hooks:
                if h.layer_idx == layer_idx:
                    h.handle.remove()
                    removed = True
                else:
                    remaining.append(h)
            self._hooks = remaining
            return removed

    def summary(self) -> str:
        """Human-readable summary of active hooks."""
        with self._lock:
            if not self._hooks:
                return "No active hooks"
            lines = [f"Active hooks ({len(self._hooks)}):"]
            for h in self._hooks:
                lines.append(f"  {h.description}")
            return "\n".join(lines)
