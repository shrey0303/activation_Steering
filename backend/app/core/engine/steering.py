"""Orchestrates steering hooks on the loaded model with circuit breaking and diagnostics."""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger

from app.core.loader import ModelManager
from app.core.engine.hook import SteeringHook


class SteeringEngine:
    """Orchestrates steering hooks on the loaded model with circuit breaking and diagnostics."""

    # Default entropy threshold for circuit breaker.
    # Normal generation entropy ≈ 2-4 nats. Spike above 6 = model confused.
    DEFAULT_ENTROPY_THRESHOLD = 6.0
    DEFAULT_COOLDOWN_TOKENS = 5


    _instance: Optional["SteeringEngine"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_manager: Optional[ModelManager] = None) -> "SteeringEngine":
        """
        Return the shared SteeringEngine singleton.

        If no instance exists yet a ModelManager must be provided.
        If the model manager changes (e.g. model was reloaded),
        the existing hooks are cleared and the reference is updated.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if model_manager is None:
                        raise RuntimeError(
                            "SteeringEngine has not been initialised yet — "
                            "pass a ModelManager on the first call."
                        )
                    cls._instance = cls(model_manager)
                    logger.info("SteeringEngine singleton created")
        elif model_manager is not None and model_manager is not cls._instance.mm:
            with cls._lock:
                # Model manager changed → clear stale hooks
                cls._instance.clear_interventions()
                cls._instance.mm = model_manager
                logger.info("SteeringEngine singleton updated with new ModelManager")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton (used when unloading a model)."""
        if cls._instance is not None:
            cls._instance.clear_interventions()
            cls._instance = None
            logger.info("SteeringEngine singleton reset")

    def __init__(self, model_manager: ModelManager) -> None:
        self.mm = model_manager
        self._hooks: Dict[int, SteeringHook] = {}
        self.entropy_threshold = self.DEFAULT_ENTROPY_THRESHOLD
        self.cooldown_tokens = self.DEFAULT_COOLDOWN_TOKENS

    # --- Hook Management ---

    def add_intervention(
        self,
        layer_idx: int,
        strength: float,
        direction_vector: Optional[torch.Tensor] = None,
        gate_threshold: Optional[float] = None,
        norm_tolerance: float = 0.05,
        decay_rate: float = 0.006,
        mode: str = "steer",
    ) -> None:
        """Register a production-grade steering hook at the given layer."""
        if not self.mm.loaded:
            raise RuntimeError("Model not loaded")


        self.remove_intervention(layer_idx)

        layers = self.mm.get_layer_modules()
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(
                f"Layer index {layer_idx} out of range [0, {len(layers)-1}]"
            )

        hook = SteeringHook(
            layer_idx=layer_idx,
            strength=strength,
            direction_vector=direction_vector,
            gate_threshold=gate_threshold,
            norm_tolerance=norm_tolerance,
            decay_rate=decay_rate,
            mode=mode,
        )
        handle = layers[layer_idx].register_forward_hook(hook.get_hook_fn())
        hook.handle = handle
        self._hooks[layer_idx] = hook


        logger.info(
            f"Hook registered: layer={layer_idx}, strength={strength:.1f}, "
            f"mode={mode}, norm_tol={norm_tolerance}, decay={decay_rate}, "
            f"hidden_dim={self.mm.hidden_dim}"
        )

    def remove_intervention(self, layer_idx: int) -> None:
        """Remove hook from a specific layer."""
        if layer_idx in self._hooks:
            hook = self._hooks.pop(layer_idx)
            if hook.handle is not None:
                hook.handle.remove()
            logger.debug(f"Hook removed from layer {layer_idx}")

    def clear_interventions(self) -> None:
        """Remove all hooks."""
        for layer_idx in list(self._hooks.keys()):
            self.remove_intervention(layer_idx)
        logger.info("All steering hooks cleared")

    @property
    def active_interventions(self) -> List[Dict[str, Any]]:
        """Serialisable list of currently active interventions."""
        return [
            {
                "layer": h.layer_idx,
                "strength": h.strength,
                "fired": h.fired,
                "overhead_ms": round(h.overhead_ms, 2),
                "token_count": h.token_count,
                "cooldown_remaining": h.cooldown_remaining,
                "gate_threshold": round(h.gate_threshold, 4),
                "diagnostics": {
                    "gated": h.last_diagnostics.gated,
                    "cooldown_active": h.last_diagnostics.cooldown_active,
                    "norm_deviation": round(h.last_diagnostics.norm_deviation, 4),
                    "effective_strength": round(h.last_diagnostics.effective_strength, 2),
                    "cosine_similarity": round(h.last_diagnostics.cosine_similarity, 4),
                },
            }
            for h in self._hooks.values()
        ]

    @property
    def total_overhead_ms(self) -> float:
        return sum(h.overhead_ms for h in self._hooks.values())

    def _reset_hooks_for_generation(self) -> None:
        """Reset all hooks and orthogonalize vectors for new generation."""
        for h in self._hooks.values():
            h.reset_token_count()

        self._orthogonalize_hooks()

    def _orthogonalize_hooks(self) -> None:
        """
        Gram-Schmidt orthogonalization for multi-vector composition.

        When multiple hooks are active, their direction vectors may overlap.
        This ensures each vector steers an independent axis — zero interference.

        Paper: "Multi-Attribute Orthogonal Subspace Steering"
        """
        hooks = [h for h in self._hooks.values() if h.direction_vector is not None]
        if len(hooks) < 2:
            return


        device = None
        for h in hooks:
            if h._v_cached is not None:
                device = h._v_cached.device
                break
        if device is None:
            return  # vectors not yet cached — will be done on first forward


        for i in range(1, len(hooks)):
            vi = hooks[i]._v_cached
            if vi is None:
                continue
            for j in range(i):
                vj = hooks[j]._v_cached
                if vj is None:
                    continue

                vi = vi - (vi @ vj / (vj @ vj + 1e-8)) * vj
            hooks[i]._v_cached = F.normalize(vi, dim=-1)

        logger.debug(f"Gram-Schmidt: orthogonalized {len(hooks)} steering vectors")

    def _trigger_cooldown(self) -> None:
        """Activate cooldown on all hooks (circuit breaker)."""
        for h in self._hooks.values():
            h.cooldown_remaining = self.cooldown_tokens
        logger.warning(
            f"Circuit breaker: cooldown {self.cooldown_tokens} tokens on all hooks"
        )

    # --- Text Generation ---

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Run generation with production-grade hooks.

        Returns dict with: text, tokens_generated, latency_ms,
        tokens_per_sec, steering_applied, steering_overhead_ms,
        interventions.
        """
        if not self.mm.loaded:
            raise RuntimeError("Model not loaded")

        model = self.mm.model
        tokenizer = self.mm.tokenizer
        device = self.mm.device

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        self._reset_hooks_for_generation()

        t0 = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

        elapsed = time.perf_counter() - t0

        new_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        tokens_generated = len(new_token_ids)

        return {
            "text": generated_text,
            "prompt": prompt,
            "tokens_generated": tokens_generated,
            "latency_ms": round(elapsed * 1000, 1),
            "tokens_per_sec": round(
                tokens_generated / max(elapsed, 0.001), 1
            ),
            "steering_applied": len(self._hooks) > 0,
            "steering_overhead_ms": round(self.total_overhead_ms, 2),
            "interventions": self.active_interventions,
        }

    # --- Activation Capture ---

    @torch.no_grad()
    def capture_activations(
        self,
        prompt: str,
        layers: Optional[List[int]] = None,
        aggregation: str = "mean",
    ) -> Dict[int, float]:
        """
        Forward pass to capture per-layer activation magnitudes.

        Parameters
        ----------
        prompt : str
        layers : list[int] | None
            Specific layers to capture.  None → all layers.
        aggregation : {"mean", "max", "l2norm"}

        Returns
        -------
        dict mapping layer_index → normalised scalar value
        """
        if not self.mm.loaded:
            raise RuntimeError("Model not loaded")

        model = self.mm.model
        tokenizer = self.mm.tokenizer
        device = self.mm.device

        all_layers = self.mm.get_layer_modules()
        target_indices = layers if layers else list(range(len(all_layers)))

        activations: Dict[int, float] = {}
        handles: List[torch.utils.hooks.RemovableHook] = []

        for idx in target_indices:
            if idx < 0 or idx >= len(all_layers):
                continue

            def make_hook(layer_idx: int):
                def _capture(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h.float()
                    if aggregation == "mean":
                        val = h.abs().mean().item()
                    elif aggregation == "max":
                        val = h.abs().max().item()
                    elif aggregation == "l2norm":
                        val = torch.norm(h).item()
                    else:
                        val = h.abs().mean().item()
                    activations[layer_idx] = val
                return _capture

            handle = all_layers[idx].register_forward_hook(make_hook(idx))
            handles.append(handle)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model(**inputs)

        for h in handles:
            h.remove()

        if activations:
            max_val = max(activations.values()) or 1.0
            activations = {k: v / max_val for k, v in activations.items()}

        return activations

    # --- Streaming Generation ---

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Yield tokens one-by-one with entropy circuit breaker.

        NOTE: This implementation does NOT use KV caching — each token
        requires a full forward pass over the entire sequence (O(n²)).
        For production throughput, use generate() which delegates to
        model.generate() with internal KV caching.

        Yields dicts:
        {
            "text": str,
            "token_id": int,
            "diagnostics": {
                "entropy": float,
                "steering_gated": bool,
                "cooldown_active": bool,
                "norm_deviation": float,
                "effective_strength": float,
                "token_index": int,
            }
        }
        """
        if not self.mm.loaded:
            raise RuntimeError("Model not loaded")

        model = self.mm.model
        tokenizer = self.mm.tokenizer
        device = self.mm.device

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        self._reset_hooks_for_generation()

        for token_idx in range(max_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]

            # Entropy is intentionally measured BEFORE top-p filtering.
            # We want to detect model confusion at the raw logit level,
            # not after we've already masked low-probability tokens.
            probs = torch.softmax(logits, dim=-1)

            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()

            if entropy > self.entropy_threshold and len(self._hooks) > 0:
                # DON'T re-run the model. Just cooldown the hooks.
                self._trigger_cooldown()


            if temperature > 0:
                logits = logits / max(temperature, 0.01)


            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            final_probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(final_probs, num_samples=1)
            token_id = next_token.item()

            if token_id == tokenizer.eos_token_id:
                break

            token_text = tokenizer.decode(
                [token_id], skip_special_tokens=True
            )
            input_ids = torch.cat([input_ids, next_token], dim=-1)


            diag = {
                "entropy": round(entropy, 3),
                "steering_gated": False,
                "cooldown_active": False,
                "norm_deviation": 0.0,
                "effective_strength": 0.0,
                "token_index": token_idx,
            }
            if self._hooks:
                any_gated = any(h.last_diagnostics.gated for h in self._hooks.values())
                any_cooldown = any(h.last_diagnostics.cooldown_active for h in self._hooks.values())
                max_norm_dev = max(
                    (h.last_diagnostics.norm_deviation for h in self._hooks.values()),
                    default=0.0,
                )
                max_strength = max(
                    (h.last_diagnostics.effective_strength for h in self._hooks.values()),
                    default=0.0,
                )
                diag.update({
                    "steering_gated": any_gated,
                    "cooldown_active": any_cooldown,
                    "norm_deviation": round(max_norm_dev, 4),
                    "effective_strength": round(max_strength, 2),
                })

            yield {
                "text": token_text,
                "token_id": token_id,
                "diagnostics": diag,
            }
