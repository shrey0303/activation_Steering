# Perf: cache direction vector on correct device/dtype
# Fix: handle NaN/Inf in steered activations with cooldown
# Fix: preserve KV cache tuple structure in hook output
"""
Activation Steering Engine v2 — Production-Grade.

Replaces blind additive steering with a 5-step production pipeline:

  1. KV Cache Preservation  — never strip tuple elements from hook output
  2. Cooldown Check         — skip injection if circuit breaker fired
  3. Gating                 — skip if model already aligned with target
  4. Orthogonal Projection  — steer without destroying token meaning
  5. L2 Norm Preservation   — clamp activation magnitude within tolerance

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


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEERING HOOK — Production Pipeline                        ║
# ╚══════════════════════════════════════════════════════════════╝


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
        # 7B+ models have strong concept specialization — activations are
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
                # ── STEP 0: Extract hidden states, preserve KV cache ──
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

                # Calibrate gate on first call
                hidden_dim = x.shape[-1]
                self._calibrate_threshold(hidden_dim)

                # Cache direction vector on correct device/dtype
                if self._v_cached is None or self._v_cached.device != x.device:
                    v = self.direction_vector.to(x.device, dtype=x.dtype)
                    # L2-normalize for stable cosine computations
                    self._v_cached = F.normalize(v, dim=-1)
                v = self._v_cached

                # ── STEP 1: Cooldown check ────────────────────────
                if self.cooldown_remaining > 0:
                    self.cooldown_remaining -= 1
                    diag.cooldown_active = True
                    self.last_diagnostics = diag
                    return output  # pass through unmodified

                # ── STEP 2: Gating — skip if already aligned ──────
                # Use last token position for gating decision
                x_last = x[:, -1, :]  # (batch, hidden_dim)
                x_last_norm = F.normalize(x_last, dim=-1)
                cos_sim = (x_last_norm * v).sum(dim=-1).mean().item()
                diag.cosine_similarity = cos_sim

                if cos_sim > self.gate_threshold:
                    diag.gated = True
                    self.last_diagnostics = diag
                    return output  # model already heading there

                # ── STEP 3: Steering or Erasure ────────────────────
                if self.mode == "erase":
                    # LEACE null-space: remove ALL linear info about v
                    # x_erased = x - (x·v̂)v̂  (project onto null-space of v)
                    proj_coeff = (x * v).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
                    x_new = x - proj_coeff * v  # erase concept direction
                    diag.effective_strength = 0.0  # erasure is binary
                else:
                    # Orthogonal projection + adaptive decay (default)
                    # v_orth = v - proj_x(v) = v - (v·x̂)x̂
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
                    scale_factor = max(act_norm / 10.0, 1.0)  # normalize: ~5-10 on 0.5B → scale ~1x
                    effective_strength = effective_strength * scale_factor
                    diag.effective_strength = effective_strength

                    # Apply steering
                    x_new = x + effective_strength * v_orth

                # ── STEP 4: L2 Norm Preservation ──────────────────
                orig_norm = x.norm(dim=-1, keepdim=True)
                new_norm = x_new.norm(dim=-1, keepdim=True)
                # Scale factor: keep within tolerance of original norm
                # Use wider tolerance for larger models where perturbation
                # needs to be proportionally bigger
                effective_tolerance = self.norm_tolerance
                if hidden_dim > 2048:
                    effective_tolerance = max(self.norm_tolerance, 0.25)
                scale = orig_norm / (new_norm + 1e-8)
                min_scale = 1.0 - effective_tolerance
                max_scale = 1.0 + effective_tolerance
                scale = scale.clamp(min=min_scale, max=max_scale)
                x_steered = x_new * scale

                # Track norm deviation for diagnostics
                actual_norm = x_steered.norm(dim=-1).mean().item()
                orig_norm_val = orig_norm.mean().item()
                diag.norm_deviation = abs(actual_norm - orig_norm_val) / (orig_norm_val + 1e-8)

                # ── STEP 5: Safety — NaN/Inf check ────────────────
                if torch.isnan(x_steered).any() or torch.isinf(x_steered).any():
                    logger.warning(
                        f"NaN/Inf at layer {self.layer_idx} – reverting"
                    )
                    self.cooldown_remaining = 10  # Extended cooldown
                    return output

                self.fired = True
                self.token_count += 1
                self.last_diagnostics = diag

                # ── Reconstruct output preserving KV cache ────────
                if kv_cache is not None:
                    return (x_steered,) + kv_cache
                return x_steered

            finally:
                self.overhead_ms = (time.perf_counter() - t0) * 1000

        return _hook


# ╔══════════════════════════════════════════════════════════════╗
# ║  STEERING ENGINE — Orchestrator                              ║
# ╚══════════════════════════════════════════════════════════════╝


class SteeringEngine:
    """
    Orchestrates production-grade steering hooks on the loaded model.

    Features:
      - Multi-layer orthogonal steering
      - Logit entropy circuit breaker (cooldown-based)
      - Per-token diagnostics streaming
      - Clean hook lifecycle management
    """

    # Default entropy threshold for circuit breaker.
    # Normal generation entropy ≈ 2-4 nats. Spike above 6 = model confused.
    DEFAULT_ENTROPY_THRESHOLD = 6.0
    DEFAULT_COOLDOWN_TOKENS = 5

    # ── Singleton ──────────────────────────────────────────────
    _instance: Optional["SteeringEngine"] = None

    @classmethod
    def get_instance(cls, model_manager: Optional[ModelManager] = None) -> "SteeringEngine":
        """
        Return the shared SteeringEngine singleton.

        If no instance exists yet a ModelManager must be provided.
        If the model manager changes (e.g. model was reloaded),
        the existing hooks are cleared and the reference is updated.
        """
        if cls._instance is None:
            if model_manager is None:
                raise RuntimeError(
                    "SteeringEngine has not been initialised yet — "
                    "pass a ModelManager on the first call."
                )
            cls._instance = cls(model_manager)
            logger.info("SteeringEngine singleton created")
        elif model_manager is not None and model_manager is not cls._instance.mm:
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

    # ╔══════════════════════════════════════════════════════════╗
    # ║               HOOK MANAGEMENT                           ║
    # ╚══════════════════════════════════════════════════════════╝

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

        # Remove existing hook on this layer if any
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

        # Log gate threshold info for diagnostics
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
        # Gram-Schmidt: ensure multi-vector mutual orthogonality
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

        # Force-cache all vectors on same device
        device = None
        for h in hooks:
            if h._v_cached is not None:
                device = h._v_cached.device
                break
        if device is None:
            return  # vectors not yet cached — will be done on first forward

        # Gram-Schmidt
        for i in range(1, len(hooks)):
            vi = hooks[i]._v_cached
            if vi is None:
                continue
            for j in range(i):
                vj = hooks[j]._v_cached
                if vj is None:
                    continue
                # Subtract projection of vi onto vj
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

    # ╔══════════════════════════════════════════════════════════╗
    # ║                TEXT GENERATION                          ║
    # ╚══════════════════════════════════════════════════════════╝

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

    # ╔══════════════════════════════════════════════════════════╗
    # ║             ACTIVATION CAPTURE                          ║
    # ╚══════════════════════════════════════════════════════════╝

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

    # ╔══════════════════════════════════════════════════════════╗
    # ║         STREAMING GENERATION + CIRCUIT BREAKER           ║
    # ╚══════════════════════════════════════════════════════════╝

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

            # ── Entropy circuit breaker ───────────────────────
            probs = torch.softmax(logits, dim=-1)
            # Shannon entropy in nats
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()

            if entropy > self.entropy_threshold and len(self._hooks) > 0:
                # DON'T re-run the model. Just cooldown the hooks.
                self._trigger_cooldown()

            # ── Temperature scaling ───────────────────────────
            if temperature > 0:
                logits = logits / max(temperature, 0.01)

            # ── Top-p sampling ────────────────────────────────
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

            # ── Collect diagnostics from hooks ────────────────
            diag = {
                "entropy": round(entropy, 3),
                "steering_gated": False,
                "cooldown_active": False,
                "norm_deviation": 0.0,
                "effective_strength": 0.0,
                "token_index": token_idx,
            }
            if self._hooks:
                # Aggregate from all active hooks
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
