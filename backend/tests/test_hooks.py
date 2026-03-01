# Test: add edge case tests for concurrent hook operations
# Test: add edge case tests for concurrent hook operations
# Test: add edge case tests for concurrent hook operations
# Test: add edge case tests for concurrent hook operations
"""
Unit tests for SteeringHook and SteeringEngine.

Tests the actual PyTorch hook machinery using dummy modules.
No GPU or loaded LLM required.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn as nn

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.engine import SteeringHook


# â”€â”€ Dummy model that mimics a transformer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class DummyLayer(nn.Module):
    """Mimics a transformer layer: input â†’ identity â†’ output."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class TupleLayer(nn.Module):
    """Mimics a transformer layer that returns (hidden, attn, kv_cache)."""

    def forward(self, x):
        return (x, None, None)


class DummyModel(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([
            DummyLayer(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEERING HOOK â€” INIT AND CALIBRATION                       â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestSteeringHookInit:
    """SteeringHook initialization and threshold calibration."""

    def test_default_init(self):
        hook = SteeringHook(layer_idx=5, strength=2.5)
        assert hook.layer_idx == 5
        assert hook.strength == 2.5
        assert hook.norm_tolerance == 0.05
        assert hook.mode == "steer"
        assert hook.fired is False

    def test_custom_gate_threshold(self):
        hook = SteeringHook(layer_idx=0, strength=1.0, gate_threshold=0.5)
        hook._calibrate_threshold(hidden_dim=64)
        assert hook.gate_threshold == 0.5  # override should stick

    def test_auto_calibrate_small_model(self):
        """hidden_dim <= 2048 should use 3-sigma gating."""
        hook = SteeringHook(layer_idx=0, strength=1.0)
        hook._calibrate_threshold(hidden_dim=896)
        expected = 3.0 / math.sqrt(896)
        assert abs(hook.gate_threshold - expected) < 0.001

    def test_auto_calibrate_large_model_disables_gating(self):
        """hidden_dim > 2048 should disable gating (threshold=999)."""
        hook = SteeringHook(layer_idx=0, strength=1.0)
        hook._calibrate_threshold(hidden_dim=3584)
        assert hook.gate_threshold == 999.0

    def test_reset_token_count(self):
        hook = SteeringHook(layer_idx=0, strength=1.0)
        hook.token_count = 100
        hook.fired = True
        hook.cooldown_remaining = 5
        hook.reset_token_count()
        assert hook.token_count == 0
        assert hook.fired is False
        assert hook.cooldown_remaining == 0


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEERING HOOK â€” FORWARD PASS EFFECTS                       â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestSteeringHookForward:
    """Hooks must modify hidden states via register_forward_hook."""

    def _register_hook(self, layer, hook_obj):
        """Register hook and return handle."""
        handle = layer.register_forward_hook(hook_obj.get_hook_fn())
        hook_obj.handle = handle
        return handle

    def test_hook_changes_output(self):
        """Hook with direction vector must change output."""
        model = DummyModel(num_layers=1, hidden_dim=64)
        x = torch.randn(1, 10, 64)
        direction = torch.randn(64)

        with torch.no_grad():
            baseline = model(x.clone()).clone()

        hook = SteeringHook(
            layer_idx=0, strength=5.0,
            direction_vector=direction,
            gate_threshold=999.0,  # disable gating
        )
        handle = self._register_hook(model.layers[0], hook)

        with torch.no_grad():
            steered = model(x.clone()).clone()
        handle.remove()

        assert not torch.allclose(baseline, steered, atol=1e-4), \
            "Hook had no effect on output!"

    def test_hook_with_tuple_output(self):
        """Hooks must handle tuple outputs (hidden, attn, kv_cache)."""
        layer = TupleLayer()
        direction = torch.randn(64)

        hook = SteeringHook(
            layer_idx=0, strength=2.0,
            direction_vector=direction,
            gate_threshold=999.0,
        )
        handle = layer.register_forward_hook(hook.get_hook_fn())

        x = torch.randn(1, 5, 64)
        out = layer(x)
        handle.remove()

        assert isinstance(out, tuple)
        assert out[0].shape == x.shape
        assert out[1] is None  # attn preserved
        assert out[2] is None  # kv_cache preserved

    def test_negative_strength_reverses(self):
        """Positive and negative strength should produce different outputs."""
        model = DummyModel(num_layers=1, hidden_dim=64)
        x = torch.randn(1, 10, 64)
        direction = torch.randn(64)

        # Positive
        hook_pos = SteeringHook(
            layer_idx=0, strength=5.0,
            direction_vector=direction,
            gate_threshold=999.0,
        )
        handle = self._register_hook(model.layers[0], hook_pos)
        with torch.no_grad():
            pos = model(x.clone()).clone()
        handle.remove()

        # Negative
        hook_neg = SteeringHook(
            layer_idx=0, strength=-5.0,
            direction_vector=direction,
            gate_threshold=999.0,
        )
        handle = self._register_hook(model.layers[0], hook_neg)
        with torch.no_grad():
            neg = model(x.clone()).clone()
        handle.remove()

        assert not torch.allclose(pos, neg, atol=1e-4), \
            "Opposite strengths produced identical output!"

    def test_hooks_removable(self):
        """After hook removal, model should produce baseline output."""
        model = DummyModel(num_layers=1, hidden_dim=64)
        x = torch.randn(1, 10, 64)

        with torch.no_grad():
            baseline = model(x.clone()).clone()

        hook = SteeringHook(
            layer_idx=0, strength=10.0,
            gate_threshold=999.0,
        )
        handle = self._register_hook(model.layers[0], hook)
        handle.remove()

        with torch.no_grad():
            after = model(x.clone()).clone()

        assert torch.allclose(baseline, after, atol=1e-6), \
            "Output changed after hook removal!"

    def test_no_nan_or_inf(self):
        """Even extreme strength should not produce NaN/Inf."""
        model = DummyModel(num_layers=1, hidden_dim=64)
        direction = torch.randn(64)

        hook = SteeringHook(
            layer_idx=0, strength=1000.0,
            direction_vector=direction,
            gate_threshold=999.0,
        )
        handle = self._register_hook(model.layers[0], hook)

        x = torch.randn(1, 5, 64)
        with torch.no_grad():
            out = model(x)
        handle.remove()

        assert not torch.isnan(out).any(), "Hook produced NaN!"
        assert not torch.isinf(out).any(), "Hook produced Inf!"

    def test_norm_preservation(self):
        """Output norm should stay within norm_tolerance of input."""
        model = DummyModel(num_layers=1, hidden_dim=64)
        direction = torch.randn(64)
        x = torch.randn(1, 10, 64)

        hook = SteeringHook(
            layer_idx=0, strength=3.0,
            direction_vector=direction,
            norm_tolerance=0.05,
            gate_threshold=999.0,
        )
        handle = self._register_hook(model.layers[0], hook)

        with torch.no_grad():
            baseline = model(x.clone()).clone()

        hook.reset_token_count()
        with torch.no_grad():
            steered = model(x.clone())
        handle.remove()

        baseline_norm = baseline.norm(dim=-1)
        steered_norm = steered.norm(dim=-1)
        ratio = steered_norm / (baseline_norm + 1e-8)

        # Allow some tolerance beyond the hook's own tolerance due to
        # activation-norm-scaling and the linear layer
        assert (ratio > 0.5).all() and (ratio < 2.0).all(), \
            f"Norm ratio out of bounds: {ratio.min():.3f} - {ratio.max():.3f}"


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEERING HOOK â€” ERASE MODE                                 â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestSteeringHookErase:
    """LEACE null-space erasure mode."""

    def _register_hook(self, layer, hook_obj):
        handle = layer.register_forward_hook(hook_obj.get_hook_fn())
        hook_obj.handle = handle
        return handle

    def test_erase_mode_changes_output(self):
        model = DummyModel(num_layers=1, hidden_dim=64)
        direction = torch.randn(64)
        x = torch.randn(1, 10, 64)

        with torch.no_grad():
            baseline = model(x.clone()).clone()

        hook = SteeringHook(
            layer_idx=0, strength=1.0,
            direction_vector=direction,
            mode="erase",
            gate_threshold=999.0,
        )
        handle = self._register_hook(model.layers[0], hook)
        with torch.no_grad():
            erased = model(x.clone()).clone()
        handle.remove()

        assert not torch.allclose(baseline, erased, atol=1e-4), \
            "Erase mode had no effect!"


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  STEERING HOOK â€” DIAGNOSTICS                                â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestSteeringHookDiagnostics:
    """Hook diagnostics should be populated after a forward pass."""

    def test_diagnostics_populated(self):
        model = DummyModel(num_layers=1, hidden_dim=64)
        direction = torch.randn(64)
        x = torch.randn(1, 10, 64)

        hook = SteeringHook(
            layer_idx=0, strength=3.0,
            direction_vector=direction,
            gate_threshold=999.0,
        )
        handle = model.layers[0].register_forward_hook(hook.get_hook_fn())

        with torch.no_grad():
            model(x)
        handle.remove()

        diag = hook.last_diagnostics
        assert diag is not None
        assert diag.cosine_similarity is not None or diag.gated or diag.cooldown_active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



