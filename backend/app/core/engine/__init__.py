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

from app.core.engine.diagnostics import SteeringDiagnostics
from app.core.engine.hook import SteeringHook
from app.core.engine.steering import SteeringEngine

__all__ = ["SteeringDiagnostics", "SteeringHook", "SteeringEngine"]
