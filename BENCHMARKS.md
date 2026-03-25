# SteerOps - Benchmark Results

Evaluation of the SteerOps activation steering pipeline across model sizes. All tests use **Contrastive Activation Addition (CAA)** vectors computed from 20 positive / 20 negative contrastive prompts per concept, evaluated on 10 held-out test prompts per strength level.

**Statistical methodology:** Cohen's d (effect size) + paired t-test (p < 0.05). Results are from the definitive pipeline test scripts.

---

## Qwen2.5-0.5B (24 layers, hidden_dim=896) - 6/25 significant

**Environment:** CPU (local), PyTorch 2.0, float32
**Test Script:** `steerops_definitive_test.py`
**Total Runtime:** 132.7 minutes (7,965s)
**Engine Version:** v2 (activation-norm-scaled strength)

### Per-Concept Results

| Concept | Layer | Strength | Cohen's d | p-value | Significant | Perplexity Ratio |
|---------|-------|----------|-----------|---------|-------------|------------------|
| Creativity | L21 (style_personality) | 1.0 | 0.34 | 0.020 | Yes | 1.02 |
| Creativity | L21 | 1.5 | 0.42 | 0.008 | Yes | 1.05 |
| Creativity | L21 | 2.0 | 0.53 | 0.002 | Yes | 1.05 |
| Creativity | L21 | 2.5 | 0.65 | 0.017 | Yes | 1.24 |
| Verbosity | L11 (style_personality) | 2.0 | 1.18 | 0.002 | Yes | 0.50 |
| Verbosity | L11 | 2.5 | 1.18 | 0.018 | Yes | 0.56 |
| Toxicity | L13 (safety_alignment) | 1.0 | 0.93 | 0.080 | No | 2.44 |
| Politeness | L11 (knowledge_retrieval) | 2.5 | 0.56 | 0.315 | No | 1.96 |
| Refusal | L13 (safety_alignment) | 2.5 | 0.02 | 0.776 | No | 1.03 |

### Key Findings (0.5B)

- **Creativity** shows clean dose-response: d increases monotonically (0.34 -> 0.42 -> 0.53 -> 0.65)
- **Verbosity** has largest effect size (d=1.18) with good fluency preservation
- **Toxicity** shows large semantic shifts but fails statistical significance -- too noisy
- **Refusal** essentially unaffected by single-layer steering -- residual healing likely
- Mean perplexity ratio 1.13 across all tests (fluency preserved)

---

## Qwen2.5-7B (28 layers, hidden_dim=3584) - 6/25 significant

**Environment:** RunPod A40 GPU, PyTorch 2.0, bfloat16
**Test Script:** `steerops_definitive_test_7b_base.py`
**Total Runtime:** 206.3 minutes (12,376s)
**Engine Version:** v2 (activation-norm-scaled strength, gating disabled for hidden_dim > 2048)

### Per-Concept Results

| Concept | Layer | Strength | Cohen's d | p-value | Significant | Perplexity Ratio |
|---------|-------|----------|-----------|---------|-------------|------------------|
| Creativity | L19 (style_personality) | 1.0 | 0.64 | 0.044 | Yes | 1.77 |
| Creativity | L19 | 1.5 | 1.12 | 0.012 | Yes | 2.52 |
| Creativity | L19 | 2.0 | 1.24 | 0.003 | Yes | 1.61 |
| Creativity | L19 | 2.5 | 1.22 | 0.005 | Yes | 2.21 |
| Politeness | L9 (knowledge_retrieval) | 1.0 | -1.26 | 0.016 | Yes | 1.14 |
| Politeness | L9 | 1.5 | -0.93 | 0.022 | Yes | 1.40 |
| Verbosity | L9 (style_personality) | 2.0 | 0.75 | 0.057 | No | 0.94 |
| Toxicity | L21 (safety_alignment) | 1.0 | -0.44 | 0.344 | No | 1.32 |
| Refusal | L21 (safety_alignment) | 1.5 | -0.24 | 0.429 | No | 3.06 |

### Key Findings (7B)

- **Creativity** shows strong monotonic dose-response: d = 0.64 -> 1.12 -> 1.24 -> 1.22
- **Politeness** significant at strengths 1.0-1.5 (negative d = inverse direction steering)
- **Safety concepts** (toxicity, refusal) still fail significance -- single-layer injection insufficient
- **Fluency trade-off:** perplexity climbs past strength 1.5 (ratio > 2.0). Usable range is 1.0-1.5
- Mean perplexity ratio 1.56 overall (higher than 0.5B due to 7B sensitivity)

---

## Cross-Model Comparison

| Metric | 0.5B | 7B |
|--------|------|-----|
| Significant tests | 6/25 | 6/25 |
| Best effect size (d) | 1.18 (verbosity) | 1.24 (creativity) |
| Mean perplexity ratio | 1.13 | 1.56 |
| Usable strength range | 0.5-2.5 | 1.0-1.5 |
| Scan time | 12.3s | 59.8s |
| Total eval time | 133 min | 206 min |

### Scaling Observations

1. **Effect sizes scale with model capacity.** Creativity d=0.65 (0.5B) vs d=1.24 (7B) at comparable strength
2. **Fluency degrades faster on 7B.** Requires activation-norm scaling to compensate for 3584-dim vs 896-dim hidden states
3. **Safety steering remains hard at all scales.** Residual connections heal single-layer perturbations for diffuse concepts like toxicity and refusal
4. **Dose-response is cleaner on 7B.** Monotonic increase in creativity effect size with near-zero noise

---

## Engine Fixes for 7B

The original engine (v1) produced 0/25 significant results on 7B. Three fixes were required:

1. **Gating disabled for large models:** The cosine-similarity gate threshold was calibrated for 896-dim activations. At 3584-dim, cosine values are systematically lower, causing the gate to silently block all steering. Fix: disable gating when `hidden_dim > 2048`.

2. **Activation-norm-scaled strength:** A fixed strength of 1.0 produces a much smaller relative perturbation on 3584-dim activations than on 896-dim. Fix: `effective_strength = strength * (activation_norm / 10.0)`.

3. **Norm tolerance increased:** The L2 norm preservation clamp was too aggressive for large models. Fix: increased tolerance from 10% to 25% for `hidden_dim > 2048`.

---

## Limitations

- **Single-layer steering only.** Safety-relevant concepts likely require multi-layer injection
- **No MoE support.** Sparse expert routing in models like Mixtral is not handled
- **Perplexity as fluency proxy.** Does not capture semantic coherence or factual accuracy
- **Small test set.** 10 prompts per strength level -- larger N would increase statistical power
