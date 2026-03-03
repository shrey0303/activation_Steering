# SteerOps €" Benchmark Results

Evaluation of the SteerOps activation steering pipeline across model sizes. All tests use **Contrastive Activation Addition (CAA)** with 20 positive / 20 negative contrastive prompts per concept, evaluated on 10 held-out test prompts per strength level.

---

## Qwen2.5-0.5B (24 layers, hidden_dim=896) €" 6/25 significant

**Environment:** CPU (local) . PyTorch 2.0 . float32  
**Test Script:** [`steerops_definitive_test.py`](backend/steerops_definitive_test.py)  
**Total Runtime:** 132.7 minutes (7,965s)  
**Engine Version:** v2 (activation-norm-scaled strength)

### Layer Scanner Results

| Category | Layers | Count |
|----------|--------|-------|
| Token Embedding | 0 | 1 |
| Positional / Morphological | 1, 2, 5 | 3 |
| Syntactic Processing | 3, 4, 6 | 3 |
| Safety / Alignment | 7, 13, 17, 18 | 4 |
| Entity / Semantic | 8 | 1 |
| Reasoning / Planning | 9, 10, 12, 14 | 4 |
| Knowledge Retrieval | 11 | 1 |
| Information Integration | 15, 16 | 2 |
| Style / Personality | 19, 20, 21, 22 | 4 |
| Output Distribution | 23 | 1 |

### Steering Results (5 Concepts Ã-- 5 Strengths)

| Concept | Target Layer | Vector Mag | Best Strength | Best d | p-value | Sig? | Sig Count |
|---------|-------------|-----------|---------------|--------|---------|------|-----------|
| **Creativity** | 21 | 4.22 | **2.5** | **0.647** | **0.017** | **œ…** | **4/5** |
| **Verbosity** | 11 | 1.42 | **2.0** | **1.179** | **0.002** | **œ…** | **2/5** |
| Toxicity | 13 | 1.39 | 1.0 | 0.928 | 0.080 | Œ | 0/5 |
| Politeness | 11 | 2.18 | 2.5 | 0.559 | 0.315 | Œ | 0/5 |
| Refusal | 13 | 1.76 | 0.5 | 0.029 | 0.634 | Œ | 0/5 |

> **6/25 tests reached statistical significance (p < 0.05).** Creativity showed monotonic dose-response across 4 strengths. Verbosity d=1.18 is a large effect (Cohen's threshold: 0.8+).

### Creativity Dose-Response (0.5B)

| Strength | Cohen's d | p-value | Significant |
|----------|----------|---------|-------------|
| 0.5 | 0.053 | 0.272 | Œ |
| 1.0 | 0.344 | 0.020 | œ… |
| 1.5 | 0.423 | 0.008 | œ… |
| 2.0 | 0.532 | 0.002 | œ… |
| 2.5 | 0.647 | 0.017 | œ… |

### Pipeline-Wide Metrics (0.5B)

| Metric | Value |
|--------|-------|
| Semantic Shift (mean) | 0.366 |
| Effect Size (mean \|d\|) | 0.420 ± 0.346 |
| Perplexity Ratio (mean) | 1.135 |
| Significant tests | 6/25 |
| Fluency Preserved | œ… |
| All Pipeline Steps Pass | œ… |

---

## Qwen2.5-7B (28 layers, hidden_dim=3584) €" 6/25 significant

**Environment:** Google Colab T4 (16GB VRAM) . auto device_map . float16  
**Test Script:** [`steerops_definitive_test_7b_base.py`](backend/steerops_definitive_test_7b_base.py)  
**Total Runtime:** 206.3 minutes (12,376s)  
**Engine Version:** v2 (gating disabled for large models, activation-norm-scaled strength, 25% norm tolerance)

### Layer Scanner Results

Scanner completed in 59.8s across 28 layers.

| Category | Layers | Count |
|----------|--------|-------|
| Token Embedding | 0 | 1 |
| Positional / Morphological | 1 | 1 |
| Syntactic Processing | 2 | 1 |
| Entity / Semantic | 3, 4, 5 | 3 |
| Safety / Alignment | 6, 7, 13, 15, 21, 24 | 6 |
| Reasoning / Planning | 8, 10, 11 | 3 |
| Knowledge Retrieval | 9 | 1 |
| Information Integration | 12, 14, 16, 17, 18, 20, 22, 23 | 8 |
| Style / Personality | 19 | 1 |
| Output Distribution | 25, 26, 27 | 3 |

### Steering Results (5 Concepts Ã-- 5 Strengths)

| Concept | Target Layer | Vector Mag | Best Strength | Best d | p-value | Sig? | Sig Count |
|---------|-------------|-----------|---------------|--------|---------|------|-----------|
| **Creativity** | 19 | 39.72 | **2.0** | **1.236** | **0.003** | **œ…** | **4/5** |
| **Politeness** | 9 | 116.55 | **1.0** | **-1.258** | **0.016** | **œ…** | **2/5** |
| Toxicity | 21 | 82.32 | 2.0 | 0.440 | 0.333 | Œ | 0/5 |
| Refusal | 21 | 185.29 | -0.242 | 0.530 | 0.429 | Œ | 0/5 |
| Verbosity | 9 | 76.47 | 2.5 | 0.753 | 0.057 | Œ | 0/5 |

> **6/25 tests reached statistical significance.** Creativity d=1.24 at p=0.003 is the strongest result in the dataset. Politeness at strength 1.0 was the cleanest result €" d=-1.26, perplexity 1.14 (fluency intact).

### Creativity Dose-Response (7B)

| Strength | Cohen's d | p-value | Perplexity | Significant |
|----------|----------|---------|-----------|-------------|
| 0.5 | 0.046 | 0.808 | 1.04 | Œ |
| 1.0 | 0.640 | 0.044 | 1.77 | œ… |
| 1.5 | 1.119 | 0.012 | 2.52 | œ… |
| 2.0 | 1.236 | 0.003 | 1.61 | œ… |
| 2.5 | 1.224 | 0.005 | 2.21 | œ… |

### Pipeline-Wide Metrics (7B)

| Metric | Value |
|--------|-------|
| Semantic Shift (mean) | 0.445 |
| Effect Size (mean \|d\|) | 0.542 ± 0.416 |
| Perplexity Ratio (mean) | 1.560 |
| Significant tests | 6/25 |
| Fluency Preserved | Œ (above 1.15 threshold) |
| Hooks Functional | œ… |

### Fluency Trade-off (7B)

Above strength 1.5, coherence degrades on some concepts:

| Concept | Strength | Perplexity | Significant | Usable? |
|---------|----------|-----------|-------------|---------|
| Creativity | 1.0 | 1.77 | œ… | š ï¸ |
| Creativity | 1.5 | 2.52 | œ… | Œ |
| Politeness | 1.0 | 1.14 | œ… | œ… |
| Politeness | 1.5 | 1.40 | œ… | š ï¸ |
| Refusal | 1.5 | 3.06 | Œ | Œ |

**Practical operating range:** strength 1.0€"1.5 depending on concept. Politeness at 1.0 is the sweet spot €" significant effect with fluency genuinely intact.

---

## 7B Debugging History

### Initial failure (0/25 significant)

First 7B evaluation produced mean shift 0.004 €" steered outputs were byte-for-byte identical to baseline. Pipeline integrity checker correctly flagged: `steering_hooks_functional: false`.

### Root cause

Gating threshold and strength parameters were calibrated on 896-dimensional hidden space (0.5B). On 3584-dimensional space (7B):

1. **Gating killed every hook call.** Auto-calibrated threshold `5/ˆš3584 = 0.084` was below the natural cosine similarity between 7B activations and concept-aligned steering vectors. Every hook returned output unmodified.
2. **Perturbation was invisible.** `strength=2.5` on an activation with norm ~90 = ~2.8% perturbation. Below the threshold to change greedy argmax token.

### Diagnostic
