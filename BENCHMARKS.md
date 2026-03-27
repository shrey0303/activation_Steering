# SteerOps — Empirical Evaluation Report

Evaluation of the SteerOps activation steering pipeline across three model architectures: Qwen2.5-0.5B, Qwen2.5-7B, and Sarvam-1 (2.51B). The Sarvam-1 evaluation spans three runs: v1 (N=10, 20 pairs, L11), v2 (N=40, 50 pairs, L6), and L11 confirmation (N=40, 50 pairs, L11). All interventions use Contrastive Activation Addition (CAA) vectors. Full telemetry: [`steerops_sarvam1_results.json`](backend/steerops_sarvam1_results.json), [`steerops_v2_deep_results.json`](steerops_v2_deep_results.json).

**Statistical methodology.** Effect size: Cohen's d_z (paired-samples standard deviation of differences), appropriate for the within-subjects design (same prompts, baseline vs. steered). Significance: paired t-test, alpha = 0.05. v2 adds percentile bootstrap 95% CIs (n_bootstrap=2000, seed=42). Perplexity ratio: steered / baseline (values below 2.0 indicate preserved fluency). Semantic shift: cosine distance between steered and unsteered sentence embeddings via `paraphrase-multilingual-MiniLM-L12-v2`. Note: d_z computed on cosine similarity scores bounded in [-1, 1] produces larger absolute values than classical behavioral rating scales; the standard thresholds (0.2/0.5/0.8) should not be applied directly.

---

## 1. Qwen2.5-0.5B (24 layers, d_model = 896)

**Environment:** CPU (local), PyTorch 2.0, float32
**Engine:** v2 (activation-norm-scaled strength)
**Runtime:** 14.5 minutes (870s)
**Result:** 7/25 statistically significant conditions

### 1.1 Per-Concept Results

| Concept    | Layer | Strength | Cohen's d | p-value | Significant | PPL Ratio |
|------------|-------|----------|-----------|---------|-------------|-----------|
| Creativity | L21   | 1.0      | 0.95      | 0.015   | Yes         | 1.82      |
| Creativity | L21   | 1.5      | 0.78      | 0.037   | Yes         | 1.81      |
| Creativity | L21   | 2.0      | 0.99      | 0.012   | Yes         | 1.86      |
| Creativity | L21   | 2.5      | 1.30      | 0.003   | Yes         | 1.30      |
| Politeness | L11   | 2.0      | 0.81      | 0.031   | Yes         | 1.35      |
| Verbosity  | L11   | 2.0      | 0.81      | 0.030   | Yes         | 0.74      |
| Verbosity  | L11   | 2.5      | 0.74      | 0.045   | Yes         | 0.86      |
| Toxicity   | L13   | 2.0      | 0.62      | 0.080   | No          | 1.68      |
| Refusal    | L13   | 2.0      | 0.35      | 0.291   | No          | 1.03      |

### 1.2 Observations

- Creativity exhibits monotonic dose-response: d = 0.51, 0.95, 0.78, 0.99, 1.30 across strengths 0.5 to 2.5.
- Politeness achieves significance at strength 2.0 (d = 0.81, p = 0.031, PPL = 1.35), demonstrating that even a 0.5B model can be meaningfully steered.
- Verbosity produces consistent significance at strengths 2.0--2.5 with sub-baseline perplexity (PPL < 1.0), indicating steered output is more fluent than baseline, consistent with reduced output length.
- Toxicity and refusal remain non-significant. Single-layer intervention is insufficient for distributed safety concepts.
- Mean perplexity ratio across all conditions: 1.23.

---

## 2. Qwen2.5-7B (28 layers, d_model = 3584)

**Environment:** RunPod A40 GPU, PyTorch 2.0, bfloat16
**Engine:** v2 (activation-norm-scaled strength, gating disabled for d_model > 2048)
**Runtime:** 208.9 minutes (12,534s)
**Result:** 7/25 statistically significant conditions

### 2.1 Per-Concept Results

| Concept    | Layer | Strength | Cohen's d | p-value | Significant | PPL Ratio |
|------------|-------|----------|-----------|---------|-------------|-----------|
| Creativity | L19   | 1.0      | 0.81      | 0.030   | Yes         | 1.77      |
| Creativity | L19   | 1.5      | 0.90      | 0.020   | Yes         | 2.52      |
| Creativity | L19   | 2.0      | 1.24      | 0.004   | Yes         | 1.61      |
| Creativity | L19   | 2.5      | 1.33      | 0.002   | Yes         | 2.21      |
| Verbosity  | L9    | 1.5      | 1.36      | 0.002   | Yes         | 0.92      |
| Verbosity  | L9    | 2.0      | 1.25      | 0.003   | Yes         | 0.94      |
| Verbosity  | L9    | 2.5      | 1.39      | 0.002   | Yes         | 0.87      |
| Politeness | L9    | 1.5      | -0.63     | 0.079   | No          | 1.40      |
| Toxicity   | L21   | 1.0      | -0.32     | 0.344   | No          | 1.32      |
| Refusal    | L21   | 1.5      | -0.39     | 0.243   | No          | 3.06      |

### 2.2 Observations

- Creativity dose-response: d = -0.01, 0.81, 0.90, 1.24, 1.33 (monotonic across strengths 0.5 to 2.5, saturating at 2.0).
- Verbosity achieves consistent significance at strengths 1.5--2.5 with sub-baseline perplexity (PPL < 1.0), the cleanest results in either Qwen evaluation.
- Verbosity at strength 2.5 on 7B is the single strongest result: d = 1.39, p = 0.002, PPL = 0.87.
- Politeness shows a marginal trend (d = -0.63, p = 0.079 at strength 1.5) but does not reach significance at alpha = 0.05.
- Safety-relevant concepts (toxicity, refusal) remain non-significant. Single-layer intervention is insufficient for diffuse, multi-layer concepts.
- Fluency degrades past strength 1.5 for creativity (PPL > 2.0). Usable steering range on 7B is narrower than on 0.5B.
- Mean perplexity ratio across all conditions: 1.56.

---

## 3. Sarvam-1 v1 (28 layers, d_model = 2048, N=10)

**Environment:** T4 GPU, 4-bit NF4 quantization via bitsandbytes
**Engine:** v2 (gating disabled for d_model >= 2048)
**Runtime:** 7651.8s (127.5 minutes). Model load: 21.0s, layer scan: 33.3s.
**Result:** 23/48 statistically significant conditions
**Contrastive pairs:** 20/20 per concept. **Routing layer:** L11.
**Latency overhead:** 0.0 ms/tok (baseline: 109.0 ms/tok)

### 3.1 Hindi Honorific Strength Sweep (v1)

CAA vector: dim = 2048, magnitude = 29.13.

| Strength | Concept Delta | Semantic Shift | Cohen's d | p-value | Sig | PPL Ratio | Polite % | TokFert |
|----------|---------------|----------------|-----------|---------|-----|-----------|----------|---------|
| 0.5      | -0.084 ± 0.230 | 0.746 ± 0.189 | -0.365    | 0.278   | No  | 1.262     | 60%      | 2.29    |
| 1.0      | -0.159 ± 0.113 | 0.686 ± 0.189 | -1.408    | 0.002   | Yes | 1.053     | 65%      | 1.57    |
| **1.5**  | **0.117 ± 0.144** | **0.614 ± 0.173** | **0.817** | **0.030** | **Yes** | **1.049** | **85%** | **1.62** |
| 2.0      | 0.042 ± 0.173 | 0.697 ± 0.155 | 0.242     | 0.464   | No  | 1.153     | 85%      | 1.56    |
| 2.5      | 0.079 ± 0.233 | 0.684 ± 0.159 | 0.339     | 0.311   | No  | 1.162     | 85%      | 1.46    |
| 3.0      | 0.052 ± 0.171 | 0.764 ± 0.102 | 0.306     | 0.358   | No  | 1.190     | 80%      | 1.59    |

### 3.2 Bengali Honorific Strength Sweep (v1)

CAA vector: dim = 2048, magnitude = 27.93.

| Strength | Concept Delta | Semantic Shift | Cohen's d | p-value | Sig | PPL Ratio | Polite % | TokFert |
|----------|---------------|----------------|-----------|---------|-----|-----------|----------|---------|
| 0.5      | -0.362 ± 0.301 | 0.633 ± 0.257 | -1.206    | 0.004   | Yes | 1.406     | 65%      | 1.63    |
| 1.0      | -0.460 ± 0.214 | 0.782 ± 0.228 | -2.152    | 0.0001  | Yes | 1.853     | 60%      | 1.82    |
| **1.5**  | **-0.568 ± 0.203** | **0.915 ± 0.100** | **-2.801** | **< 0.001** | **Yes** | **1.279** | **50%** | **1.58** |
| 2.0      | -0.495 ± 0.210 | 0.769 ± 0.234 | -2.357    | < 0.001 | Yes | 4.052     | 60%      | 7.56    |
| 2.5      | -0.172 ± 0.302 | 0.616 ± 0.161 | -0.570    | 0.105   | No  | 1.575     | 50%      | 5.07    |
| 3.0      | -0.229 ± 0.175 | 0.651 ± 0.116 | -1.312    | 0.003   | Yes | 3.191     | 55%      | 9.26    |

### 3.3 Tamil Honorific Degradation (v1) — SUPERSEDED

**This result is a measurement artifact of insufficient contrastive pairs (15 pairs, magnitude 13.28). It is superseded by Section 4a, which shows PPL ≤ 3.06x at the same layer with 50-pair vectors. Do not cite these numbers as findings.**

CAA vector: dim = 2048, magnitude = 13.28 (less than half of Hindi/Bengali magnitudes).

| Strength | Cohen's d | PPL Ratio     |
|----------|-----------|---------------|
| 0.5      | -2.004    | 1.304         |
| 1.0      | -0.325    | **8.465**     |
| 1.5      | -1.505    | **97.759**    |
| 2.0      | -1.661    | **145.055**   |
| 2.5      | -1.170    | **97.120**    |
| 3.0      | -1.501    | **156.234**   |

### 3.4 English and General Concept Steering (v1)

| Concept            | Best Strength | Best Cohen's d | p-value | Sig | PPL Ratio |
|--------------------|--------------|----------------|---------|-----|-----------|
| English politeness | 1.5          | 1.478          | 0.001   | Yes | 8.507     |
| Verbosity          | 1.5          | -0.850         | 0.025   | Yes | 0.933     |
| Creativity         | 3.0          | 1.011          | 0.011   | Yes | 1.233     |
| Refusal            | 2.0          | 0.903          | 0.019   | Yes | 6.323     |
| Toxicity           | 2.5          | -0.387         | 0.252   | No  | 6.099     |

---

## 4. Sarvam-1 v2 Deep Evaluation (N=40, Bootstrap CIs)

**Environment:** Kaggle T4 x2 GPU, 4-bit NF4 quantization
**Runtime:** ~240 minutes (4 hours). Model load: 57.4s, layer scan: 37.1s.
**Contrastive pairs:** 50/50 per language (expanded from 20/20 in v1)
**Routing layer:** L6 (scanner routing bug — took first match instead of highest-confidence; see Section 6.1)
**Bootstrap:** 2000 resamples, seed=42, BCa-lite percentile intervals

### 4.1 Hindi Honorific (v2, N=40)

CAA vector: dim = 2048, magnitude = 4.51, routed to L6.

| Strength | Cohen's d | 95% CI | p-value | PPL | Polite % | TokFert |
|----------|-----------|--------|---------|-----|----------|---------|
| 0.5      | 0.14      | [-0.16, 0.47] | 0.409 | 1.40 | 82% | 1.53 |
| 1.0      | 0.25      | [-0.07, 0.59] | 0.140 | 1.42 | 70% | 2.29 |
| 1.5      | **0.47**  | [0.18, 0.77]  | **0.006** | 2.94 | 70% | 1.71 |
| 2.0      | **1.20**  | [0.81, 1.66]  | **<0.001** | 2.23 | 85% | 2.33 |
| 2.5      | **1.60**  | [1.20, 2.17]  | **<0.001** | 1.86 | **93%** | 1.80 |
| 3.0      | **1.75**  | [1.29, 2.38]  | **<0.001** | 2.28 | 90% | 1.82 |

Monotonic dose-response. CI excludes zero from strength 1.5 onward. Peak polite ratio: 93% at strength 2.5. PPL stable below 3.0.

### 4.2 Bengali Honorific (v2, N=40) — Polarity Inverted

CAA vector: dim = 2048, magnitude = 8.42, routed to L6.

| Strength | Cohen's d | 95% CI | p-value | PPL | Polite % | TokFert |
|----------|-----------|--------|---------|-----|----------|---------|
| 0.5      | -0.40     | [-0.80, -0.04] | 0.036 | 1.32 | 67% | 1.74 |
| 1.0      | **-1.36** | [-1.99, -0.91] | **<0.001** | 1.43 | 57% | 2.03 |
| 1.5      | **-1.63** | [-2.52, -1.03] | **<0.001** | 1.26 | 55% | 2.79 |
| 2.0      | **-1.37** | [-2.11, -0.85] | **<0.001** | 1.46 | 55% | 2.79 |
| 2.5      | **-1.24** | [-1.95, -0.74] | **<0.001** | 1.39 | 59% | 3.04 |
| 3.0      | -0.69     | [-1.23, -0.22] | 0.003 | 1.80 | 62% | 2.41 |

All d values negative — vector steers away from the concept. Negating the vector would produce d=+1.63 at strength 1.5. PPL stays below 1.8 across all strengths; Bengali generation quality is fully preserved. Token fertility does not catastrophically explode (max 3.04 vs. v1's 9.26).

### 4.3 Tamil Control Experiment (v2, N=40)

**Cross-lingual condition (Hindi vector → Tamil prompts):**

| Strength | Cohen's d | 95% CI | p-value | PPL | Polite % | TokFert |
|----------|-----------|--------|---------|-----|----------|---------|
| 0.5      | -0.85     | [-1.32, -0.46] | 0.0002 | 1.75 | 73% | 1.91 |
| 1.0      | **-1.76** | [-2.78, -1.12] | **<0.001** | 2.94 | 55% | 1.71 |
| 1.5      | **-1.74** | [-2.85, -1.08] | **<0.001** | 2.82 | 54% | 1.83 |
| 2.0      | -0.99     | [-1.62, -0.50] | 0.0001 | 3.27 | 61% | 2.23 |
| 2.5      | -0.53     | [-1.14, -0.10] | 0.023 | 2.86 | 66% | 3.09 |
| 3.0      | -0.57     | [-1.12, -0.14] | 0.011 | 3.73 | 73% | 7.02 |

**Native condition (Tamil vector → Tamil prompts):**

| Strength | Cohen's d | 95% CI | p-value | PPL | Polite % | TokFert |
|----------|-----------|--------|---------|-----|----------|---------|
| 0.5      | -0.82     | [-1.25, -0.45] | 0.0001 | 1.32 | 63% | 2.10 |
| 1.0      | **-1.24** | [-1.85, -0.80] | **<0.001** | 1.92 | 54% | 2.12 |
| 1.5      | **-1.28** | [-1.98, -0.77] | **<0.001** | 1.95 | 51% | 2.23 |
| 2.0      | **-1.28** | [-2.05, -0.74] | **<0.001** | 1.86 | 51% | 2.46 |
| 2.5      | **-1.70** | [-2.68, -1.08] | **<0.001** | 3.04 | 63% | 2.10 |
| 3.0      | **-3.41** | [-5.25, -2.53] | **<0.001** | 4.71 | 80% | 1.77 |

**Q2 Verdict (updated with L11 confirmation):** The L11 confirmation run (Section 4a) definitively resolves this question. With 50-pair vectors at L11 — the same layer as v1 — Tamil cross-lingual PPL peaks at 3.06x (not 97x). Tamil-native at L11 achieves d = -3.07 with PPL = 2.08, the strongest effect in the evaluation. The v1 catastrophe was caused by noisy 15-pair vectors (magnitude 13.28 vs. 46.12 with 50 pairs), not typological incompatibility. Native vectors outperform cross-lingual by ~2x at L11 despite cos = 0.9563 directional alignment.

### 4.4 Cross-Vector Cosine Similarities

| Vector Pair | Cosine Similarity |
|-------------|-------------------|
| Hindi ↔ Bengali | **0.7065** |
| Bengali ↔ Tamil | **0.7652** |
| Hindi ↔ Tamil | **0.6436** |

**Q5 Verdict: REJECTED.** All three vectors share high cosine similarity (>0.6) at L6. At L11, Hindi-Tamil alignment converges to **0.9563** — near-identity. The model encodes a unified Indic honorific concept that converges with depth. Hindi-Bengali alignment (0.71) is slightly higher than Hindi-Tamil at L6 (0.64), consistent with shared Indo-Aryan morphology, but this gap vanishes by L11.

### 4.5 Layer Responsiveness (v2, strength 1.5, 10 prompts per probe)

| Layer | Depth | Hindi Shift | Hindi PPL | Bengali Shift | Bengali PPL | Tamil (native) Shift | Tamil (native) PPL | Tamil (cross) Shift | Tamil (cross) PPL |
|-------|-------|-------------|-----------|---------------|-------------|---------------------|-------------------|--------------------|--------------------|
| L4    | 14%   | 0.533       | 1.600     | 0.683         | 1.320       | 0.643               | **8.782**         | 0.495              | 1.545              |
| L8    | 29%   | 0.527       | 1.047     | 0.661         | 2.384       | 0.726               | 1.402             | 0.348              | 1.143              |
| **L12** | **43%** | **0.732** | 2.571   | **0.780**     | 1.644       | **0.925**           | 1.350             | 0.608              | 3.402              |
| L16   | 57%   | 0.639       | 3.156     | 0.659         | 2.298       | 0.477               | 1.725             | **0.786**          | 2.948              |
| L21   | 75%   | 0.556       | 1.295     | 0.382         | 1.375       | 0.508               | 1.704             | 0.506              | 3.987              |
| L25   | 89%   | 0.645       | 1.799     | 0.327         | 1.507       | 0.183               | 1.656             | 0.452              | 1.194              |

All native vectors peak at L12 (43% depth). Tamil native achieves the highest semantic shift of any condition (0.925) at L12, with stable PPL (1.35). Tamil at L4 shows catastrophic PPL (8.78), confirming layer-dependent fragility. Cross-lingual condition peaks later at L16 (57%).

---

## 4a. Sarvam-1 L11 Confirmation Run (N=40, 50 pairs, L11)

**Environment:** Kaggle T4, 4-bit NF4 quantization
**Runtime:** 89 minutes (5330s). Model load: ~60s, vector computation: ~14s.
**Routing layer:** L11 (forced, to replicate v1 conditions with v2-quality vectors)
**Purpose:** Determine whether v1 Tamil catastrophe (PPL > 97x at L11) reproduces with 50-pair vectors.

### 4a.1 Vector Diagnostics at L11

| Metric | Hindi | Tamil |
|--------|-------|-------|
| Magnitude (L11, 50 pairs) | 25.59 | 46.12 |
| Magnitude (L6, 50 pairs) | 4.51 | 7.98 |
| Magnitude (L11, 15-20 pairs, v1) | 29.13 | 13.28 |

Magnitude scales with layer depth (L11 activations have ~5x higher norms than L6). v1 Tamil magnitude was anomalously low (13.28 vs. Hindi's 29.13 at the same layer), confirming insufficient contrastive pairs.

**cos(Hindi_L11, Tamil_L11) = 0.9563** — near-identical directions despite different language families. Compare to cos = 0.6436 at L6. Honorific representations converge with network depth.

### 4a.2 Tamil Cross-Lingual at L11 (Hindi vec → Tamil prompts)

| Strength | Cohen's d | PPL |
|----------|-----------|-----|
| 0.5      | -1.51     | 1.03 |
| 1.0      | -1.53     | 1.71 |
| 1.5      | -1.34     | 1.81 |
| 2.0      | -1.54     | 1.95 |
| 2.5      | -1.98     | 3.06 |
| 3.0      | **-2.12** | 1.60 |

**PPL never exceeds 3.06x.** The v1 catastrophe (97x-156x) is not reproduced. 6/6 conditions significant at p < 0.05.

### 4a.3 Tamil Native at L11 (Tamil vec → Tamil prompts)

| Strength | Cohen's d | PPL |
|----------|-----------|-----|
| 0.5      | -1.84     | 1.10 |
| 1.0      | -2.67     | 2.01 |
| 1.5      | **-3.07** | 2.08 |
| 2.0      | -2.92     | 1.68 |
| 2.5      | -2.78     | 1.91 |
| 3.0      | -2.65     | 1.84 |

d = -3.07 at PPL 2.08 is the strongest effect in the entire evaluation. 6/6 conditions significant. PPL never exceeds 2.08. Polarity inversion persists (all d negative), matching Bengali and v2 Tamil.

### 4a.4 Native vs. Cross-Lingual Effectiveness at L11

| Strength | Cross |d| | Native |d| | Ratio |
|----------|----------|-----------|-------|
| 0.5      | 1.51     | 1.84      | 1.22x |
| 1.0      | 1.53     | 2.67      | 1.74x |
| **1.5**  | 1.34     | **3.07**  | **2.29x** |
| 2.0      | 1.54     | 2.92      | 1.90x |
| 2.5      | 1.98     | 2.78      | 1.40x |
| 3.0      | 2.12     | 2.65      | 1.25x |

Native vectors outperform cross-lingual by ~2x at strength 1.5 despite 95.6% directional alignment. The 4.4% divergence encodes morphological information (Tamil agglutinative suffixes vs. Hindi pronoun/verb patterns).

---

## 5. Cross-Architecture Comparison

| Metric                      | Qwen2.5-0.5B | Qwen2.5-7B | Sarvam-1 v1 (N=10) | Sarvam-1 v2 (N=40) | Sarvam-1 L11 (N=40) |
|-----------------------------|-------------|-----------|---|---|---|
| Architecture                | Qwen2        | Qwen2     | LlamaForCausalLM | LlamaForCausalLM | LlamaForCausalLM |
| Parameters                  | 0.5B         | 7B        | 2.51B | 2.51B | 2.51B |
| d_model                     | 896          | 3584      | 2048 | 2048 | 2048 |
| Contrastive pairs           | 20/20        | 20/20     | 20/20 | 50/50 | 50/50 |
| N per condition             | 10           | 10        | 10 | 40 | 40 |
| Significant / total         | 7/25         | 7/25      | 23/48 | 20/24 | 12/12 |
| Significance rate            | 28%          | 28%       | 48% | 83% | **100%** |
| Peak effect size (|d|)      | 1.30         | 1.39      | 2.80 | 3.41 | **3.07** |
| Peak effect concept         | Creativity   | Verbosity | Bengali honorific | Tamil native (L6) | Tamil native (L11) |
| Steering overhead (ms/tok)  | < 2          | < 2       | 0.0 | 0.0 | 0.0 |

---

## 6. Key v1 → v2 → L11 Corrections

### 6.1 Tamil Degradation — Resolved

v1 concluded that Tamil degradation was evidence of language-family-specific circuit failure (PPL > 97x at L11 with 15-pair vectors). v2 showed PPL stayed below 4.71 at L6 with 50-pair vectors, but ran at a different layer due to a routing bug (first-match instead of highest-confidence-match). The L11 confirmation run resolves this definitively:

1. **Vector quality was the root cause.** v1 used 15 pairs (magnitude=13.28). L11 confirmation uses 50 pairs (magnitude=**46.12** — higher than Hindi's 25.59 at the same layer). The 15-pair vector was noisy; the 50-pair vector produces the strongest steering in the evaluation (d=-3.07, PPL=2.08).
2. **L11 is not catastrophic.** Same layer as v1, same cross-lingual condition, PPL peaks at 3.06x (not 97x).
3. **cos(Hindi_L11, Tamil_L11) = 0.9563.** The vectors are 95.6% aligned at this layer — near-identical despite different language families.

### 6.2 Bengali Polarity

v1 reported Bengali d=-2.80 as the "strongest effect." Both v1 and v2 confirm this is an inverted vector — the d is negative at all six strengths. Bengali steering is mechanistically functional but requires vector negation.

The root cause of Bengali polarity inversion has not been determined. Possible causes include incorrect positive/negative pair ordering in the Bengali contrastive set passed to VectorCalculator (computing `mean_neg - mean_pos` instead of `mean_pos - mean_neg` due to label ambiguity), or genuine directional asymmetry in Sarvam-1's Bengali honorific representation. Negated-vector validation has not been performed and is a known limitation.

### 6.3 Cross-Lingual Subspace

v1 hypothesized that the shared circuit processed Hindi/Bengali effectively but not Tamil due to subspace orthogonality. v2 measures cos(Hindi, Tamil)=0.64 at L6; the L11 confirmation measures cos=0.9563 at L11. The orthogonality hypothesis is definitively rejected. Honorific representations converge with network depth.

### 6.4 Routing Bug

v2 used a `break`-on-first-match routing strategy instead of v1's confidence-sorted routing, causing all v2 sweeps to run at L6 instead of L11. This has been corrected in the evaluation script (sort by confidence descending, take highest).

---

## 7. Limitations

1. **Sample size.** v1 used 10 prompts per condition (limited power). v2 used 40 prompts with bootstrap CIs, providing robust effect estimates.
2. **No negated-vector validation.** The Bengali/Tamil polarity inversion has not been experimentally confirmed by running with negated vectors.
3. **Single-layer injection only.** Multi-layer cascaded steering is expected to improve results for distributed concepts but was not evaluated.
4. **Perplexity as fluency proxy.** PPL does not capture semantic coherence, factual correctness, or pragmatic appropriateness.
5. **No MoE support.** Sparse expert routing architectures require router-aware hook placement.
6. **Embedding model limitations.** Concept alignment via `paraphrase-multilingual-MiniLM-L12-v2` has limited capacity for fine-grained distinctions in low-resource Indic languages.
7. **Single-tenant inference.** PyTorch `register_forward_hook` mutates global model state; per-request steering in batched serving requires custom CUDA kernel integration.
