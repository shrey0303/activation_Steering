# SteerOps -- Empirical Evaluation Report

Evaluation of the SteerOps activation steering pipeline across three model architectures: Qwen2.5-0.5B, Qwen2.5-7B, and Sarvam-1 (2.51B). All interventions use Contrastive Activation Addition (CAA) vectors computed from 20 positive / 20 negative contrastive prompt pairs per concept. Evaluation is performed on 10 held-out test prompts per strength level per concept. Full telemetry: [`steerops_sarvam1_results.json`](steerops_sarvam1_results.json).

**Statistical methodology.** Effect size: Cohen's d (pooled standard deviation). Significance: paired t-test, alpha = 0.05. Perplexity ratio: steered / baseline (values below 2.0 indicate preserved fluency). Semantic shift: cosine distance between steered and unsteered sentence embeddings via `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 1. Qwen2.5-0.5B (24 layers, d_model = 896)

**Environment:** CPU (local), PyTorch 2.0, float32
**Engine:** v2 (activation-norm-scaled strength)
**Runtime:** 132.7 minutes (7,965s)
**Result:** 6/25 statistically significant conditions

### 1.1 Per-Concept Results

| Concept    | Layer | Strength | Cohen's d | p-value | Significant | PPL Ratio |
|------------|-------|----------|-----------|---------|-------------|-----------|
| Creativity | L21   | 1.0      | 0.34      | 0.020   | Yes         | 1.02      |
| Creativity | L21   | 1.5      | 0.42      | 0.008   | Yes         | 1.05      |
| Creativity | L21   | 2.0      | 0.53      | 0.002   | Yes         | 1.05      |
| Creativity | L21   | 2.5      | 0.65      | 0.017   | Yes         | 1.24      |
| Verbosity  | L11   | 2.0      | 1.18      | 0.002   | Yes         | 0.50      |
| Verbosity  | L11   | 2.5      | 1.18      | 0.018   | Yes         | 0.56      |
| Toxicity   | L13   | 1.0      | 0.93      | 0.080   | No          | 2.44      |
| Politeness | L11   | 2.5      | 0.56      | 0.315   | No          | 1.96      |
| Refusal    | L13   | 2.5      | 0.02      | 0.776   | No          | 1.03      |

### 1.2 Observations

- Creativity exhibits monotonic dose-response: d = 0.34, 0.42, 0.53, 0.65 across strengths 1.0 to 2.5.
- Verbosity produces the largest single effect size (d = 1.18) with perplexity below 1.0 (steered output is more fluent than baseline, consistent with reduced output length).
- Toxicity shows large semantic shifts but fails significance (high within-group variance).
- Refusal is unaffected by single-layer steering (d = 0.02). Residual stream healing across subsequent layers nullifies the perturbation.
- Mean perplexity ratio across all conditions: 1.13.

---

## 2. Qwen2.5-7B (28 layers, d_model = 3584)

**Environment:** RunPod A40 GPU, PyTorch 2.0, bfloat16
**Engine:** v2 (activation-norm-scaled strength, gating disabled for d_model > 2048)
**Runtime:** 206.3 minutes (12,376s)
**Result:** 6/25 statistically significant conditions

### 2.1 Per-Concept Results

| Concept    | Layer | Strength | Cohen's d | p-value | Significant | PPL Ratio |
|------------|-------|----------|-----------|---------|-------------|-----------|
| Creativity | L19   | 1.0      | 0.64      | 0.044   | Yes         | 1.77      |
| Creativity | L19   | 1.5      | 1.12      | 0.012   | Yes         | 2.52      |
| Creativity | L19   | 2.0      | 1.24      | 0.003   | Yes         | 1.61      |
| Creativity | L19   | 2.5      | 1.22      | 0.005   | Yes         | 2.21      |
| Politeness | L9    | 1.0      | -1.26     | 0.016   | Yes         | 1.14      |
| Politeness | L9    | 1.5      | -0.93     | 0.022   | Yes         | 1.40      |
| Verbosity  | L9    | 2.0      | 0.75      | 0.057   | No          | 0.94      |
| Toxicity   | L21   | 1.0      | -0.44     | 0.344   | No          | 1.32      |
| Refusal    | L21   | 1.5      | -0.24     | 0.429   | No          | 3.06      |

### 2.2 Observations

- Creativity dose-response: d = 0.64, 1.12, 1.24, 1.22 (monotonic with saturation at strength 2.0).
- Politeness achieves significance at strengths 1.0--1.5 with negative Cohen's d, indicating inverse-direction steering is effective for suppression.
- Politeness at strength 1.0 on 7B is the cleanest single result in either Qwen evaluation: d = -1.26, p = 0.016, PPL ratio = 1.14.
- Safety-relevant concepts (toxicity, refusal) remain non-significant. Single-layer intervention is insufficient for diffuse, multi-layer concepts.
- Fluency degrades past strength 1.5 (PPL ratio > 2.0). Usable steering range on 7B is narrower than on 0.5B.
- Mean perplexity ratio across all conditions: 1.56.

---

## 3. Sarvam-1 (28 layers, d_model = 2048, LlamaForCausalLM)

**Environment:** T4 GPU, 4-bit NF4 quantization via bitsandbytes
**Engine:** v2 (gating disabled for d_model >= 2048)
**Runtime:** 7160.5s (119.3 minutes). Model load: 21.0s, layer scan: 31.1s.
**Result:** 23/48 statistically significant conditions
**Latency overhead:** 0.0 ms/tok (baseline: 90.5 ms/tok)

### 3.1 Indic Honorific Steering -- Summary

Best result per language (selected by highest effect size with PPL ratio < 2.0):

| Concept           | Layer | Strength | Cohen's d  | p-value  | PPL Ratio | Polite Ratio | Token Fertility |
|-------------------|-------|----------|-----------|----------|-----------|-------------|-----------------|
| Hindi honorific   | L11   | 1.5      | 0.783     | 0.030    | 1.049     | 0.85        | 1.62            |
| Bengali honorific | L11   | 1.5      | -4.212    | < 0.001  | 1.279     | 0.50        | 1.58            |
| Tamil honorific   | L11   | 0.5      | -2.781    | 0.0001   | 1.304     | 0.50        | 1.72            |

Tamil is the only language where PPL < 2.0 is achievable (at strength 0.5 only). All higher strengths produce catastrophic perplexity (see Section 3.4).

### 3.2 Full Hindi Honorific Strength Sweep

CAA vector: dim = 2048, magnitude = 29.13.

| Strength | Concept Delta (mu +/- sigma) | Semantic Shift (mu +/- sigma) | Cohen's d | p-value | Sig | PPL Ratio | Polite % | TokFert |
|----------|------------------------------|-------------------------------|-----------|---------|-----|-----------|----------|---------|
| 0.5      | -0.084 +/- 0.230             | 0.746 +/- 0.189               | -0.615    | 0.278   | No  | 1.262     | 60%      | 2.29    |
| 1.0      | -0.159 +/- 0.113             | 0.686 +/- 0.189               | -1.682    | 0.002   | Yes | 1.053     | 65%      | 1.57    |
| **1.5**  | **0.117 +/- 0.144**          | **0.614 +/- 0.173**           | **0.783** | **0.030** | **Yes** | **1.049** | **85%** | **1.62** |
| 2.0      | 0.042 +/- 0.173              | 0.697 +/- 0.155               | 0.301     | 0.464   | No  | 1.153     | 85%      | 1.56    |
| 2.5      | 0.079 +/- 0.233              | 0.684 +/- 0.159               | 0.428     | 0.311   | No  | 1.162     | 85%      | 1.46    |
| 3.0      | 0.052 +/- 0.171              | 0.764 +/- 0.102               | 0.388     | 0.358   | No  | 1.190     | 80%      | 1.59    |

Optimal operating point: strength 1.5. Polite marker ratio peaks at 85% with PPL 1.049. Effect saturates at strength 2.0+ (no further polite ratio gain, slight perplexity increase).

### 3.3 Full Bengali Honorific Strength Sweep

CAA vector: dim = 2048, magnitude = 27.93.

| Strength | Concept Delta (mu +/- sigma) | Semantic Shift (mu +/- sigma) | Cohen's d  | p-value  | Sig | PPL Ratio | Polite % | TokFert |
|----------|------------------------------|-------------------------------|-----------|----------|-----|-----------|----------|---------|
| 0.5      | -0.362 +/- 0.301             | 0.633 +/- 0.257               | -1.858    | 0.004    | Yes | 1.406     | 65%      | 1.63    |
| 1.0      | -0.460 +/- 0.214             | 0.782 +/- 0.228               | -2.873    | 0.0001   | Yes | 1.853     | 60%      | 1.82    |
| **1.5**  | **-0.568 +/- 0.203**         | **0.915 +/- 0.100**           | **-4.212**| **< 0.001** | **Yes** | **1.279** | **50%** | **1.58** |
| 2.0      | -0.495 +/- 0.210             | 0.769 +/- 0.234               | -2.688    | < 0.001  | Yes | 4.052     | 60%      | 7.56    |
| 2.5      | -0.172 +/- 0.302             | 0.616 +/- 0.161               | -0.946    | 0.105    | No  | 1.575     | 50%      | 5.07    |
| 3.0      | -0.229 +/- 0.175             | 0.651 +/- 0.116               | -1.555    | 0.003    | Yes | 3.191     | 55%      | 9.26    |

Peak effect: d = -4.21 at strength 1.5. Beyond strength 2.0, token fertility degrades from 1.58 to 7.56 tokens/word, indicating tokenizer-level output degeneration. The narrow optimal window (strength 1.0--1.5) is characteristic of high-effect-size steering: the representation is highly sensitive to perturbation in this direction.

### 3.4 Tamil Honorific Degradation

CAA vector: dim = 2048, magnitude = 13.28 (notably lower than Hindi/Bengali, suggesting weaker contrastive signal in the training data).

| Strength | Concept Delta (mu +/- sigma) | Semantic Shift (mu +/- sigma) | Cohen's d | p-value | Sig | PPL Ratio     |
|----------|------------------------------|-------------------------------|-----------|---------|-----|---------------|
| 0.5      | -0.685 +/- 0.342             | 0.862 +/- 0.304               | -2.781    | 0.0001  | Yes | 1.304         |
| 1.0      | -0.106 +/- 0.324             | 0.424 +/- 0.184               | -0.511    | 0.330   | No  | **8.465**     |
| 1.5      | -0.377 +/- 0.250             | 0.616 +/- 0.221               | -1.677    | 0.001   | Yes | **97.759**    |
| 2.0      | -0.452 +/- 0.272             | 0.703 +/- 0.255               | -1.813    | 0.0005  | Yes | **145.055**   |
| 2.5      | -0.420 +/- 0.359             | 0.711 +/- 0.249               | -1.717    | 0.005   | Yes | **97.120**    |
| 3.0      | -0.406 +/- 0.270             | 0.704 +/- 0.107               | -2.228    | 0.001   | Yes | **156.234**   |

Perplexity exceeds 97x baseline at strength >= 1.5. The significant Cohen's d values at these PPL ratios are measurement artifacts: cosine distance between coherent baseline text and near-random steered output will always be large, but this does not indicate behavioral modulation.

**Root cause:** The CAA vector encodes Indo-Aryan honorific morphological patterns (verb conjugation, pronoun substitution). Tamil (Dravidian family) uses agglutinative honorific marking ("-nga" pluralization suffixes) occupying different residual stream subspaces. Injecting the Indo-Aryan vector at the shared L4/L8 circuit destructively interferes with Tamil generation.

**Note:** The Tamil CAA vector magnitude (13.28) is less than half the Hindi (29.13) and Bengali (27.93) magnitudes. This indicates that the contrastive pairs produce a weaker directional signal for Tamil, likely because Sarvam-1's training data contains fewer Tamil honorific examples, resulting in less differentiated positive/negative activation distributions.

### 3.5 English and General Concept Steering

**English Politeness (Control)** -- CAA vector magnitude: 24.49:

| Strength | Cohen's d | p-value | Sig | PPL Ratio |
|----------|----------|---------|-----|-----------|
| 0.5      | -0.049   | 0.869   | No  | 1.258     |
| 1.0      | 1.090    | 0.061   | No  | 4.902     |
| 1.5      | 2.364    | 0.001   | Yes | 8.507     |
| 2.0      | 1.898    | 0.005   | Yes | 9.490     |
| 2.5      | 2.150    | 0.001   | Yes | 11.185    |
| 3.0      | 1.843    | 0.002   | Yes | 20.324    |

Large effect sizes at strength >= 1.5, but PPL ratios consistently above 4.9. English politeness steering on an Indic-primary model produces measurable behavioral shift at the cost of coherent generation. This is expected: the model's English generation capability is secondary to its Indic competence.

**Verbosity Control** -- CAA vector magnitude: 54.05 (highest of all concepts):

| Strength | Cohen's d | p-value | Sig | PPL Ratio |
|----------|----------|---------|-----|-----------|
| 0.5      | -0.191   | 0.668   | No  | 2.748     |
| 1.0      | -1.012   | 0.042   | Yes | 1.463     |
| 1.5      | -1.017   | 0.025   | Yes | 0.933     |
| 2.0      | -0.188   | 0.703   | No  | 1.207     |
| 2.5      | -0.294   | 0.432   | No  | 0.866     |
| 3.0      | -0.339   | 0.527   | No  | 0.846     |

Significant at strengths 1.0--1.5. PPL ratio below 1.0 at strengths 1.5+ indicates steered output is *more fluent* than baseline, consistent with reduced output length.

**Creativity Enhancement** -- CAA vector magnitude: 37.68:

| Strength | Cohen's d | p-value | Sig | PPL Ratio |
|----------|----------|---------|-----|-----------|
| 0.5      | -0.127   | 0.596   | No  | 1.291     |
| 1.0      | -0.128   | 0.795   | No  | 1.237     |
| 1.5      | 0.263    | 0.477   | No  | 2.185     |
| 2.0      | 0.891    | 0.032   | Yes | 1.551     |
| 2.5      | 0.985    | 0.026   | Yes | 1.658     |
| 3.0      | 1.065    | 0.011   | Yes | 1.233     |

Monotonic dose-response from strength 1.5 onward, with stable fluency (PPL 1.23 at peak). Creativity reaches significance later (strength 2.0+) than honorific concepts, suggesting it requires stronger perturbation to shift the output distribution.

**Refusal Behavior** -- CAA vector magnitude: 8.13 (lowest of all concepts):

| Strength | Cohen's d | p-value | Sig | PPL Ratio |
|----------|----------|---------|-----|-----------|
| 0.5      | -0.205   | 0.695   | No  | 1.180     |
| 1.0      | 0.654    | 0.206   | No  | 1.921     |
| 1.5      | 0.911    | 0.114   | No  | 3.670     |
| 2.0      | 0.941    | 0.019   | Yes | 6.323     |
| 2.5      | 1.254    | 0.055   | No  | 10.258    |
| 3.0      | 1.257    | 0.038   | Yes | 11.682    |

Achieves significance only at strength 2.0 and 3.0, both with PPL > 6.0. Effect sizes are coupled with degraded generation, confirming that refusal behavior is not amenable to single-layer activation steering. The low CAA vector magnitude (8.13 vs. 29.13 for Hindi) indicates weak directional separation between refusal and non-refusal activations.

**Toxicity Reduction** -- CAA vector magnitude: 15.18:

| Strength | Cohen's d | p-value | Sig | PPL Ratio |
|----------|----------|---------|-----|-----------|
| 0.5      | -0.052   | 0.894   | No  | 1.234     |
| 1.0      | -0.184   | 0.707   | No  | 2.194     |
| 1.5      | -0.069   | 0.892   | No  | 2.597     |
| 2.0      | 0.124    | 0.780   | No  | 3.622     |
| 2.5      | -0.499   | 0.252   | No  | 6.099     |
| 3.0      | -0.244   | 0.533   | No  | 6.581     |

Zero significant results across all six strength levels. Toxicity reduction is the least steerable concept tested. This is a distributed, multi-layer behavior that cannot be modulated by single-layer intervention.

### 3.6 Circuit Localization (Layer Responsiveness)

Semantic shift measured at six equidistant layers across the 28-layer architecture (strength 1.5, per-concept CAA vector):

**Indic Honorific Languages:**

| Layer | Depth | Hindi Shift | Hindi PPL | Bengali Shift | Bengali PPL | Tamil Shift | Tamil PPL |
|-------|-------|-------------|-----------|---------------|-------------|-------------|-----------|
| L4    | 14%   | 0.708       | 1.192     | 0.532         | 1.293       | 0.627       | 5.733     |
| L8    | 29%   | 0.528       | 1.105     | **0.958**     | 1.787       | **0.818**   | 8.534     |
| L12   | 43%   | 0.628       | 1.159     | 0.928         | 1.172       | 0.609       | 12.695    |
| L16   | 57%   | 0.644       | 2.916     | 0.433         | 4.111       | 0.278       | 4.251     |
| L21   | 75%   | 0.531       | 1.241     | 0.283         | 1.860       | 0.476       | 1.964     |
| L25   | 89%   | 0.627       | 1.569     | 0.202         | 1.589       | 0.180       | 1.974     |

All three Indic languages exhibit peak responsiveness at L4 and/or L8 (14--29% depth). Hindi distributes responsiveness more uniformly across layers; Bengali and Tamil concentrate sharply at L8. Tamil PPL degrades at every layer, confirming the degradation is not layer-specific but vector-specific.

**English and General Concepts:**

| Layer | Depth | Politeness Shift | Politeness PPL | Verbosity Shift | Verbosity PPL | Creativity Shift | Creativity PPL | Refusal Shift | Refusal PPL | Toxicity Shift | Toxicity PPL |
|-------|-------|-----------------|---------------|----------------|--------------|-----------------|---------------|--------------|------------|---------------|-------------|
| L4    | 14%   | 0.161           | 0.962         | 0.504          | 1.066        | 0.699           | 1.836         | 0.465        | 2.680      | 0.713         | 1.539       |
| L8    | 29%   | 0.891           | 1.436         | 0.422          | 0.976        | 0.759           | 1.524         | 0.867        | 3.740      | 0.927         | 2.026       |
| L12   | 43%   | **1.008**       | 8.018         | 0.558          | 1.092        | 0.557           | 1.498         | 0.780        | 4.513      | 0.862         | 32.479      |
| L16   | 57%   | 0.774           | 4.604         | **0.562**      | 1.739        | 0.577           | 2.208         | **0.838**    | 1.855      | **0.894**     | 7.993       |
| L21   | 75%   | 0.815           | 1.989         | 0.527          | 1.916        | 0.502           | 1.902         | 0.761        | 2.085      | 0.740         | 1.324       |
| L25   | 89%   | 0.568           | 1.806         | 0.564          | 1.098        | **0.732**       | 2.506         | 0.451        | 2.706      | 0.589         | 3.533       |

English politeness peaks at L12 (43% depth) with massive PPL (8.0), consistent with mid-network semantic processing. Verbosity distributes responsiveness uniformly (range: 0.42--0.56). Creativity peaks at L25 (89%), indicating late-layer style processing. Refusal peaks at L16/L8, but with PPL > 1.8 at all layers. Toxicity peaks at L8/L16, but L12 produces PPL of 32.5, further confirming toxicity's unstable response to single-layer perturbation.

---

## 4. Cross-Architecture Comparison

| Metric                      | Qwen2.5-0.5B | Qwen2.5-7B | Sarvam-1 (2.51B) |
|-----------------------------|-------------|-----------|-------------------  |
| Architecture                | Qwen2        | Qwen2     | LlamaForCausalLM  |
| Parameters                  | 0.5B         | 7B        | 2.51B             |
| d_model                     | 896          | 3584      | 2048              |
| Layers                      | 24           | 28        | 28                |
| Significant / total         | 6/25         | 6/25      | 23/48             |
| Significance rate            | 24%          | 24%       | 48%               |
| Peak effect size (|d|)      | 1.18         | 1.26      | 4.21              |
| Peak effect concept         | Verbosity    | Politeness| Bengali honorific  |
| Mean PPL ratio              | 1.13         | 1.56      | 13.68 (all), 1.27 (Hindi/Bengali optimal) |
| Usable strength range       | 0.5--2.5     | 1.0--1.5  | 0.5--1.5          |
| Steering overhead (ms/tok)  | < 2          | < 2       | 0.0               |
| Scan time (s)               | 12.3         | 59.8      | 31.1              |
| Total evaluation runtime    | 133 min      | 206 min   | 119 min           |

### 4.1 Scaling Observations

1. **Effect sizes scale with domain specificity.** The largest effects occur when the model's training data is aligned with the steering concept. Sarvam-1 (Indic-primary) produces d = 4.21 for Bengali honorifics. General-purpose Qwen models produce d = 1.24 at best (creativity on 7B).

2. **Fluency sensitivity increases with model scale.** The usable strength range narrows from [0.5, 2.5] on 0.5B to [0.5, 1.5] on 2.51B and [1.0, 1.5] on 7B. Larger models require less perturbation for equivalent behavioral effect, but are also more sensitive to over-perturbation.

3. **Safety-relevant concepts resist single-layer steering at all scales.** Toxicity and refusal fail significance on all three models. These concepts are distributed across multiple layers; residual stream connections heal single-layer perturbations.

4. **Cross-lingual transfer is partial and asymmetric.** Hindi/Bengali (both Indo-Aryan) share sufficient morphological structure for effective steering from a shared circuit. Tamil (Dravidian) requires language-family-specific vectors to avoid catastrophic perplexity degradation.

5. **Token fertility is a model-specific constraint.** Bengali token fertility explodes at strength >= 2.0 (1.58 to 7.56 tokens/word) while Hindi fertility remains stable up to strength 3.0. This suggests Bengali tokenization in Sarvam-1 is more fragile under activation perturbation.

---

## 5. Engine Corrections for Scale Invariance

The original engine (v1) produced 0/25 significant results on Qwen2.5-7B. Three corrections were applied:

**5.1 Gating threshold recalibration.** The cosine-similarity gate threshold `tau = 3 / sqrt(d_model)` was calibrated for d_model = 896 (tau = 0.100). At d_model = 3584, tau = 0.050, which fell below the natural cosine similarity between random activations and the steering vector, causing 100% of hook invocations to be silently gated. For d_model >= 2048, gating is disabled.

**5.2 Activation-norm-scaled strength.** A fixed strength of 1.0 produces a perturbation magnitude of approximately `1.0 * ||v_orth||` regardless of the activation norm. On 0.5B (||x|| ~ 9), this is ~11% of the activation. On 7B (||x|| ~ 35), this is ~3%. The correction scales strength by `||x||_2 / 10`, making the perturbation proportional to the activation magnitude.

**5.3 Norm tolerance widening.** The L2 norm preservation clamp (tau_norm = 0.05) restricted steered activations to within 5% of the original norm. For d_model > 2048, the required perturbation to produce measurable behavioral effect exceeds this tolerance. tau_norm is widened to 0.25 for large models.

---

## 6. Limitations of This Evaluation

1. **Sample size.** 10 test prompts per condition. Statistical power is limited; effects near the significance boundary (0.05 < p < 0.10) may achieve significance with larger N.

2. **Single-layer injection only.** Multi-layer cascaded steering is expected to improve significance rates for distributed concepts (toxicity, refusal, safety alignment) but was not evaluated.

3. **Perplexity as fluency proxy.** PPL does not capture semantic coherence, factual correctness, or pragmatic appropriateness. Human evaluation is required to validate that steered outputs are meaningfully different (not merely statistically different).

4. **Polysemantic vectors.** CAA direction vectors are dense and may encode multiple behavioral concepts simultaneously. The measured behavioral shift may include unintended side effects not captured by the concept alignment metric.

5. **No MoE support.** Sparse expert routing architectures (Mixtral, DeepSeek-V3) require router-aware hook placement. The current architecture applies hooks uniformly to transformer blocks.

6. **Tamil vectors are not independently optimized.** The Tamil degradation may be partially attributable to inadequate contrastive pairs in the Tamil training set, not solely to typological mismatch. The CAA vector magnitude (13.28) is less than half that of Hindi (29.13), indicating weaker directional signal.

7. **Embedding model limitations.** Concept alignment is computed via `paraphrase-multilingual-MiniLM-L12-v2`, which has limited capacity for fine-grained semantic distinctions in low-resource Indic languages.

8. **Global fluency is NOT preserved.** The `fluency_preserved` flag in the pipeline summary is `false`. Mean perplexity across all 48 conditions is 13.675, driven by Tamil degradation and English generation on an Indic-primary model. Fluency is preserved only when operating within the validated domain (Indo-Aryan honorifics at strength <= 1.5).
