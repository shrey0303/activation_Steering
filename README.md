<div align="center">

# SteerOps

### Activation Steering for LLMs


[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

</div>

---

## Table of Contents

- [What is SteerOps?](#what-is-steerops)
- [How It Works](#how-it-works)
  - [Phase 1: Mathematical Layer Scanning](#phase-1-mathematical-layer-scanning-scannerpy)
  - [Phase 1.5: Offline PCA Feature Dictionary](#phase-15-offline-pca-feature-dictionary-feature_extractorpy)
  - [Phase 2: Behavior Interpretation + Intent Router](#phase-2-behavior-interpretation--intent-router-interpreterpy)
  - [Phase 3: Layer Resolution](#phase-3-layer-resolution-resolverpy)
  - [Phase 4: Activation Steering](#phase-4-activation-steering-enginepy)
  - [Phase 5: Direction Vector Computation](#phase-5-direction-vector-computation-vector_calculatorpy)
- [Research-Backed Improvements (v2.0)](#research-backed-improvements-v20)
- [Supported Models](#supported-models)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Web UI Guide](#web-ui-guide)
  - [Steering Diagnostics](#steering-diagnostics-panel-right-bottom)
- [API Reference](#api-reference)
- [Python Library (`steerops`)](#python-library-steerops)
- [Benchmarks](#benchmarks)
- [Roadmap](#roadmap)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## What is SteerOps?

SteerOps is an **activation-level debugger and steering tool** for transformer-based language models. Traditional approaches to controlling model behavior rely on prompt engineering or fine-tuning. SteerOps takes a fundamentally different approach:

1. **Scans** model weight matrices mathematically (SVD, attention entropy, FFN norms)
2. **Extracts** an offline PCA Feature Dictionary of orthogonal steering directions per layer
3. **Maps** each layer to a functional category based on mechanistic interpretability research
4. **Interprets** user-provided behavior descriptions using semantic embedding matching
5. **Steers** activations at specific layers in real-time using PyTorch forward hooks with orthogonal projection, Lâ€šâ€š norm preservation, and adaptive decay
6. **Exports** validated interventions as portable JSON patches

**Key Innovation:** SteerOps uses a hybrid offline-then-runtime architecture. Heavy computation (PCA, auto-labeling) runs offline once per model. Runtime steering is O(1) lookup + forward hook injection â‚¬" no re-running forward passes, no gradient computation.

---

## How It Works

### Phase 1: Mathematical Layer Scanning (`scanner.py`)

> **Note (v2.0):** Phase 1 is now used strictly for **diagnostic visualization** â‚¬" populating the Activation Heatmap and layer categorization display in the UI. It is **no longer in the steering logic path**. Steering now flows through Phase 1.5 (PCA) â€ ' Phase 2 (Intent Router) â€ ' Phase 3 (O(1) Lookup) â€ ' Phase 4 (Engine). The K-Means categorization remains useful for understanding model structure at a glance, but all runtime steering decisions use the PCA Feature Dictionary directly.

The scanner profiles every transformer layer by analyzing weight matrices directly. No forward pass is needed.

#### Metrics Computed Per Layer

| Metric | Method | What It Reveals |
|--------|--------|-----------------|
| **SVD Effective Rank** | Singular Value Decomposition of attention weight matrices | Higher rank â€ ' more diverse learned transformations â€ ' complex processing |
| **Attention Entropy** | Shannon entropy of `W_Q @ W_K.T` eigenvalues | High entropy â€ ' distributed attention â€ ' semantic/global processing |
| **FFN Norm** | Frobenius norm of feed-forward weight matrices | Large norms â€ ' strong non-linear transforms â€ ' reasoning/retrieval |
| **Inter-Layer CKA** | Centered Kernel Alignment between adjacent layers | Sharp drops â€ ' functional boundary between processing stages |

#### How SVD Effective Rank Works

```
Given weight matrix W Ë†Ë† â€žÂ^(dÃƒ--d):
  1. Compute singular values: ÃÆ’â€šÂ â€°Â¥ ÃÆ’â€šâ€š â€°Â¥ ... â€°Â¥ ÃÆ’_n
  2. Normalize: p_i = ÃÆ’_i / ÃŽÂ£ÃÆ’_j
  3. Compute entropy: H = -ÃŽÂ£ p_i . log(p_i)
  4. Effective rank = e^H

Higher effective rank â€ ' the layer learned a more complex transformation
Lower effective rank â€ ' dominated by few singular values â€ ' simpler function
```

#### How Attention Entropy Works

```
For attention weight matrices W_Q, W_K:
  1. Compute W = W_Q @ W_K.T (attention pattern proxy)
  2. Extract eigenvalues ÃŽÂ»â€šÂ, ÃŽÂ»â€šâ€š, ...
  3. Normalize to probability distribution
  4. Shannon entropy H = -ÃŽÂ£ p_i . logâ€šâ€š(p_i)

High entropy â€ ' attention distributed broadly â€ ' global/semantic processing
Low entropy â€ ' attention focused narrowly â€ ' local/syntactic processing
```

#### Layer Categorization

After computing features, layers are categorized using K-Means clustering validated against positional heuristics from mechanistic interpretability research:

| Category | Position Range | Function | Citation |
|----------|---------------|----------|----------|
| `token_embedding` | 0â‚¬"5% | Raw vocabulary lookup | Universal |
| `positional_morphological` | 5â‚¬"12% | Position encoding, morphology | Logit Lens (nostalgebraist 2020) |
| `syntactic_processing` | 12â‚¬"25% | Grammar, clause structure | Hewitt & Manning 2019 |
| `entity_semantic` | 25â‚¬"40% | Entity tracking, polysemy | Attention head analysis |
| `knowledge_retrieval` | 40â‚¬"55% | Factual recall via FF key-value | Geva et al. 2021 |
| `reasoning_planning` | 55â‚¬"70% | Multi-step inference | Mid-layer attention studies |
| `safety_alignment` | 70â‚¬"78% | Refusal circuits, guardrails | Anthropic activation patching |
| `information_integration` | 78â‚¬"88% | Cross-layer signal merging | Lawson et al. 2025 |
| `style_personality` | 88â‚¬"95% | Tone, register, personality | Late-layer generation studies |
| `output_distribution` | 95â‚¬"100% | Final token probabilities | Logit Lens studies |

Scan results are cached in SQLite for instant subsequent access.

---

### Phase 1.5: Offline PCA Feature Dictionary (`feature_extractor.py`)

**New in v2.0** â‚¬" Builds an indexed dictionary of orthogonal steering directions via PCA, run offline once per model.

#### Pipeline

```
1. Run 165 diverse prompts through the model
2. Capture residual stream activations at every layer (mean-pooled)
3. Center the data + run SVD-based PCA per layer
4. Extract top-K principal components (with min-variance noise filter)
5. Auto-label top 5 components per layer via contrastive generation probing
6. Store in SQLite (metadata) + NumPy files (vectors)
```

#### PCA Math

```
Given activations A Ë†Ë† â€žÂ^(n_prompts Ãƒ-- hidden_dim) at layer â€ž":

  1. Center: Ã„â‚¬ = A - mean(A)
  2. SVD:    Ã„â‚¬ = U . S . VÃ¡Âµâ‚¬
  3. Top-K:  components = VÃ¡Âµâ‚¬[:k]       # (k, hidden_dim)
  4. Variance: explained_i = SÂ²Ã¡ÂµÂ¢ / ÃŽÂ£SÂ²  # fraction per component

Components are orthogonal by construction (SVD guarantees this).
```

#### Min-Variance Noise Filter

Components explaining < 0.1% variance are automatically dropped, preventing noise PCA components from polluting the feature dictionary. In practice, this reduces 20 requested components to 3â‚¬"10 meaningful directions per layer.

#### Contrastive Auto-Labeling (CAA-Style)

Each top component is labeled using a contrastive approach (Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition"):

1. **Amplify** (+5.0 strength) the component during generation across probe prompts
2. **Suppress** (-5.0 strength) the same component during generation
3. Compute `delta = embed(amplified_outputs) - embed(suppressed_outputs)`
4. Match `delta` against behavioral keywords via cosine similarity

This contrastive approach **doubles the signal strength** vs. single-sided (steered vs. baseline), producing more reliable labels.

#### Feature ID Format

```
L{layer_idx}_PC{component_idx}

Examples:
  L14_PC0  â€ '  Layer 14, highest-variance component
  L22_PC3  â€ '  Layer 22, 4th component
```

#### Files

| File | Purpose |
|------|---------|
| `feature_extractor.py` | Full offline pipeline + `FeatureDictionary` class with O(1) lookup |
| `feature_dataset.py` | 165 diverse prompts, labeling prompts, behavioral keywords |

---

### Phase 2: Behavior Interpretation + Intent Router (`interpreter.py`)

The interpreter uses **per-layer bidirectional semantic matching** to determine which layers to target and in what direction.

#### Legacy Mode: Semantic Matching

For each of the 10 layer categories, we maintain two semantic descriptions:
- **Enhance description**: What behaviors INCREASE this layer's function
- **Suppress description**: What behaviors DECREASE this layer's function

When the user provides a behavior (e.g., "be very angry"):

```
1. Embed the user's input using all-MiniLM-L6-v2 sentence transformer
2. For each layer category:
   a. Compute cosine similarity against the ENHANCE description
   b. Compute cosine similarity against the SUPPRESS description
   c. Direction = whichever side scores higher
   d. Magnitude = calibrated similarity score
3. Return layers sorted by magnitude with independent directions
```

#### New: Intent Router with NLI Cross-Encoder (v2.0)

The `IntentRouter` uses a **two-stage retrieve-then-classify** pipeline:

**Stage 1 â‚¬" Retrieval (Bi-Encoder):**
```
1. User types: "make the model less toxic"
2. Embed the text using all-MiniLM-L6-v2 (bi-encoder)
3. Cosine similarity against all labeled feature embeddings
4. Return top-K matching features by topic
```

**Stage 2 â‚¬" Direction Classification (NLI Cross-Encoder):**
```
For each top-K candidate:
  1. Construct hypothesis: "Amplify {feature_label}"
  2. Feed [user_text, hypothesis] into cross-encoder/nli-deberta-v3-small
  3. Model outputs: [contradiction, entailment, neutral] probabilities
  4. contradiction > entailment â€ ' suppress (-1.0)
     entailment > contradiction â€ ' enhance (+1.0)
```

**Why both models?** NLI alone cannot do retrieval â‚¬" it finds spurious cross-concept relationships. For example, `"Make it more toxic"` vs `"Amplify anger"` scores contradiction=0.998, incorrectly beating the correct match. The bi-encoder correctly isolates topic matching (WHICH feature); the NLI correctly determines direction (enhance vs suppress). Together: 7/7 test accuracy. Separately: 2/7.

Falls back to keyword detection if the cross-encoder is unavailable.

#### Example Results

| Input | Layer Effects |
|-------|---------------|
| "be very angry" | suppress reasoning (-0.79), enhance style (+0.58), suppress safety (-0.55) |
| "be very polite" | enhance safety (+0.66), enhance style (+0.60) |
| "be very intelligent" | enhance knowledge (+0.79), enhance reasoning (+0.66) |
| "be very helpful" | enhance reasoning (+0.85), enhance knowledge (+0.70) |
| "be extremely cautious" | enhance safety (+1.00) |

**Key Design Decision:** Direction is determined per-layer, not globally. "Be angry" correctly enhances style (anger IS personality expression) while suppressing helpfulness (angry = not helpful) and safety (anger bypasses caution).

#### Confidence Measurement

Confidence is calculated based on:
- **Semantic similarity strength** â‚¬" how closely the behavior matches layer descriptions
- **Dominance gap** â‚¬" how much the top match exceeds other matches
- **Dynamic threshold** â‚¬" `mean + 0.3 Ãƒ-- std` of all similarity scores

---

### Phase 3: Layer Resolution (`resolver.py`)

Maps the interpreter's output (category + direction + magnitude) to specific layer indices in the loaded model.

#### Legacy Mode: Scan-Based Resolution

```
1. Look up which layers belong to each matched category (from scan results)
2. Sort by the layer's scan anomaly score (higher = more impactful)
3. Select top N layers per category
4. Return with direction vectors ready for steering
```

#### New: Feature Dictionary Direct Lookup (v2.0)

O(1) lookup: `feature_id â€ ' (layer_index, vector, strength, direction)` â‚¬" no heuristics needed.

#### Bell-Curve Layer-Aware Strength (v2.0)

Default strength is automatically scaled by a Gaussian bell curve centered at **60% model depth**:

```python
relative_pos = feature.layer_idx / total_layers
multiplier = exp(-((relative_pos - 0.6)Â² / 0.1))
```

Research across Llama 2, GPT-2, and Mistral confirms:
- **Early layers (0â‚¬"20%)**: handle tokens/syntax â€ ' low strength to avoid breaking grammar
- **Middle layers (~60%)**: optimal injection point â€ ' full strength
- **Late layers (80â‚¬"100%)**: diminishing returns â€ ' reduced strength

Example for a 30-layer model:

| Layer | Relative Position | Bell Multiplier | Effective Strength (base=2.0) |
|-------|-------------------|-----------------|-------------------------------|
| L0    | 0.00              | 0.027           | 0.05                          |
| L10   | 0.33              | 0.491           | 0.98                          |
| L18   | 0.60              | **1.000**       | **2.00** (peak)               |
| L25   | 0.83              | 0.580           | 1.16                          |
| L29   | 0.97              | 0.261           | 0.52                          |

---

### Phase 4: Activation Steering (`engine.py`)

The steering engine modifies model activations in real-time using PyTorch forward hooks.

#### Steering Pipeline (Per Token)

```
For each registered hook at layer â€ž":

  STEP 0: Extract x from output, preserve KV cache
  STEP 1: Cooldown check (circuit breaker recovery)
  STEP 2: Gating â‚¬" skip injection if x is already aligned with v
           (cosine similarity > auto-calibrated gate_threshold)
  STEP 3: Steering or Erasure
           Mode "steer": Orthogonal projection + adaptive decay
           Mode "erase": LEACE null-space projection
  STEP 4: Lâ€šâ€š norm preservation (clamp scale within Â±5%)
  STEP 5: NaN/Inf safety check â€ ' revert + cooldown on failure
```

#### Orthogonal Projection (Default Mode)

Instead of naive additive steering (`x + ÃŽÂ±.v`), which doubles the component already present:

```
v_orth = v - proj_xÃŒâ€š(v) = v - (v . xÃŒâ€š)xÃŒâ€š
x_steered = x + strength Ãƒ-- v_orth
```

Only the **orthogonal component** is injected â‚¬" the part of the steering direction not already in the activation.

#### LEACE Concept Erasure (Mode: "erase")

Based on Belrose et al., "LEACE: Perfect Linear Concept Erasure in Closed Form":

```
x_erased = x - (x . vÃŒâ€š)vÃŒâ€š
```

Provably removes **all** linear information about concept v from the activation. Unlike suppression (which can overshoot and invert), erasure is binary and mathematically guaranteed.

#### Adaptive Strength Decay

```
decay = max(min_decay, 1.0 - token_count Ãƒ-- decay_rate)
effective_strength = base_strength Ãƒ-- decay
```

Full strength on early tokens, decaying over time â‚¬" prevents late-token drift.

#### Logit Entropy Circuit Breaker

After each token, compute Shannon entropy of the logit distribution. If entropy > 6.0 nats, all hooks enter cooldown (5 tokens of pass-through).

#### Gram-Schmidt Multi-Vector Composition (v2.0)

When multiple hooks are active simultaneously, their direction vectors may overlap. Before each generation, all active vectors are orthogonalized via Gram-Schmidt:

```
For hooks [vâ€šâ‚¬, vâ€šÂ, vâ€šâ€š]:
  vâ€šÂ' = vâ€šÂ - proj_vâ€šâ‚¬(vâ€šÂ)              # remove vâ€šâ‚¬ component from vâ€šÂ
  vâ€šâ€š' = vâ€šâ€š - proj_vâ€šâ‚¬(vâ€šâ€š) - proj_vâ€šÂ'(vâ€šâ€š) # remove both components from vâ€šâ€š
  Normalize all to unit length
```

This guarantees each vector steers an **independent axis** with zero interference.

> **When is this needed?** PCA components from the *same layer* are already strictly orthogonal by SVD construction â‚¬" Gram-Schmidt is a no-op for them. It becomes necessary when combining vectors from **different sources**: e.g., a PCA feature vector + a custom CAA vector, or PCA vectors from different layers projected into the same residual stream. The implementation handles all cases uniformly.

#### Auto-Calibrated Gating

In high-dimensional spaces (4096D for Llama-3), cosine similarity naturally shrinks toward zero. Gate threshold is auto-calibrated:

```python
gate_threshold = 2.0 / sqrt(hidden_dim)
# 768D â€ ' 0.072,  4096D â€ ' 0.031
```

**Safety Features:**
- NaN/Inf detection and clamping
- Automatic hook cleanup after generation
- Thread-safe concurrent generation support
- Steering overhead measurement (typically <2ms per token)

---

### Phase 5: Direction Vector Computation (`vector_calculator.py`)

Direction vectors are computed via **Contrastive Activation Addition (CAA)**:

```
1. Define contrastive prompt pairs:
   Positive: "Please help me with this task"
   Negative: "I refuse to help you with anything"

2. Run both through the model, capture activations at the target layer

3. Direction vector = mean(positive_activations) - mean(negative_activations)

4. L2-normalize for stable steering magnitude
```

The direction vector is a property of the layer and model, not the specific behavior. It represents the "direction" in activation space that corresponds to the behavioral concept.

> **Note:** In v2.0, PCA Feature Extraction (Phase 1.5) provides an alternative path to direction vectors â‚¬" unsupervised, orthogonal by construction, and auto-labeled. CAA remains available for targeted contrastive scenarios.

---

## Research-Backed Improvements (v2.0)

All improvements are backed by published research and verified with integration tests (6/6 passing):

| # | Improvement | Paper/Source | What It Does | Verified |
|---|-------------|-------------|--------------|----------|
| 1 | **Gram-Schmidt Orthogonalization** | "Multi-Attribute Orthogonal Subspace Steering" | Ensures multi-vector composition has zero interference (dot products: 0.90 â€ ' 0.000000) | Å“â€¦ |
| 2 | **Min-Variance Noise Filter** | Standard PCA practice | Drops components < 0.1% variance â‚¬" 20 PCs â€ ' 3 kept when signal concentrated | Å“â€¦ |
| 3 | **Contrastive Auto-Labeling** | Rimsky et al., "Steering Llama 2 via CAA" | Amplify(+) vs Suppress(-) delta doubles signal vs baseline-only approach | Å“â€¦ |
| 4 | **Bell-Curve Layer Strength** | Llama 2, GPT-2, Mistral layer studies | Gaussian peak at 60% depth â‚¬" L0=0.05, L18=2.00, L29=0.52 | Å“â€¦ |
| 5 | **LEACE Concept Erasure** | Belrose et al., "Perfect Linear Concept Erasure" | Null-space projection removes concept completely (3.44 â€ ' 0.000000), preserves perpendicular info exactly | Å“â€¦ |
| 6 | **NLI Cross-Encoder Direction** | Standard NLI (DeBERTa-v3) | Replaces keyword hacks â‚¬" handles negation natively (13/13 tests passed) | Å“â€¦ |

Additional runtime improvements (implemented but not separately tested):
- **Adaptive Strength Decay** â‚¬" prevents auto-regressive generation drift
- **Auto-Calibrated Gating** â‚¬" threshold adjusted per hidden dimension
- **Lâ€šâ€š Norm Preservation** â‚¬" activation magnitude clamped within Â±5%
- **Entropy Circuit Breaker** â‚¬" kills steering on confused logits (entropy > 6.0 nats)

---

## Supported Models

SteerOps works with **any HuggingFace transformer model** that uses the standard architecture (attention + FFN layers). Tested and recommended models:

| Model | Parameters | VRAM Required | Quality | Notes |
|-------|-----------|---------------|---------|-------|
| `HuggingFaceTB/SmolLM2-135M` | 135M | ~300MB | Demo | 30 layers â‚¬" limited differentiation |
| `HuggingFaceTB/SmolLM2-360M` | 360M | ~700MB | Better | More distinct layer boundaries |
| `HuggingFaceTB/SmolLM2-1.7B` | 1.7B | ~3.5GB | Good | Recommended for development |
| `meta-llama/Llama-2-7b-hf` | 7B | ~14GB (7GB quantized) | Excellent | Production-quality steering |
| `mistralai/Mistral-7B-v0.1` | 7B | ~14GB (7GB quantized) | Excellent | Strong reasoning layers |
| `meta-llama/Meta-Llama-3-8B` | 8B | ~16GB (8GB quantized) | Best | 80 layers â‚¬" finest granularity |
| `meta-llama/Llama-2-70b-hf` | 70B | ~140GB (35GB 4-bit) | Research | Maximum layer resolution |

### Model Requirements

- Must be a **decoder-only transformer** (GPT-style)
- Must be available on HuggingFace or as a local checkpoint
- Architecture must have named `model.layers[i]` (standard HuggingFace format)
- Quantized models (4-bit, 8-bit via bitsandbytes) are fully supported

### How Model Size Affects Quality

```
Small models (< 1B):
  "Å“"â‚¬"â‚¬ Few layers (â€°Â¤ 30) â€ ' multiple categories share same layers
  "Å“"â‚¬"â‚¬ Lower confidence scores â€ ' behaviors overlap
  """â‚¬"â‚¬ Good for demos and development

Medium models (1Bâ‚¬"7B):
  "Å“"â‚¬"â‚¬ 32â‚¬"48 layers â€ ' better category separation
  "Å“"â‚¬"â‚¬ Clear functional boundaries at CKA drops
  """â‚¬"â‚¬ Good for production use

Large models (7B+):
  "Å“"â‚¬"â‚¬ 32â‚¬"80+ layers â€ ' fine-grained layer specialization
  "Å“"â‚¬"â‚¬ Distinct safety, reasoning, and style zones
  """â‚¬"â‚¬ Best steering accuracy and confidence
```

---

## Architecture

```
"Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â
"â€š                      Frontend (React + Vite)                      "â€š
"â€š  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â   "â€š
"â€š  "â€š Control Panel"â€š  "â€š     Chat     "â€š  "â€š  Activation Heatmap  "â€š   "â€š
"â€š  "â€š + Feature    "â€š  "â€š  Interface   "â€š  "â€š  + Diagnostics Panel "â€š   "â€š
"â€š  "â€š   Browser    "â€š  "â€š  (WebSocket) "â€š  "â€š  (real-time metrics) "â€š   "â€š
"â€š  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ   "â€š
"""â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ
                            "â€š REST + WebSocket
"Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬-Â¼"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â
"â€š                     Backend (FastAPI + PyTorch)                    "â€š
"â€š                                                                   "â€š
"â€š  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â  "â€š
"â€š  "â€š   Scanner    "â€š  "â€š  PCA Feature    "â€š  "â€š   Intent Router    "â€š  "â€š
"â€š  "â€š  (SVD, CKA,  "â€š  "â€š  Extractor     "â€š  "â€š  (Bi-Encoder +     "â€š  "â€š
"â€š  "â€š   Entropy)   "â€š  "â€š  (Offline)      "â€š  "â€š   NLI Cross-Enc.)  "â€š  "â€š
"â€š  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ  "â€š
"â€š         "â€š                   "â€š                      "â€š             "â€š
"â€š         "â€š         "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬-Â¼"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â           "â€š             "â€š
"â€š         """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬-Âº"â€š   Layer Resolver   "â€š--â€ž"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ             "â€š
"â€š                   "â€š  (O(1) Lookup +    "â€š                         "â€š
"â€š                   "â€š   Bell-Curve Str.) "â€š                         "â€š
"â€š                   """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ                         "â€š
"â€š                             "â€š                                    "â€š
"â€š  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬-Â¼"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â  "â€š
"â€š  "â€š               Steering Engine (PyTorch Hooks)              "â€š  "â€š
"â€š  "â€š  â‚¬Â¢ Orthogonal Projection    â‚¬Â¢ Lâ€šâ€š Norm Preservation        "â€š  "â€š
"â€š  "â€š  â‚¬Â¢ Adaptive Decay           â‚¬Â¢ Entropy Circuit Breaker      "â€š  "â€š
"â€š  "â€š  â‚¬Â¢ Gram-Schmidt Multi-Vec   â‚¬Â¢ LEACE Concept Erasure        "â€š  "â€š
"â€š  "â€š  â‚¬Â¢ Bell-Curve Strength      â‚¬Â¢ Auto-Calibrated Gating       "â€š  "â€š
"â€š  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ  "â€š
"â€š                                                                   "â€š
"â€š  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â "â€š
"â€š  "â€š  Storage: SQLite (scan cache + features) + NumPy (vectors)  "â€š "â€š
"â€š  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ "â€š
"â€š                                                                   "â€š
"â€š  "Å’"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Â "â€š
"â€š  "â€š  Vector Calculator (CAA) â‚¬" legacy contrastive pairs path    "â€š "â€š
"â€š  """â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ "â€š
"""â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"â‚¬"Ëœ

---

## Production Viability & Architectural Trade-offs

SteerOps is designed as an agile, mathematical debugger for LLMs, but makes explicit architectural trade-offs that differ from billion-dollar multi-tenant inference engines.

### 1. Global PyTorch Hooks vs. Multi-Tenancy

Standard PyTorch forward hooks (`register_forward_hook`) mutate the global model state in VRAM. In a massive multi-tenant SaaS environment, batched inference requires highly complex custom CUDA kernels (like vLLM's PagedAttention) to apply different steering vectors to different sequences within the same batch.

**The SteerOps Solution:** Rather than pretending standard PyTorch supports zero-cost multi-tenant vector injection, SteerOps embraces its identity as a **single-tenant diagnostic tool**. When deployed via `STEEROPS_DEPLOY_MODE=production`, it uses a custom `SessionLockMiddleware`. This enforces a strict 1-user GPU queue, gracefully returning a 503 error to concurrent API requests, guaranteeing zero cross-contamination of steering hooks across users.

### 2. CAA vs. Sparse Autoencoders (SAEs)

State-of-the-art interpretability currently favors Sparse Autoencoders (SAEs), which extract millions of monosemantic features (e.g., Anthropic's Golden Gate Claud). 

**Why SteerOps uses CAA and PCA:** Training an SAE requires massive datasets and heavy GPU compute for *every single layer* of a specific model. SteerOps optimizes for **deployment agility and zero-training inference**. Using Contrastive Activation Addition (CAA), an engineer can take a brand new open-source model and generate a steering vector for "toxicity" in 60 seconds. SteerOps is intervention-first, providing immediate runtime patching without the millions of dollars required for SAE training.
```

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Node.js 20+** (for frontend)
- **CUDA GPU** recommended (works on CPU, but 10x slower)
- **4GB+ VRAM** for small models, 8GB+ for 7B models

### Option 1: Local Development

#### Backend

```bash
# Clone the repository
git clone https://github.com/shrey0303/activation_Steering.git
cd activation_Steering

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Configure (optional â‚¬" defaults work out of the box)
cp backend/.env.example backend/.env
# Edit backend/.env to set MODEL_NAME, DEVICE, etc.

# Start the backend server
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

#### Backend API docs available at **http://localhost:8000/docs**

### Option 2: Docker (Production)

```bash
docker-compose up --build
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### Deployment Modes

SteerOps has two deployment modes controlled by the `STEEROPS_DEPLOY_MODE` environment variable:

| Mode | Env Value | Concurrency Guards | Use Case |
|------|-----------|-------------------|----------|
| **Local** | `local` (default) | None â‚¬" full access | Personal dev, research |
| **Production** | `production` | Session lock + rate limiting | Public demos, LinkedIn showcase |

#### Production Mode Features

When `STEEROPS_DEPLOY_MODE=production`:

1. **Session Lock** â‚¬" Only one user can interact with GPU endpoints at a time. Others receive a `503 Service Unavailable` with a friendly message. Lock auto-expires after 5 minutes of inactivity.
2. **Rate Limiting** â‚¬" Per-IP sliding window: 30 GPU requests/min, 120 lightweight requests/min. Prevents abuse.
3. **Exempt Endpoints** â‚¬" Health checks, metrics, model listing, and docs are always accessible.

```bash
# Enable production mode locally for testing:
set STEEROPS_DEPLOY_MODE=production  # Windows
export STEEROPS_DEPLOY_MODE=production  # Linux/Mac
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker Compose automatically sets `production` mode.

### First Run Walkthrough

```
1. Open http://localhost:5173
2. The app loads with default model (SmolLM2-135M)
3. Wait for model download and loading (~30s first time)
4. Model scan runs automatically after loading
5. Switch to "Expected Behavior" tab
6. Type a behavior: "be very polite"
7. Click "Analyze & Detect Layers"
8. Review detected layers (e.g., L27â€ â€˜ for style, L23â€ â€˜ for safety)
9. Adjust steering strength slider if needed
10. Type a prompt in the chat and send
11. See steered output with real-time token streaming
12. Export the intervention as a JSON patch
```

### Run PCA Feature Extraction (Optional, v2.0)

```bash
cd backend
python -m app.core.feature_extractor --model HuggingFaceTB/SmolLM2-135M --top-k 20
```

### Web UI Guide

The SteerOps interface has 5 panels:

#### Control Panel (Left)

| Section | What It Does |
|---------|-------------|
| **Model Selector** | Type any HuggingFace model ID (e.g., `HuggingFaceTB/SmolLM2-135M`), click Load |
| **Layer Map** | After scanning, shows all layers colored by category (reasoning, safety, style, etc.) |
| **Behavior Analysis** | Type a behavior description (e.g., "be very angry") â€ ' click "Analyze & Detect Layers" â€ ' see which layers activate |
| **Feature Dictionary** | Browse PCA features, relabel them, search by keyword (requires running feature extraction first) |

#### Chat (Center)

Type prompts to generate text. When steering hooks are active (layers selected from analysis), the output is steered in real-time. The chat shows token-by-token streaming via WebSocket.

#### Layer Activation Map (Right Top)

Heatmap of per-layer activations. Send a prompt in chat to see live activation magnitudes across all layers. Layers that fire strongly are highlighted.

#### Steering Diagnostics Panel (Right Bottom)

**This panel shows real-time per-token steering metrics during generation.** It only activates when steering hooks are applied.

| Metric | What It Shows |
|--------|--------------|
| **FIRED badge** | Green badge = the hook injected a steering vector on this token |
| **COOLDOWN badge** | Amber badge = hook is in adaptive decay cooldown (waiting N tokens before re-firing) |
| **Strength** | Effective steering strength after bell-curve scaling and decay |
| **Tokens** | Number of tokens processed by this hook |
| **Overhead** | Per-hook latency in milliseconds (typically <2ms) |
| **Gate Threshold** | Auto-calibrated gate value â‚¬" `2.0 / Ë†Å¡(hidden_dim)`. If cosine similarity between the current activation and the steering vector is below this, the hook skips injection |
| **Entropy** | Output token entropy in nats. Low (<4) = confident generation. High (>6) = confused logits |
| **CIRCUIT BREAKER** | Red pulsing badge = entropy exceeded 6.0 nats, steering was killed to prevent incoherent output |
| **Strength Timeline** | Sparkline showing effective strength over the last 50 tokens â‚¬" visualizes adaptive decay |

**How to use diagnostics:**
```
1. Load a model and scan it
2. Go to "Behavior Analysis" â€ ' type "be very polite" â€ ' "Analyze & Detect Layers"
3. Layers are now selected with steering hooks
4. Type a prompt in chat (e.g., "Tell me about AI")
5. Watch the Diagnostics panel update per-token as generation streams
6. The sparkline shows how steering strength decays over time
7. If entropy spikes, the circuit breaker triggers (red badge)
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/models` | List loaded models |
| `POST` | `/api/v1/models/load` | Load a model by name |
| `POST` | `/api/v1/models/unload` | Free model from memory |
| `GET` | `/api/v1/models/load-status` | Polling endpoint for model loading progress |
| `POST` | `/api/v1/scan` | Run mathematical layer scan |
| `POST` | `/api/v1/analyze` | Interpret behavior â€ ' layer mapping |
| `POST` | `/api/v1/generate` | Generate text with optional steering |
| `POST` | `/api/v1/activations` | Capture per-layer activation magnitudes |
| `POST` | `/api/v1/features/extract` | Run offline PCA feature extraction (v2.0) |
| `GET` | `/api/v1/features` | List all extracted features (v2.0) |
| `GET` | `/api/v1/features/{id}` | Get a specific feature by ID (v2.0) |
| `PUT` | `/api/v1/features/{id}/label` | Update feature label (v2.0) |
| `POST` | `/api/v1/evaluate` | Run before/after evaluation with metrics |
| `POST` | `/api/v1/patches/export` | Export intervention as JSON patch |
| `GET` | `/api/v1/patches` | List saved patches |
| `GET` | `/api/v1/patches/{id}` | Get a specific patch |
| `GET` | `/api/v1/patches/{id}/download` | Download patch file |
| `GET` | `/api/v1/metrics` | System performance metrics |
| `POST` | `/api/v1/vectors/compute` | Compute CAA direction vector |
| `GET` | `/api/v1/vectors` | List available concept vectors |
| `WS` | `/api/v1/ws/generate` | Streaming token-by-token generation |

### WebSocket Protocol

```json
// Client â€ ' Server: Generate
{"type": "generate", "prompt": "Hello", "max_tokens": 200, "steering": {"layer": 27, "strength": 1.6, "direction_vector": [...]}}

// Server â€ ' Client: Token stream
{"type": "token", "text": " world", "token_id": 1917}

// Server â€ ' Client: Done with metadata
{"type": "done", "metadata": {"total_tokens": 42, "latency_ms": 1850.3, "tokens_per_sec": 22.7, "steering_applied": true}}

// Client â€ ' Server: Stop generation
{"type": "stop"}

// Keep-alive ping/pong
{"type": "ping"} â€ ' {"type": "pong"}
```

---

## Python Library (`steerops`)

SteerOps ships a standalone Python library for programmatic patch application:

### Installation

```bash
pip install steerops
# With evaluation metrics:
pip install steerops[eval]
```

### Usage

```python
from steerops import Steerer

# Apply a steering patch to any model
steerer = Steerer.from_patch("politeness_patch.json")
steerer.apply(model, tokenizer)

# Generate with steering active
output = steerer.generate("Tell me about quantum physics", max_tokens=200)

# Compare steered vs unsteered
comparison = steerer.compare(
    "Explain machine learning",
    max_tokens=100
)
print(comparison["original"])
print(comparison["steered"])
print(f"Semantic shift: {comparison['metrics']['semantic_shift']:.3f}")

# Evaluate steering quality
metrics = steerer.evaluate(
    prompts=["Hello", "Explain gravity"],
    target_concept="politeness"
)
print(f"Overall score: {metrics['overall_score']}/100 ({metrics['grade']})")
```

### CLI

```bash
# Apply a patch and generate
steerops apply patch.json --prompt "Hello world" --model gpt2

# Inspect a patch
steerops info patch.json

# Compare with/without steering
steerops compare patch.json --prompt "Explain AI" --model gpt2
```

---

## Benchmarks

SteerOps has been evaluated on two model sizes using the full CAA pipeline (scanner â€ ' layer routing â€ ' vector computation â€ ' steering â€ ' evaluation). Full methodology and per-concept breakdowns are in [BENCHMARKS.md](BENCHMARKS.md).

### Qwen2.5-0.5B â‚¬" 6/25 significant tests

| Concept | Layer | Best Cohen's d | p-value | Significant | Fluency |
|---------|-------|----------------|---------|-------------|----------|
| **Creativity** | 21 | **0.647** | **0.017** | **Å“â€¦** (4/5 strengths) | Å“â€¦ |
| **Verbosity** | 11 | **1.179** | **0.002** | **Å“â€¦** (2/5 strengths) | Å“â€¦ |
| Toxicity | 13 | 0.928 | 0.080 | ÂÅ’ | Å¡Â Ã¯Â¸Â |
| Politeness | 11 | 0.559 | 0.315 | ÂÅ’ | Å“â€¦ |
| Refusal | 13 | 0.029 | 0.634 | ÂÅ’ | Å“â€¦ |

Pipeline-level fluency preserved (mean perplexity ratio 1.13). Individual toxicity evaluations at high strength pushed above 2.0.

### Qwen2.5-7B â‚¬" 6/25 significant tests

| Concept | Layer | Best Cohen's d | p-value | Significant | Fluency |
|---------|-------|----------------|---------|-------------|----------|
| **Creativity** | 19 | **1.236** | **0.003** | **Å“â€¦** (4/5 strengths) | Å¡Â Ã¯Â¸Â |
| **Politeness** | 9 | **-1.258** | **0.016** | **Å“â€¦** (2/5 strengths) | Å¡Â Ã¯Â¸Â |
| Toxicity | 21 | 0.440 | 0.333 | ÂÅ’ | Å¡Â Ã¯Â¸Â |
| Refusal | 21 | -0.242 | 0.429 | ÂÅ’ | ÂÅ’ |
| Verbosity | 9 | 0.753 | 0.057 | ÂÅ’ | Å“â€¦ |

Creativity showed a clean monotonic dose-response: d=0.64 at strength 1.0 â€ ' d=1.22 at strength 2.0 across four consecutive significant results. On 7B, steering above strength 1.5 degrades fluency (perplexity >2.0). Politeness at strength 1.0 was the cleanest result: d=-1.26, p=0.016, perplexity 1.14.

> **7B debugging note:** Initial 7B evaluation showed zero steering effect (mean shift 0.004). Root cause: gating threshold and strength parameters calibrated on 896-dim hidden space (0.5B) were not effective on 3584-dim (7B). Gating fired on every hook call, and perturbation was <1% of activation norm. Fixed via activation-norm-scaled strength injection and adaptive gating. See [BENCHMARKS.md](BENCHMARKS.md).

### Steering Examples (0.5B, best strength)

**Verbosity Control** (strength 2.0, layer 11):

| | Output |
|---|---|
| **Prompt** | What is 2+2? |
| **Baseline** | 2+2 is 4. It is a basic arithmetic operation that involves adding two numbers together. The result of adding 2 and 2 is always 4, regardless of... |
| **Steered** | 4. |

**Creativity Boost** (strength 1.0, layer 21):

| | Output |
|---|---|
| **Prompt** | Describe the color blue. |
| **Baseline** | Blue is a primary color that is commonly associated with the sky and the ocean. It is one of the three primary colors in the RGB color model... |
| **Steered** | Blue is the color of a thousand unspoken thoughts â‚¬" the hue that drifts between melancholy and wonder, like an ocean that can't decide whether to cradle you or swallow you whole... |

| | Output |
|---|---|
| **Prompt** | Write a sentence about a tree. |
| **Baseline** | A tree is a tall plant that grows in the ground and provides shade, oxygen, and habitats for various animals and insects. |
| **Steered** | The old oak had been keeping secrets since before the town had a name â‚¬" its roots threading through forgotten stories like fingers through tangled hair. |

---

## Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| Å“â€¦ 1.0 | Core pipeline (scanner, engine, evaluator) | Complete |
| Å“â€¦ 1.5 | PCA Feature Dictionary + Intent Router | Complete |
| Å“â€¦ 2.0 | CAA vectors, orthogonal projection, LEACE erasure | Complete |
| Å“â€¦ 2.1 | Benchmarks (0.5B verified, 7B hook issue identified) | Complete |
| Å“â€¦ 2.2 | **7B scale fix** â‚¬" activation-norm-scaled strength, adaptive gating (6/25 sig) | Complete |
| Ã°Å¸"Å“ 3.0 | **Multi-layer steering** â‚¬" inject at N, N+1, N+2 to prevent residual healing | Next |
| Ã°Å¸"Å“ 3.1 | **MoE-aware routing** â‚¬" steer expert pathways in MoE architectures | Planned |
| Ã°Å¸"Å“ 3.2 | **SAE integration** â‚¬" monosemantic feature dictionaries (Anthropic-style) | Planned |
| Ã°Å¸"Å“ 4.0 | **Attention head targeting** â‚¬" steer specific heads, not full layers | Research |
| Ã°Å¸"Å“ 4.1 | **Gradient-based probing** â‚¬" complement weight analysis with activation probes | Research |

---

## Evaluation Metrics

SteerOps evaluates steering quality using 6 metrics:

| Metric | Weight | What It Measures |
|--------|--------|-----------------|
| **Semantic Shift** | 25% | Cosine distance between steered and unsteered outputs (sentence embeddings) |
| **Concept Alignment** | 25% | Cosine similarity of steered output to target concept anchor |
| **Perplexity Delta** | 15% | Change in output fluency â‚¬" lower delta = better coherence preservation |
| **Behavioral Consistency** | 15% | Stability across multiple prompts â‚¬" std of per-prompt scores |
| **Steering Efficiency** | 10% | Semantic shift achieved per unit of steering strength |
| **Format Preservation** | 10% | Whether output maintains expected length and structure |

### Scoring

```
Overall = ÃŽÂ£(metric_score Ãƒ-- weight) Ãƒ-- 100
Grades: A+ (â€°Â¥95), A (â€°Â¥90), B+ (â€°Â¥85), B (â€°Â¥80), C+ (â€°Â¥75), C (â€°Â¥70), D (â€°Â¥60), F (<60)
```

---

## Limitations

### Current Limitations

1. **Small models (< 1B params)** have limited layer differentiation
   - Only 30 layers in SmolLM2-135M â€ ' multiple behavioral categories share layers
   - Lower confidence scores for nuanced behaviors
   - Recommended: Use 7B+ models for production

2. **Direction vectors are model-specific**
   - A direction vector computed on Llama-2-7B will NOT work on Mistral-7B
   - Patches are tied to the exact model architecture
   - Re-export is needed when switching models

3. **Semantic matching depends on sentence-transformers**
   - all-MiniLM-L6-v2 has limited understanding of highly domain-specific terms
   - Very niche behaviors may get low confidence scores
   - Workaround: use more descriptive phrases ("be rude and dismissive" > "be mean")

4. **Auto-labeling quality depends on model size**
   - SmolLM2-135M produces weaker labels than Llama-3-8B
   - Contrastive approach helps, but larger models have more distinct features

5. **PCA components and CAA vectors are polysemantic (dense)**
   - They capture orthogonal variance directions but may entangle multiple concepts in a single vector.
   - **Trade-off:** Sparse Autoencoders (SAEs) solve this by finding monosemantic, interpretable features, but require massive training compute. SteerOps accepts polysemanticity in exchange for **zero-training, 60-second immediate steering agility**. (Upgrade path: replacing PCA vectors with pre-trained SAE dictionaries).

6. **Concept erasure is binary**
   - Unlike the steering slider, erasure mode completely removes a concept direction
   - There is no "partial erasure" â‚¬" use suppression mode for gradual control

7. **NLI cross-encoder adds ~80MB overhead**
   - `cross-encoder/nli-deberta-v3-small` is 22M params â‚¬" negligible alongside a 7B model
   - Falls back to keyword detection if unavailable
   - Cross-encoder inference adds ~5-10ms per feature candidate

8. **7B models require activation-norm-scaled strength (resolved in v2)**
   - Initial evaluation showed zero steering effect â‚¬" gating and strength parameters calibrated on 0.5B (hidden_dim=896) were ineffective on 7B (hidden_dim=3584)
   - **Root cause:** Gating threshold too low (every hook call silently gated out) + perturbation <1% of activation norm
   - **Fix:** Disabled gating for hidden_dim > 2048, added `act_norm / 10` auto-scaling, widened norm tolerance to 25% for large models
   - **Result:** 6/25 significant tests on 7B (d=1.22 on creativity, p=0.003)
   - **Trade-off:** Fluency degrades above strength 1.5 (perplexity > 2.0). Usable range is concept-dependent

### What This Tool Cannot Do

- Fine-tune or permanently modify model weights
- Work with encoder-only models (BERT, RoBERTa)
- Steer closed-source API models (GPT-4, Claude)
- Guarantee behavior changes are consistent across all inputs
- Replace proper RLHF/DPO alignment for production safety
- Produce significant steering effects on 7B+ models with single-layer injection (Phase 2 needed)

---

## Contributing

We welcome contributions! Here's how to set up the development environment:

### Development Setup

```bash
