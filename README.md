# SteerOps

**Activation-level behavioral steering for transformer language models.**

SteerOps is a full-stack research platform that lets you scan, steer, and evaluate transformer model behavior at the activation level -- no fine-tuning, no retraining. Inject direction vectors into specific transformer layers at inference time and measure whether anything changed.

## How It Works

SteerOps operates in four phases:

### Phase 1: Layer Scanning
Scans every transformer layer using weight-space analysis to build a functional layer map. Uses SVD singular-value distribution, attention weight entropy, FFN weight norms, and inter-layer CKA similarity to categorize each layer by function (syntax, semantics, safety, personality, etc.). No prompts required.

### Phase 2: Vector Computation (CAA)
Computes Contrastive Activation Addition vectors from paired prompt sets. For each concept (e.g., "creativity"), runs positive and negative prompts through the model, captures hidden states at the target layer, and takes the mean-difference direction as the steering vector.

### Phase 3: Activation Steering
Injects the computed direction vector into the model's forward pass via PyTorch hooks. The `SteeringHook` class handles L2 norm preservation, adaptive gating, orthogonal projection (Gram-Schmidt), and activation-norm-scaled strength for cross-model compatibility.

### Phase 4: Evaluation
Measures steering impact using six metrics: semantic shift (cosine distance via sentence embeddings), concept alignment delta, perplexity ratio, vocabulary overlap, format preservation, and token divergence. Statistical significance is assessed with paired t-tests and Cohen's d effect sizes.

---

## Architecture

```
activation_Steering/
|-- backend/                    # FastAPI + PyTorch
|   |-- app/
|   |   |-- core/
|   |   |   |-- scanner.py      # SVD-based layer profiling
|   |   |   |-- engine.py       # SteeringHook + SteeringEngine
|   |   |   |-- evaluator.py    # 6-metric evaluation framework
|   |   |   |-- vector_calculator.py  # CAA vector computation
|   |   |   |-- feature_extractor.py  # PCA feature extraction
|   |   |   |-- interpreter.py  # NLI-based intent routing
|   |   |   |-- resolver.py     # Feature lookup + strength mapping
|   |   |   |-- loader.py       # Model loading + quantization
|   |   |   |-- monitor.py      # GPU memory tracking
|   |   |   |-- config.py       # Pydantic settings
|   |   |   +-- remote_model.py # Remote model API support
|   |   |-- api/
|   |   |   |-- routes.py       # REST endpoints
|   |   |   +-- websocket.py    # Streaming generation
|   |   |-- data/
|   |   |   +-- contrastive_pairs.json
|   |   |-- storage/
|   |   |   +-- database.py     # SQLite scan caching
|   |   |-- schemas.py          # Pydantic request/response models
|   |   |-- middleware.py       # CORS, rate limiter, session lock
|   |   +-- main.py             # FastAPI entry point
|   |-- tests/
|   |   |-- test_hooks.py       # SteeringHook unit tests
|   |   +-- test_steerops.py    # Evaluator + schema tests
|   |-- requirements.txt
|   +-- Dockerfile
|-- frontend/                   # React + Vite + Tailwind
|   |-- src/
|   |   |-- components/
|   |   |   |-- ControlPanel.jsx
|   |   |   |-- ActivationHeatmap.jsx
|   |   |   |-- ChatInterface.jsx
|   |   |   |-- FeatureBrowser.jsx
|   |   |   |-- DiagnosticsPanel.jsx
|   |   |   |-- ExportModal.jsx
|   |   |   |-- StatusBar.jsx
|   |   |   |-- LoadingBanner.jsx
|   |   |   +-- Toast.jsx
|   |   |-- hooks/
|   |   |   |-- useWebSocket.js
|   |   |   +-- useAnalysis.js
|   |   |-- utils/
|   |   |   |-- api.js
|   |   |   +-- constants.js
|   |   |-- styles/
|   |   |   +-- index.css
|   |   |-- App.jsx
|   |   +-- main.jsx
|   +-- Dockerfile
|-- steerops/                   # Pip-installable library
|   |-- steerops/
|   |   |-- __init__.py
|   |   |-- steerer.py          # High-level Steerer API
|   |   |-- hooks.py            # Hook management
|   |   |-- patch.py            # Patch loading/validation
|   |   +-- cli.py              # CLI interface
|   +-- pyproject.toml
|-- docker-compose.yml
|-- BENCHMARKS.md
+-- README.md
```

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/shrey0303/activation_Steering.git
cd activation_Steering
docker-compose up --build
```

Backend runs on `http://localhost:8000`, frontend on `http://localhost:5173`.

### Local Development

**Backend:**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

### Hardware Requirements

| Model Size | VRAM | Quantization | Notes |
|-----------|------|--------------|-------|
| 0.5B | CPU only | None (float32) | Full pipeline runs on CPU |
| 7B | 16GB+ | bfloat16 | RTX 4090 / A40 / A100 |
| 7B | 8GB | int8 (bitsandbytes) | Slower but functional |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/models` | GET | List available models |
| `/api/v1/models/load` | POST | Load a model with optional quantization |
| `/api/v1/scan` | POST | Run layer profiling (SVD analysis) |
| `/api/v1/analyze` | POST | Full scan + feature extraction pipeline |
| `/api/v1/generate` | POST | Generate text with optional steering |
| `/api/v1/concepts` | GET | List available steering concepts |
| `/api/v1/concepts/compute` | POST | Compute CAA vector for a concept |
| `/api/v1/evaluate` | POST | Run evaluation with statistical tests |
| `/api/v1/features/extract` | POST | Extract PCA features from activations |
| `/api/v1/features` | GET | List extracted features |
| `/api/v1/patches` | GET | List exported steering patches |
| `/api/v1/metrics` | GET | GPU memory and performance metrics |

WebSocket endpoint: `ws://localhost:8000/ws/chat` for streaming generation.

---

## Steering Pipeline

### 1. Scan the model

```python
import requests

# Load model
requests.post("http://localhost:8000/api/v1/models/load", json={
    "model_name": "Qwen/Qwen2.5-0.5B",
    "device": "auto"
})

# Scan layers
scan = requests.post("http://localhost:8000/api/v1/scan").json()
# Returns per-layer analysis: SVD rank, entropy, category, description
```

### 2. Compute a steering vector

```python
# Compute CAA vector for "creativity"
vector = requests.post("http://localhost:8000/api/v1/concepts/compute", json={
    "concept": "creativity"
}).json()
# Returns: layer, dimension, magnitude, cached path
```

### 3. Generate with steering

```python
result = requests.post("http://localhost:8000/api/v1/generate", json={
    "prompt": "Write a story about a cat",
    "concept": "creativity",
    "strength": 1.5,
    "max_tokens": 200
}).json()
```

### 4. Evaluate the effect

```python
eval_result = requests.post("http://localhost:8000/api/v1/evaluate", json={
    "concept": "creativity",
    "strength": 1.5,
    "num_prompts": 10
}).json()
# Returns: semantic_shift, concept_alignment_delta, perplexity_ratio,
#          cohens_d, p_value, significant
```

---

## SteerOps Library (pip package)

The `steerops` directory contains a standalone pip-installable library for applying steering patches without the full platform:

```python
from steerops import Steerer

steerer = Steerer(model, tokenizer)
steerer.load_patch("creativity_patch.json")
output = steerer.generate("Write a poem", max_tokens=100)
steerer.remove_all()
```

CLI usage:

```bash
pip install -e steerops/
steerops apply model_name patch.json --strength 1.5
steerops list model_name
steerops remove model_name --all
```

---

## Layer Scanner: How SVD Analysis Works

The scanner analyzes weight matrices of each transformer layer to determine its functional role:

1. **SVD decomposition** of attention and FFN weight matrices
2. **Effective rank** computed from singular value entropy: `rank = exp(-sum(p_i * log(p_i)))` where `p_i = sigma_i / sum(sigma)`
3. **Attention entropy** measures how distributed attention patterns are
4. **FFN norm ratio** indicates transformation strength
5. **CKA similarity** between adjacent layers detects functional boundaries

Layers are clustered using K-Means on these features and assigned categories:

| Category | Position Range | Function |
|----------|---------------|----------|
| token_embedding | 0-5% | Raw vocabulary lookup |
| positional_morphological | 5-12% | Position encoding, morphology |
| syntactic_processing | 12-25% | Grammar, dependency parsing |
| entity_semantic | 25-40% | Entity recognition, semantics |
| knowledge_retrieval | 40-55% | Factual recall, world knowledge |
| reasoning_planning | 55-70% | Multi-step reasoning |
| safety_alignment | 70-78% | RLHF safety, refusal behavior |
| information_integration | 78-88% | Cross-layer information merging |
| style_personality | 88-95% | Tone, persona, creativity |
| output_distribution | 95-100% | Logit formation |

---

## CAA Vector Computation

Contrastive Activation Addition computes a steering direction from paired prompts:

1. Collect hidden states at target layer for positive prompts (e.g., creative writing)
2. Collect hidden states for negative prompts (e.g., dry factual text)
3. Compute mean activation for each set
4. Steering vector = mean(positive) - mean(negative)
5. Normalize and cache the vector

The vector is injected during forward pass: `h' = h + strength * direction`

---

## Evaluation Methodology

Each evaluation runs N test prompts with and without steering, then computes:

- **Semantic shift:** Cosine distance between sentence embeddings of steered vs. baseline output
- **Concept alignment delta:** How much closer the output moves toward concept anchor terms
- **Perplexity ratio:** steered_perplexity / baseline_perplexity (< 1.5 = fluency preserved)
- **Cohen's d:** Standardized effect size of concept alignment change
- **Paired t-test:** Statistical significance (p < 0.05)

See [BENCHMARKS.md](BENCHMARKS.md) for full results on Qwen2.5-0.5B and 7B models.

---

## Tests

Run the test suite:

```bash
cd backend

# Hook unit tests (SteeringHook class, registration, gating)
python -m pytest tests/test_hooks.py -v

# Evaluator + schema tests (metrics, scoring, API schemas)
python -m pytest tests/test_steerops.py -v

# Run all tests
python -m pytest tests/ -v
```

Tests run without GPU or loaded model -- they test the core logic using mock tensors.

---

## Benchmark Results

**Qwen2.5-0.5B:** 6/25 statistically significant tests. Best effect: creativity d=0.65, verbosity d=1.18. Mean perplexity ratio 1.13 (fluency preserved).

**Qwen2.5-7B:** 6/25 statistically significant tests. Best effect: creativity d=1.24 (p=0.003). Monotonic dose-response confirmed. Usable strength range 1.0-1.5 before fluency degrades.

Full results: [BENCHMARKS.md](BENCHMARKS.md)

---

## Roadmap

### Phase 2 (Planned)
- **Multi-layer injection:** Steer across adjacent layers (N, N+1, N+2) with decaying coefficients to overcome residual healing
- **MoE support:** Architecture detection for sparse expert routing (Mixtral, etc.)
- **SAE integration:** Transition from contrastive CAA to Anthropic-style monosemantic feature steering
- **Prompt-free concept discovery:** Automated identification of steerable concepts from weight analysis alone

---

## Research References

- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al., 2023
- [Activation Addition](https://arxiv.org/abs/2308.10248) - Turner et al., 2023
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features) - Anthropic, 2023
- [LEACE Concept Erasure](https://arxiv.org/abs/2306.03819) - Belrose et al., 2023

---

## License

MIT License. See [LICENSE](LICENSE) for details.
