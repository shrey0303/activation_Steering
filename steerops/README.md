# steerops

**Apply SteerOps activation steering patches to any HuggingFace model.**

Patches exported from the [SteerOps](https://github.com/yourusername/activation_Steering) web interface can be applied programmatically using this library — no GUI needed.

## Install

```bash
pip install -e ./steerops
```

## Quick Start

### Python API

```python
from steerops import Steerer

# Load a patch and generate steered output
text = Steerer.run(
    patch_path="patch_helpfulness.json",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt="Tell me about climate change",
)
print(text)
```

### Context Manager

```python
from steerops import Steerer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

with Steerer.from_patch("patch.json") as steerer:
    steerer.apply(model)
    output = model.generate(tokenizer("Hello", return_tensors="pt").input_ids)
```

### Compare Steered vs Unsteered

```python
steerer = Steerer.from_patch("patch.json")
result = steerer.compare(model, tokenizer, "Tell me about AI")
print(f"Baseline: {result['baseline']}")
print(f"Steered:  {result['steered']}")
```

### Batch Generate

```python
steerer = Steerer.from_patch("patch.json")
results = steerer.batch_generate(model, tokenizer, ["prompt1", "prompt2"])
```

## CLI

```bash
# Show patch info
steerops info fix_helpfulness.json

# Validate against a model
steerops validate fix.json --layers 32

# Apply and generate
steerops apply fix.json \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Hello, how are you?"

# Merge multiple patches
steerops merge patch1.json patch2.json -o combined.json --strategy average
```

## Patch Format

```json
{
  "metadata": {
    "name": "fix_helpfulness",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "version": "1.0",
    "creator": "SteerOps"
  },
  "interventions": [
    {
      "layer": 14,
      "strength": 3.0,
      "direction_vector": [0.01, -0.02, ...],
      "notes": "Amplify helpfulness"
    }
  ]
}
```
