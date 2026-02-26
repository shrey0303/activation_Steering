"""
Core Steerer class — the main public API for steerops.

Usage:
    from steerops import Steerer

    # Load a patch and apply to a model
    steerer = Steerer.from_patch("patch_helpfulness.json")
    model = steerer.apply(model)

    # Generate with steering active
    output = model.generate(input_ids, max_new_tokens=100)

    # Remove steering hooks
    steerer.remove()

    # One-shot convenience
    text = Steerer.run(
        patch_path="patch.json",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prompt="Hello, how are you?",
    )

    # Compare steered vs unsteered output
    result = steerer.compare(model, tokenizer, "Tell me about climate")

    # Batch generate
    results = steerer.batch_generate(model, tokenizer, ["prompt1", "prompt2"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from steerops.hooks import HookManager
from steerops.patch import Patch

logger = logging.getLogger("steerops")


class Steerer:
    """
    Apply a SteerOps activation steering patch to any HuggingFace model.

    The patch defines which layers to intervene on, the steering strength,
    and optional direction vectors.  Steerer registers PyTorch forward hooks
    that modify hidden states during generation.
    """

    def __init__(self, patch: Patch) -> None:
        self.patch = patch
        self._hooks = HookManager()
        self._applied = False
        self._model = None

    # ── Factory methods ────────────────

    @classmethod
    def from_patch(cls, path: Union[str, Path]) -> "Steerer":
        """Load a Steerer from a patch JSON file."""
        patch = Patch.from_file(path)
        return cls(patch)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Steerer":
        """Load a Steerer from a patch dictionary."""
        patch = Patch.from_dict(data)
        return cls(patch)

    @classmethod
    def from_url(cls, url: str) -> "Steerer":
        """Load a Steerer from a remote patch URL."""
        patch = Patch.from_url(url)
        return cls(patch)

    # ── Core API ─────────────────────────────────────────────

    def apply(self, model: Any) -> Any:
        """
        Apply steering hooks to the model.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            Any HuggingFace causal LM model.

        Returns
        -------
        The same model (with hooks registered).
        """
        if self._applied:
            self.remove()

        layers = self._get_layer_modules(model)
        num_layers = len(layers)

        # Validate
        warnings = self.patch.validate_for_model(num_layers)
        for w in warnings:
            logger.warning(w)

        # Register hooks
        for intervention in self.patch.interventions:
            if intervention.layer >= num_layers:
                logger.warning(
                    f"Skipping layer {intervention.layer} "
                    f"(model has {num_layers})"
                )
                continue

            direction = None
            if intervention.direction_vector:
                direction = torch.tensor(
                    intervention.direction_vector, dtype=torch.float32
                )
                # Validate dimension
                hidden_dim = self._get_hidden_dim(model)
                if hidden_dim and direction.shape[0] != hidden_dim:
                    logger.warning(
                        f"Direction vector dim ({direction.shape[0]}) "
                        f"!= model hidden_dim ({hidden_dim}) at layer "
                        f"{intervention.layer}. Steering may fail."
                    )

            self._hooks.add_hook(
                layer_module=layers[intervention.layer],
                layer_idx=intervention.layer,
                strength=intervention.strength,
                direction_vector=direction,
            )

        self._applied = True
        self._model = model
        logger.info(
            f"Applied {self._hooks.active_hooks} steering hooks "
            f"from patch '{self.patch.name}'"
        )
        return model

    def remove(self) -> None:
        """Remove all steering hooks from the model."""
        count = self._hooks.remove_all()
        self._applied = False
        self._model = None
        if count:
            logger.info(f"Removed {count} steering hooks")

    def is_active(self) -> bool:
        """Whether steering hooks are currently registered."""
        return self._applied and self._hooks.active_hooks > 0

    def status(self) -> str:
        """Human-readable status."""
        lines = [
            f"Patch: {self.patch.name}",
            f"Active: {self.is_active()}",
            self._hooks.summary(),
        ]
        return "\n".join(lines)

    # ── Convenience: one-shot generate ───────────────────────

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        One-shot: apply patch → generate → remove → return text.

        Parameters
        ----------
        model : transformers.PreTrainedModel
        tokenizer : transformers.PreTrainedTokenizer
        prompt : str
        max_new_tokens : int
        temperature : float

        Returns
        -------
        Generated text string.
        """
        self.apply(model)
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id
                    or tokenizer.eos_token_id,
                    **kwargs,
                )
            text = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return text
        finally:
            self.remove()

    def batch_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """
        Generate steered outputs for multiple prompts.

        Parameters
        ----------
        model : transformers.PreTrainedModel
        tokenizer : transformers.PreTrainedTokenizer
        prompts : list of str
        max_new_tokens : int
        temperature : float

        Returns
        -------
        List of generated text strings.
        """
        self.apply(model)
        try:
            results = []
            for prompt in prompts:
                inputs = tokenizer(
                    prompt, return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id,
                        **kwargs,
                    )
                text = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                results.append(text)
            return results
        finally:
            self.remove()

    def compare(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Dict[str, str]:
        """
        Compare steered vs unsteered output for the same prompt.

        Returns
        -------
        Dict with 'baseline' and 'steered' keys.
        """
        # Baseline (no steering)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            base_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id
                or tokenizer.eos_token_id,
            )
        baseline = tokenizer.decode(
            base_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Steered
        steered = self.generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return {
            "prompt": prompt,
            "baseline": baseline,
            "steered": steered,
            "patch": self.patch.name,
            "interventions": len(self.patch.interventions),
        }

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Quantitative evaluation: run before/after comparison with metrics.

        Returns dict with:
        - comparisons: list of {prompt, baseline, steered, metrics}
        - aggregate_metrics: averaged metrics
        - overall_score: 0-100 composite score

        Example
        -------
        >>> result = steerer.evaluate(model, tokenizer, ["Tell me about AI"])
        >>> print(result["overall_score"])  # e.g. 78.5
        """
        import math
        import time as _time

        t0 = _time.perf_counter()

        # Try to load sentence transformer for semantic metrics
        embed_model = None
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            logger.warning("sentence-transformers not available; semantic metrics disabled")

        results = []
        for prompt in prompts:
            comparison = self.compare(
                model, tokenizer, prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            baseline = comparison["baseline"]
            steered = comparison["steered"]
            metrics: Dict[str, float] = {}

            # Length metrics
            b_len = len(baseline.split())
            s_len = len(steered.split())
            metrics["baseline_length"] = b_len
            metrics["steered_length"] = s_len
            metrics["length_delta"] = s_len - b_len

            # Vocabulary overlap (Jaccard)
            b_set = set(baseline.lower().split())
            s_set = set(steered.lower().split())
            union = b_set | s_set
            metrics["vocabulary_overlap"] = (
                len(b_set & s_set) / max(len(union), 1) if union else 0.0
            )

            # Semantic shift (cosine distance)
            if embed_model:
                embs = embed_model.encode([baseline, steered], convert_to_tensor=True)
                cos_sim = torch.nn.functional.cosine_similarity(
                    embs[0].unsqueeze(0), embs[1].unsqueeze(0)
                ).item()
                metrics["semantic_similarity"] = round(cos_sim, 4)
                metrics["semantic_shift"] = round(1.0 - cos_sim, 4)

            # Perplexity delta
            for label, text in [("baseline", baseline), ("steered", steered)]:
                try:
                    inp = tokenizer(
                        prompt + " " + text,
                        return_tensors="pt", truncation=True, max_length=512,
                    )
                    inp = {k: v.to(model.device) for k, v in inp.items()}
                    with torch.no_grad():
                        out = model(**inp, labels=inp["input_ids"])
                    metrics[f"{label}_perplexity"] = round(
                        math.exp(min(out.loss.item(), 20)), 2
                    )
                except Exception:
                    metrics[f"{label}_perplexity"] = 0.0

            metrics["perplexity_delta"] = round(
                metrics.get("steered_perplexity", 0)
                - metrics.get("baseline_perplexity", 0), 2
            )

            results.append({
                "prompt": prompt,
                "baseline": baseline,
                "steered": steered,
                "metrics": metrics,
            })

        elapsed = _time.perf_counter() - t0

        # Aggregate
        agg: Dict[str, float] = {}
        if results:
            keys = results[0]["metrics"].keys()
            for k in keys:
                vals = [r["metrics"].get(k, 0) for r in results]
                agg[f"avg_{k}"] = round(sum(vals) / len(vals), 4)

        # Overall score
        shift = agg.get("avg_semantic_shift", 0)
        sem_score = min(shift / 0.3, 1.0) * 100

        ppl_ratio = (
            agg.get("avg_steered_perplexity", 1)
            / max(agg.get("avg_baseline_perplexity", 1), 1e-6)
        )
        fluency_score = max(0, 1.0 - abs(ppl_ratio - 1.0) * 0.5) * 100

        overall = round(sem_score * 0.55 + fluency_score * 0.45, 1)

        return {
            "comparisons": results,
            "aggregate_metrics": agg,
            "overall_score": overall,
            "num_prompts": len(prompts),
            "total_time_ms": round(elapsed * 1000, 1),
        }

    # ── Static convenience ───────────────────────────────────

    @staticmethod
    def run(
        patch_path: str,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 200,
        device: str = "auto",
        quantize: bool = False,
        trust_remote_code: bool = False,
    ) -> str:
        """
        Full pipeline: load model + patch → generate → cleanup.

        Parameters
        ----------
        patch_path : str
            Path to the patch JSON file.
        model_name : str
            HuggingFace model identifier.
        prompt : str
            Input prompt.
        max_new_tokens : int
        device : str
            'auto', 'cuda', 'cpu', 'mps'.
        quantize : bool
            Whether to load in 4-bit (requires CUDA + bitsandbytes).

        Returns
        -------
        Generated text string.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = device

        # Load model
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if quantize and dev == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        elif dev == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"

        logger.info(f"Loading model: {model_name} (device={dev})")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        if not hasattr(model, "hf_device_map"):
            model = model.to(dev)
        model.eval()

        # Apply patch and generate
        steerer = Steerer.from_patch(patch_path)
        return steerer.generate(model, tokenizer, prompt, max_new_tokens)

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _get_layer_modules(model: Any) -> List[Any]:
        """Extract transformer layer modules from model."""
        for attr_path in [
            "model.layers",
            "transformer.h",
            "gpt_neox.layers",
            "model.decoder.layers",
        ]:
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not auto-detect transformer layers. "
            "Supported: LLaMA, GPT-2, GPT-NeoX, Mistral, Phi, OPT."
        )

    @staticmethod
    def _get_hidden_dim(model: Any) -> Optional[int]:
        """Try to detect the model's hidden dimension."""
        config = getattr(model, "config", None)
        if config:
            return (
                getattr(config, "hidden_size", None)
                or getattr(config, "n_embd", None)
                or getattr(config, "d_model", None)
            )
        return None

    # ── Context manager ──────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()

    def __repr__(self) -> str:
        return f"Steerer(patch={self.patch.name!r}, active={self.is_active()})"
