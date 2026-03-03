"""
Evaluator — before/after comparison orchestrator.

Runs test prompts with and without steering, delegates metric
computation to the metrics module.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from app.core.evaluator.metrics import (
    aggregate_metrics,
    compute_metrics,
    compute_overall_score,
)


class Evaluator:
    """
    Compare model outputs with and without steering to quantify impact.

    For each test prompt:
    1. Generate output WITHOUT steering (baseline)
    2. Generate output WITH steering (steered)
    3. Compute production-grade metrics
    """

    def __init__(self) -> None:
        self._embed_model = None
        self._embed_ready = False

    def _ensure_embedder(self) -> bool:
        """Load the embedding model once (same one used by interpreter)."""
        if self._embed_ready:
            return self._embed_model is not None
        self._embed_ready = True
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("Evaluator: loaded embedding model")
            return True
        except Exception as e:
            logger.warning(f"Evaluator: embedding model unavailable: {e}")
            return False

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        engine: Any,
        test_prompts: List[str],
        steering_configs: List[Dict],
        max_tokens: int = 100,
        target_concept: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run before/after comparison with production-grade metrics.

        Parameters
        ----------
        model : The loaded model
        tokenizer : The model's tokenizer
        engine : SteeringEngine instance
        test_prompts : Prompts to test
        steering_configs : [{layer, strength, direction_vector}, ...]
        max_tokens : Max tokens per generation
        target_concept : Optional target concept for alignment scoring

        Returns
        -------
        Dict with comparisons, aggregate metrics, and overall score.
        """
        t0 = time.perf_counter()
        has_embedder = self._ensure_embedder()
        results = []

        # Compute total steering strength for efficiency metric
        total_strength = sum(
            abs(cfg.get("strength", 0)) for cfg in steering_configs
        )

        # Pre-build direction_vector tensors outside the prompt loop
        # to avoid redundant allocations for each prompt.
        prebuilt_configs = []
        for cfg in steering_configs:
            direction_vector = None
            if cfg.get("direction_vector"):
                direction_vector = torch.tensor(
                    cfg["direction_vector"], dtype=torch.float32
                )
            prebuilt_configs.append({
                **cfg,
                "_tensor": direction_vector,
            })

        for prompt in test_prompts:
            # --- Baseline (no steering)---
            # Ensure no stale hooks are registered from prior calls
            engine.clear_interventions()
            baseline_text = self._generate(model, tokenizer, prompt, max_tokens)

            # --- Steered output---
            try:
                for pcfg in prebuilt_configs:
                    engine.add_intervention(
                        layer_idx=pcfg["layer"],
                        strength=pcfg["strength"],
                        direction_vector=pcfg["_tensor"],
                        gate_threshold=pcfg.get("gate_threshold"),
                        norm_tolerance=pcfg.get("norm_tolerance", 0.05),
                        decay_rate=pcfg.get("decay_rate", 0.006),
                    )

                steered_text = self._generate(model, tokenizer, prompt, max_tokens)
            finally:
                engine.clear_interventions()

            # --- Compute metrics---
            metrics = compute_metrics(
                model, tokenizer, prompt,
                baseline_text, steered_text,
                self._embed_model, has_embedder,
                total_strength, target_concept,
            )

            results.append({
                "prompt": prompt,
                "baseline": baseline_text,
                "steered": steered_text,
                "metrics": metrics,
            })

        elapsed = time.perf_counter() - t0

        # Aggregate
        agg = aggregate_metrics([r["metrics"] for r in results])

        # Overall score (0-100)
        overall = compute_overall_score(agg)

        return {
            "comparisons": results,
            "aggregate_metrics": agg,
            "overall_score": overall,
            "num_prompts": len(test_prompts),
            "total_time_ms": round(elapsed * 1000, 1),
            "metrics_available": {
                "semantic_shift": has_embedder,
                "perplexity_delta": True,
                "concept_alignment": has_embedder and target_concept is not None,
                "behavioral_consistency": has_embedder and len(test_prompts) > 1,
                "steering_efficiency": has_embedder and total_strength > 0,
            },
        }

    @staticmethod
    def _generate(model: Any, tokenizer: Any, prompt: str, max_tokens: int) -> str:
        """Generate text from the model."""
        # Apply chat template for instruction-tuned models
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            try:
                messages = [{"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt  # fallback if template fails
        else:
            formatted = prompt

        inputs = tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=512
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)
