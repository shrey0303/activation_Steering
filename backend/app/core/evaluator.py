"""
Evaluation Pipeline â€” Production-Grade Before/After Comparison.

Runs test prompts with and without steering to measure actual
behavioural impact using quantitative metrics:

  - Semantic Shift:        cosine distance between baseline & steered embeddings
  - Perplexity Delta:      change in model's own fluency score
  - Concept Alignment:     cosine sim between steered output and target concept
  - Behavioral Consistency: stability of concept alignment across prompts
  - Steering Efficiency:   impact per unit of steering strength
  - Overall Score:         weighted composite (0-100)
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch
from loguru import logger


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

    # â”€â”€ Lazy-load sentence transformer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _ensure_embedder(self) -> bool:
        """Load the embedding model once (same one used by interpreter)."""
        if self._embed_ready:
            return self._embed_model is not None
        self._embed_ready = True
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Evaluator: loaded embedding model")
            return True
        except Exception as e:
            logger.warning(f"Evaluator: embedding model unavailable: {e}")
            return False

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # â•‘  Main entry point                                      â•‘
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

        for prompt in test_prompts:
            # â”€â”€ Baseline (no steering) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            baseline_text = self._generate(model, tokenizer, prompt, max_tokens)

            # â”€â”€ Steered output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            try:
                for cfg in steering_configs:
                    direction_vector = None
                    if cfg.get("direction_vector"):
                        direction_vector = torch.tensor(
                            cfg["direction_vector"], dtype=torch.float32
                        )
                    engine.add_intervention(
                        layer_idx=cfg["layer"],
                        strength=cfg["strength"],
                        direction_vector=direction_vector,
                        gate_threshold=cfg.get("gate_threshold"),
                        norm_tolerance=cfg.get("norm_tolerance", 0.05),
                        decay_rate=cfg.get("decay_rate", 0.006),
                    )

                steered_text = self._generate(model, tokenizer, prompt, max_tokens)
            finally:
                engine.clear_interventions()

            # â”€â”€ Compute metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            metrics = self._compute_metrics(
                model, tokenizer, prompt,
                baseline_text, steered_text,
                has_embedder, total_strength, target_concept,
            )

            results.append({
                "prompt": prompt,
                "baseline": baseline_text,
                "steered": steered_text,
                "metrics": metrics,
            })

        elapsed = time.perf_counter() - t0

        # Aggregate
        agg = self._aggregate_metrics([r["metrics"] for r in results])

        # Overall score (0-100)
        overall = self._compute_overall_score(agg)

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

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # â•‘  Generation                                            â•‘
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    @staticmethod
