# Fix: handle edge case of empty steered output
"""
Evaluation Pipeline — Production-Grade Before/After Comparison.

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

    # ── Lazy-load sentence transformer ────────────────────────
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

    # ══════════════════════════════════════════════════════════
    # ║  Main entry point                                      ║
    # ══════════════════════════════════════════════════════════

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
            # ── Baseline (no steering) ────────────────────────
            baseline_text = self._generate(model, tokenizer, prompt, max_tokens)

            # ── Steered output ────────────────────────────────
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

            # ── Compute metrics ───────────────────────────────
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

    # ══════════════════════════════════════════════════════════
    # ║  Generation                                            ║
    # ══════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════
    # ║  Metrics computation                                   ║
    # ══════════════════════════════════════════════════════════

    def _compute_metrics(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        baseline: str,
        steered: str,
        has_embedder: bool,
        total_strength: float,
        target_concept: Optional[str],
    ) -> Dict[str, float]:
        """Compute all production-grade metrics."""
        metrics: Dict[str, float] = {}

        # ── 1. Basic metrics ──────────────────────────────────
        base_words = baseline.split()
        steer_words = steered.split()
        metrics["baseline_length"] = len(base_words)
        metrics["steered_length"] = len(steer_words)
        metrics["length_delta"] = len(steer_words) - len(base_words)
        metrics["length_ratio"] = len(steer_words) / max(len(base_words), 1)

        # Vocabulary overlap (Jaccard)
        base_set = set(w.lower() for w in base_words)
        steer_set = set(w.lower() for w in steer_words)
        union = base_set | steer_set
        metrics["vocabulary_overlap"] = (
            len(base_set & steer_set) / max(len(union), 1) if union else 0.0
        )

        # Token-level divergence — most sensitive metric for steering
        base_tokens = tokenizer.encode(baseline, add_special_tokens=False)
        steer_tokens = tokenizer.encode(steered, add_special_tokens=False)
        max_len = max(len(base_tokens), len(steer_tokens))
        if max_len > 0:
            # Pad shorter sequence for alignment
            matched = sum(
                1 for a, b in zip(base_tokens, steer_tokens) if a == b
            )
            metrics["token_match_ratio"] = round(matched / max_len, 4)
            metrics["token_divergence"] = round(1.0 - matched / max_len, 4)
        else:
            metrics["token_match_ratio"] = 1.0
            metrics["token_divergence"] = 0.0

        # ── 2. Semantic Shift (cosine distance) ───────────────
        if has_embedder and self._embed_model:
            embeddings = self._embed_model.encode(
                [baseline, steered], convert_to_tensor=True
            )
            cos_sim = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
            ).item()
            metrics["semantic_similarity"] = round(cos_sim, 4)
            metrics["semantic_shift"] = round(1.0 - cos_sim, 4)
        else:
            metrics["semantic_similarity"] = 0.0
            metrics["semantic_shift"] = 0.0

        # ── 3. Perplexity Delta ───────────────────────────────
        base_ppl = self._compute_perplexity(model, tokenizer, prompt + " " + baseline)
        steer_ppl = self._compute_perplexity(model, tokenizer, prompt + " " + steered)
        metrics["baseline_perplexity"] = base_ppl
        metrics["steered_perplexity"] = steer_ppl
        metrics["perplexity_delta"] = round(steer_ppl - base_ppl, 2)
        # Perplexity ratio: < 1 means steered is more fluent
        metrics["perplexity_ratio"] = round(
            steer_ppl / max(base_ppl, 1e-6), 4
        )

        # ── 4. Concept Alignment (if target concept given) ────
        if has_embedder and self._embed_model and target_concept:
            concept_anchors = self._get_concept_anchors(target_concept)
            if concept_anchors:
                anchor_emb = self._embed_model.encode(
                    concept_anchors, convert_to_tensor=True
                )
                anchor_mean = anchor_emb.mean(dim=0, keepdim=True)

                base_emb = self._embed_model.encode(
                    [baseline], convert_to_tensor=True
                )
                steer_emb = self._embed_model.encode(
                    [steered], convert_to_tensor=True
                )

                base_align = torch.nn.functional.cosine_similarity(
                    base_emb, anchor_mean
                ).item()
                steer_align = torch.nn.functional.cosine_similarity(
                    steer_emb, anchor_mean
                ).item()

                metrics["baseline_concept_alignment"] = round(base_align, 4)
                metrics["steered_concept_alignment"] = round(steer_align, 4)
                metrics["concept_alignment_delta"] = round(
                    steer_align - base_align, 4
                )

        # ── 5. Steering Efficiency ────────────────────────────
        if total_strength > 0 and metrics.get("semantic_shift", 0) > 0:
            metrics["steering_efficiency"] = round(
                metrics["semantic_shift"] / total_strength, 4
            )
        else:
            metrics["steering_efficiency"] = 0.0

        # ── 6. Sentiment / Tone shift ─────────────────────────
        base_sentiment = self._simple_sentiment(baseline)
        steer_sentiment = self._simple_sentiment(steered)
        metrics["baseline_sentiment"] = base_sentiment
        metrics["steered_sentiment"] = steer_sentiment
        metrics["sentiment_delta"] = round(steer_sentiment - base_sentiment, 4)

        return metrics

    # ══════════════════════════════════════════════════════════
    # ║  Perplexity                                            ║
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _compute_perplexity(model: Any, tokenizer: Any, text: str) -> float:
        """Compute model's own perplexity on a text."""
        try:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            return round(math.exp(min(loss, 20)), 2)  # Cap at e^20
        except Exception:
            return 0.0

    # ══════════════════════════════════════════════════════════
    # ║  Concept anchors for alignment scoring                 ║
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _get_concept_anchors(concept: str) -> List[str]:
        """Return short anchor sentences for a steering concept."""
        anchors = {
            "politeness": [
                "Thank you for your question, I'd be happy to help.",
                "I appreciate your patience. Let me explain clearly.",
                "That's a great observation. Here's what I think.",
                "I understand your concern and want to address it thoughtfully.",
            ],
            "toxicity": [
                "I won't engage with harmful language or hate speech.",
                "Let me provide a safe and constructive response.",
                "I'll address this respectfully while being honest.",
                "I want to ensure my response is helpful and non-harmful.",
            ],
            "creativity": [
                "Imagine a world where colors could sing and melodies could paint.",
                "The idea danced at the edge of possibility like twilight.",
                "Let me weave you a tapestry of unexpected connections.",
                "Picture this: a garden of thoughts blooming in parallel.",
            ],
            "refusal": [
                "I can't help with that request due to safety concerns.",
                "I'd prefer to redirect our conversation to safer topics.",
                "I'm not able to provide that information.",
                "Let me suggest an alternative approach instead.",
            ],
            "verbosity": [
                "To provide a thorough answer, let me cover several aspects.",
                "There are multiple dimensions to consider in this question.",
                "Let me elaborate extensively on each point for clarity.",
                "A comprehensive analysis requires examining the details.",
            ],
        }
        return anchors.get(concept, [])

    # ══════════════════════════════════════════════════════════
    # ║  Sentiment heuristic                                   ║
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _simple_sentiment(text: str) -> float:
        """Return sentiment polarity in [-1, 1] using TextBlob or fallback."""
        try:
            from textblob import TextBlob
            return round(TextBlob(text).sentiment.polarity, 4)
        except Exception:
            return 0.0

    # ══════════════════════════════════════════════════════════
    # ║  Aggregation                                           ║
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across all comparisons + compute consistency."""
        if not all_metrics:
            return {}

        keys = all_metrics[0].keys()
        agg: Dict[str, float] = {}

        for key in keys:
            values = [m.get(key, 0.0) for m in all_metrics]
            agg[f"avg_{key}"] = round(sum(values) / len(values), 4)

            # Standard deviation for consistency metrics
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                agg[f"std_{key}"] = round(variance ** 0.5, 4)

        # Behavioral consistency = 1 - std_dev(concept_alignment) if present
        if "std_steered_concept_alignment" in agg:
            std = agg["std_steered_concept_alignment"]
            agg["behavioral_consistency"] = round(max(0, 1.0 - std * 5), 4)

        return agg

    # ══════════════════════════════════════════════════════════
    # ║  Overall score (0-100)                                 ║
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _compute_overall_score(agg: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute a weighted overall steering effectiveness score (0-100).

        Components:
        - Semantic impact (30%): How much the meaning changed
        - Fluency preservation (25%): Perplexity didn't explode
        - Concept alignment (20%): Steered toward target concept
        - Consistency (15%): Stable behavior across prompts
        - Efficiency (10%): Impact per unit of strength
        """
        scores: Dict[str, float] = {}

        # Semantic impact: shift of 0.3+ is strong, 0.0 is nothing
        shift = agg.get("avg_semantic_shift", 0)
        scores["semantic_impact"] = round(min(shift / 0.3, 1.0) * 100, 1)

        # Fluency preservation: perplexity ratio near 1.0 is ideal
        ratio = agg.get("avg_perplexity_ratio", 1.0)
        if ratio == 0:
            ratio = 1.0
        # Penalize if perplexity increased a lot (ratio >> 1)
        fluency = max(0, 1.0 - abs(ratio - 1.0) * 0.5)
        scores["fluency_preservation"] = round(fluency * 100, 1)

        # Concept alignment delta: positive means better alignment
        align_delta = agg.get("avg_concept_alignment_delta", 0)
        scores["concept_alignment"] = round(
            min(max(align_delta + 0.5, 0) / 0.5, 1.0) * 100, 1
        )

        # Behavioral consistency
        consistency = agg.get("behavioral_consistency", 0.5)
        scores["behavioral_consistency"] = round(consistency * 100, 1)

        # Steering efficiency: higher = better (normalize to 0-100)
        eff = agg.get("avg_steering_efficiency", 0)
        scores["steering_efficiency"] = round(min(eff / 0.1, 1.0) * 100, 1)

        # Weighted composite
        weights = {
            "semantic_impact": 0.30,
            "fluency_preservation": 0.25,
            "concept_alignment": 0.20,
            "behavioral_consistency": 0.15,
            "steering_efficiency": 0.10,
        }

        total = sum(scores.get(k, 0) * w for k, w in weights.items())

        return {
            "score": round(total, 1),
            "grade": (
                "A+" if total >= 90 else
                "A" if total >= 80 else
                "B+" if total >= 70 else
                "B" if total >= 60 else
                "C" if total >= 50 else
                "D" if total >= 40 else "F"
            ),
            "breakdown": scores,
            "weights": weights,
        }
