"""
Evaluation metrics — computation, aggregation, and scoring.

All metric functions are stateless and operate on raw text inputs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from app.core.evaluator.anchors import get_concept_anchors


def compute_metrics(
    model: Any,
    tokenizer: Any,
    prompt: str,
    baseline: str,
    steered: str,
    embed_model: Any,
    has_embedder: bool,
    total_strength: float,
    target_concept: Optional[str],
) -> Dict[str, float]:
    """Compute all production-grade metrics for a single prompt comparison."""
    metrics: Dict[str, float] = {}

    base_words = baseline.split()
    steer_words = steered.split()
    metrics["baseline_length"] = len(base_words)
    metrics["steered_length"] = len(steer_words)
    metrics["length_delta"] = len(steer_words) - len(base_words)
    metrics["length_ratio"] = len(steer_words) / max(len(base_words), 1)

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
        matched = sum(
            1 for a, b in zip(base_tokens, steer_tokens) if a == b
        )
        metrics["token_match_ratio"] = round(matched / max_len, 4)
        metrics["token_divergence"] = round(1.0 - matched / max_len, 4)
    else:
        metrics["token_match_ratio"] = 1.0
        metrics["token_divergence"] = 0.0

    if has_embedder and embed_model:
        embeddings = embed_model.encode(
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

    # --- Perplexity Delta---
    base_ppl = compute_perplexity(model, tokenizer, prompt + " " + baseline)
    steer_ppl = compute_perplexity(model, tokenizer, prompt + " " + steered)
    metrics["baseline_perplexity"] = base_ppl
    metrics["steered_perplexity"] = steer_ppl
    metrics["perplexity_delta"] = round(steer_ppl - base_ppl, 2)
    # Perplexity ratio: < 1 means steered is more fluent
    metrics["perplexity_ratio"] = round(
        steer_ppl / max(base_ppl, 1e-6), 4
    )

    if has_embedder and embed_model and target_concept:
        concept_anchors = get_concept_anchors(target_concept)
        if concept_anchors:
            anchor_emb = embed_model.encode(
                concept_anchors, convert_to_tensor=True
            )
            anchor_mean = anchor_emb.mean(dim=0, keepdim=True)

            base_emb = embed_model.encode(
                [baseline], convert_to_tensor=True
            )
            steer_emb = embed_model.encode(
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

            # Directional shift: does the (steered - baseline) delta
            # actually point toward the concept anchor centroid?
            base_emb_vec = embed_model.encode(
                [baseline], convert_to_tensor=True
            )
            steer_emb_vec = embed_model.encode(
                [steered], convert_to_tensor=True
            )
            delta_vec = steer_emb_vec - base_emb_vec
            delta_norm = delta_vec.norm()
            if delta_norm > 1e-6:
                directional = torch.nn.functional.cosine_similarity(
                    delta_vec, anchor_mean
                ).item()
                metrics["directional_shift"] = round(directional, 4)
            else:
                metrics["directional_shift"] = 0.0

    if total_strength > 0 and metrics.get("semantic_shift", 0) > 0:
        metrics["steering_efficiency"] = round(
            metrics["semantic_shift"] / total_strength, 4
        )
    else:
        metrics["steering_efficiency"] = 0.0

    # --- Sentiment / Tone shift---
    base_sentiment = simple_sentiment(baseline)
    steer_sentiment = simple_sentiment(steered)
    metrics["baseline_sentiment"] = base_sentiment
    metrics["steered_sentiment"] = steer_sentiment
    metrics["sentiment_delta"] = round(steer_sentiment - base_sentiment, 4)

    return metrics


def compute_perplexity(model: Any, tokenizer: Any, text: str) -> float:
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
    except Exception as e:
        logger.debug(f"Perplexity computation failed: {e}")
        return 0.0


def simple_sentiment(text: str) -> float:
    """Return sentiment polarity in [-1, 1] using TextBlob or fallback."""
    try:
        from textblob import TextBlob
        return round(TextBlob(text).sentiment.polarity, 4)
    except Exception:
        logger.debug("TextBlob not available for sentiment analysis")
        return 0.0


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
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
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            agg[f"std_{key}"] = round(variance ** 0.5, 4)

    # Behavioral consistency = 1 - std_dev(concept_alignment) if present
    if "std_steered_concept_alignment" in agg:
        std = agg["std_steered_concept_alignment"]
        agg["behavioral_consistency"] = round(max(0, 1.0 - std * 5), 4)

    return agg


def compute_overall_score(agg: Dict[str, float]) -> Dict[str, Any]:
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

    # Concept alignment delta: positive means better alignment, zero means no effect
    align_delta = agg.get("avg_concept_alignment_delta", 0)
    scores["concept_alignment"] = round(
        min(max(align_delta, 0) / 0.3, 1.0) * 100, 1
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
