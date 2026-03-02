"""
Layer categorization logic — assigns functional categories to
transformer layers based on extracted features plus clustering.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans

from app.core.constants import (
    CATEGORIES, CATEGORY_CITATIONS, BEHAVIORAL_ROLES,
    position_to_category,
)


def compute_layer_similarities(
    features: List[Dict[str, float]],
) -> List[float]:
    """
    Compute pairwise similarity between adjacent layers.

    Large drops indicate functional boundaries between layer groups.
    """
    if len(features) < 2:
        return [0.0]

    feature_keys = [
        "svd_rank", "svd_entropy", "attn_entropy",
        "ffn_norm_mean", "weight_std_mean", "sparsity",
    ]

    def to_vec(f: Dict[str, float]) -> np.ndarray:
        return np.array([f.get(k, 0.0) for k in feature_keys])

    similarities: List[float] = [0.0]  # first layer has no predecessor
    for i in range(1, len(features)):
        v1, v2 = to_vec(features[i - 1]), to_vec(features[i])
        norm1 = np.linalg.norm(v1) + 1e-10
        norm2 = np.linalg.norm(v2) + 1e-10
        cosine_sim = float(np.dot(v1, v2) / (norm1 * norm2))
        similarities.append(cosine_sim)

    return similarities



def position_based_clusters(n_layers: int) -> np.ndarray:
    """Fallback clustering based purely on position (10 zones)."""
    boundaries = [0.05, 0.12, 0.25, 0.40, 0.55, 0.70, 0.78, 0.88, 0.95]
    labels = np.zeros(n_layers, dtype=int)
    for i in range(n_layers):
        pos = i / max(n_layers - 1, 1)
        label = len(boundaries)  # default = last category
        for b_idx, b in enumerate(boundaries):
            if pos < b:
                label = b_idx
                break
        labels[i] = label
    return labels


def generate_layer_description(
    layer_idx: int,
    category: str,
    stats: Dict[str, float],
    relative_pos: float,
) -> str:
    """
    Generate a dynamic, stats-driven description for a layer.
    Goes beyond the generic BEHAVIORAL_ROLES text by interpreting
    the actual weight statistics of this specific layer.
    """
    parts = [f"Layer {layer_idx} ({category.replace('_', ' ').title()})"]

    # SVD rank interpretation
    svd_rank = stats.get("svd_rank", 0.0)
    if svd_rank > 0.7:
        parts.append(
            f"High expressive rank ({svd_rank:.2f}) — this layer "
            "uses diverse feature combinations"
        )
    elif svd_rank > 0.3:
        parts.append(
            f"Moderate rank ({svd_rank:.2f}) — balanced feature usage"
        )
    elif svd_rank > 0:
        parts.append(
            f"Low rank ({svd_rank:.2f}) — highly specialized, "
            "compresses information into fewer dimensions"
        )

    # Attention entropy
    attn_ent = stats.get("attn_entropy", 0.0)
    if attn_ent > 0.7:
        parts.append(
            f"Broad attention (entropy={attn_ent:.2f}) — attends "
            "to many positions, suggests global context aggregation"
        )
    elif attn_ent > 0.3:
        parts.append(
            f"Focused attention (entropy={attn_ent:.2f}) — "
            "selectively attends to relevant positions"
        )
    elif attn_ent > 0:
        parts.append(
            f"Sharp attention (entropy={attn_ent:.2f}) — highly "
            "local patterns, likely syntactic or positional"
        )

    # FFN norm
    ffn_norm = stats.get("ffn_norm_mean", 0.0)
    if ffn_norm > 20:
        parts.append(
            f"Strong FF transformation (norm={ffn_norm:.1f}) — "
            "significant feature manipulation"
        )
    elif ffn_norm > 5:
        parts.append(
            f"Moderate FF activity (norm={ffn_norm:.1f})"
        )

    # Sparsity
    sparsity = stats.get("sparsity", 0.0)
    if sparsity > 0.3:
        parts.append(
            f"High sparsity ({sparsity:.0%}) — selective activation, "
            "specialized function"
        )
    elif sparsity > 0.1:
        parts.append(f"Moderate sparsity ({sparsity:.0%})")

    return ". ".join(parts) + "."


def categorise_layers(
    features: List[Dict[str, float]],
    similarities: List[float],
    n_layers: int,
) -> List[Dict[str, Any]]:
    """
    Assign each layer a functional category using a combination of:
    1. Relative position in the model
    2. Extracted weight features (SVD rank, attention entropy, FFN norms)
    3. Inter-layer similarity boundaries
    4. K-Means clustering on feature vectors

    Categories are adaptive: models with <20 layers get merged zones,
    larger models get all 10 categories.
    """
    profiles: List[Dict[str, Any]] = []

    # Build feature matrix for clustering
    feature_keys = [
        "svd_rank", "svd_entropy", "svd_decay", "attn_entropy",
        "ffn_norm_mean", "ffn_norm_max", "weight_std_mean", "sparsity",
    ]
    feature_matrix = np.array(
        [[f.get(k, 0.0) for k in feature_keys] for f in features]
    )

    col_std = feature_matrix.std(axis=0)
    col_std[col_std < 1e-10] = 1.0
    col_mean = feature_matrix.mean(axis=0)
    feature_matrix_norm = (feature_matrix - col_mean) / col_std

    # Adaptive cluster count based on model depth
    n_clusters = min(len(CATEGORIES), n_layers)
    if n_layers >= len(CATEGORIES):
        try:
            kmeans = KMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            )
            cluster_labels = kmeans.fit_predict(feature_matrix_norm)
        except Exception:
            cluster_labels = position_based_clusters(n_layers)
    else:
        cluster_labels = position_based_clusters(n_layers)

    # Map cluster IDs to categories using average position
    cluster_positions: Dict[int, list] = {}
    for idx, cl in enumerate(cluster_labels):
        cluster_positions.setdefault(cl, [])
        cluster_positions[cl].append(idx / max(n_layers - 1, 1))

    cluster_avg_pos = {
        cl: np.mean(positions)
        for cl, positions in cluster_positions.items()
    }

    # Sort clusters by average position → assign categories in order
    sorted_clusters = sorted(
        cluster_avg_pos.keys(), key=lambda c: cluster_avg_pos[c]
    )
    cluster_to_category: Dict[int, str] = {}
    for rank, cl in enumerate(sorted_clusters):
        cat_idx = int(rank * len(CATEGORIES) / len(sorted_clusters))
        cat_idx = min(cat_idx, len(CATEGORIES) - 1)
        cluster_to_category[cl] = CATEGORIES[cat_idx]

    # --- Build final profiles---
    for idx in range(n_layers):
        # Category from clustering (primary)
        cluster = int(cluster_labels[idx])
        category = cluster_to_category.get(
            cluster, "knowledge_retrieval"
        )

        # Refine with position heuristic (secondary validation)
        pos = idx / max(n_layers - 1, 1)
        pos_category = position_to_category(pos)

        # Confidence: multi-signal scoring using all extracted features
        # Base confidence from feature quality (how much info we extracted)
        svd_rank = features[idx].get("svd_rank", 0.0)
        svd_entropy = features[idx].get("svd_entropy", 0.0)
        attn_entropy_val = features[idx].get("attn_entropy", 0.0)
        ffn_norm = features[idx].get("ffn_norm_mean", 0.0)
        total_params = features[idx].get("total_params", 0.0)

        # Feature quality score: how much useful signal was extracted
        # Each contributes up to 0.15, total up to 0.60
        feature_quality = 0.0
        if svd_rank > 0:
            feature_quality += min(svd_rank, 1.0) * 0.15
        if svd_entropy > 0:
            feature_quality += min(svd_entropy, 1.0) * 0.15
        if attn_entropy_val > 0:
            feature_quality += min(attn_entropy_val, 1.0) * 0.15
        if ffn_norm > 0:
            feature_quality += min(ffn_norm / 50.0, 1.0) * 0.15

        # Position certainty: layers near boundaries get slight penalty
        pos_certainty = 1.0 - 0.3 * abs(pos - 0.5)  # range ~0.85 to 1.0

        if category == pos_category:
            # Cluster & position agree → high confidence
            confidence = 0.80 + 0.10 * pos_certainty + feature_quality * 0.15
        else:
            # Disagree → moderate confidence, boosted by feature quality
            confidence = 0.60 + feature_quality * 0.30 + 0.05 * pos_certainty

        # Bonus for layers with strong extracted params (not position-only fallback)
        if total_params > 0:
            confidence += 0.03

        confidence = round(min(max(confidence, 0.40), 0.95), 3)

        # Weight stats for storage
        weight_stats = {
            k: round(features[idx].get(k, 0.0), 6) for k in feature_keys
        }
        weight_stats["similarity_to_prev"] = round(similarities[idx], 4)

        # Dynamic description from weight stats
        description = generate_layer_description(
            idx, category, weight_stats, pos
        )

        profiles.append({
            "layer_index": idx,
            "category": category,
            "confidence": confidence,
            "behavioral_role": BEHAVIORAL_ROLES.get(category, ""),
            "citation": CATEGORY_CITATIONS.get(category, ""),
            "weight_stats": weight_stats,
            "description": description,
        })

    return profiles
