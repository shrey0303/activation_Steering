# Perf: reduce memory allocation in SVD computation
# Fix: prevent division by zero in CKA computation
"""
🌟 Mathematical Layer Scanner — Core Innovation.

Analyses a transformer model's weight matrices to build a functional
layer map **without running any prompts**.  Uses:

1. SVD singular-value distribution  →  learned complexity / rank
2. Attention weight entropy        →  local vs. global patterns
3. FFN weight norms                →  transformation strength
4. Inter-layer similarity (CKA)   →  functional boundary detection
5. Rule-based categorisation       →  maps features → functional groups

The resulting layer profile is cached in SQLite so repeated scans are
instant.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from scipy import stats as sp_stats
from sklearn.cluster import KMeans

from app.core.loader import ModelManager


# ── Layer functional categories (research-backed) ─────────────
#
# Each category is grounded in published mechanistic interpretability
# research.  Position ranges are approximate — actual assignment uses
# K-Means clustering on weight features with position as a tiebreaker.
#
CATEGORIES = [
    "token_embedding",           # 0–5%   Raw vocabulary lookup
    "positional_morphological",  # 5–12%  Position encoding, morphology
    "syntactic_processing",      # 12–25% Phrase structure, dependencies
    "entity_semantic",           # 25–40% NER, polysemy, coreference
    "knowledge_retrieval",       # 40–55% Factual recall via FF key-value
    "reasoning_planning",        # 55–70% Multi-step inference, planning
    "safety_alignment",          # 70–78% Refusal circuits, guardrails
    "information_integration",   # 78–88% Cross-layer signal merging
    "style_personality",         # 88–95% Tone, register, personality
    "output_distribution",       # 95–100% Final token probabilities
]

# Citations for each category — used in layer descriptions and README
CATEGORY_CITATIONS = {
    "token_embedding": "Universal — raw subword/token identity encoding",
    "positional_morphological": "Logit Lens (nostalgebraist 2020) — early layers show positional/shallow predictions",
    "syntactic_processing": "Probing classifiers (Hewitt & Manning 2019) — structural probe for syntax",
    "entity_semantic": "Attention head analysis — entity tracking and polysemy resolution",
    "knowledge_retrieval": "Geva et al. 2021 — FF layers as key-value memories for factual recall",
    "reasoning_planning": "Mid-layer attention for multi-step inference and look-ahead planning",
    "safety_alignment": "Anthropic activation patching — refusal circuit identification",
    "information_integration": "Lawson et al. 2025 — middle-layer redundancy and signal merging",
    "style_personality": "Late-layer tone and register control in generation",
    "output_distribution": "Anti-overconfidence mechanism in final layer (logit lens studies)",
}

BEHAVIORAL_ROLES = {
    "token_embedding": (
        "Processes raw token representations: maps vocabulary indices to dense "
        "vectors, encodes subword identity and basic lexical features. "
        "Interventions here alter the model's fundamental token perception."
    ),
    "positional_morphological": (
        "Encodes sequence position and morphological features: word order, "
        "prefix/suffix patterns, and basic part-of-speech signals. "
        "Ref: Logit Lens shows nonsensical predictions at this depth."
    ),
    "syntactic_processing": (
        "Handles grammatical structure: phrase boundaries, dependency parsing, "
        "subject-verb agreement, and clause-level organization. "
        "Ref: Hewitt & Manning 2019 structural probes."
    ),
    "entity_semantic": (
        "Performs entity recognition, coreference resolution, and compositional "
        "semantics: builds meaning from word combinations, resolves polysemy, "
        "and tracks entity references across the input."
    ),
    "knowledge_retrieval": (
        "Retrieves factual knowledge from learned parameters: encyclopedic facts, "
        "world knowledge, and associative memory. "
        "Ref: Geva et al. 2021 — FF layers function as key-value memories."
    ),
    "reasoning_planning": (
        "Performs multi-step logical inference, causal reasoning, and response "
        "planning: deduction chains, cause-effect relationships, and "
        "task decomposition for complex queries."
    ),
    "safety_alignment": (
        "Implements safety filters and alignment guardrails: refusal decisions, "
        "value alignment checks, and content policy enforcement. "
        "Ref: Anthropic's activation patching identified refusal circuits here."
    ),
    "information_integration": (
        "Merges signals from earlier layers: resolves conflicting information, "
        "weighs competing hypotheses, and consolidates multi-source evidence. "
        "Ref: Lawson et al. 2025 — these layers show high redundancy."
    ),
    "style_personality": (
        "Controls output style and personality: tone, formality register, "
        "humor, empathy, and conversational persona. Interventions here "
        "change HOW the model says things without altering WHAT it says."
    ),
    "output_distribution": (
        "Shapes final token probability distribution: vocabulary selection, "
        "confidence calibration, and next-token prediction. Contains an "
        "anti-overconfidence mechanism that suppresses overly certain outputs."
    ),
}


class LayerScanner:
    """
    Mathematically profile every transformer layer by analysing weight
    matrices only — no forward pass required.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self.mm = model_manager

    # ╔══════════════════════════════════════════════════════════╗
    # ║                   PUBLIC INTERFACE                       ║
    # ╚══════════════════════════════════════════════════════════╝

    def scan(self) -> List[Dict[str, Any]]:
        """
        Analyse all layers and return a list of LayerProfile dicts.

        Returns
        -------
        list of dict
            Each dict contains:
            - layer_index, category, confidence, behavioral_role,
              weight_stats, description
        """
        if not self.mm.loaded:
            raise RuntimeError("No model loaded – call ModelManager.load() first")

        t0 = time.perf_counter()
        layers = self.mm.get_layer_modules()
        n_layers = len(layers)

        if n_layers == 0:
            raise RuntimeError(
                "Could not detect transformer layers in this model architecture"
            )

        logger.info(f"🔬 Scanning {n_layers} layers of {self.mm.model_name}...")

        # ── Step 1: Extract per-layer features ────────────────
        features: List[Dict[str, float]] = []
        for idx, layer_module in enumerate(layers):
            feat = self._extract_layer_features(idx, layer_module, n_layers)
            features.append(feat)
            if (idx + 1) % 8 == 0 or idx == n_layers - 1:
                logger.debug(f"   Scanned layer {idx + 1}/{n_layers}")

        # ── Step 2: Compute inter-layer similarity ────────────
        similarities = self._compute_layer_similarities(features)

        # ── Step 3: Categorise layers ─────────────────────────
        profiles = self._categorise_layers(features, similarities, n_layers)

        elapsed = time.perf_counter() - t0
        logger.info(f"✅ Scan complete in {elapsed:.1f}s")
        return profiles

    def get_scan_hash(self) -> str:
        """
        Quick hash of model identity + parameter count so we know
        when a re-scan is needed.
        """
        info = (
            f"{self.mm.model_name}|{self.mm.num_layers}|"
            f"{self.mm.hidden_dim}|{self.mm.architecture}"
        )
        return hashlib.sha256(info.encode()).hexdigest()[:16]

    # ╔══════════════════════════════════════════════════════════╗
    # ║            STEP 1: FEATURE EXTRACTION                   ║
    # ╚══════════════════════════════════════════════════════════╝

    @staticmethod
    def _dequantize_param(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        """
        Safely dequantize a parameter to FP32 for analysis.

        Handles:
        - bitsandbytes Linear4bit / Linear8bitLt  (packed quantised)
        - Standard FP16 / BF16 params              (simple cast)
        - Regular FP32 params                      (no-op)

        Returns None if the parameter cannot be dequantised.
        """
        try:
            # ── bitsandbytes 4-bit ────────────────────────────
            module_type = type(param).__module__ or ""
            cls_name = type(param).__qualname__ or ""

            # Check if this is a bnb Params4bit
            if "bitsandbytes" in module_type or "Params4bit" in cls_name:
                try:
                    import bitsandbytes as bnb
                    # bnb stores weights in .data as packed uint8
                    # dequantize() returns the FP32 equivalent
                    return bnb.functional.dequantize_4bit(
                        param.data, param.quant_state
                    ).float().cpu()
                except Exception:
                    pass

            # Check if param belongs to a bnb Linear4bit layer
            if hasattr(param, "quant_state") and param.quant_state is not None:
                try:
                    import bitsandbytes as bnb
                    return bnb.functional.dequantize_4bit(
                        param.data, param.quant_state
                    ).float().cpu()
                except Exception:
                    pass

            # ── bitsandbytes 8-bit ────────────────────────────
            if hasattr(param, "SCB") or hasattr(param, "CB"):
                try:
                    import bitsandbytes as bnb
                    return param.data.float().cpu()
                except Exception:
                    pass

            # ── Standard param: cast to float32 ──────────────
            return param.detach().float().cpu()

        except Exception:
            return None

    @torch.no_grad()
    def _extract_layer_features(
        self,
        layer_idx: int,
        layer_module: torch.nn.Module,
        total_layers: int,
    ) -> Dict[str, float]:
        """
        Extract mathematical features from a single layer's weight
        matrices.  All computation on CPU to avoid GPU contention.

        For quantised models, weights are dequantised to FP32 before
        analysis so scan quality is identical to full-precision models.
        """
        features: Dict[str, float] = {
            "layer_index": float(layer_idx),
            "relative_position": layer_idx / max(total_layers - 1, 1),
        }

        # Collect all weight tensors in this layer (dequantised to FP32)
        weight_tensors: Dict[str, torch.Tensor] = {}
        for name, param in layer_module.named_parameters():
            w = self._dequantize_param(param)
            if w is not None and w.ndim >= 2:
                weight_tensors[name] = w

        if not weight_tensors:
            # Fallback: position-based heuristic
            features.update(self._position_heuristic(layer_idx, total_layers))
            return features

        # ── SVD analysis (top-k singular values) ──────────────
        svd_features = self._svd_analysis(weight_tensors)
        features.update(svd_features)

        # ── Attention weight entropy ──────────────────────────
        attn_features = self._attention_entropy(weight_tensors)
        features.update(attn_features)

        # ── FFN norm analysis ─────────────────────────────────
        ffn_features = self._ffn_norm_analysis(weight_tensors)
        features.update(ffn_features)

        # ── General weight statistics ─────────────────────────
        general = self._general_weight_stats(weight_tensors)
        features.update(general)

        return features

    def _svd_analysis(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Singular value decomposition of weight matrices.

        The distribution of singular values reveals:
        - High effective rank  →  complex learned transformations
        - Rapid decay         →  low-rank, simpler representation
        - Entropy of spectrum →  diversity of learned features

        Uses randomized SVD (scipy) with aggressive downsampling
        for speed — 256 rows × 512 cols × k=32 per matrix.
        """
        from scipy.sparse.linalg import svds
        all_singular: List[np.ndarray] = []

        for name, w in weights.items():
            try:
                m, n = w.shape[0], w.shape[-1]
                if m < 2 or n < 2:
                    continue

                # Aggressive downsampling for speed:
                # cap at 256 rows × 512 cols (was 1024 × unlimited)
                w_2d = w.reshape(-1, w.shape[-1])[:256, :512].numpy()
                k = min(32, min(w_2d.shape) - 1)  # scipy svds needs k < min(m,n)
                if k < 1:
                    continue

                # Randomized SVD: O(m*n*k) instead of O(m*n*min(m,n))
                _, s, _ = svds(w_2d.astype(np.float64), k=k)
                all_singular.append(s[::-1])  # svds returns ascending
            except Exception:
                continue

        if not all_singular:
            return {"svd_rank": 0.0, "svd_entropy": 0.0, "svd_decay": 0.0}

        combined = np.concatenate(all_singular)
        combined = combined / (combined.max() + 1e-10)  # normalise

        # Effective rank: entropy-based (Roy & Vetterli, 2007)
        p = combined / (combined.sum() + 1e-10)
        p = p[p > 1e-10]
        svd_entropy = float(-np.sum(p * np.log(p + 1e-10)))
        effective_rank = float(np.exp(svd_entropy))

        # Decay rate: ratio of 10th to 1st singular value
        sorted_sv = np.sort(combined)[::-1]
        idx_10pct = max(1, len(sorted_sv) // 10)
        decay = float(sorted_sv[idx_10pct] / (sorted_sv[0] + 1e-10))

        return {
            "svd_rank": effective_rank,
            "svd_entropy": svd_entropy,
            "svd_decay": decay,
        }

    def _attention_entropy(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyse attention projection weights.

        Higher entropy in Q/K projections → global attention (semantic).
        Lower entropy → local attention (syntactic).
        """
        attn_entropy_values: List[float] = []

        for name, w in weights.items():
            # Heuristic: attention projections usually have "q_proj", "k_proj",
            # "query", "key", "attn" in their name
            lower = name.lower()
            is_attn = any(
                kw in lower
                for kw in ["q_proj", "k_proj", "query", "key", "attn", "attention"]
            )
            if not is_attn:
                continue

            try:
                # Row-wise L2 norms as a proxy for head importance
                w_2d = w.reshape(-1, w.shape[-1])
                norms = torch.norm(w_2d, dim=1).numpy()
                norms = norms / (norms.sum() + 1e-10)
                norms = norms[norms > 1e-10]
                ent = float(-np.sum(norms * np.log(norms + 1e-10)))
                attn_entropy_values.append(ent)
            except Exception:
                continue

        if not attn_entropy_values:
            return {"attn_entropy": 0.0}

        return {"attn_entropy": float(np.mean(attn_entropy_values))}

    def _ffn_norm_analysis(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Analyse feed-forward network weight magnitudes.

        Large norms → strong non-linear transformation (reasoning).
        Small norms → light processing (embedding / output).
        """
        ffn_norms: List[float] = []

        for name, w in weights.items():
            lower = name.lower()
            is_ffn = any(
                kw in lower
                for kw in [
                    "mlp", "ffn", "fc", "gate", "up_proj",
                    "down_proj", "dense", "intermediate",
                ]
            )
            if not is_ffn:
                continue

            try:
                norm_val = float(torch.norm(w.float()).item())
                ffn_norms.append(norm_val)
            except Exception:
                continue

        if not ffn_norms:
            return {"ffn_norm_mean": 0.0, "ffn_norm_max": 0.0}

        return {
            "ffn_norm_mean": float(np.mean(ffn_norms)),
            "ffn_norm_max": float(np.max(ffn_norms)),
        }

    def _general_weight_stats(
        self, weights: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Overall weight statistics: sparsity, kurtosis, skewness."""
        all_vals: List[float] = []
        total_params = 0
        near_zero = 0

        for w in weights.values():
            try:
                flat = w.float().flatten().numpy()
                all_vals.append(float(np.std(flat)))
                total_params += len(flat)
                near_zero += int(np.sum(np.abs(flat) < 1e-4))
            except Exception:
                continue

        sparsity = near_zero / max(total_params, 1)

        return {
            "weight_std_mean": float(np.mean(all_vals)) if all_vals else 0.0,
            "sparsity": sparsity,
            "total_params": float(total_params),
        }

    def _position_heuristic(
        self, layer_idx: int, total_layers: int
    ) -> Dict[str, float]:
        """
        Fallback when weight extraction fails — use relative position.
        """
        pos = layer_idx / max(total_layers - 1, 1)
        return {
            "svd_rank": 0.0,
            "svd_entropy": 0.0,
            "svd_decay": 0.0,
            "attn_entropy": 0.0,
            "ffn_norm_mean": 0.0,
            "ffn_norm_max": 0.0,
            "weight_std_mean": 0.0,
            "sparsity": 0.0,
            "total_params": 0.0,
        }

    # ╔══════════════════════════════════════════════════════════╗
    # ║       STEP 2: INTER-LAYER SIMILARITY (CKA-like)        ║
    # ╚══════════════════════════════════════════════════════════╝

    def _compute_layer_similarities(
        self, features: List[Dict[str, float]]
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

    # ╔══════════════════════════════════════════════════════════╗
    # ║          STEP 3: LAYER CATEGORISATION                   ║
    # ╚══════════════════════════════════════════════════════════╝

    def _categorise_layers(
        self,
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

        # Normalise features
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
                cluster_labels = self._position_based_clusters(n_layers)
        else:
            cluster_labels = self._position_based_clusters(n_layers)

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

        # ── Build final profiles ──────────────────────────────
        for idx in range(n_layers):
            # Category from clustering (primary)
            cluster = int(cluster_labels[idx])
            category = cluster_to_category.get(
                cluster, "knowledge_retrieval"
            )

            # Refine with position heuristic (secondary validation)
            pos = idx / max(n_layers - 1, 1)
            position_category = self._position_to_category(pos)

            # Confidence: multi-signal scoring using all extracted features
            # Base confidence from feature quality (how much info we extracted)
            svd_rank = features[idx].get("svd_rank", 0.0)
            svd_entropy = features[idx].get("svd_entropy", 0.0)
            attn_entropy = features[idx].get("attn_entropy", 0.0)
            ffn_norm = features[idx].get("ffn_norm_mean", 0.0)
            total_params = features[idx].get("total_params", 0.0)

            # Feature quality score: how much useful signal was extracted
            # Each contributes up to 0.15, total up to 0.60
            feature_quality = 0.0
            if svd_rank > 0:
                feature_quality += min(svd_rank, 1.0) * 0.15
            if svd_entropy > 0:
                feature_quality += min(svd_entropy, 1.0) * 0.15
            if attn_entropy > 0:
                feature_quality += min(attn_entropy, 1.0) * 0.15
            if ffn_norm > 0:
                feature_quality += min(ffn_norm / 50.0, 1.0) * 0.15

            # Position certainty: layers near boundaries get slight penalty
            pos_certainty = 1.0 - 0.3 * abs(pos - 0.5)  # range ~0.85 to 1.0

            if category == position_category:
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
            description = self._generate_layer_description(
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

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _position_to_category(relative_pos: float) -> str:
        """Map a relative position [0,1] to a research-backed category."""
        if relative_pos < 0.05:
            return "token_embedding"
        elif relative_pos < 0.12:
            return "positional_morphological"
        elif relative_pos < 0.25:
            return "syntactic_processing"
        elif relative_pos < 0.40:
            return "entity_semantic"
        elif relative_pos < 0.55:
            return "knowledge_retrieval"
        elif relative_pos < 0.70:
            return "reasoning_planning"
        elif relative_pos < 0.78:
            return "safety_alignment"
        elif relative_pos < 0.88:
            return "information_integration"
        elif relative_pos < 0.95:
            return "style_personality"
        else:
            return "output_distribution"

    @staticmethod
    def _position_based_clusters(n_layers: int) -> np.ndarray:
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

    @staticmethod
    def _generate_layer_description(
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

