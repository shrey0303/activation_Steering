"""
Intent Router — maps user text to PCA feature IDs via cosine similarity + NLI.

Two modes:
  - Browse: User picks features from the dictionary directly
  - Text:   "be angry" → cosine sim → top 3 matching features
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from app.core.interpreter.models import (
    FeatureMatch,
    IntentRouterResult,
    split_clauses,
)


class IntentRouter:
    """
    Map user text to PCA feature IDs using a two-stage pipeline.

    Stage 1 — Retrieval (Bi-Encoder):
        Sentence-BERT (all-MiniLM-L6-v2) embeds user text and feature labels
        independently, then ranks by cosine similarity.
        This finds WHICH features match the topic (e.g., "toxic" → toxicity).

    Stage 2 — Direction (NLI Cross-Encoder):
        For each top-K candidate, NLI cross-encoder scores user text against
        "Amplify {label}" to determine enhance vs suppress.
        This handles NEGATION natively (e.g., "less toxic" → suppress).

    Why both? NLI alone finds spurious cross-concept relationships — "more
    toxic" vs "Amplify anger" scores contradiction=0.998, beating the correct
    match. Bi-encoder correctly isolates topic matching; NLI correctly
    determines direction. Together they're accurate. Separately they aren't.

    Models:
        - all-MiniLM-L6-v2 (22M params, ~80MB) — retrieval
        - cross-encoder/nli-deberta-v3-small (22M params, ~80MB) — direction

    Usage:
        router = IntentRouter(feature_dict)
        result = router.route("make it less toxic")
        # Bi-encoder finds "toxicity" feature
        # NLI: "less toxic" vs "Amplify toxicity" → contradiction → suppress
    """

    def __init__(self, feature_dict=None) -> None:
        self.feature_dict = feature_dict
        self._embedder = None          # Bi-encoder for retrieval
        self._cross_encoder = None     # NLI cross-encoder for direction
        self._label_embeddings = None
        self._labeled_features = None

    def _ensure_embedder(self) -> None:
        """Lazy-load bi-encoder for topic retrieval."""
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("IntentRouter: loaded bi-encoder (all-MiniLM-L6-v2)")
        except ImportError:
            logger.warning("sentence-transformers not available")

    def _ensure_cross_encoder(self) -> None:
        """Lazy-load NLI cross-encoder for direction classification."""
        if self._cross_encoder is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(
                "cross-encoder/nli-deberta-v3-small"
            )
            logger.info(
                "IntentRouter: loaded NLI cross-encoder "
                "(nli-deberta-v3-small)"
            )
        except Exception as e:
            logger.warning(f"NLI cross-encoder not available: {e}")

    def _cache_label_embeddings(self) -> None:
        """Pre-compute bi-encoder embeddings for all labeled features."""
        if self.feature_dict is None:
            return
        if self._label_embeddings is not None:
            return

        self._ensure_embedder()
        if self._embedder is None:
            return

        self._labeled_features = self.feature_dict.get_labeled()
        if not self._labeled_features:
            self._labeled_features = self.feature_dict.all_features

        labels = [f.label for f in self._labeled_features]
        self._label_embeddings = self._embedder.encode(labels)
        logger.info(
            f"Cached {len(labels)} label embeddings for intent routing"
        )

    def route(self, text: str, top_k: int = 3) -> IntentRouterResult:
        """
        Route user intent to features via retrieve-then-classify.

        Stage 1: Bi-encoder finds top-K features by topic similarity.
        Stage 2: NLI cross-encoder classifies enhance vs suppress.

        Parameters
        ----------
        text : User intent, e.g., "make it less toxic"
        top_k : Number of top matching features to return

        Returns
        -------
        IntentRouterResult with matched features and NLI-based directions
        """
        if self.feature_dict is None:
            return IntentRouterResult(query=text, method="no_dictionary")

        self._cache_label_embeddings()
        if self._embedder is None or self._label_embeddings is None:
            return IntentRouterResult(query=text, method="no_embedder")

        # --- Stage 1: Bi-encoder retrieval ---
        text_emb = self._embedder.encode([text])[0]

        text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        labels_norm = self._label_embeddings / (
            np.linalg.norm(self._label_embeddings, axis=1, keepdims=True)
            + 1e-8
        )

        sims = labels_norm @ text_norm
        top_indices = np.argsort(sims)[::-1][:top_k]

        # --- Stage 2: NLI direction classification ---
        self._ensure_cross_encoder()

        matches = []
        for idx in top_indices:
            if sims[idx] < 0.05:
                continue
            feat = self._labeled_features[idx]


            direction = self._classify_direction(text, feat.label)

            matches.append(
                FeatureMatch(
                    feature_id=feat.feature_id,
                    label=feat.label,
                    layer_idx=feat.layer_idx,
                    similarity=float(sims[idx]),
                    direction=direction,
                    variance_explained=feat.variance_explained,
                )
            )

        confidence = matches[0].similarity if matches else 0.0

        return IntentRouterResult(
            matches=matches,
            query=text,
            method="nli_cross_encoder",
            confidence=confidence,
        )

    def _classify_direction(
        self, user_text: str, feature_label: str
    ) -> float:
        """
        Determine enhance vs suppress using dual-hypothesis NLI.

        Tests BOTH:
          - 'This text requests an increase in {label}' (enhance)
          - 'This text requests a decrease in {label}' (suppress)

        Whichever hypothesis scores higher entailment wins.
        Handles compound intents by splitting clauses first.
        """
        if self._cross_encoder is None:
            return self._keyword_fallback(user_text)

        enhance_h = f"This text requests an increase in {feature_label}."
        suppress_h = f"This text requests a decrease in {feature_label}."

        # Split compound text into clauses (handles 'angry and calm', 'toxic but friendly')
        clauses = split_clauses(user_text)
        if not clauses:
            clauses = [user_text]

        pairs = []
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            pairs.append((clause, enhance_h))
            pairs.append((clause, suppress_h))

        if not pairs:
            return 1.0  # default enhance

        raw_scores = self._cross_encoder.predict(pairs, apply_softmax=True)


        best_enh = -999.0
        best_sup = -999.0
        for i in range(0, len(raw_scores), 2):
            enh_entail = float(raw_scores[i][1])
            sup_entail = float(raw_scores[i + 1][1])
            if enh_entail > best_enh:
                best_enh = enh_entail
            if sup_entail > best_sup:
                best_sup = sup_entail

        direction = -1.0 if best_sup > best_enh else 1.0

        logger.debug(
            f"NLI direction for '{user_text}' | '{feature_label}': "
            f"enhance={best_enh:.3f}, suppress={best_sup:.3f} → "
            f"{'enhance' if direction > 0 else 'suppress'}"
        )
        return direction

    @staticmethod
    def _keyword_fallback(text: str) -> float:
        """Fallback direction inference when NLI is unavailable."""
        suppress_kw = [
            "less", "reduce", "suppress", "remove", "stop",
            "don't", "not", "no ", "without", "anti",
        ]
        text_lower = text.lower()
        for kw in suppress_kw:
            if kw in text_lower:
                return -1.0
        return 1.0

    def reload(self, feature_dict) -> None:
        """Hot-reload with a new feature dictionary (e.g. after model swap or feature extraction)."""
        self.feature_dict = feature_dict
        self._label_embeddings = None
        self._labeled_features = None
        logger.info("IntentRouter: reloaded feature dictionary")

    @classmethod
    def from_db(cls, model_name: str) -> "IntentRouter":
        """Create an IntentRouter by loading features from the features.db for a given model."""
        from app.core.feature_extractor import FeatureDictionary
        fd = FeatureDictionary.load(model_name)
        if not fd.all_features:
            logger.info(
                f"IntentRouter.from_db: no features found for '{model_name}' — "
                f"router will use static fallback in ResponseInterpreter"
            )
            return cls(feature_dict=None)
        logger.info(
            f"IntentRouter.from_db: loaded {len(fd.all_features)} features "
            f"for '{model_name}' ({len(fd.get_labeled())} labeled)"
        )
        return cls(feature_dict=fd)
