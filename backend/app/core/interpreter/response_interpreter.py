"""
Response Interpreter — two-stage bi-encoder + NLI direction classifier.

Legacy system kept for backward compatibility with the analyze endpoint.
Per-layer bidirectional matching using sentence-transformers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False

from app.core.interpreter.models import (
    CATEGORY_HYPOTHESES,
    CATEGORY_LABELS,
    LAYER_CATEGORIES,
    InterpretationResult,
)
from app.core.interpreter.intent_router import IntentRouter


class ResponseInterpreter:
    """
    Two-stage interpreter: bi-encoder retrieval + NLI direction.

    Stage 1 (Retrieval): Bi-encoder (all-MiniLM-L6-v2) compares user text
        against category LABEL embeddings. No hardcoded descriptions —
        just simple names like "safety and toxicity", "personality and style".

    Stage 2 (Direction): NLI cross-encoder (nli-deberta-v3-small) determines
        enhance vs suppress per matched category. The NLI model natively
        understands that "be more toxic" CONTRADICTS "Amplify safety"
        without needing the word "toxic" in a hardcoded description.

    Falls back to keyword-based direction if NLI model is unavailable.
    """

    def __init__(self) -> None:
        self._embedder = None
        self._nli_model = None
        self._category_embeddings: Dict[str, Any] = {}
        # Dynamic routing via FeatureDictionary (injected from routes.py after scan/extract)
        self._intent_router: Optional["IntentRouter"] = None

    def set_intent_router(self, router: "IntentRouter") -> None:
        """Inject a FeatureDictionary-backed IntentRouter for dynamic routing.
        Called from routes.py after model load + feature extraction.
        """
        self._intent_router = router
        labels = []
        if router.feature_dict is not None:
            labels = router.feature_dict.all_labels
        logger.info(
            f"ResponseInterpreter: dynamic router injected "
            f"({len(labels)} feature labels available)"
        )

    def set_feature_dict(self, feature_dict) -> None:
        """Convenience: wrap a FeatureDictionary in an IntentRouter and inject it."""
        router = IntentRouter(feature_dict=feature_dict)
        self.set_intent_router(router)

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Interpreter: loaded bi-encoder (all-MiniLM-L6-v2)")
            self._precache_category_embeddings()
        except ImportError:
            logger.warning("sentence-transformers not available")

    def _ensure_nli(self) -> None:
        if self._nli_model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
            logger.info("Interpreter: loaded NLI cross-encoder")
        except Exception as e:
            logger.warning(f"NLI model unavailable: {e}")

    def _precache_category_embeddings(self) -> None:
        """Embed category labels for retrieval."""
        if self._embedder is None:
            return
        labels = [CATEGORY_LABELS[cat] for cat in LAYER_CATEGORIES]
        embeddings = self._embedder.encode(labels)
        for cat, emb in zip(LAYER_CATEGORIES, embeddings):
            self._category_embeddings[cat] = emb

    def _classify_direction_nli(
        self, clause: str, category: str
    ) -> Tuple[float, float]:
        """
        Use NLI cross-encoder to determine direction for a category.

        Tests both enhance and suppress hypotheses, picks whichever
        the NLI model more strongly entails. This is more robust than
        a single 'Amplify X' hypothesis, especially for:
          - Ambiguous words (e.g. 'ethical' for 'safety_alignment')
          - Negation (e.g. 'not safe' should suppress)
          - Indirect phrasing

        Returns (direction, confidence):
          direction: +1.0 (enhance) or -1.0 (suppress)
          confidence: entailment score of winning hypothesis
        """
        if self._nli_model is None:
            return self._classify_direction_fallback(clause)

        hyps = CATEGORY_HYPOTHESES.get(category)
        if not hyps:
            return self._classify_direction_fallback(clause)

        enhance_h = hyps["enhance_h"]
        suppress_h = hyps["suppress_h"]


        scores = self._nli_model.predict([
            (clause, enhance_h),
            (clause, suppress_h),
        ])

        enh_entail = float(scores[0][1])
        sup_entail = float(scores[1][1])

        if enh_entail >= sup_entail:
            return +1.0, enh_entail
        else:
            return -1.0, sup_entail

    @staticmethod
    def _split_clauses(text: str) -> List[str]:
        """
        Split compound text into clauses by natural connectors.

        Handles: and, but, though, yet, while, even, however, although,
                 or, nor, comma-separated phrases, semicolons.
        """
        import re
        pattern = r'\s*(?:\band\b|\bbut\b|\bthough\b|\byet\b|\bwhile\b|\beven\b|\bhowever\b|\balthough\b|\bor\b|\bnor\b|[;,])\s*'
        clauses = re.split(pattern, text, flags=re.IGNORECASE)
        return [c.strip() for c in clauses if len(c.strip()) > 2]

    @staticmethod
    def _classify_direction_fallback(text: str) -> Tuple[float, float]:
        """Keyword fallback when NLI unavailable."""
        suppress_words = {"less", "reduce", "don't", "not", "no", "without",
                         "suppress", "remove", "decrease", "stop", "avoid"}
        words = set(text.lower().split())
        if words & suppress_words:
            return -1.0, 0.6
        return +1.0, 0.6


    def interpret(
        self,
        text: str,
        prompt: str = "",
    ) -> InterpretationResult:
        """
        Interpret user text into layer-level intent scores.

        Primary path (when FeatureDictionary is available):
          Uses IntentRouter to route to PCA feature IDs from features.db.
          Zero hardcoding — labels are whatever the DB contains.

        Fallback path (no feature extraction run yet):
          Three-stage static-category pipeline (LAYER_CATEGORIES + NLI).
          Provides full functionality out-of-the-box for new users.
        """
        # --- Sentiment---
        polarity = 0.0
        subjectivity = 0.0
        if _HAS_TEXTBLOB and text:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

       
        if (
            self._intent_router is not None
            and self._intent_router.feature_dict is not None
            and self._intent_router.feature_dict.get_labeled()
        ):
            self._intent_router._ensure_embedder()
            self._intent_router._ensure_cross_encoder()

            result = self._intent_router.route(text, top_k=5)

          
            intent_scores: Dict[str, float] = {}
            feature_matches = []
            for m in result.matches:
               
                intent_scores[m.feature_id] = m.direction * m.similarity
                feature_matches.append({
                    "feature_id": m.feature_id,
                    "label": m.label,
                    "layer_idx": m.layer_idx,
                    "direction": m.direction,
                    "similarity": m.similarity,
                })

            dominant = sorted(
                intent_scores.keys(),
                key=lambda k: abs(intent_scores[k]),
                reverse=True,
            )

            summary_parts = []
            for fid in dominant[:3]:
                sc = intent_scores[fid]
                match = next((m for m in result.matches if m.feature_id == fid), None)
                label = match.label if match else fid
                dir_label = "enhance" if sc > 0 else "suppress"
                summary_parts.append(f"{dir_label} {label} ({abs(sc):.2f})")

            return InterpretationResult(
                sentiment_polarity=polarity,
                sentiment_subjectivity=subjectivity,
                intent_scores=intent_scores,
                dominant_intents=dominant,
                summary="; ".join(summary_parts) if summary_parts else "No matches",
                method="intent_router_db",
                feature_matches=feature_matches,
                routed_by_features=True,
            )

        # --- FALLBACK PATH: Static category + NLI---
        # Used when no feature extraction has been run yet.
        self._ensure_embedder()
        self._ensure_nli()

        if self._embedder is None:
            return InterpretationResult(
                sentiment_polarity=polarity,
                sentiment_subjectivity=subjectivity,
                method="fallback",
            )

        # --- Stage 1: Split into clauses ---
        clauses = self._split_clauses(text)
        if not clauses:
            clauses = [text]

        # --- Stage 2: Per-clause category matching ---
        cat_scores: Dict[str, Tuple[float, float]] = {}

        for clause in clauses:
            clause_emb = self._embedder.encode([clause])[0]
            clause_norm = clause_emb / (np.linalg.norm(clause_emb) + 1e-8)

          
            clause_sims = {}
            all_clause_sims = []
            for cat in LAYER_CATEGORIES:
                cat_emb = self._category_embeddings[cat]
                cat_norm = cat_emb / (np.linalg.norm(cat_emb) + 1e-8)
                sim = float(np.dot(clause_norm, cat_norm))
                clause_sims[cat] = sim
                all_clause_sims.append(sim)

           
            sim_arr = np.array(all_clause_sims)
            threshold = float(sim_arr.mean() + 0.3 * sim_arr.std())

           
            for cat, sim in clause_sims.items():
                if sim <= threshold:
                    continue

             
                direction, nli_conf = self._classify_direction_nli(clause, cat)
                score = direction * sim

                
                if cat not in cat_scores or abs(score) > abs(cat_scores[cat][0]):
                    cat_scores[cat] = (score, sim)

        intent_scores = {cat: s[0] for cat, s in cat_scores.items()}
        layer_sims = {cat: s[1] for cat, s in cat_scores.items()}
        dominant = sorted(
            intent_scores.keys(),
            key=lambda c: abs(intent_scores[c]),
            reverse=True,
        )

        summary_parts = []
        for cat in dominant[:3]:
            score = intent_scores[cat]
            dir_label = "enhance" if score > 0 else "suppress"
            display = cat.replace("_", " ").title()
            summary_parts.append(f"{dir_label} {display} ({abs(score):.2f})")

        method = "nli_clause_v6" if self._nli_model else "embedding_fallback"

        return InterpretationResult(
            sentiment_polarity=polarity,
            sentiment_subjectivity=subjectivity,
            intent_scores=intent_scores,
            dominant_intents=dominant,
            summary="; ".join(summary_parts) if summary_parts else "No strong matches",
            method=method,
            layer_similarities=layer_sims,
        )
