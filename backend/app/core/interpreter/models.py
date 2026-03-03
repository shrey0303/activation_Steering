"""
Interpreter data models, constants, and shared utilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


# --- Data Classes ---


@dataclass
class FeatureMatch:
    """A matched feature from the intent router."""
    feature_id: str
    label: str
    layer_idx: int
    similarity: float
    direction: float = 1.0
    variance_explained: float = 0.0


@dataclass
class IntentRouterResult:
    """Output of the intent router."""
    matches: List[FeatureMatch] = field(default_factory=list)
    query: str = ""
    method: str = "cosine_similarity"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matches": [
                {
                    "feature_id": m.feature_id,
                    "label": m.label,
                    "layer_idx": m.layer_idx,
                    "similarity": round(m.similarity, 4),
                    "direction": m.direction,
                    "variance_explained": round(m.variance_explained, 6),
                }
                for m in self.matches
            ],
            "query": self.query,
            "method": self.method,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class InterpretationResult:
    """Structured output of the response interpreter."""
    sentiment_polarity: float = 0.0
    sentiment_subjectivity: float = 0.0
    intent_scores: Dict[str, float] = field(default_factory=dict)
    dominant_intents: List[str] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)
    summary: str = ""
    method: str = "embedding"
    layer_similarities: Dict[str, float] = field(default_factory=dict)
    # Feature-dict routing results (populated when FeatureDictionary is available)
    feature_matches: List[Dict] = field(default_factory=list)  # [{feature_id, label, layer_idx, direction}]
    routed_by_features: bool = False  # True when IntentRouter (DB) was used

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentiment_polarity": round(self.sentiment_polarity, 3),
            "sentiment_subjectivity": round(self.sentiment_subjectivity, 3),
            "intent_scores": {
                k: round(v, 3) for k, v in self.intent_scores.items()
            },
            "dominant_intents": self.dominant_intents,
            "keywords_found": self.keywords_found,
            "summary": self.summary,
            "method": self.method,
            "layer_similarities": {
                k: round(v, 3) for k, v in self.layer_similarities.items()
            },
        }


# --- Legacy Category Labels ---

LAYER_CATEGORIES = [
    "token_embedding",
    "positional_morphological",
    "syntactic_processing",
    "entity_semantic",
    "knowledge_retrieval",
    "reasoning_planning",
    "safety_alignment",
    "information_integration",
    "style_personality",
    "output_distribution",
]

# Neutral category names for bi-encoder retrieval step
CATEGORY_LABELS = {
    "token_embedding": "vocabulary and word choice",
    "positional_morphological": "grammar and syntax",
    "syntactic_processing": "sentence structure",
    "entity_semantic": "entity and reference understanding",
    "knowledge_retrieval": "factual knowledge",
    "reasoning_planning": "reasoning and logic",
    "safety_alignment": "safe and appropriate behavior",
    "information_integration": "information coherence",
    "style_personality": "personality and style",
    "output_distribution": "output confidence",
}

# NLI hypothesis pairs for direction classification.
# Entailment score for enhance_h vs suppress_h determines +1 or -1 direction.
CATEGORY_HYPOTHESES = {
    "token_embedding": {
        "enhance_h": "The speaker wants clearer and more precise vocabulary.",
        "suppress_h": "The speaker wants vague and imprecise language.",
    },
    "positional_morphological": {
        "enhance_h": "The speaker wants better grammar and correct syntax.",
        "suppress_h": "The speaker wants broken grammar and bad syntax.",
    },
    "syntactic_processing": {
        "enhance_h": "The speaker wants more complex and sophisticated sentence structure.",
        "suppress_h": "The speaker wants simple and choppy sentences.",
    },
    "entity_semantic": {
        "enhance_h": "The speaker wants better entity recognition and accurate references.",
        "suppress_h": "The speaker wants confused entities and wrong references.",
    },
    "knowledge_retrieval": {
        "enhance_h": "The speaker wants a more knowledgeable and factually accurate response.",
        "suppress_h": "The speaker wants a less knowledgeable and less accurate response.",
    },
    "reasoning_planning": {
        "enhance_h": "The speaker wants more logical and analytical reasoning.",
        "suppress_h": "The speaker wants illogical and confused reasoning.",
    },
    "safety_alignment": {
        "enhance_h": "The speaker wants a safer, more polite, and harmless response.",
        "suppress_h": "The speaker wants a more toxic, offensive, or harmful response.",
    },
    "information_integration": {
        "enhance_h": "The speaker wants more coherent and well-integrated information.",
        "suppress_h": "The speaker wants disjointed and incoherent output.",
    },
    "style_personality": {
        "enhance_h": "The speaker wants a more expressive, friendly, and engaging personality.",
        "suppress_h": "The speaker wants a bland, dry, and robotic response.",
    },
    "output_distribution": {
        "enhance_h": "The speaker wants more confident and decisive output.",
        "suppress_h": "The speaker wants uncertain and hedging output.",
    },
}


# --- Utilities ---


def split_clauses(text: str) -> List[str]:
    """
    Split compound text into clauses by natural connectors.

    Handles: and, but, though, yet, while, even, however, although,
             or, nor, plus, comma-separated phrases, semicolons.

    Examples:
        'be angry and calm'    → ['be angry', 'calm']
        'toxic but friendly'   → ['toxic', 'friendly']
        'more dangerous; less polite' → ['more dangerous', 'less polite']
    """
    pattern = (
        r'\s*(?:\band\b|\bbut\b|\bthough\b|\byet\b|\bwhile\b|\beven\b|'
        r'\bhowever\b|\balthough\b|\bor\b|\bnor\b|\bplus\b|[;,])\s*'
    )
    clauses = re.split(pattern, text, flags=re.IGNORECASE)
    return [c.strip() for c in clauses if len(c.strip()) > 2]
