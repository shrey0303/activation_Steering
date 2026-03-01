"""
Shared constants for the SteerOps backend.

All magic numbers, timeout values, and category mappings live here
so they can be imported from a single location.
"""

from __future__ import annotations

from typing import Dict, List


# --- API Timeout Constants (seconds) ---

SCAN_TIMEOUT: int = 600
GENERATE_TIMEOUT: int = 120
ANALYZE_TIMEOUT: int = 120
COMPUTE_TIMEOUT: int = 120
FEATURE_EXTRACTION_TIMEOUT: int = 300


# --- Behavior → Concept Keyword Map ---
# Used by the analyze endpoint to match a user-provided behavior
# description to the closest CAA concept in contrastive_pairs.json.

KEYWORD_MAP: Dict[str, str] = {
    "polite": "politeness", "rude": "politeness", "courteous": "politeness",
    "honest": "politeness", "kind": "politeness", "respectful": "politeness",
    "helpful": "politeness", "friendly": "politeness",
    "toxic": "toxicity", "harmful": "toxicity", "offensive": "toxicity",
    "hateful": "toxicity", "angry": "toxicity", "aggressive": "toxicity",
    "safe": "toxicity", "unsafe": "toxicity",
    "creative": "creativity", "imaginative": "creativity",
    "artistic": "creativity", "inventive": "creativity", "original": "creativity",
    "verbose": "verbosity", "concise": "verbosity", "brief": "verbosity",
    "detailed": "verbosity", "short": "verbosity", "long": "verbosity",
    "refuse": "refusal", "reject": "refusal", "decline": "refusal",
    "comply": "refusal", "obey": "refusal", "obedient": "refusal",
}


# --- Layer Category → CAA Concept Mapping ---
# Maps scanner-assigned layer categories to the best CAA concept
# for auto-computing direction vectors during patch export.

CATEGORY_TO_CONCEPT: Dict[str, str] = {
    "style_personality": "politeness",
    "safety_alignment": "toxicity",
    "reasoning_planning": "refusal",
    "entity_semantic": "creativity",
    "information_integration": "verbosity",
    "knowledge_retrieval": "refusal",
    "output_distribution": "verbosity",
    "syntactic_processing": "creativity",
    "positional_morphological": "creativity",
    "token_embedding": "verbosity",
}


# --- Layer Functional Categories ---
# Research-backed categories for transformer layer classification.
# Position ranges are approximate — actual assignment uses K-Means
# clustering on weight features with position as a tiebreaker.

CATEGORIES: List[str] = [
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

# Citations for each category
CATEGORY_CITATIONS: Dict[str, str] = {
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

BEHAVIORAL_ROLES: Dict[str, str] = {
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


# --- Steering Engine Defaults ---

DEFAULT_ENTROPY_THRESHOLD: float = 6.0
DEFAULT_COOLDOWN_TOKENS: int = 5
DEFAULT_GATE_THRESHOLD: float = 0.15
DEFAULT_NORM_TOLERANCE: float = 0.05
DEFAULT_DECAY_RATE: float = 0.006
DEFAULT_MIN_DECAY: float = 0.4


# --- Position-Based Layer Categorisation ---

def position_to_category(relative_pos: float) -> str:
    """Map relative layer position [0, 1] to a research-backed functional category."""
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
