"""
Labeling prompts and behavioral keyword vocabulary for PCA auto-labeling.
"""

from __future__ import annotations

from typing import List


# Probing prompts specifically designed to show behavioral differences
# when a PCA component is amplified.
LABELING_PROMPTS: List[str] = [
    "Tell me a story about a brave knight.",
    "Explain how gravity works.",
    "Give me advice about saving money.",
    "Write a poem about the ocean.",
    "What is the meaning of life?",
    "Describe your favorite food in detail.",
    "Help me solve this math problem: 2x + 5 = 17",
    "Debate whether social media is good or bad.",
    "Tell me something interesting about space.",
    "How should I handle a disagreement with a friend?",
]

# Target labels we try to match PCA components against.
BEHAVIORAL_KEYWORDS: List[str] = [
    "formality",
    "emotion",
    "verbosity",
    "toxicity",
    "creativity",
    "logic",
    "safety",
    "politeness",
    "confidence",
    "humor",
    "aggression",
    "helpfulness",
    "technicality",
    "simplicity",
    "enthusiasm",
    "caution",
    "empathy",
    "directness",
    "storytelling",
    "analytical",
    "knowledge",
    "syntax",
    "repetition",
    "coherence",
    "sentiment",
    "persuasion",
    "instruction_following",
    "abstractness",
    "concreteness",
    "question_answering",
]
