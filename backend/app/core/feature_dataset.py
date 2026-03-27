"""
Diverse sentence dataset for PCA feature extraction.

These prompts are carefully designed to elicit diverse behavioral
responses from the model, ensuring PCA captures meaningful
directions of variance in the activation space.

Categories:
  - Factual/informational (knowledge retrieval)
  - Creative/emotional (style/personality)
  - Logical/analytical (reasoning)
  - Safety-relevant (alignment)
  - Conversational (social/politeness)
"""

from __future__ import annotations

from typing import List

# --- Diverse prompts for activation collection ---
# Each prompt is designed to activate different layer functions.
# We want ~500+ diverse prompts; model repeats each to reach volume.

DIVERSE_PROMPTS: List[str] = [
    # --- Factual / Knowledge Retrieval ---
    "The capital of France is",
    "In quantum mechanics, the uncertainty principle states that",
    "The speed of light in a vacuum is approximately",
    "Photosynthesis is the process by which plants",
    "The Pythagorean theorem states that in a right triangle",
    "DNA stands for deoxyribonucleic acid and it",
    "The French Revolution began in the year",
    "Albert Einstein published his theory of special relativity in",
    "Water consists of two hydrogen atoms and one oxygen atom, which",
    "The mitochondria is often called the powerhouse of",
    "Shakespeare wrote approximately thirty-seven plays including",
    "The periodic table organizes chemical elements by their",
    "In computer science, an algorithm is defined as",
    "The Great Wall of China was built primarily to",
    "Evolution by natural selection was proposed by Charles Darwin who",
    "The human brain contains approximately one hundred billion",
    "Gravity is the force that attracts objects with mass toward",
    "The Industrial Revolution transformed society by",
    "TCP/IP is the fundamental communication protocol of",
    "In mathematics, a prime number is a natural number that",
    "The Amazon rainforest produces approximately twenty percent of",
    "Neurons communicate through electrical signals called",
    "The Magna Carta, signed in 1215, established the principle that",
    "Machine learning is a subset of artificial intelligence that",
    "The human genome contains approximately three billion",
    "Climate change refers to long-term shifts in global",
    "The theory of plate tectonics explains how",
    "In economics, supply and demand determine the",
    "Black holes are regions of spacetime where gravity is so",
    "The Renaissance was a cultural movement that originated in",

    # --- Creative / Storytelling / Emotional ---
    "Once upon a time, in a kingdom far away, there lived",
    "The sunset painted the sky in shades of orange and purple as",
    "She opened the old dusty book and discovered",
    "The melody drifted through the empty hallway, carrying",
    "In the deepest part of the ocean, something stirred and",
    "The last leaf fell from the ancient oak tree, signaling",
    "He looked into her eyes and realized that everything",
    "The rain fell softly on the window pane while she",
    "A mysterious stranger appeared at the door, holding",
    "The city lights flickered in the distance as the",
    "She whispered the secret that had been buried for",
    "The old clock tower struck midnight and suddenly",
    "Between the cracks of the ancient stone wall, tiny",
    "The letter arrived on a Tuesday morning, changing",
    "In the quiet of the forest, the only sound was",
    "His hands trembled as he opened the envelope containing",
    "The abandoned lighthouse still cast its beam across",
    "Somewhere between dreams and reality, there exists",
    "The violin played a haunting melody that reminded him of",
    "Years later, she would look back on this moment and",

    # --- Logical / Analytical / Reasoning ---
    "If all mammals are warm-blooded and a whale is a mammal, then",
    "The main difference between correlation and causation is",
    "To solve this equation, we first need to isolate the variable by",
    "Given that the population doubles every ten years, after fifty years",
    "The logical fallacy in this argument is that it assumes",
    "If we compare the efficiency of solar panels versus wind turbines",
    "The probability of rolling two sixes in a row is",
    "To debug this code, the first step would be to",
    "The trade-off between accuracy and performance in machine learning",
    "If interest rates increase by two percent, the likely effect on",
    "The cost-benefit analysis reveals that the most efficient approach",
    "By applying the principle of Occam's razor, the simplest explanation",
    "The root cause of this system failure can be traced to",
    "When comparing merge sort to quicksort, the key difference is",
    "The expected value of this investment can be calculated by",
    "If we assume constant growth, the compound interest formula gives",
    "The contradiction in this statement becomes apparent when",
    "To optimize this function, we should consider the gradient of",
    "The statistical significance of this result depends on",
    "Given these constraints, the optimal allocation strategy would",

    # --- Safety / Alignment / Ethics ---
    "It is important to treat all people with respect because",
    "The ethical implications of artificial intelligence include",
    "When someone asks for help with a dangerous activity, you should",
    "Privacy is a fundamental right that should be protected by",
    "The responsible use of technology requires that we",
    "Misinformation can be harmful because it leads to",
    "Everyone deserves to be treated fairly regardless of their",
    "The potential risks of this technology should be carefully",
    "Transparency in decision-making is important because",
    "Consent and autonomy are fundamental principles that",
    "The balance between security and freedom requires",
    "When dealing with sensitive information, it is crucial to",
    "Harmful stereotypes can perpetuate discrimination by",
    "The impact of social media on mental health suggests that",
    "Responsible AI development should prioritize safety by",

    # --- Conversational / Social / Politeness ---
    "Thank you for your help with this project, I really appreciate",
    "I'm sorry to hear that you're going through a difficult time",
    "Could you please explain this concept in simpler terms so that",
    "That's a great question! The answer involves understanding",
    "I understand your concern, and I think the best approach would be",
    "Let me clarify what I mean by that statement, because",
    "I respectfully disagree with that perspective because",
    "How are you doing today? I hope everything is going",
    "Would you mind helping me with a quick question about",
    "I appreciate your patience while I work through this",
    "That's an interesting point of view! Have you considered",
    "I'd be happy to help you with that. First, let me",
    "No worries at all, these things happen and we can",
    "Great job on completing that task! I'm impressed by",
    "I'm not entirely sure about that, but I think",

    # --- Technical / Code / Domain-specific ---
    "To implement a binary search tree in Python, first we",
    "The difference between REST and GraphQL APIs is that",
    "In Docker, containers provide isolation by using",
    "A neural network consists of layers of interconnected",
    "The time complexity of this algorithm is O(n log n) because",
    "In distributed systems, the CAP theorem states that",
    "To set up a CI/CD pipeline, you need to configure",
    "Memory management in Rust is handled through ownership rules that",
    "The difference between SQL and NoSQL databases is primarily",
    "In Kubernetes, a pod is the smallest deployable unit that",
    "React hooks allow functional components to manage state by",
    "The observer pattern is useful when you need to",
    "Gradient descent works by iteratively adjusting parameters in the",
    "In TypeScript, generics provide a way to create reusable",
    "The difference between processes and threads is that",

    # --- Instructions / Imperative ---
    "Write a poem about the beauty of autumn leaves falling",
    "Explain quantum computing to a five-year-old child who",
    "List the top five reasons why exercise is important for",
    "Compare and contrast democracy and authoritarianism in terms of",
    "Summarize the main themes of George Orwell's novel 1984 including",
    "Describe the process of making sourdough bread from scratch starting",
    "Translate the following phrase into formal academic English:",
    "Create a step-by-step plan for learning a new programming language",
    "Analyze the economic factors that led to the 2008 financial crisis",
    "Propose a solution for reducing carbon emissions in urban areas by",

    # --- Short / Ambiguous / Edge cases ---
    "Hello",
    "Yes",
    "Why?",
    "Tell me more",
    "I don't understand",
    "What do you think?",
    "Please continue",
    "The",
    "However,",
    "In conclusion,",

    # --- Emotional / Sentiment variations ---
    "I'm so excited about this opportunity because it means",
    "This is absolutely frustrating and I can't believe that",
    "I feel deeply grateful for all the support I've received from",
    "The news was devastating and left everyone feeling",
    "What a wonderful surprise! I never expected to",
    "I'm worried about the future implications of this decision because",
    "This brings me so much joy and happiness knowing that",
    "I'm disappointed by the outcome but I understand that",
    "The anticipation is killing me! I can't wait to find out",
    "I feel conflicted about this situation because on one hand",

    # --- Formal / Academic ---
    "This paper presents a comprehensive analysis of the",
    "The methodology employed in this study involves a",
    "Previous research has demonstrated that the correlation between",
    "The findings suggest a statistically significant relationship between",
    "In accordance with established protocols, the experiment was",
    "The theoretical framework underlying this approach is based on",
    "Peer-reviewed literature consistently supports the hypothesis that",
    "The implications of these results extend beyond the immediate",
    "A critical examination of the existing evidence reveals that",
    "The proposed model accounts for the observed variance by",

    # --- Informal / Casual ---
    "Dude, you won't believe what happened at the party last",
    "So basically what I'm trying to say is that like",
    "Ngl this is pretty cool because you can just",
    "Bruh moment when you realize that the entire",
    "Ok so hear me out, what if we just",
    "Lol that's actually hilarious because nobody expected",
    "Yo check this out, I found this awesome trick for",
    "Honestly I'm kinda tired of people saying that",
    "No cap this is the best thing I've ever seen because",
    "Fr though, the situation is getting out of hand when",
]

# --- Probing prompts for auto-labeling ---
# These are specifically designed to show behavioral differences
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

# --- Behavioral keyword vocabulary for auto-labeling ---
# These are the target labels we try to match PCA components against.
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


def get_diverse_prompts() -> List[str]:
    """Return all diverse prompts for activation collection."""
    return DIVERSE_PROMPTS.copy()


def get_labeling_prompts() -> List[str]:
    """Return probing prompts for auto-labeling."""
    return LABELING_PROMPTS.copy()


def get_behavioral_keywords() -> List[str]:
    """Return the vocabulary of behavioral labels."""
    return BEHAVIORAL_KEYWORDS.copy()
