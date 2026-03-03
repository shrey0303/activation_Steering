"""Concept anchor sentences for evaluation alignment scoring."""

from __future__ import annotations

from typing import Dict, List


CONCEPT_ANCHORS: Dict[str, List[str]] = {
    "politeness": [
        "Thank you for your question, I'd be happy to help.",
        "I appreciate your patience. Let me explain clearly.",
        "That's a great observation. Here's what I think.",
        "I understand your concern and want to address it thoughtfully.",
    ],
    "hindi_honorific": [
        "आपका बहुत-बहुत धन्यवाद, कृपया बताइए मैं आपकी क्या सहायता कर सकता हूँ।",
        "श्रीमान जी, आपका अनुरोध स्वीकार है। कृपया थोड़ा धैर्य रखें।",
        "आदरणीय महोदय, आपकी सेवा में हम सदैव तत्पर हैं।",
        "नमस्कार जी, कृपया अपनी बात विस्तार से बताइए।",
        "आपकी कृपा है जो आपने हमसे संपर्क किया। हम आपकी पूरी मदद करेंगे।",
        "जी हाँ, आप बिलकुल सही कह रहे हैं। कृपया आगे बताइए।",
    ],
    "bengali_honorific": [
        "আপনাকে অনেক ধন্যবাদ, দয়া করে বলুন আমি কীভাবে সাহায্য করতে পারি।",
        "মহাশয়, আপনার অনুরোধ গ্রহণ করা হয়েছে। অনুগ্রহ করে একটু ধৈর্য রাখুন।",
        "আদরণীয় মহাশয়, আপনার সেবায় আমরা সদা প্রস্তুত।",
        "নমস্কার, দয়া করে আপনার বিষয়টি বিস্তারিত বলুন।",
        "আপনার কৃতজ্ঞতা জানাই যে আপনি আমাদের সাথে যোগাযোগ করেছেন।",
    ],
    "tamil_honorific": [
        "மிக்க நன்றி, தயவுசெய்து நான் எவ்வாறு உதவ முடியும் என்று சொல்லுங்கள்.",
        "ஐயா, உங்கள் கோரிக்கை ஏற்றுக்கொள்ளப்பட்டது. தயவுசெய்து பொறுமையாக இருங்கள்.",
        "மரியாதைக்குரிய ஐயா, உங்கள் சேவையில் நாங்கள் எப்போதும் தயாராக இருக்கிறோம்.",
        "வணக்கம், தயவுசெய்து உங்கள் விஷயத்தை விரிவாகக் கூறுங்கள்.",
        "நீங்கள் எங்களைத் தொடர்பு கொண்டதற்கு நன்றி தெரிவிக்கிறோம்.",
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


def get_concept_anchors(concept: str) -> List[str]:
    """Return short anchor sentences for a steering concept."""
    return CONCEPT_ANCHORS.get(concept, [])
