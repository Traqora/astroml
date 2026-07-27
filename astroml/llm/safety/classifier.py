"""Content classification for safety — categorises text by harm type.

Resolves #455: Multi-category content classifier with confidence scoring.
"""
from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Sequence


class ContentCategory(str, Enum):
    """Harm categories for LLM content classification."""

    SAFE = "safe"
    TOXIC = "toxic"
    HATE_SPEECH = "hate_speech"
    HARMFUL = "harmful"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    BIAS = "bias"
    NSFW = "nsfw"
    MISINFORMATION = "misinformation"


# ─── Lightweight rule-based classifier ──────────────────────────────────────
# In production, swap `_classify_heuristic` with a real moderation API call
# (e.g. OpenAI Moderation, LlamaGuard, NeMo Guardrails).

_PROMPT_INJECTION_PATTERNS: Sequence[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(?:evil|unrestricted|DAN|jailbreak)", re.I),
    re.compile(r"act\s+as\s+if\s+you\s+(have\s+no|don.t\s+have)\s+(restrictions?|rules?|guidelines?)", re.I),
    re.compile(r"system\s*:\s*you\s+must", re.I),
    re.compile(r"</?(system|user|assistant)>", re.I),
]

_JAILBREAK_PATTERNS: Sequence[re.Pattern] = [
    re.compile(r"\bDAN\b"),
    re.compile(r"do\s+anything\s+now", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"disregard\s+(your|all)\s+(rules?|ethics?|training)", re.I),
    re.compile(r"pretend\s+you.re\s+(not\s+an?\s+ai|human|unrestricted)", re.I),
]

_HARMFUL_PATTERNS: Sequence[re.Pattern] = [
    re.compile(r"\b(how\s+to\s+make|synthesize|build)\s+(a\s+)?(bomb|weapon|explosive|poison)", re.I),
    re.compile(r"\b(hack|exploit|bypass)\s+(a\s+)?(bank|system|server)", re.I),
    re.compile(r"\b(suicide|self.harm)\s+method", re.I),
]

_TOXIC_PATTERNS: Sequence[re.Pattern] = [
    re.compile(r"\b(kill|murder|assault)\s+(all\s+)?\w+", re.I),
    re.compile(r"\bhate\s+all\s+\w+", re.I),
]


class ContentClassifier:
    """Rule-based content classifier with extensible backend.

    Returns a *(category, confidence)* tuple.  Confidence is in [0, 1].
    A confidence of 0 means definitely not that category, 1 means certain.

    For production deployments, replace ``classify`` with a call to
    OpenAI Moderation API, LlamaGuard, or NeMo Guardrails.
    """

    def classify(self, text: str) -> tuple[ContentCategory, float]:
        """Classify *text* and return *(category, confidence)*."""
        # Prompt injection
        inj_conf = self._pattern_confidence(text, _PROMPT_INJECTION_PATTERNS)
        if inj_conf > 0:
            return ContentCategory.PROMPT_INJECTION, min(inj_conf, 1.0)

        # Jailbreak
        jb_conf = self._pattern_confidence(text, _JAILBREAK_PATTERNS)
        if jb_conf > 0:
            return ContentCategory.JAILBREAK, min(jb_conf, 1.0)

        # Harmful
        harm_conf = self._pattern_confidence(text, _HARMFUL_PATTERNS)
        if harm_conf > 0:
            return ContentCategory.HARMFUL, min(harm_conf, 1.0)

        # Toxic
        toxic_conf = self._pattern_confidence(text, _TOXIC_PATTERNS)
        if toxic_conf > 0:
            return ContentCategory.TOXIC, min(toxic_conf, 1.0)

        return ContentCategory.SAFE, 0.0

    @staticmethod
    def _pattern_confidence(text: str, patterns: Sequence[re.Pattern]) -> float:
        """Return average match confidence across *patterns*."""
        matches = sum(1 for p in patterns if p.search(text))
        if matches == 0:
            return 0.0
        # Scale by ratio of matched patterns, boosted by first match certainty
        base_confidence = 0.65
        extra = 0.1 * (matches - 1)
        return min(base_confidence + extra, 1.0)

    def deterministic_hash(self, text: str) -> str:
        """Deterministic content hash for caching and test fixtures."""
        return hashlib.sha256(text.encode()).hexdigest()
