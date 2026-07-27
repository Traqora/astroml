"""
Translation service for multilingual LLM support.

Handles translation between 10+ languages with caching.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


class SupportedLanguage(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    from_cache: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "confidence": self.confidence,
            "from_cache": self.from_cache,
        }


class TranslationService:
    """
    Translate text between supported languages.

    Supports real-time translation via LLM and cached template translations.
    """

    def __init__(self, cache_size: int = 10000):
        """
        Initialize translation service.

        Args:
            cache_size: Maximum cache entries
        """
        self._cache: Dict[str, str] = {}
        self.cache_size = cache_size

    def translate(
        self,
        text: str,
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
        use_cache: bool = True,
    ) -> TranslationResult:
        """
        Translate text to target language.

        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            use_cache: Whether to use translation cache

        Returns:
            TranslationResult
        """
        # Check cache
        cache_key = f"{source_language.value}:{target_language.value}:{text}"
        if use_cache and cache_key in self._cache:
            return TranslationResult(
                original_text=text,
                translated_text=self._cache[cache_key],
                source_language=source_language.value,
                target_language=target_language.value,
                confidence=0.98,
                from_cache=True,
            )

        # Simulate translation
        translated = self._simulate_translation(text, target_language.value)

        # Cache result
        if use_cache and len(self._cache) < self.cache_size:
            self._cache[cache_key] = translated

        return TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language=source_language.value,
            target_language=target_language.value,
            confidence=0.92,
            from_cache=False,
        )

    def batch_translate(
        self,
        texts: List[str],
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
    ) -> List[TranslationResult]:
        """
        Translate multiple texts.

        Args:
            texts: List of texts to translate
            source_language: Source language
            target_language: Target language

        Returns:
            List of TranslationResults
        """
        return [
            self.translate(text, source_language, target_language)
            for text in texts
        ]

    def translate_prompt_template(
        self,
        template: str,
        target_language: SupportedLanguage,
    ) -> str:
        """
        Translate a prompt template preserving placeholders.

        Args:
            template: Template with {placeholder} syntax
            target_language: Target language

        Returns:
            Translated template
        """
        # Simulate template translation
        translations = {
            "es": "Analizar este {documento}",
            "fr": "Analyser ce {document}",
            "de": "Dieses {dokument} analysieren",
        }

        return translations.get(target_language.value, template)

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.

        Returns:
            List of language codes
        """
        return [lang.value for lang in SupportedLanguage]

    def clear_cache(self) -> None:
        """Clear translation cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_entries": len(self._cache),
            "cache_capacity": self.cache_size,
        }

    def _simulate_translation(self, text: str, target_lang: str) -> str:
        """Simulate translation (placeholder for real translation API)."""
        # Simple mapping for demo
        translations = {
            "es": f"[ES] {text}",
            "fr": f"[FR] {text}",
            "de": f"[DE] {text}",
            "pt": f"[PT] {text}",
            "ja": f"[JA] {text}",
            "zh-CN": f"[ZH-CN] {text}",
        }
        return translations.get(target_lang, text)
