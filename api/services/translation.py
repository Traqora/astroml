"""Translation service for multi-language LLM output support (Issue 1)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

try:
    import redis
except ImportError:
    redis = None

try:
    import babel.dates
    import babel.numbers
    from babel.core import Locale
except ImportError:
    babel = None
    Locale = None


SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English", "native": "English", "locale": "en_US"},
    "es": {"name": "Spanish", "native": "Español", "locale": "es_ES"},
    "fr": {"name": "French", "native": "Français", "locale": "fr_FR"},
    "de": {"name": "German", "native": "Deutsch", "locale": "de_DE"},
    "zh": {"name": "Chinese (Simplified)", "native": "中文（简体）", "locale": "zh_CN"},
    "ja": {"name": "Japanese", "native": "日本語", "locale": "ja_JP"},
    "ko": {"name": "Korean", "native": "한국어", "locale": "ko_KR"},
    "pt": {"name": "Portuguese", "native": "Português", "locale": "pt_BR"},
    "it": {"name": "Italian", "native": "Italiano", "locale": "it_IT"},
    "ru": {"name": "Russian", "native": "Русский", "locale": "ru_RU"},
    "ar": {"name": "Arabic", "native": "العربية", "locale": "ar_SA"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "locale": "hi_IN"},
    "nl": {"name": "Dutch", "native": "Nederlands", "locale": "nl_NL"},
    "pl": {"name": "Polish", "native": "Polski", "locale": "pl_PL"},
    "tr": {"name": "Turkish", "native": "Türkçe", "locale": "tr_TR"},
}


@dataclass
class TranslationCacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }


class TranslationCache:
    """Multi-layer translation cache with Redis backend and in-memory fallback."""

    def __init__(self, ttl: int = 86400, max_memory_entries: int = 10000):
        self.ttl = ttl
        self.max_memory_entries = max_memory_entries
        self._memory_cache: Dict[str, tuple[str, float]] = {}
        self._lock = threading.RLock()
        self._stats = TranslationCacheStats()
        self._redis_client = None

        if redis is not None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
            try:
                self._redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis_client.ping()
            except Exception:
                self._redis_client = None

    def _make_key(self, source_text: str, target_lang: str, source_lang: str = "auto") -> str:
        content = f"{source_lang}:{target_lang}:{source_text}"
        return f"trans:{hashlib.sha256(content.encode()).hexdigest()[:32]}"

    def get(self, source_text: str, target_lang: str, source_lang: str = "auto") -> Optional[str]:
        key = self._make_key(source_text, target_lang, source_lang)

        with self._lock:
            if key in self._memory_cache:
                value, expiry = self._memory_cache[key]
                if time.time() < expiry:
                    self._stats.hits += 1
                    return value
                else:
                    del self._memory_cache[key]

        if self._redis_client:
            try:
                value = self._redis_client.get(key)
                if value:
                    with self._lock:
                        self._memory_cache[key] = (value, time.time() + self.ttl)
                        self._enforce_memory_limit()
                    self._stats.hits += 1
                    return value
            except Exception:
                pass

        self._stats.misses += 1
        return None

    def set(self, source_text: str, target_lang: str, translation: str, source_lang: str = "auto") -> None:
        key = self._make_key(source_text, target_lang, source_lang)
        expiry = time.time() + self.ttl

        with self._lock:
            self._memory_cache[key] = (translation, expiry)
            self._enforce_memory_limit()
        self._stats.sets += 1

        if self._redis_client:
            try:
                self._redis_client.setex(key, self.ttl, translation)
            except Exception:
                pass

    def _enforce_memory_limit(self) -> None:
        if len(self._memory_cache) > self.max_memory_entries:
            now = time.time()
            expired = [k for k, (_, exp) in self._memory_cache.items() if exp < now]
            for k in expired:
                del self._memory_cache[k]
            if len(self._memory_cache) > self.max_memory_entries:
                oldest = min(self._memory_cache.items(), key=lambda x: x[1][1])[0]
                del self._memory_cache[oldest]
                self._stats.evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.to_dict()

    def clear(self) -> int:
        with self._lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
        if self._redis_client:
            try:
                keys = self._redis_client.keys("trans:*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception:
                pass
        return count


class LocaleFormatter:
    """Locale-aware formatting for dates, numbers, and currencies."""

    def __init__(self, locale_code: str = "en_US"):
        self.locale_code = locale_code
        self._locale = None
        if babel and Locale:
            try:
                self._locale = Locale.parse(locale_code)
            except Exception:
                self._locale = Locale.parse("en_US")

    @property
    def locale(self):
        if self._locale is None and babel and Locale:
            self._locale = Locale.parse("en_US")
        return self._locale

    def format_date(self, dt: datetime, format: str = "medium") -> str:
        if not babel or not self.locale:
            return dt.strftime("%Y-%m-%d")
        try:
            return babel.dates.format_date(dt, format=format, locale=self.locale)
        except Exception:
            return dt.strftime("%Y-%m-%d")

    def format_datetime(self, dt: datetime, format: str = "medium") -> str:
        if not babel or not self.locale:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            return babel.dates.format_datetime(dt, format=format, locale=self.locale)
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format_number(self, value: float, decimals: int = 2) -> str:
        if not babel or not self.locale:
            return f"{value:,.{decimals}f}"
        try:
            return babel.numbers.format_number(value, locale=self.locale)
        except Exception:
            return f"{value:,.{decimals}f}"

    def format_currency(self, amount: float, currency: str = "USD") -> str:
        if not babel or not self.locale:
            return f"{currency} {amount:,.2f}"
        try:
            return babel.numbers.format_currency(amount, currency, locale=self.locale)
        except Exception:
            return f"{currency} {amount:,.2f}"

    def format_percent(self, value: float, decimals: int = 1) -> str:
        if not babel or not self.locale:
            return f"{value * 100:.{decimals}f}%"
        try:
            return babel.numbers.format_percent(value, locale=self.locale)
        except Exception:
            return f"{value * 100:.{decimals}f}%"


class TranslationService:
    """Main translation service with LLM-based translation and caching."""

    def __init__(self, llm_provider=None):
        self.llm_provider = llm_provider
        self.cache = TranslationCache()
        self._formatters: Dict[str, LocaleFormatter] = {}
        self._lock = threading.Lock()

    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        return SUPPORTED_LANGUAGES.copy()

    def get_formatter(self, locale_code: str) -> LocaleFormatter:
        with self._lock:
            if locale_code not in self._formatters:
                self._formatters[locale_code] = LocaleFormatter(locale_code)
            return self._formatters[locale_code]

    def _build_translation_prompt(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
    ) -> str:
        target_info = SUPPORTED_LANGUAGES.get(target_lang, {"name": target_lang, "native": target_lang})
        target_name = target_info["name"]
        target_native = target_info["native"]

        context_part = f"\nContext: {context}" if context else ""

        return f"""Translate the following text to {target_name} ({target_native}).
Source language: {source_lang if source_lang != "auto" else "auto-detect"}{context_part}

Text to translate:
{text}

Return ONLY the translated text, no explanations or metadata."""

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        if target_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {target_lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")

        if use_cache:
            cached = self.cache.get(text, target_lang, source_lang)
            if cached is not None:
                return {
                    "translated_text": cached,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "cached": True,
                    "latency_ms": 0.0,  # Cached responses have zero latency
                }

        start_time = time.time()
        if self.llm_provider:
            prompt = self._build_translation_prompt(text, target_lang, source_lang, context)
            translation = self.llm_provider.generate(prompt)
        else:
            translation = f"[{target_lang}] {text}"
        latency_ms = (time.time() - start_time) * 1000

        if use_cache:
            self.cache.set(text, target_lang, translation, source_lang)

        return {
            "translated_text": translation.strip(),
            "source_language": source_lang,
            "target_language": target_lang,
            "cached": False,
            "latency_ms": round(latency_ms, 2),
        }

    def translate_batch(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        return [self.translate(t, target_lang, source_lang, context, use_cache) for t in texts]

    async def translate_async(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.translate(text, target_lang, source_lang, context, use_cache)
        )

    async def translate_batch_async(
        self,
        texts: List[str],
        target_lang: str,
        source_lang: str = "auto",
        context: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Async batch translate multiple texts to target language."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [self.translate(t, target_lang, source_lang, context, use_cache) for t in texts]
        )

    def format_locale(
        self,
        value: Union[float, int, str],
        locale: str,
        format_type: str,
        currency_code: Optional[str] = None,
    ) -> str:
        """Format a single value according to locale and format type."""
        formatter = self.get_formatter(locale)
        
        if format_type == "number":
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    return value
            return formatter.format_number(float(value))
        elif format_type == "currency":
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    return value
            currency = currency_code or "USD"
            return formatter.format_currency(float(value), currency)
        elif format_type == "percent":
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    return value
            return formatter.format_percent(float(value))
        elif format_type == "date":
            if isinstance(value, str):
                try:
                    from datetime import datetime
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return value
            if isinstance(value, datetime):
                return formatter.format_date(value)
            return str(value)
        elif format_type == "datetime":
            if isinstance(value, str):
                try:
                    from datetime import datetime
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    return value
            if isinstance(value, datetime):
                return formatter.format_datetime(value)
            return str(value)
        else:
            return str(value)

    def get_cache_stats(self) -> Dict[str, Any]:
        return self.cache.get_stats()

    def invalidate_cache(self, text: str) -> bool:
        """Invalidate cache for given text across all target languages (using source language auto-detection)."""
        # We'll use a special key pattern or just clear all for simplicity
        # For a more sophisticated approach, we'd need to iterate through all language combinations
        # For now, let's clear entries that start with the text hash
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        
        with self.cache._lock:
            # Find and remove keys that match this text
            keys_to_delete = [
                k for k in self.cache._memory_cache.keys() 
                if k.startswith(f"trans:{text_hash}")
            ]
            for k in keys_to_delete:
                del self.cache._memory_cache[k]
        
        if self.cache._redis_client:
            try:
                # Use SCAN to find matching keys
                cursor = 0
                deleted_count = 0
                while True:
                    cursor, keys = self.cache._redis_client.scan(
                        cursor, match=f"trans:*{text_hash}*", count=100
                    )
                    if keys:
                        deleted_count += self.cache._redis_client.delete(*keys)
                    if cursor == 0:
                        break
                return deleted_count > 0
            except Exception:
                return False
        return len(keys_to_delete) > 0 if 'keys_to_delete' in locals() else False

    def invalidate_all_cache(self) -> int:
        return self.cache.clear()


translation_service = TranslationService()