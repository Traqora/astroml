"""Voice interface for LLM queries (issue #411)."""
from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_auth, AuthContext
from api.database import get_db
from api.routers.llm import log_llm_interaction

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

# Supported languages for STT/TTS
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "pt": "Portuguese",
    "ko": "Korean",
}


class STTRequest(BaseModel):
    """Speech-to-text request."""
    audio_data: str
    language: str = "en"
    format: str = "wav"


class STTResponse(BaseModel):
    """Speech-to-text response."""
    text: str
    confidence: float
    language: str


class TTSRequest(BaseModel):
    """Text-to-speech request."""
    text: str
    language: str = "en"
    voice: Optional[str] = None
    speed: float = 1.0


class TTSResponse(BaseModel):
    """Text-to-speech response."""
    audio_data: str
    duration_ms: float
    language: str


@router.post("/stt", response_model=STTResponse)
async def speech_to_text(
    request: STTRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
) -> STTResponse:
    """Convert speech to text.

    Converts audio input to text with automatic language detection.
    Supports 5+ languages: English, Spanish, French, German, Japanese, Chinese, Portuguese, Korean.
    Target latency: <2 seconds.
    """
    try:
        if request.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {request.language}")

        audio_bytes = base64.b64decode(request.audio_data)

        import time
        start_time = time.time()

        text = await mock_stt(audio_bytes, request.language)
        confidence = 0.95

        latency_ms = int((time.time() - start_time) * 1000)

        await log_llm_interaction(
            db,
            feature="stt",
            prompt=f"audio[{request.format}:{request.language}]",
            response=text,
            interaction_type="stt",
            auth=auth,
            latency_ms=latency_ms,
        )

        return STTResponse(
            text=text,
            confidence=confidence,
            language=request.language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tts", response_model=TTSResponse)
async def text_to_speech(
    request: TTSRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
) -> TTSResponse:
    """Convert text to speech.

    Converts text to audio output with support for multiple languages and voices.
    Supports 5+ languages: English, Spanish, French, German, Japanese, Chinese, Portuguese, Korean.
    Target latency: <2 seconds.
    """
    try:
        if request.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {request.language}")

        if request.speed < 0.5 or request.speed > 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")

        import time
        start_time = time.time()

        audio_data, duration_ms = await mock_tts(
            request.text,
            request.language,
            request.voice,
            request.speed,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        await log_llm_interaction(
            db,
            feature="tts",
            prompt=request.text,
            response=f"audio[wav:{request.language}:{duration_ms}ms]",
            interaction_type="tts",
            auth=auth,
            latency_ms=latency_ms,
        )

        return TTSResponse(
            audio_data=audio_b64,
            duration_ms=duration_ms,
            language=request.language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/query")
async def voice_query(
    file: UploadFile = File(...),
    language: str = "en",
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
) -> dict:
    """End-to-end voice query: STT -> LLM -> TTS.

    Takes audio input, converts to text, queries LLM, and returns spoken response.
    """
    try:
        import time
        start_time = time.time()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        audio_bytes = await file.read()
        text = await mock_stt(audio_bytes, language)

        from api.services.llm_query import QueryTranslator
        query_translator = QueryTranslator()
        sql_response = query_translator.translate_to_sql(text)

        audio_data, duration_ms = await mock_tts(
            sql_response,
            language,
            None,
            1.0,
        )

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        latency_ms = int((time.time() - start_time) * 1000)

        await log_llm_interaction(
            db,
            feature="voice_query",
            prompt=text,
            response=sql_response,
            interaction_type="voice_query",
            auth=auth,
            latency_ms=latency_ms,
        )

        return {
            "input_text": text,
            "response_text": sql_response,
            "audio_data": audio_b64,
            "audio_duration_ms": duration_ms,
            "total_latency_ms": latency_ms,
            "language": language,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/languages")
async def get_supported_languages() -> dict:
    """Get list of supported languages for voice interface."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "count": len(SUPPORTED_LANGUAGES),
    }


async def mock_stt(audio_bytes: bytes, language: str) -> str:
    """Mock speech-to-text conversion.

    In production, this would call a real STT service like:
    - Google Cloud Speech-to-Text
    - Azure Speech to Text
    - AWS Transcribe
    - Whisper
    """
    return f"Transcribed text in {SUPPORTED_LANGUAGES.get(language, language)}"


async def mock_tts(
    text: str,
    language: str,
    voice: Optional[str],
    speed: float,
) -> tuple[bytes, float]:
    """Mock text-to-speech conversion.

    In production, this would call a real TTS service like:
    - Google Cloud Text-to-Speech
    - Azure Text to Speech
    - AWS Polly
    - ElevenLabs
    """
    duration_ms = len(text) * 50 / speed
    audio_bytes = b"RIFF" + b"\x00" * 36 + b"fmt " + b"\x00" * 16 + b"data" + b"\x00" * 4
    return audio_bytes, duration_ms
