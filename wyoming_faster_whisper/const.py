"""Constants."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class SttLibrary(str, Enum):
    """Speech-to-text library."""

    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    TRANSFORMERS = "transformers"
    SHERPA = "sherpa"
    ONNX_ASR = "onnx-asr"
    FUNASR = "funasr"
    QWEN3_ASR = "qwen3-asr"


AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"

# Where to look for Home Assistant when biasing toward its names.
HASS_API_URL = "http://homeassistant.local:8123/api"

# SenseVoice (FunASR) can be told to decode these languages explicitly;
# otherwise it auto-detects. Maps the locale-style codes that Home Assistant /
# intent-sentences use (e.g. "zh-CN") onto the base SenseVoice language.
_SENSE_VOICE_LANGUAGES = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hk": "yue",  # Hong Kong audio is typically Cantonese
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "en": "en",
}


def sense_voice_language(language: Optional[str]) -> Optional[str]:
    """Normalize a language code to a SenseVoice language, or None if unsupported."""
    if not language:
        return None

    return _SENSE_VOICE_LANGUAGES.get(language.lower())


PARAKEET_LANGUAGES = {
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
}


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        pass

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        """Count the tokens ``text`` would use as an initial_prompt.

        Returns None when this backend has no tokenizer to ask, in which case
        callers fall back to an estimate. Used to fit as many entity names as
        possible into the prompt without crossing the model's context limit.
        """
        return None

    @property
    def supports_streaming(self) -> bool:
        """Whether this transcriber can process audio chunks incrementally.

        When False (the default), callers must buffer the entire utterance and
        use transcribe(). When True, start_stream() returns a StreamingSession
        that transcribes audio as it arrives.
        """
        return False

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> "StreamingSession":
        """Begin a new streaming transcription session.

        The returned session holds all per-utterance state, so a single
        (shared) transcriber can drive multiple concurrent sessions.

        Only valid when supports_streaming is True.
        """
        raise NotImplementedError


class StreamingSession(ABC):
    """A single in-progress streaming transcription.

    Created by Transcriber.start_stream(). Holds per-utterance state so it is
    safe to use one session per client connection even when the underlying
    transcriber is shared.
    """

    @abstractmethod
    def accept_chunk(self, audio_bytes: bytes) -> None:
        """Feed a chunk of audio (16Khz 16-bit mono PCM) to the stream."""

    @abstractmethod
    def finish(self) -> str:
        """Finish the stream and return the final transcript."""
