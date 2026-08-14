"""Tests for what the dispatch handler actually hands the transcriber.

Drives real Wyoming events through ``handle_event`` with a fake transcriber and
loader, so the whole biasing path is covered in-process: AudioStart starts the
refresh, AudioStop waits for it, and the resulting prompt reaches
``transcribe()``. No model and no network.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

pytest.importorskip("aiohttp")

# pylint: disable=wrong-import-position
from wyoming.asr import Transcribe, Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.info import Info  # noqa: E402

from wyoming_faster_whisper.const import StreamingSession, Transcriber  # noqa: E402
from wyoming_faster_whisper.dispatch_handler import DispatchEventHandler  # noqa: E402
from wyoming_faster_whisper.hass_api import HomeAssistantError  # noqa: E402
from wyoming_faster_whisper.name_cache import HassNameCache  # noqa: E402
from wyoming_faster_whisper.vocabulary import RecognitionContext  # noqa: E402

RATE = 16000


class FakeSession(StreamingSession):
    """Records the prompt its stream was opened with."""

    def __init__(self, initial_prompt: Optional[str]) -> None:
        self.initial_prompt = initial_prompt
        self.chunks: List[bytes] = []

    def accept_chunk(self, audio_bytes: bytes) -> None:
        self.chunks.append(audio_bytes)

    def finish(self) -> str:
        return "streamed transcript"


class FakeTranscriber(Transcriber):
    """Captures every call, and can pretend to have (or lack) a tokenizer."""

    def __init__(
        self,
        streaming: bool = False,
        tokens_per_word: Optional[int] = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.sessions: List[FakeSession] = []
        self._streaming = streaming
        self._tokens_per_word = tokens_per_word

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        self.calls.append(
            {
                "language": language,
                "beam_size": beam_size,
                "initial_prompt": initial_prompt,
            }
        )
        return "fake transcript"

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        if self._tokens_per_word is None:
            # Stands in for a backend with no tokenizer to ask.
            return None

        return len(text.split()) * self._tokens_per_word

    @property
    def supports_streaming(self) -> bool:
        return self._streaming

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> StreamingSession:
        session = FakeSession(initial_prompt)
        self.sessions.append(session)
        return session

    @property
    def prompt(self) -> Optional[str]:
        """The prompt from the last transcribe() call."""
        assert self.calls, "transcribe() was never called"
        return self.calls[-1]["initial_prompt"]


class FakeLoader:
    """Just the attributes DispatchEventHandler reads off ModelLoader."""

    def __init__(
        self, transcriber: FakeTranscriber, initial_prompt: Optional[str] = None
    ) -> None:
        self.transcriber = transcriber
        self.initial_prompt = initial_prompt
        self.beam_size = 5
        self.preferred_language = "en"
        self.vad_clip = False
        self.vad_clip_threshold = 0.5
        self.vad_clip_pad_ms = 400

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        return self.transcriber


class FakeHass:
    """Stands in for HomeAssistant."""

    def __init__(self, context: Optional[RecognitionContext] = None, delay=0.0) -> None:
        self.context = context if context is not None else RecognitionContext()
        self.delay = delay
        self.fail = False
        self.calls = 0

    async def get_context(self) -> RecognitionContext:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)

        if self.fail:
            raise HomeAssistantError("unreachable")

        return self.context


class Handler(DispatchEventHandler):
    """Captures written events instead of needing a socket."""

    def __init__(self, *args, **kwargs) -> None:
        self.written: List[Any] = []
        super().__init__(*args, **kwargs)

    async def write_event(self, event) -> None:
        self.written.append(event)

    @property
    def transcript(self) -> str:
        for event in self.written:
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text

        raise AssertionError("no transcript was written")


def _names(*entities) -> RecognitionContext:
    return RecognitionContext(entities=list(entities))


def _handler(transcriber, names=None, initial_prompt=None) -> Handler:
    return Handler(Info(), FakeLoader(transcriber, initial_prompt), names, None, None)


def _cache(hass, **kwargs) -> HassNameCache:
    kwargs.setdefault("max_tokens", 1000)
    return HassNameCache(hass, **kwargs)


async def _utterance(handler: Handler, chunks: int = 3, pace: float = 0.0) -> None:
    """One full request, the way Home Assistant sends it."""
    await handler.handle_event(Transcribe(language="en").event())
    await handler.handle_event(AudioStart(rate=RATE, width=2, channels=1).event())
    for _ in range(chunks):
        await handler.handle_event(
            AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
        )
        # Let background work run, as it does between real chunks.
        await asyncio.sleep(pace)

    await handler.handle_event(AudioStop().event())


# --- the prompt that reaches the transcriber ------------------------------


async def test_without_a_token_the_configured_prompt_is_used():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=None, initial_prompt="Vocabulary:")

    await _utterance(handler)

    assert transcriber.prompt == "Vocabulary:"


async def test_without_a_token_or_a_prompt_nothing_is_sent():
    transcriber = FakeTranscriber()
    await _utterance(_handler(transcriber))

    assert transcriber.prompt is None


async def test_home_assistant_names_reach_the_transcriber():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee", "Nanit"))))

    await _utterance(handler)

    assert transcriber.prompt == "Ecobee, Nanit."


async def test_names_are_appended_to_the_configured_prompt():
    transcriber = FakeTranscriber()
    cache = _cache(FakeHass(_names("Ecobee")), prefix="Vocabulary:")
    handler = _handler(transcriber, names=cache, initial_prompt="Vocabulary:")

    await _utterance(handler)

    assert transcriber.prompt == "Vocabulary: Ecobee."


async def test_the_transcript_still_comes_back():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee"))))

    await _utterance(handler)

    assert handler.transcript == "fake transcript"


# --- the AudioStart window ------------------------------------------------


async def test_audio_start_starts_the_fetch_and_audio_stop_uses_it():
    # Nothing is cached up front, so the names in the prompt can only have come
    # from a fetch that began at AudioStart and landed by AudioStop.
    hass = FakeHass(_names("Ecobee"))
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass))

    await _utterance(handler)

    assert hass.calls == 1
    assert transcriber.prompt == "Ecobee."


async def test_an_utterance_fetches_once_even_over_many_paced_chunks():
    # The regression: a finished fetch is immediately due again, so without a
    # per-utterance latch every chunk would start another one. Pacing matters --
    # the fetch has to be able to finish mid-utterance to re-arm.
    hass = FakeHass(_names("Ecobee"))
    handler = _handler(FakeTranscriber(), names=_cache(hass))

    await _utterance(handler, chunks=30, pace=0.001)

    assert hass.calls == 1


async def test_a_second_utterance_fetches_again():
    hass = FakeHass(_names("Ecobee"))
    handler = _handler(FakeTranscriber(), names=_cache(hass))

    await _utterance(handler)
    await _utterance(handler)

    assert hass.calls == 2


async def test_a_rename_between_utterances_is_picked_up():
    hass = FakeHass(_names("Ecobee"))
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass))

    await _utterance(handler)
    assert transcriber.prompt == "Ecobee."

    hass.context = _names("Ecobee", "Nanit")
    await _utterance(handler)
    assert transcriber.prompt == "Ecobee, Nanit."


async def test_chunks_without_audio_start_still_refresh():
    # Safety net for clients that skip AudioStart.
    hass = FakeHass(_names("Ecobee"))
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass))

    await handler.handle_event(Transcribe(language="en").event())
    for _ in range(3):
        await handler.handle_event(
            AudioChunk(audio=b"\x00\x00" * 160, rate=RATE, width=2, channels=1).event()
        )
        await asyncio.sleep(0)

    await handler.handle_event(AudioStop().event())

    assert hass.calls == 1
    assert transcriber.prompt == "Ecobee."


# --- token budget --------------------------------------------------------


async def test_the_budget_uses_the_models_own_tokenizer():
    # 10 tokens per word: "Ecobee, Nanit." is 2 words -> 20, over a 15 budget,
    # so only the first name fits.
    transcriber = FakeTranscriber(tokens_per_word=10)
    cache = _cache(FakeHass(_names("Ecobee", "Nanit")), max_tokens=15)
    handler = _handler(transcriber, names=cache)

    await _utterance(handler)

    assert transcriber.prompt == "Ecobee."


async def test_a_backend_without_a_tokenizer_still_gets_a_prompt():
    # count_prompt_tokens returns None, so the character estimate is used.
    transcriber = FakeTranscriber(tokens_per_word=None)
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee", "Nanit"))))

    await _utterance(handler)

    assert transcriber.prompt == "Ecobee, Nanit."


# --- degradation ---------------------------------------------------------


async def test_an_unreachable_home_assistant_still_transcribes():
    hass = FakeHass(_names("Ecobee"))
    hass.fail = True
    transcriber = FakeTranscriber()
    handler = _handler(
        transcriber,
        names=_cache(hass, prefix="Vocabulary:"),
        initial_prompt="Vocabulary:",
    )

    await _utterance(handler)

    assert handler.transcript == "fake transcript"
    assert transcriber.prompt == "Vocabulary:"


async def test_a_slow_fetch_does_not_stop_the_transcript():
    hass = FakeHass(_names("Ecobee"), delay=5.0)
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(hass, wait_timeout=0.01))

    await _utterance(handler)

    assert handler.transcript == "fake transcript"
    assert transcriber.prompt is None, "no names had arrived yet"


async def test_audio_stop_with_no_audio_returns_an_empty_transcript():
    transcriber = FakeTranscriber()
    handler = _handler(transcriber, names=_cache(FakeHass(_names("Ecobee"))))

    await handler.handle_event(Transcribe(language="en").event())
    await handler.handle_event(AudioStop().event())

    assert handler.transcript == ""
    assert not transcriber.calls


# --- streaming path ------------------------------------------------------


async def test_a_streaming_session_gets_the_names_already_on_hand():
    hass = FakeHass(_names("Ecobee"))
    cache = _cache(hass)
    await cache.refresh()

    transcriber = FakeTranscriber(streaming=True)
    handler = _handler(transcriber, names=cache)

    await _utterance(handler)

    assert transcriber.sessions
    assert transcriber.sessions[0].initial_prompt == "Ecobee."
    assert handler.transcript == "streamed transcript"


async def test_a_streaming_session_does_not_wait_for_a_slow_fetch():
    # The prompt is needed at the start of the audio, so there is nothing to
    # wait for: the session opens with the prefix alone, immediately.
    #
    # The timing assert is the real check. Waiting would still end up with
    # "Vocabulary:" once the timeout expired, so only elapsed time distinguishes
    # "did not wait" from "waited and gave up".
    hass = FakeHass(_names("Ecobee"), delay=30.0)
    transcriber = FakeTranscriber(streaming=True)
    handler = _handler(
        transcriber,
        names=_cache(hass, prefix="Vocabulary:", wait_timeout=30.0),
        initial_prompt="Vocabulary:",
    )

    start = time.monotonic()
    await _utterance(handler)
    elapsed = time.monotonic() - start

    assert transcriber.sessions[0].initial_prompt == "Vocabulary:"
    assert elapsed < 1.0, f"streaming start blocked for {elapsed:.1f}s"
