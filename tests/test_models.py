"""Tests for pure model/library selection logic.

These are dependency-free: guess_stt_library takes backend-availability flags
as arguments, so the real STT backends need not be installed.
"""

import pytest

from wyoming_faster_whisper.const import SttLibrary
from wyoming_faster_whisper.models import guess_stt_library, vad_clip_enabled

_ALL_AVAILABLE = dict(
    has_transformers=True,
    has_sherpa=True,
    has_onnx_asr=True,
    has_funasr=True,
    has_qwen3_asr=True,
)


def _guess(preferred, language, model=None, **avail):
    flags = {**_ALL_AVAILABLE, **avail}
    return guess_stt_library(preferred, model, language, **flags)


# --- AUTO: per-language backend selection ---------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        # FunASR (SenseVoice) languages, including locale-style codes.
        ("zh", SttLibrary.FUNASR),
        ("zh-CN", SttLibrary.FUNASR),
        ("zh-TW", SttLibrary.FUNASR),
        ("zh-HK", SttLibrary.FUNASR),  # Hong Kong -> Cantonese
        ("yue", SttLibrary.FUNASR),
        ("ja", SttLibrary.FUNASR),
        ("ko", SttLibrary.FUNASR),
        # Other specialized backends are unaffected.
        ("en", SttLibrary.SHERPA),
        ("ru", SttLibrary.ONNX_ASR),
        # Everything else defaults to faster-whisper.
        ("de", SttLibrary.FASTER_WHISPER),
        (None, SttLibrary.FASTER_WHISPER),
    ],
)
def test_auto_selects_per_language_backend(language, expected) -> None:
    assert _guess(SttLibrary.AUTO, language) == expected


def test_auto_funasr_languages_fall_back_when_funasr_missing() -> None:
    # zh would route to FunASR, but it isn't installed -> faster-whisper.
    assert (
        _guess(SttLibrary.AUTO, "zh-CN", has_funasr=False) == SttLibrary.FASTER_WHISPER
    )


def test_auto_with_explicit_model_skips_per_language_selection() -> None:
    # A forced model disables auto backend selection (stays faster-whisper).
    assert (
        _guess(SttLibrary.AUTO, "zh-CN", model="some/model")
        == SttLibrary.FASTER_WHISPER
    )


# --- Explicit library: dependency fallback --------------------------------


def test_explicit_funasr_kept_when_available() -> None:
    assert _guess(SttLibrary.FUNASR, "zh") == SttLibrary.FUNASR


def test_explicit_funasr_falls_back_when_missing() -> None:
    assert (
        _guess(SttLibrary.FUNASR, "zh", has_funasr=False) == SttLibrary.FASTER_WHISPER
    )


def test_explicit_qwen3_asr_kept_when_available() -> None:
    assert _guess(SttLibrary.QWEN3_ASR, "en") == SttLibrary.QWEN3_ASR


def test_explicit_qwen3_asr_falls_back_when_missing() -> None:
    assert (
        _guess(SttLibrary.QWEN3_ASR, "en", has_qwen3_asr=False)
        == SttLibrary.FASTER_WHISPER
    )


def test_auto_never_selects_qwen3_asr() -> None:
    # Qwen3-ASR is opt-in only: it is large and slow relative to the
    # per-language defaults, so AUTO must not route to it.
    for language in ("en", "ru", "zh", "de", None):
        assert _guess(SttLibrary.AUTO, language) != SttLibrary.QWEN3_ASR


def test_explicit_faster_whisper_is_passthrough() -> None:
    assert (
        _guess(SttLibrary.FASTER_WHISPER, "en", has_sherpa=False)
        == SttLibrary.FASTER_WHISPER
    )


# --- --vad-clip library selection -----------------------------------------


def test_vad_clip_off_when_flag_absent() -> None:
    for library in SttLibrary:
        assert not vad_clip_enabled(library, vad_clip=False)


def test_bare_vad_clip_applies_to_every_library() -> None:
    # `--vad-clip` with no values -> vad_clip_libraries is None.
    for library in SttLibrary:
        assert vad_clip_enabled(library, vad_clip=True, vad_clip_libraries=None)


def test_named_library_is_clipped() -> None:
    assert vad_clip_enabled(
        SttLibrary.QWEN3_ASR, vad_clip=True, vad_clip_libraries={SttLibrary.QWEN3_ASR}
    )


def test_unnamed_libraries_are_not_clipped() -> None:
    # `--vad-clip qwen3-asr` must leave faster-whisper alone, where clipping
    # buys nothing (audio is padded to 30s internally regardless).
    assert not vad_clip_enabled(
        SttLibrary.FASTER_WHISPER,
        vad_clip=True,
        vad_clip_libraries={SttLibrary.QWEN3_ASR},
    )


def test_several_libraries_can_be_named() -> None:
    named = {SttLibrary.QWEN3_ASR, SttLibrary.SHERPA}
    assert vad_clip_enabled(SttLibrary.SHERPA, vad_clip=True, vad_clip_libraries=named)
    assert vad_clip_enabled(
        SttLibrary.QWEN3_ASR, vad_clip=True, vad_clip_libraries=named
    )
    assert not vad_clip_enabled(
        SttLibrary.FUNASR, vad_clip=True, vad_clip_libraries=named
    )


def test_named_libraries_are_ignored_when_flag_is_off() -> None:
    # Defensive: a stale library set must not enable clipping on its own.
    assert not vad_clip_enabled(
        SttLibrary.QWEN3_ASR, vad_clip=False, vad_clip_libraries={SttLibrary.QWEN3_ASR}
    )
