"""Tests for the Qwen3-ASR prompt/front-end helpers.

The model files are multi-GB, so these cover the pure logic only: the chat
template the decoder is prompted with, and the mel front-end. Skipped when
onnxruntime is not installed (the handler imports it at module scope).
"""

import numpy as np
import pytest

qwen3_asr_handler = pytest.importorskip(
    "wyoming_faster_whisper.qwen3_asr_handler",
    reason="qwen3-asr extra not installed",
)

_build_prompt_ids = qwen3_asr_handler.Qwen3AsrTranscriber._build_prompt_ids
_log_mel_spectrogram = qwen3_asr_handler._log_mel_spectrogram
_mel_filterbank = qwen3_asr_handler._mel_filterbank
qwen3_asr_language = qwen3_asr_handler.qwen3_asr_language

_AUDIO_PAD = qwen3_asr_handler._AUDIO_PAD


# --- language normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "English"),
        ("EN", "English"),
        ("en-US", "English"),  # locale-style
        ("zh-CN", "Chinese"),
        ("yue", "Cantonese"),
        ("xx", None),  # unsupported
        (None, None),
        ("", None),
    ],
)
def test_language_normalization(code, expected) -> None:
    assert qwen3_asr_language(code) == expected


# --- prompt construction ---------------------------------------------------


def test_prompt_has_one_audio_pad_per_encoder_frame() -> None:
    ids = _build_prompt_ids(None, 7, [], [])
    assert ids.count(_AUDIO_PAD) == 7


def test_prompt_matches_chat_template_layout() -> None:
    ids = _build_prompt_ids(None, 1, [], [])
    # <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    assert ids[:9] == [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
    # ...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    assert ids[10:] == [151670, 151645, 198, 151644, 77091, 198]


def test_context_is_placed_in_the_system_turn() -> None:
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 1, context, [])
    # Context sits after "<|im_start|>system\n" and before "<|im_end|>".
    assert ids[:3] == [151644, 8948, 198]
    assert ids[3:5] == context
    assert ids[5] == 151645


def test_forced_language_suffix_goes_after_the_assistant_turn() -> None:
    language_ids = [3333, qwen3_asr_handler._ASR_TEXT]
    ids = _build_prompt_ids(None, 1, [], language_ids)
    assert ids[-2:] == language_ids
    assert ids[-3] == 198  # newline closing "<|im_start|>assistant\n"


# --- mel front-end ---------------------------------------------------------


def test_mel_filterbank_shape_and_normalization() -> None:
    filters = _mel_filterbank()
    assert filters.shape == (128, (400 // 2) + 1)
    assert filters.dtype == np.float32
    assert (filters >= 0).all()
    # Every filter carries some weight (no empty bands).
    assert (filters.sum(axis=1) > 0).all()


def test_log_mel_frame_count_and_range() -> None:
    # 1 second of 16kHz audio -> 100 frames (10ms hop), minus the dropped frame.
    audio = np.zeros(16000, dtype=np.float32)
    mel = _log_mel_spectrogram(audio, _mel_filterbank())
    assert mel.shape == (1, 128, 100)
    assert mel.dtype == np.float32
    # The log scale is clamped to an 8-decade window then rescaled.
    assert mel.max() - mel.min() <= 2.0 + 1e-6
