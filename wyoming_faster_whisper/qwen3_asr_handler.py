"""Code for transcription using Qwen3-ASR ONNX exports.

Qwen3-ASR is a speech LLM: an audio encoder feeds projected audio embeddings into
a Qwen3 decoder that generates the transcript token by token. The ONNX export
splits it into three graphs (encoder, decoder prefill, decoder step), so unlike
the other backends there is no library that drives it for us - the greedy decode
loop lives here.

The payoff is context biasing. Qwen3-ASR accepts free-form text in the chat
template's system turn to bias decoding toward specific spellings, so
``--initial-prompt`` can carry Home Assistant entity names ("Vocabulary: Ecobee,
office lamp.") and fix names that would otherwise be transcribed phonetically.

onnxruntime is imported at module scope (not lazily inside __init__) so that
importing this module raises ImportError when it is absent. ModelLoader relies on
that to detect the backend and fall back to faster-whisper, matching
onnx_asr_handler/funasr_handler.
"""

import json
import logging
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from .const import Transcriber

_LOGGER = logging.getLogger(__name__)

_RATE = 16000

# Mel front-end (identical to Whisper's).
_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 128
_FMIN = 0.0
_FMAX = 8000.0

# Chat template token ids. NOTE: the reference prompt builder in
# andrewleech/qwen3-asr-onnx hardcodes 9125 and 882 here, which decode to
# " Current" and " time" rather than "system" and "user".
_IM_START = 151644
_IM_END = 151645
_SYSTEM = 8948
_USER = 872
_ASSISTANT = 77091
_NEWLINE = 198
_AUDIO_START = 151669
_AUDIO_END = 151670
_AUDIO_PAD = 151676
_ASR_TEXT = 151704  # <asr_text>: everything before it is the language preamble
_EOS_IDS = frozenset((151643, 151645))

_MAX_NEW_TOKENS = 256

# Only these files are needed for int4 inference; the repo also holds multi-GB
# FP32 weights and tarballs that we must not pull down.
_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "embed_tokens.bin",
    "encoder.int4.onnx",
    "encoder.int4.onnx.data",
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",
]

# Language names the model was trained to accept in the forced-language suffix.
_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def qwen3_asr_language(language: Optional[str]) -> Optional[str]:
    """Normalize a language code to a Qwen3-ASR language name, or None."""
    if not language:
        return None

    code = language.lower()
    name = _LANGUAGE_NAMES.get(code)
    if name is None:
        # Accept locale-style codes like "en-US" or "zh-CN".
        name = _LANGUAGE_NAMES.get(code.split("-", maxsplit=1)[0])

    return name


def _mel_filterbank() -> np.ndarray:
    """Slaney-normalized mel filterbank, matching librosa.filters.mel(norm="slaney")."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        mels = freq / f_sp
        above = freq >= min_log_hz
        mels[above] = min_log_mel + np.log(freq[above] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        freqs = mels * f_sp
        above = mels >= min_log_mel
        freqs[above] = min_log_hz * np.exp(logstep * (mels[above] - min_log_mel))
        return freqs

    fft_freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / _RATE)
    mel_points = np.linspace(
        hz_to_mel(np.array([_FMIN]))[0], hz_to_mel(np.array([_FMAX]))[0], _N_MELS + 2
    )
    band_hz = mel_to_hz(mel_points)

    diff = np.diff(band_hz)
    ramps = band_hz[:, np.newaxis] - fft_freqs[np.newaxis, :]
    weights = np.zeros((_N_MELS, len(fft_freqs)), dtype=np.float32)
    for i in range(_N_MELS):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: equal area per filter.
    weights *= (2.0 / (band_hz[2 : _N_MELS + 2] - band_hz[:_N_MELS]))[:, np.newaxis]
    return weights


def _log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper-compatible log-mel spectrogram, shape [1, n_mels, time]."""
    padded = np.pad(audio, _N_FFT // 2, mode="reflect")
    # Periodic Hann (np.hanning is symmetric).
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)

    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP_LENGTH]
    spec = np.fft.rfft(frames * window, n=_N_FFT, axis=-1)
    magnitudes = (np.abs(spec) ** 2).T.astype(np.float32)

    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # match WhisperFeatureExtractor's frame count
    return log_spec[np.newaxis, :, :].astype(np.float32)


class Qwen3AsrTranscriber(Transcriber):
    """Wrapper for a Qwen3-ASR ONNX export (encoder + split decoder)."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        cpu_threads: int = 4,
    ) -> None:
        """Initialize model."""
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    model_id,
                    cache_dir=str(Path(cache_dir).resolve()),
                    local_files_only=local_files_only,
                    allow_patterns=_ALLOW_PATTERNS,
                )
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_args = {
            "sess_options": options,
            "providers": ["CPUExecutionProvider"],
        }

        self._encoder = ort.InferenceSession(
            str(model_dir / "encoder.int4.onnx"), **session_args
        )
        self._decoder_init = ort.InferenceSession(
            str(model_dir / "decoder_init.int4.onnx"), **session_args
        )
        self._decoder_step = ort.InferenceSession(
            str(model_dir / "decoder_step.int4.onnx"), **session_args
        )

        with open(model_dir / "config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        # Keep the embedding table in fp16 and cast a single row per generated
        # token. Casting the whole table up front costs ~300MB of RSS for nothing.
        hidden_size = config["decoder"]["hidden_size"]
        self._embed_tokens = np.fromfile(
            model_dir / "embed_tokens.bin", dtype=np.float16
        ).reshape(-1, hidden_size)

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._mel_filters = _mel_filterbank()
        self._encoder_inputs = [inp.name for inp in self._encoder.get_inputs()]
        self._prompt_cache: Dict[Tuple[str, Optional[str]], List[int]] = {}

    def _context_ids(
        self, initial_prompt: Optional[str], language: Optional[str]
    ) -> Tuple[List[int], List[int]]:
        """Tokenize the biasing context and the forced-language suffix."""
        key = (initial_prompt or "", language)
        cached = self._prompt_cache.get(key)
        if cached is None:
            context = (
                self._tokenizer.encode(initial_prompt, add_special_tokens=False).ids
                if initial_prompt
                else []
            )
            language_name = qwen3_asr_language(language)
            suffix = (
                self._tokenizer.encode(
                    f"language {language_name}", add_special_tokens=False
                ).ids
                + [_ASR_TEXT]
                if language_name
                else []
            )
            # Cache as one list so a single dict lookup covers both.
            cached = [len(context)] + context + suffix
            self._prompt_cache[key] = cached

        n_context = cached[0]
        return cached[1 : 1 + n_context], cached[1 + n_context :]

    def _build_prompt_ids(
        self,
        n_audio_tokens: int,
        context_ids: Sequence[int],
        language_ids: Sequence[int],
    ) -> List[int]:
        """Build the chat-template prompt around the audio placeholders.

        <|im_start|>system\n{context}<|im_end|>\n
        <|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n
        <|im_start|>assistant\n{language {Name}<asr_text>}
        """
        return (
            [_IM_START, _SYSTEM, _NEWLINE, *context_ids, _IM_END, _NEWLINE]
            + [_IM_START, _USER, _NEWLINE, _AUDIO_START]
            + [_AUDIO_PAD] * n_audio_tokens
            + [_AUDIO_END, _IM_END, _NEWLINE]
            + [_IM_START, _ASSISTANT, _NEWLINE, *language_ids]
        )

    def _generate(self, audio_features: np.ndarray, prompt_ids: List[int]) -> List[int]:
        """Greedy decode: prefill the prompt, then step one token at a time."""
        logits, past_keys, past_values = self._decoder_init.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_ids": np.array(prompt_ids, dtype=np.int64)[np.newaxis, :],
                "position_ids": np.arange(len(prompt_ids), dtype=np.int64)[
                    np.newaxis, :
                ],
                "audio_features": audio_features,
                "audio_offset": np.array(
                    [prompt_ids.index(_AUDIO_PAD)], dtype=np.int64
                ),
            },
        )

        token = int(np.argmax(logits[0, -1, :]))
        tokens = [token]
        position = len(prompt_ids)

        while (token not in _EOS_IDS) and (len(tokens) < _MAX_NEW_TOKENS):
            logits, past_keys, past_values = self._decoder_step.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": self._embed_tokens[token].astype(np.float32)[
                        np.newaxis, np.newaxis, :
                    ],
                    "position_ids": np.array([[position]], dtype=np.int64),
                    "past_keys": past_keys,
                    "past_values": past_values,
                },
            )
            token = int(np.argmax(logits[0, -1, :]))
            tokens.append(token)
            position += 1

        return tokens

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio. Decoding is greedy, so
        beam_size is ignored. initial_prompt is used as biasing context rather
        than as a transcript prefix.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        mel = _log_mel_spectrogram(audio, self._mel_filters)

        encoder_inputs = {self._encoder_inputs[0]: mel}
        if len(self._encoder_inputs) > 1:
            encoder_inputs[self._encoder_inputs[1]] = np.array(
                [mel.shape[2]], dtype=np.int64
            )
        audio_features = self._encoder.run(None, encoder_inputs)[0]

        context_ids, language_ids = self._context_ids(initial_prompt, language)
        prompt_ids = self._build_prompt_ids(
            audio_features.shape[1], context_ids, language_ids
        )
        tokens = self._generate(audio_features, prompt_ids)

        # The model writes the detected language before <asr_text>; drop it.
        if _ASR_TEXT in tokens:
            tokens = tokens[tokens.index(_ASR_TEXT) + 1 :]

        return self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
