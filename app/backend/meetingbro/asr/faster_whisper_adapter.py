from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..schemas import OriginalLanguage
from .base import ASRAdapter, ASRSegment

logger = logging.getLogger(__name__)

_SUPPORTED: tuple[OriginalLanguage, ...] = ("zh", "en", "de")


def _normalize_language(code: Optional[str]) -> OriginalLanguage:
    if not code:
        return "unknown"
    code = code.lower()
    if code in _SUPPORTED:
        return code  # type: ignore[return-value]
    return "unknown"


class FasterWhisperAdapter(ASRAdapter):
    """ASR via faster-whisper.

    Auto-detect is used when ``forced_language`` is ``None``. The first
    targeted first-class languages are Chinese, English, and German, matching
    the product requirements.
    """

    def __init__(
        self,
        *,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        cpu_threads: int = 0,
        num_workers: int = 1,
        vad_threshold: float = 0.3,
        vad_min_speech_ms: int = 100,
        vad_min_silence_ms: int = 300,
        vad_speech_pad_ms: int = 400,
        multilingual: bool = False,
        language_detection_threshold: float = 0.5,
        language_detection_segments: int = 1,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._beam_size = beam_size
        self._cpu_threads = max(0, cpu_threads)
        self._num_workers = max(1, num_workers)
        self._vad_threshold = vad_threshold
        self._vad_min_speech_ms = max(0, vad_min_speech_ms)
        self._vad_min_silence_ms = max(0, vad_min_silence_ms)
        self._vad_speech_pad_ms = max(0, vad_speech_pad_ms)
        self._multilingual = multilingual
        self._language_detection_threshold = language_detection_threshold
        self._language_detection_segments = max(1, language_detection_segments)
        self._model = None  # lazy
        self._oom_fallback_activated = False

    def _compute_type_fallback_chain(self) -> list[str]:
        preferred = (self._compute_type or "").strip().lower() or "int8"
        device = (self._device or "cpu").strip().lower()

        if device == "cpu":
            # float16 on CPU is often unsupported/slow; prefer int8/float32 fallback.
            order = [preferred, "int8", "float32"]
        else:
            order = [preferred, "float16", "int8", "float32"]

        seen: set[str] = set()
        chain: list[str] = []
        for item in order:
            if item not in seen:
                seen.add(item)
                chain.append(item)
        return chain

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            last_exc: Exception | None = None
            for idx, compute_type in enumerate(self._compute_type_fallback_chain()):
                if idx == 0:
                    logger.info(
                        "loading faster-whisper model size=%s device=%s compute=%s cpu_threads=%d num_workers=%d",
                        self._model_size,
                        self._device,
                        compute_type,
                        self._cpu_threads,
                        self._num_workers,
                    )
                else:
                    logger.warning(
                        "retrying faster-whisper model load with fallback compute=%s device=%s",
                        compute_type,
                        self._device,
                    )

                try:
                    self._model = WhisperModel(
                        self._model_size,
                        device=self._device,
                        compute_type=compute_type,
                        cpu_threads=self._cpu_threads,
                        num_workers=self._num_workers,
                    )
                    if compute_type != self._compute_type:
                        logger.warning(
                            "faster-whisper compute fallback active: %s -> %s on device=%s",
                            self._compute_type,
                            compute_type,
                            self._device,
                        )
                        self._compute_type = compute_type
                    break
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    # Continue only for known compute-type incompatibility.
                    if not (
                        "float16" in msg
                        or "compute type" in msg
                        or "not support" in msg
                        or "unsupported" in msg
                        or "efficient" in msg
                    ):
                        raise
            if self._model is None and last_exc is not None:
                raise last_exc
        return self._model

    @staticmethod
    def _is_cuda_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda failed" in msg and "memory" in msg
            or "cublas" in msg and "alloc" in msg
        )

    def _maybe_activate_oom_fallback(self, exc: Exception) -> bool:
        """Switch CUDA Whisper to CPU/int8 once when we hit OOM.

        This keeps sessions alive on low-VRAM machines instead of repeatedly
        dropping chunks after the first CUDA OOM.
        """
        if self._oom_fallback_activated:
            return False
        if self._device.lower() != "cuda":
            return False
        if not self._is_cuda_oom_error(exc):
            return False

        self._oom_fallback_activated = True
        logger.warning(
            "faster-whisper CUDA OOM detected; falling back to CPU/int8 for the rest of this session"
        )
        self._device = "cpu"
        self._compute_type = "int8"
        self._model = None
        return True

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        forced_language: Optional[str] = None,
        offset_seconds: float = 0.0,
        initial_prompt: Optional[str] = None,
        quality_preset: str = "realtime",
    ) -> list[ASRSegment]:
        if samples.size == 0:
            return []
        if sample_rate != 16_000:
            raise ValueError(
                f"FasterWhisperAdapter expects 16 kHz input, got {sample_rate}"
            )
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        model = self._ensure_model()
        transcribe_kwargs = dict(
            language=forced_language,
            initial_prompt=initial_prompt or None,
            vad_filter=True,
            vad_parameters=dict(
                threshold=self._vad_threshold,
                min_speech_duration_ms=self._vad_min_speech_ms,
                min_silence_duration_ms=self._vad_min_silence_ms,
                speech_pad_ms=self._vad_speech_pad_ms,
            ),
            temperature=0.0,
            condition_on_previous_text=False,
            no_repeat_ngram_size=3,
            word_timestamps=False,
            multilingual=self._multilingual,
            language_detection_threshold=self._language_detection_threshold,
            language_detection_segments=self._language_detection_segments,
        )
        if quality_preset == "retry":
            transcribe_kwargs.update(
                beam_size=max(self._beam_size, 5),
                best_of=max(self._beam_size, 5),
                compression_ratio_threshold=2.8,
                no_speech_threshold=0.45,
            )
        else:
            transcribe_kwargs.update(
                beam_size=self._beam_size,
                best_of=1,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.4,
            )

        try:
            segments_iter, info = model.transcribe(samples, **transcribe_kwargs)
        except Exception as exc:
            if not self._maybe_activate_oom_fallback(exc):
                raise
            model = self._ensure_model()
            segments_iter, info = model.transcribe(samples, **transcribe_kwargs)
        detected_language = forced_language or info.language
        lang = _normalize_language(detected_language)

        out: list[ASRSegment] = []
        for s in segments_iter:
            text = (s.text or "").strip()
            if not text:
                continue
            # Whisper avg_logprob is usually negative even for decent segments, so
            # a plain sigmoid(avg_logprob) collapses most normal speech into the
            # low-confidence bucket. Re-center and steepen it to get a more useful
            # rough UI confidence signal.
            avg_logprob = getattr(s, "avg_logprob", None)
            normalized_logprob = float(avg_logprob) if avg_logprob is not None else -0.7
            conf = 1.0 / (1.0 + math.exp(-3.5 * (normalized_logprob + 0.7)))
            out.append(
                ASRSegment(
                    start_time=offset_seconds + float(s.start),
                    end_time=offset_seconds + float(s.end),
                    text=text,
                    language=lang,
                    confidence=max(0.0, min(1.0, conf)),
                    avg_logprob=float(avg_logprob) if avg_logprob is not None else None,
                    no_speech_prob=(
                        float(getattr(s, "no_speech_prob"))
                        if getattr(s, "no_speech_prob", None) is not None
                        else None
                    ),
                    compression_ratio=(
                        float(getattr(s, "compression_ratio"))
                        if getattr(s, "compression_ratio", None) is not None
                        else None
                    ),
                )
            )
        return out
