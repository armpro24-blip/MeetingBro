"""Neural speaker diarizer using sherpa-onnx speaker embeddings.

Replaces the spectral-centroid identity heuristic of :class:`EnergyDiarizer`
with a real speaker-embedding model (e.g. 3D-Speaker CAM++). Speech regions are
still found by the same silence-gap energy logic, but each region's *identity*
is decided by cosine-matching a neural embedding against speakers seen earlier
in the session — far more reliable than the centroid distance.

This keeps the per-window :class:`Diarizer` contract: a :class:`SpeakerEmbeddingManager`
accumulates speakers across ``diarize`` calls so labels are stable for the whole
meeting. The model loads lazily on first use; callers should fall back to
:class:`EnergyDiarizer` if construction/loading fails (no model downloaded).
"""
from __future__ import annotations

import logging

import numpy as np

from .base import Diarizer, DiarizationSegment

logger = logging.getLogger(__name__)

# Regions shorter than this are too short for a reliable speaker embedding.
_MIN_EMBED_SECONDS = 0.5
_MIN_SEGMENT_SECONDS = 0.3
_FRAME_SECONDS = 0.03


class NeuralDiarizer(Diarizer):
    """Speaker diarizer backed by a sherpa-onnx speaker-embedding model.

    Parameters
    ----------
    model_path
        Path to a sherpa-onnx speaker-embedding ONNX model.
    threshold
        Cosine-similarity threshold for matching a region to an existing
        speaker. Higher = stricter (more distinct speakers). 0.5 is a reasonable
        default for CAM++/3D-Speaker.
    num_threads, provider
        ONNX runtime settings.
    max_speakers
        Soft cap; beyond this, regions are assigned to the closest speaker.
    silence_gap_seconds, silence_rms_threshold
        Speech-region detection (mirrors EnergyDiarizer so behaviour is
        comparable; only the identity step changes).
    """

    def __init__(
        self,
        *,
        model_path: str,
        threshold: float = 0.5,
        num_threads: int = 1,
        provider: str = "cpu",
        max_speakers: int = 8,
        silence_gap_seconds: float = 0.8,
        silence_rms_threshold: float = 0.01,
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._num_threads = max(1, num_threads)
        self._provider = provider
        self._max_speakers = max_speakers
        self._silence_gap = silence_gap_seconds
        self._silence_rms = silence_rms_threshold
        self._extractor = None  # lazy
        self._manager = None
        self._next_speaker_idx = 1

    def _ensure_loaded(self):
        if self._extractor is None:
            import sherpa_onnx  # imported lazily; optional dependency

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self._model_path,
                num_threads=self._num_threads,
                provider=self._provider,
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
            logger.info(
                "loaded neural diarizer model=%s dim=%d threshold=%.2f",
                self._model_path, self._extractor.dim, self._threshold,
            )
        return self._extractor

    def ensure_loaded(self) -> None:
        """Eagerly materialize the model so load failures surface at startup."""
        self._ensure_loaded()

    def reset(self) -> None:
        # Drop accumulated speakers; rebuild the manager lazily.
        self._manager = None
        self._next_speaker_idx = 1
        if self._extractor is not None:
            import sherpa_onnx
            self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)

    def _embedding(self, samples: np.ndarray, sample_rate: int):
        extractor = self._ensure_loaded()
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return extractor.compute(stream)

    def _assign_speaker(self, embedding) -> str:
        manager = self._manager
        name = manager.search(embedding, threshold=self._threshold)
        if name:
            return name
        if manager.num_speakers >= self._max_speakers:
            # At the cap: pick the best existing match regardless of threshold.
            best = ""
            best_score = -1.0
            for existing in manager.all_speakers():
                sc = manager.verify(existing, embedding, threshold=-1.0)
                # verify returns bool; use score() for the similarity value.
                score = manager.score(existing, embedding) if hasattr(manager, "score") else 0.0
                if score > best_score:
                    best_score, best = score, existing
            return best or f"Speaker {self._next_speaker_idx}"
        name = f"Speaker {self._next_speaker_idx}"
        self._next_speaker_idx += 1
        manager.add(name, embedding)
        return name

    def diarize(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        offset_seconds: float = 0.0,
    ) -> list[DiarizationSegment]:
        if samples.size == 0:
            return []
        self._ensure_loaded()

        regions = _find_speech_regions(
            samples, sample_rate,
            silence_rms=self._silence_rms,
            silence_gap_seconds=self._silence_gap,
        )
        if not regions:
            return []

        result: list[DiarizationSegment] = []
        for start_sec, end_sec in regions:
            if (end_sec - start_sec) < _MIN_SEGMENT_SECONDS:
                continue
            start_sample = int(start_sec * sample_rate)
            end_sample = min(int(end_sec * sample_rate), len(samples))
            region = samples[start_sample:end_sample]
            if (end_sec - start_sec) < _MIN_EMBED_SECONDS:
                # Too short for a reliable embedding; skip identity, leave to ASR.
                continue
            try:
                embedding = self._embedding(region.astype(np.float32), sample_rate)
            except Exception as exc:  # pragma: no cover - runtime/model issues
                logger.debug("neural diarizer embedding failed: %s", exc)
                continue
            if embedding is None:
                continue
            speaker = self._assign_speaker(embedding)
            result.append(DiarizationSegment(
                start_time=offset_seconds + start_sec,
                end_time=offset_seconds + end_sec,
                speaker_label=speaker,
                confidence=0.7,  # neural identity — more reliable than energy heuristic
            ))
        return result


def _find_speech_regions(
    samples: np.ndarray,
    sample_rate: int,
    *,
    silence_rms: float,
    silence_gap_seconds: float,
) -> list[tuple[float, float]]:
    """Contiguous speech regions separated by silence gaps (adaptive threshold)."""
    frame_len = max(1, int(sample_rate * _FRAME_SECONDS))
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return []
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    noise_floor = float(np.percentile(rms, 10))
    threshold = max(silence_rms, noise_floor * 2.5)
    is_speech = rms > threshold

    regions: list[tuple[float, float]] = []
    in_speech = False
    start_frame = 0
    silence_frames = 0
    gap_frames = int(silence_gap_seconds * sample_rate / frame_len)
    for i, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                start_frame = i
                in_speech = True
            silence_frames = 0
        elif in_speech:
            silence_frames += 1
            if silence_frames >= gap_frames:
                end_frame = i - silence_frames + 1
                start_sec = start_frame * frame_len / sample_rate
                end_sec = end_frame * frame_len / sample_rate
                if end_sec > start_sec:
                    regions.append((start_sec, end_sec))
                in_speech = False
                silence_frames = 0
    if in_speech:
        start_sec = start_frame * frame_len / sample_rate
        end_sec = len(is_speech) * frame_len / sample_rate
        if end_sec > start_sec:
            regions.append((start_sec, end_sec))
    return regions
