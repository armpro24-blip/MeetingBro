"""Audio capture abstractions.

Supported sources for the MVP vertical slice:

- ``MicrophoneSource``  — Windows / macOS / Linux microphone via ``sounddevice``.
  Offline / in-person meeting mode. This is the **first targeted OS path** for
  the MVP; system-audio loopback (online meeting mode) lands in a later phase.

- ``WavFileSource`` — replays a WAV file as an audio stream. Used for the
  deterministic end-to-end vertical-slice test and for dev without a real mic.

Both sources yield fixed-duration ``AudioChunk`` objects of mono ``float32`` PCM
at a configurable sample rate (default 16 kHz to match Whisper).
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A fixed-duration slice of mono float32 PCM."""

    samples: np.ndarray  # shape (N,), dtype float32, range ~[-1, 1]
    sample_rate: int
    start_time: float  # seconds since session start
    enqueue_wall_time: Optional[float] = None  # monotonic timestamp for queue-latency diagnostics


class AudioSource(ABC):
    """Abstract audio source. Implementations stream ``AudioChunk``s."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Target sample rate of emitted chunks (Hz)."""
        ...

    @abstractmethod
    async def stream(self) -> AsyncIterator[AudioChunk]:
        ...

    def drain_drops(self) -> int:
        """Return and reset the cumulative drop count (chunks discarded due to full queue).

        Thread-safe.  Default returns 0 (sources without a queue never drop).
        """
        return 0

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None


def _resample_mono(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return np.zeros(0, dtype=np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_rate, dst_rate)
    return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)


class MicrophoneSource(AudioSource):
    """Microphone capture via ``sounddevice``.

    ``chunk_seconds`` controls how often the source emits an ``AudioChunk``. The
    default (3.0 s) is a balance between ASR quality and UI latency.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        device: Optional[int | str] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._device = device
        self._stop = threading.Event()
        # Shared queue reference so aclose() can inject a sentinel to unblock q.get().
        self._q: Optional[queue.Queue] = None
        self._drop_lock = threading.Lock()
        self._drop_count: int = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def drain_drops(self) -> int:
        with self._drop_lock:
            n = self._drop_count
            self._drop_count = 0
        return n

    async def stream(self) -> AsyncIterator[AudioChunk]:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            import sys
            extra_hint = ""
            if sys.platform.startswith("linux"):
                extra_hint = (
                    " On Linux, install PortAudio runtime/dev packages first, "
                    "for example: `sudo apt install portaudio19-dev libportaudio2` "
                    "then reinstall sounddevice in your venv."
                )
            raise RuntimeError(
                "sounddevice is required for MicrophoneSource."
                + extra_hint
            ) from exc

        loop = asyncio.get_running_loop()
        q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=32)
        self._q = q
        chunk_frames = int(self._sample_rate * self._chunk_seconds)

        def _callback(indata, frames, time_info, status):  # noqa: ARG001 - sd API
            if status:
                logger.debug("audio status: %s", status)
            # indata shape: (frames, channels). We force mono by averaging.
            mono = indata.mean(axis=1).astype(np.float32, copy=False).copy()
            try:
                q.put_nowait(mono)
            except queue.Full:
                with self._drop_lock:
                    self._drop_count += 1
                logger.warning("audio queue full — dropping chunk")

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=self._device,
            callback=_callback,
        )
        stream.start()
        logger.info(
            "mic capture started rate=%d chunk_frames=%d device=%s",
            self._sample_rate, chunk_frames, self._device,
        )
        t0 = 0.0
        try:
            while not self._stop.is_set():
                mono = await loop.run_in_executor(None, q.get)
                if mono is None:  # sentinel injected by aclose()
                    break
                yield AudioChunk(samples=mono, sample_rate=self._sample_rate, start_time=t0)
                t0 += len(mono) / self._sample_rate
        finally:
            stream.stop()
            stream.close()
            self._q = None
            logger.info("mic capture stopped")

    async def aclose(self) -> None:
        self._stop.set()
        # Unblock any pending q.get() in the stream iterator so it can exit cleanly.
        if self._q is not None:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass


class WavFileSource(AudioSource):
    """Replay a WAV file as if it were a live stream."""

    def __init__(
        self,
        path: Path,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        realtime: bool = False,
    ) -> None:
        self._path = path
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._realtime = realtime

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def stream(self) -> AsyncIterator[AudioChunk]:
        import soundfile as sf

        with sf.SoundFile(str(self._path), mode="r") as f:
            src_rate = f.samplerate
            samples = f.read(dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = _resample_mono(samples, src_rate, self._sample_rate)

        chunk_frames = int(self._sample_rate * self._chunk_seconds)
        t0 = 0.0
        for start in range(0, len(samples), chunk_frames):
            chunk = samples[start : start + chunk_frames]
            if len(chunk) == 0:
                break
            yield AudioChunk(samples=chunk, sample_rate=self._sample_rate, start_time=t0)
            t0 += len(chunk) / self._sample_rate
            if self._realtime:
                await asyncio.sleep(len(chunk) / self._sample_rate)
