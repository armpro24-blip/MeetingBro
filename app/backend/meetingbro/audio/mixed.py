from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from .capture import AudioChunk, AudioSource, MicrophoneSource
from .loopback import SystemAudioLoopbackSource


class MixedAudioSource(AudioSource):
    """Mix microphone and system loopback into one mono stream.

    This is a pragmatic local-capture mode for online meetings where the user
    both listens to remote participants and speaks into the meeting.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        microphone_gain: float = 1.2,
        system_gain: float = 1.0,
        auto_balance_enabled: bool = True,
        max_microphone_boost: float = 1.8,
        microphone_activity_floor: float = 0.008,
        balance_smoothing: float = 0.35,
        microphone_device: "int | str | None" = None,
        speaker_name: "str | None" = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._microphone_gain = max(0.0, microphone_gain)
        self._system_gain = max(0.0, system_gain)
        self._auto_balance_enabled = auto_balance_enabled
        self._max_microphone_boost = max(1.0, max_microphone_boost)
        self._microphone_activity_floor = max(1e-5, microphone_activity_floor)
        self._balance_smoothing = min(1.0, max(0.0, balance_smoothing))
        self._smoothed_microphone_boost = 1.0
        self._mic = MicrophoneSource(sample_rate=sample_rate, chunk_seconds=chunk_seconds, device=microphone_device)
        self._loopback = SystemAudioLoopbackSource(sample_rate=sample_rate, chunk_seconds=chunk_seconds, speaker_name=speaker_name)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def microphone_gain(self) -> float:
        return self._microphone_gain

    @property
    def system_gain(self) -> float:
        return self._system_gain

    @property
    def auto_balance_enabled(self) -> bool:
        return self._auto_balance_enabled

    @property
    def effective_microphone_gain(self) -> float:
        return self._microphone_gain * self._smoothed_microphone_boost

    def drain_drops(self) -> int:
        return self._mic.drain_drops() + self._loopback.drain_drops()

    def _rms(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))

    def _effective_microphone_gain(self, mic_samples: np.ndarray, loopback_samples: np.ndarray) -> float:
        base_gain = self._microphone_gain
        if not self._auto_balance_enabled:
            return base_gain

        mic_rms = self._rms(mic_samples)
        loopback_rms = self._rms(loopback_samples)
        if mic_rms < self._microphone_activity_floor or loopback_rms <= 0.0:
            target_boost = 1.0
        else:
            target_boost = min(
                self._max_microphone_boost,
                max(1.0, np.sqrt(loopback_rms / max(mic_rms, 1e-6))),
            )
        smoothing = self._balance_smoothing
        self._smoothed_microphone_boost = (
            self._smoothed_microphone_boost * (1.0 - smoothing)
            + target_boost * smoothing
        )
        return base_gain * self._smoothed_microphone_boost

    async def stream(self) -> AsyncIterator[AudioChunk]:
        mic_iter = self._mic.stream().__aiter__()
        loopback_iter = self._loopback.stream().__aiter__()
        try:
            while True:
                mic_next = mic_iter.__anext__()
                loopback_next = loopback_iter.__anext__()
                mic_chunk, loopback_chunk = await asyncio.gather(mic_next, loopback_next)
                frame_count = max(len(mic_chunk.samples), len(loopback_chunk.samples))
                mic_samples = np.zeros(frame_count, dtype=np.float32)
                loopback_samples = np.zeros(frame_count, dtype=np.float32)
                mic_samples[: len(mic_chunk.samples)] = mic_chunk.samples
                loopback_samples[: len(loopback_chunk.samples)] = loopback_chunk.samples
                effective_microphone_gain = self._effective_microphone_gain(mic_samples, loopback_samples)
                gain_sum = effective_microphone_gain + self._system_gain
                if gain_sum <= 0.0:
                    mixed = np.zeros(frame_count, dtype=np.float32)
                else:
                    mixed = np.clip(
                        (
                            (mic_samples * effective_microphone_gain)
                            + (loopback_samples * self._system_gain)
                        ) / max(1.0, gain_sum),
                        -1.0,
                        1.0,
                    )
                yield AudioChunk(
                    samples=mixed,
                    sample_rate=self._sample_rate,
                    start_time=min(mic_chunk.start_time, loopback_chunk.start_time),
                )
        except StopAsyncIteration:
            return
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        await asyncio.gather(
            self._mic.aclose(),
            self._loopback.aclose(),
            return_exceptions=True,
        )