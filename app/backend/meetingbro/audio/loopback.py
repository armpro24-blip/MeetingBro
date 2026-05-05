"""Cross-platform system-audio loopback capture.

Captures what the default output device is currently rendering — i.e. the audio
stream the user hears from Teams / Zoom / BBB / a browser tab — and emits it
through the same :class:`AudioSource` interface the rest of the pipeline uses.

Online-meeting mode: platform-specific loopback paths.

- Windows: WASAPI loopback via ``soundcard``.
- Linux: PulseAudio/PipeWire monitor capture via ``parec`` / ``pw-cat``.
- macOS: virtual-loopback input devices (for example BlackHole) via
  ``sounddevice``.

Implementation notes
--------------------
We use the `soundcard` package, which exposes each output device as a loopback
*microphone* on Windows (``soundcard.all_microphones(include_loopback=True)``).
``sounddevice`` 0.5.x does not expose the WASAPI loopback flag in its
``WasapiSettings`` binding, so we keep the mic / WAV paths on ``sounddevice``
and use ``soundcard`` only for the loopback path.

The recorder yields its native rate (typically 48 kHz stereo float32). We
downmix to mono and resample to the session's target rate (16 kHz by default)
via the shared ``_resample_mono`` helper so the downstream ASR path is
identical to mic / WAV sources.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import sys
import threading
from typing import AsyncIterator, Optional

import numpy as np
import shutil
import subprocess

from .capture import AudioChunk, AudioSource, _resample_mono

logger = logging.getLogger(__name__)


class SystemAudioLoopbackSource(AudioSource):
    """System loopback capture of meeting audio output.

    Parameters
    ----------
    sample_rate
        Target sample rate for emitted chunks (Hz). Resampled from the device's
        native rate. Defaults to 16 kHz to match Whisper.
    chunk_seconds
        Duration of each emitted ``AudioChunk`` in seconds.
    speaker_name
        Optional device hint. On Windows this is matched against the speaker
        name (WASAPI loopback). On macOS this is matched against the virtual
        loopback input device name (for example "BlackHole 2ch").
    native_sample_rate
        Capture rate requested from the loopback device. Most Windows output
        devices run at 48 kHz; that's the default here.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        speaker_name: Optional[str] = None,
        native_sample_rate: int = 48_000,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._speaker_name = speaker_name
        self._native_rate = native_sample_rate
        self._stop = threading.Event()
        self._drop_lock = threading.Lock()
        self._drop_count: int = 0

        # On Windows keep the existing soundcard-based implementation.
        # On Linux try to use PulseAudio / PipeWire tools (parec / pw-cat).
        if sys.platform.startswith("win"):
            # unchanged behavior: Windows-specific reader implemented below
            return
        if sys.platform.startswith("linux"):
            # delegate to a lightweight Linux implementation
            self._linux_impl = _LinuxLoopbackImpl(
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
                native_sample_rate=native_sample_rate,
            )
            return
        if sys.platform == "darwin":
            self._mac_impl = _MacLoopbackImpl(
                sample_rate=sample_rate,
                chunk_seconds=chunk_seconds,
                native_sample_rate=native_sample_rate,
                device_name=speaker_name,
            )
            return
        raise RuntimeError(
            "SystemAudioLoopbackSource currently supports Windows, macOS, and Linux only."
        )

    @property
    def sample_rate(self) -> int:
        # If on Linux delegate to implementation
        if hasattr(self, "_linux_impl"):
            return self._linux_impl.sample_rate
        if hasattr(self, "_mac_impl"):
            return self._mac_impl.sample_rate
        return self._sample_rate

    def drain_drops(self) -> int:
        if hasattr(self, "_linux_impl"):
            return self._linux_impl.drain_drops()
        if hasattr(self, "_mac_impl"):
            return self._mac_impl.drain_drops()
        with self._drop_lock:
            n = self._drop_count
            self._drop_count = 0
        return n

    def _resolve_loopback_mic(self, sc):
        if self._speaker_name is not None:
            return sc.get_microphone(str(self._speaker_name), include_loopback=True)
        default_speaker = sc.default_speaker()
        # Match the loopback mic whose name corresponds to the default speaker.
        mics = sc.all_microphones(include_loopback=True)
        target = default_speaker.name
        for m in mics:
            if m.isloopback and m.name == target:
                return m
        # Fallback: first loopback mic we find.
        for m in mics:
            if m.isloopback:
                return m
        raise RuntimeError("no WASAPI loopback device found")

    async def stream(self) -> AsyncIterator[AudioChunk]:
        # If Linux implementation is present, yield from it.
        if hasattr(self, "_linux_impl"):
            async for c in self._linux_impl.stream():
                yield c
            return
        if hasattr(self, "_mac_impl"):
            async for c in self._mac_impl.stream():
                yield c
            return

        # Windows implementation (unchanged)
        try:
            import soundcard as sc
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "soundcard is required for SystemAudioLoopbackSource"
            ) from exc

        mic = self._resolve_loopback_mic(sc)
        native_rate = self._native_rate
        native_block = max(1, int(native_rate * self._chunk_seconds))

        loop = asyncio.get_running_loop()
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        stop = self._stop

        def _reader() -> None:
            import ctypes
            _ole32 = ctypes.WinDLL("ole32", use_last_error=True)
            COINIT_MULTITHREADED = 0x0
            _hr = _ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
            _com_owned = _hr in (0, 1)
            try:
                with mic.recorder(samplerate=native_rate, blocksize=native_block) as rec:
                    logger.info(
                        "loopback capture started mic=%s native_rate=%d block=%d "
                        "target_rate=%d",
                        mic, native_rate, native_block, self._sample_rate,
                    )
                    while not stop.is_set():
                        data = rec.record(numframes=native_block)
                        if data is None or len(data) == 0:
                            continue
                        if data.ndim > 1:
                            mono = data.mean(axis=1).astype(np.float32, copy=False)
                        else:
                            mono = data.astype(np.float32, copy=False)
                        try:
                            q.put_nowait(mono.copy())
                        except queue.Full:
                            with self._drop_lock:
                                self._drop_count += 1
                            logger.warning("loopback queue full — dropping chunk")
            except Exception:
                logger.exception("loopback reader thread crashed")
            finally:
                if _com_owned:
                    _ole32.CoUninitialize()
                q.put(None)  # type: ignore[arg-type]

        reader = threading.Thread(target=_reader, name="loopback-reader", daemon=True)
        reader.start()

        t0 = 0.0
        try:
            while not stop.is_set():
                native_samples = await loop.run_in_executor(None, lambda: q.get())
                if native_samples is None:
                    break
                resampled = _resample_mono(
                    native_samples, native_rate, self._sample_rate
                )
                if len(resampled) == 0:
                    continue
                yield AudioChunk(
                    samples=resampled,
                    sample_rate=self._sample_rate,
                    start_time=t0,
                )
                t0 += len(resampled) / self._sample_rate
        finally:
            stop.set()
            reader.join(timeout=2.0)
            logger.info("loopback capture stopped")

    async def aclose(self) -> None:
        # Delegate to linux impl if present
        if hasattr(self, "_linux_impl"):
            await self._linux_impl.aclose()
            return
        if hasattr(self, "_mac_impl"):
            await self._mac_impl.aclose()
            return
        self._stop.set()


class _MacLoopbackImpl:
    """macOS system-audio capture from a virtual loopback input device.

    macOS does not expose direct system-output loopback by default. This path
    expects a virtual driver (for example BlackHole/Soundflower/Loopback) and
    records from that input endpoint using ``sounddevice``.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_seconds: float = 3.0,
        native_sample_rate: int = 48_000,
        device_name: Optional[str] = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._native_rate = native_sample_rate
        self._device_name = device_name
        self._stop = threading.Event()
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

    def _resolve_device(self, sd) -> tuple[int, str, int]:
        devices = sd.query_devices()
        hint = (self._device_name or "").strip().lower()

        if hint:
            for index, dev in enumerate(devices):
                max_inputs = int(dev.get("max_input_channels", 0) or 0)
                name = str(dev.get("name", ""))
                if max_inputs > 0 and hint in name.lower():
                    return index, name, max_inputs
            raise RuntimeError(
                f"No macOS loopback input device matching '{self._device_name}' was found."
            )

        preferred_markers = ("blackhole", "soundflower", "loopback", "vb-cable", "ishowu")
        for index, dev in enumerate(devices):
            max_inputs = int(dev.get("max_input_channels", 0) or 0)
            name = str(dev.get("name", ""))
            if max_inputs > 0 and any(marker in name.lower() for marker in preferred_markers):
                return index, name, max_inputs

        raise RuntimeError(
            "No macOS loopback input device found. Install a virtual audio driver "
            "(for example BlackHole), route system output into it, then retry."
        )

    async def stream(self) -> AsyncIterator[AudioChunk]:
        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("sounddevice is required for macOS loopback capture") from exc

        loop = asyncio.get_running_loop()
        q: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=32)
        self._q = q

        device_id, device_name, max_inputs = self._resolve_device(sd)
        channels = 2 if max_inputs >= 2 else 1
        native_block = max(1, int(self._native_rate * self._chunk_seconds))

        def _callback(indata, frames, time_info, status):  # noqa: ARG001 - sd API
            if status:
                logger.debug("mac loopback status: %s", status)
            if indata is None or len(indata) == 0:
                return
            mono = indata.mean(axis=1).astype(np.float32, copy=False).copy()
            try:
                q.put_nowait(mono)
            except queue.Full:
                with self._drop_lock:
                    self._drop_count += 1
                logger.warning("mac loopback queue full - dropping chunk")

        stream = sd.InputStream(
            samplerate=self._native_rate,
            channels=channels,
            dtype="float32",
            blocksize=native_block,
            device=device_id,
            callback=_callback,
        )
        stream.start()
        logger.info(
            "mac loopback started device=%s rate=%d block=%d target_rate=%d",
            device_name,
            self._native_rate,
            native_block,
            self._sample_rate,
        )

        t0 = 0.0
        try:
            while not self._stop.is_set():
                native_samples = await loop.run_in_executor(None, lambda: q.get())
                if native_samples is None:
                    break
                resampled = _resample_mono(native_samples, self._native_rate, self._sample_rate)
                if len(resampled) == 0:
                    continue
                yield AudioChunk(samples=resampled, sample_rate=self._sample_rate, start_time=t0)
                t0 += len(resampled) / self._sample_rate
        finally:
            stream.stop()
            stream.close()
            self._q = None
            logger.info("mac loopback stopped")

    async def aclose(self) -> None:
        self._stop.set()
        if self._q is not None:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass


class _LinuxLoopbackImpl:
    """Simple Linux system-audio capture using `parec` (PulseAudio) or `pw-cat` (PipeWire).

    This implementation shells out to a small helper process and reads raw PCM
    frames from stdout. It produces mono float32 numpy arrays to match the
    rest of the pipeline.
    """

    def __init__(self, *, sample_rate: int = 16_000, chunk_seconds: float = 3.0, native_sample_rate: int = 48_000) -> None:
        self._sample_rate = sample_rate
        self._chunk_seconds = chunk_seconds
        self._native_rate = native_sample_rate
        self._stop = threading.Event()
        self._drop_lock = threading.Lock()
        self._drop_count: int = 0
        self._proc: Optional[subprocess.Popen] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def drain_drops(self) -> int:
        with self._drop_lock:
            n = self._drop_count
            self._drop_count = 0
        return n

    def _select_cmd(self) -> list[str]:
        # Prefer PulseAudio's parec, fall back to pw-cat for PipeWire
        if shutil.which("parec"):
            return [
                "parec",
                "--device=@DEFAULT_MONITOR@",
                "--format=s16le",
                "--rate=%d" % self._native_rate,
                "--channels=2",
            ]
        if shutil.which("pw-cat"):
            # pw-cat: output raw s16le to stdout
            return [
                "pw-cat",
                "--record",
                "--channels=2",
                "--format=s16le",
                "--rate=%d" % self._native_rate,
            ]
        raise RuntimeError(
            "No suitable recorder found: install `pulseaudio-utils` (parec) or `pipewire-utils` (pw-cat)"
        )

    async def stream(self) -> AsyncIterator[AudioChunk]:
        native_rate = self._native_rate
        native_block = max(1, int(native_rate * self._chunk_seconds))
        bytes_per_frame = 2 * 2  # 2 channels * 2 bytes (s16le)
        read_bytes = native_block * bytes_per_frame

        loop = asyncio.get_running_loop()
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)

        def _reader() -> None:
            cmd = self._select_cmd()
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                self._proc = proc
                logger.info("linux loopback started cmd=%s native_rate=%d block=%d target_rate=%d", cmd, native_rate, native_block, self._sample_rate)
                stdout = proc.stdout
                assert stdout is not None
                while not self._stop.is_set():
                    data = stdout.read(read_bytes)
                    if not data:
                        break
                    # Interpret as s16le interleaved stereo
                    arr = np.frombuffer(data, dtype=np.int16)
                    if arr.size == 0:
                        continue
                    if arr.size % 2 != 0:
                        arr = arr[:-1]
                    arr = arr.reshape(-1, 2)
                    mono = (arr.mean(axis=1) / 32768.0).astype(np.float32, copy=False)
                    try:
                        q.put_nowait(mono.copy())
                    except queue.Full:
                        with self._drop_lock:
                            self._drop_count += 1
                        logger.warning("linux loopback queue full — dropping chunk")
            except Exception:
                logger.exception("linux loopback reader crashed")
            finally:
                if self._proc:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
                q.put(None)  # type: ignore[arg-type]

        reader = threading.Thread(target=_reader, name="linux-loopback-reader", daemon=True)
        reader.start()

        t0 = 0.0
        try:
            while not self._stop.is_set():
                native_samples = await loop.run_in_executor(None, lambda: q.get())
                if native_samples is None:
                    break
                resampled = _resample_mono(native_samples, native_rate, self._sample_rate)
                if len(resampled) == 0:
                    continue
                yield AudioChunk(samples=resampled, sample_rate=self._sample_rate, start_time=t0)
                t0 += len(resampled) / self._sample_rate
        finally:
            self._stop.set()
            reader.join(timeout=2.0)
            logger.info("linux loopback stopped")

    async def aclose(self) -> None:
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
