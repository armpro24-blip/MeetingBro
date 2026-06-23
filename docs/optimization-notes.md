# OPTIMIZATION_NOTES.md

Behaviour changes introduced by the C1–C10 fixes (ordered by priority).

> For the later 2026-06 accuracy + latency optimization round (benchmark harness,
> vocabulary biasing, neural diarization, preview lane), see
> [docs/optimization-2026-06.md](optimization-2026-06.md).

---

## P0 — Must-fix

### C1 · Rolling/Cumulative summary concurrency (session/manager.py)
**Before:** `_maybe_emit_rolling` and `_maybe_emit_cumulative` had no guard.
Every ASR chunk triggered `_schedule_summary(...)`, so multiple background tasks
could call the LLM simultaneously, multiplying token spend and causing race-updates
on `last_*_at`.
**After:** `rolling_in_flight` / `cumulative_in_flight` flags on `_State`; entering
either function sets the flag, `finally` resets it.  At most one call per type is
in-flight at any time.  Mirrors the existing `memory_in_flight` pattern.

### C2 · Resampler anti-aliasing (audio/capture.py)
**Before:** `_resample_mono` used `np.interp` (linear interpolation, no anti-alias
filter).  Downsampling 48 kHz → 16 kHz folded 8–24 kHz energy back into 0–8 kHz,
adding noise that hurt Whisper accuracy.
**After:** `scipy.signal.resample_poly` which applies an FIR anti-aliasing filter
before decimation.  Function signature and zero-length guard unchanged.

### C3 · Adaptive diarizer silence threshold (diarization/energy.py)
**Before:** `_SILENCE_RMS_THRESHOLD = 0.01` was a global constant.  High-level
loopback audio sat entirely above it (all speech, no speaker changes detected);
low-level far-field mic sat below it (all silence, no speakers assigned).
**After:** `EnergyDiarizer.diarize` computes the P10 percentile of all frame RMS
values as `noise_floor`, then uses `max(silence_rms_threshold, noise_floor * 2.5)`
as the per-call threshold.  The constructor parameter is now a floor, not a ceiling.

---

## P1 — Should-fix

### C4 · Event queue bounded (session/manager.py)
**Before:** `asyncio.Queue()` with no maxsize — a slow or disconnected frontend
caused unbounded memory growth.
**After:** `maxsize=1024`.  `_emit` checks fullness: `transcript_segment` events are
silently dropped; `summary_snapshot`, `error`, `session_state`, `speaker_update` are
always queued.  Every 50th dropped event triggers an `error` event notifying the
consumer of backpressure.

### C5 · Language vote stickiness (session/manager.py)
**Before:** Each ASR chunk was independently auto-detected.  Noise/short chunks
caused spurious `zh→unknown→en→zh` oscillations, triggering inconsistent translation.
**After:** When `forced_language` is `None`, the first 5 non-unknown batch detections
are tallied; the majority language locks as an implicit `forced_language` for
subsequent ASR calls.  Unlock requires 3 consecutive batches with a different
language at confidence > 0.8.  If the user explicitly set a language, this logic is
bypassed entirely.

### C6 · `sample_rate` public property (audio/capture.py, audio/loopback.py, session/manager.py)
**Before:** `session/manager.py` accessed `audio_source._sample_rate` via
`hasattr(…, '_sample_rate')` — fragile, private-attribute access.
**After:** `AudioSource` ABC exposes an abstract `sample_rate` property.
`MicrophoneSource`, `WavFileSource`, `SystemAudioLoopbackSource` each implement it.
`_run`'s final flush now calls `self._cfg.audio_source.sample_rate`.

### C7 · `stop()` task-cancel race (session/manager.py)
**Before:** `stop()` awaited `aclose()` (blocking), then `wait_for(task, timeout=5.0)`.
A 5 s budget was too tight for a `base`/`small` Whisper call; the task was cancelled
before ASR could return its result, silently discarding the last segment.
**After:** `aclose()` is fired as a non-blocking `create_task`; `wait_for(task, 15.0)`
gives ASR time to complete; a `WARNING` log is emitted on cancel so dropped segments
are observable.

---

## P2 — Nice-to-have

### C8 · Accumulation buffer hard cap (session/manager.py)
**Before:** If `asr_accumulation_seconds` was set very large (or small chunks
accumulated faster than expected), `accum_samples` could grow without bound.
**After:** If `accum_duration > 4 × asr_accumulation_seconds` the buffer is
force-flushed with an `asr_buffer_overflow` error event.

### C9 · Default Whisper model `small` (main.py)
**Before:** Default `MEETINGBRO_WHISPER_SIZE=base`.  Noticeably lower accuracy on
Chinese, German, and proper nouns.
**After:** Default `small`.  On a mid-range CPU (i5/Ryzen 5) `small` runs at ≈2×
real-time — adequate for the 2.5 s ASR accumulation window.  Override with
`MEETINGBRO_WHISPER_SIZE=base` if CPU budget is tight.

### C10 · Audio drop telemetry (audio/capture.py, audio/loopback.py, session/manager.py)
**Before:** Queue-full drops were only logged as `WARNING` with no observable count.
**After:** Each source carries a thread-safe `_drop_count` counter incremented on
every drop; `drain_drops()` reads and resets it atomically.  `SessionManager._run`
drains this counter after each ASR cycle and emits an `error` event with
`code="audio_drop"` every 10 cumulative dropped chunks, surfacing capture overruns
in the UI.
