# Design: Streaming Whisper caption lane (Phase 1)

Date: 2026-06-23
Status: approved (design) — pending spec review

## Problem

In `balanced` mode the Qwen3 **preview lane** gives ~1 s captions, but it
requires an optional ~700 MB model the README tells most users to skip. So the
Whisper-only majority gets captions *only* from the formal lane, whose latency
the frontier sweep measured at **~3.8 s median / ~7 s tail (p95)** on real
meeting audio — and the same sweep proved that median is **structural** (no
buffered-segmentation knob moves it; see
[docs/benchmarks/frontier-latency-accuracy-notes.md](../../benchmarks/frontier-latency-accuracy-notes.md)).

The only lever for the median is to stop waiting for whole segments: decode
incrementally and commit text as speech grows.

## Goal

Give the Whisper-only majority **accurate, low-latency captions from the model
they already have**, by adding a streaming-decode caption lane backed by the
existing formal Whisper model. No new model dependency.

### Success criteria (gated by the frontier harness)

Measured on the AMI fixture via an extension of
`scripts/benchmark_latency_accuracy_frontier.py`:

- Committed-caption **p50 well under the ~3.8 s formal median — target ~1–1.5 s**.
- **WER of the committed stream not materially worse than the formal lane**
  (the committed text is what the user reads live; it must stay trustworthy).
- **ASR RTF stays < 1 on target hardware** with both lanes running (no sustained
  resource-governor backpressure).
- The no-extra-model path works (streaming lane runs with only the formal
  Whisper model loaded).

## Non-goals (Phase 1)

- Making streaming commits the **authoritative** transcript. The formal lane
  remains the source of truth for stored transcript, summaries, and translation.
  (Authoritative streaming = Phase 2.)
- Retiring the Qwen3 preview lane (= Phase 2/D, only if streaming proves
  superior).
- Any frontend/UI change beyond reusing the existing preview rendering.
- Changing the formal lane's decode path.

## Scope decision

**Target A (captions for the Whisper-only majority), built additively, with a
path to D (one streaming lane for everyone).** A new lane behind a flag — not a
rewrite — honoring the repo's incremental ethos. The formal lane is untouched.

## Approach (chosen)

**LocalAgreement-2 on the formal Whisper model.** Re-decode the growing
unconfirmed buffer on a cadence; commit the word-prefix that agrees across two
consecutive decodes; flush the remainder on a VAD endpoint. Rejected
alternatives: a streaming-native model (the parked Nemotron path —
English-only-on-CPU or multilingual-needs-GPU, breaks CPU-first multilingual);
and raw preview-without-LocalAgreement (captions would flicker as re-decodes
rewrite earlier words).

## Architecture

Three units with clear boundaries:

### 1. `StreamingTranscriber` — pure LocalAgreement-2 logic

A standalone module with **no audio or model dependency**, so it is unit-testable
against synthetic hypotheses.

- **State:** the previous hypothesis (list of timestamped words) and the current
  commit offset (seconds into the buffer).
- **Input per step:** the current hypothesis — an ordered list of
  `(word, start, end)` over the unconfirmed buffer.
- **Rule (LocalAgreement-2):** commit the longest leading run of words that
  matches the previous hypothesis (normalized text equality; timestamps used to
  resolve buffer advance, not for equality). Everything after the agreed prefix
  is **pending**.
- **Output per step:** `(newly_committed: list[word], pending: list[word],
  commit_until_seconds: float)`.
- **Flush:** an explicit `flush()` commits all remaining words (called on a VAD
  endpoint) and resets per-utterance state.
- **What it depends on:** nothing but its inputs. **What it does:** decides
  committed vs pending. **How you use it:** feed hypotheses, read commits.

### 2. Streaming driver — loop in the session manager

- Maintains the **unconfirmed audio buffer** from the last commit point.
- Every ~1.5 s of accumulated new audio (cadence configurable), runs the
  **formal Whisper model** on the bounded buffer (≤ the 4 s `pre_vad` cap) with
  `word_timestamps=True`, `condition_on_previous_text=False` (anti-hallucination
  preserved), on a **dedicated executor** so it never blocks the formal lane.
- Feeds the hypothesis to `StreamingTranscriber`; advances the buffer past
  `commit_until_seconds`.
- On a **pre-VAD silence endpoint** (the existing signal), calls `flush()` and
  resets the buffer for the next utterance.
- Emits committed+pending text via the existing preview event path.

### 3. Integration — reuse existing preview plumbing

- Exposed as a new preview backend value `"whisper-streaming"` slotting into the
  current `preview_asr_backend_name` / `_emit_transcript_preview` path and the
  frontend's existing preview rendering (streaming text animation, preview
  stack). **No new UI.**
- **Backend selection:** when `fast_preview` is enabled and no Qwen model is
  present, default the preview backend to `whisper-streaming` (the majority gets
  fast captions for free). Qwen and `whisper-streaming` are both selectable;
  precedence is config-driven, defaulting to whichever model is available
  (Qwen if installed, else streaming).

### Adapter change (minimal, additive)

Extend `FasterWhisperAdapter` to optionally return **word-level timestamps**
(faster-whisper `word_timestamps=True`), surfaced on `ASRSegment` (or a parallel
word list). Default path unchanged; only the streaming driver requests words.

### Model sharing

The streaming lane **shares the already-loaded formal Whisper model** (no extra
~460 MB). Concurrent decodes are serialized by the dedicated executor; this is
acceptable because both are CPU-bound and the governor caps total load.

## Data flow

```
audio chunks
  └─> [formal lane: unchanged]  ──> formal segment ──> storage / summaries / translation (AUTHORITATIVE)
  └─> [streaming driver] grows unconfirmed buffer
         every ~1.5s ──> formal Whisper model (word_timestamps) ──> hypothesis
              ──> StreamingTranscriber (LocalAgreement-2) ──> (committed, pending)
              ──> emit transcript_preview (existing path) ──> live caption UI
         on VAD endpoint ──> flush() + reset buffer
```

## CPU budget & gating (primary risk)

A second Whisper decode competes with the formal lane on CPU. Guards:

- **Bounded re-decode window** (≤ 4 s, the `pre_vad_max_segment_seconds` cap).
- **Tunable cadence** (~1.5 s) — fewer re-decodes per second.
- **Hardware-gated**: reuse `hardware.py` detection; enable only on capable
  machines. `summary_only` profile stays off.
- **Governor backpressure**: under resource pressure, the streaming lane is shed
  first (it is best-effort), the formal lane is protected.

## Error handling

The streaming lane is **best-effort**. Any decode error, timeout, or
backpressure → drop that tick, log, fall back silently to formal-only captions.
It must never block, delay, or corrupt the formal lane or the stored transcript.
Degradation is exactly like today's optional preview lane being absent.

## Testing

- **Unit:** `StreamingTranscriber` commit/flush logic against synthetic
  hypothesis sequences (growing buffers, disagreement, partial-word churn,
  endpoint flush). No model required. This is the first real unit-test target in
  the repo.
- **Integration / benchmark:** extend
  `scripts/benchmark_latency_accuracy_frontier.py` (or a sibling) to run the
  streaming lane on AMI and report committed-caption p50/p95, committed-stream
  WER vs formal, and RTF. Compare against the success criteria above.
- **Regression:** the formal-lane accuracy baseline
  (`scripts/benchmark_accuracy.py`) must be unchanged (formal path untouched).

## Phase 2 (out of scope, recorded for direction)

1. Make streaming commits **authoritative** (lowers final-text/median latency —
   the option-C win), once committed-stream WER is validated at parity.
2. If streaming beats Qwen on quality+latency, **retire the Qwen preview lane**
   (option D — one accurate streaming lane for everyone).

## Config summary

- New preview backend value: `whisper-streaming`.
- New tunables (profile/env): streaming cadence seconds, enable flag, hardware
  gate threshold. Defaults: on in `balanced`/`performance` when hardware capable
  and no Qwen present; off in `summary_only`.
