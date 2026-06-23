# Optimization round — 2026-06

A summary of the accuracy + latency optimization pass run in June 2026, what
shipped, what was investigated and deliberately *not* shipped, and the measured
evidence behind each decision.

This complements two sibling documents:

- [docs/optimization-notes.md](optimization-notes.md) — the earlier C1–C10
  correctness/robustness fixes (concurrency guards, anti-aliasing, bounded
  queues, …).
- [docs/benchmarks/baseline-2026-06.md](benchmarks/baseline-2026-06.md) — the
  reproducible numbers this round measures against.

---

## Goal & constraints

Direction: **more accurate + faster**, under four self-imposed constraints:

| Constraint | Meaning |
|---|---|
| **Balanced** | Accuracy and speed weighted equally — no one-sided wins. |
| **Incremental** | Keep the dual-lane architecture (formal Whisper + Qwen3 preview). No streaming rewrite. |
| **CPU-first** | GPU / `large-v3-turbo` / batched inference are opt-in enhancements, never the default. |
| **Measure first** | No change ships without a before/after number. Issue #7 (the benchmark harness) gates everything else. |

The work was tracked as GitHub issues #7–#18 on the `armpro24-blip/MeetingBro`
remote.

---

## Methodology — measure before you touch

Before this round there was **no WER/CER baseline**, so the first deliverable
was the measurement harness, not a fix.

- `scripts/asr_metrics.py` — shared WER (Latin) / CER (CJK) implementation with a
  self-test.
- `scripts/benchmark_accuracy.py` — runs the full pipeline, reports per-fixture
  WER/CER, keyword recall, ASR real-time factor (RTF), and realtime caption
  latency; writes `docs/benchmarks/baseline-2026-06.{md,json}`. Applies the
  production `RUNTIME_PROFILE_PRESETS["balanced"]` preset so the baseline matches
  what users actually run.
- Ground truth lives in `data/benchmark/manifest.json` (committed); audio is
  gitignored.

Two classes of fixture were built so findings generalize:

- **Synthetic (offline, deterministic):** `scripts/gen_synth_fixtures.py` +
  `scripts/_sapi_render.ps1` render zh/en code-switch, two-speaker, and
  proper-noun clips from offline Windows SAPI voices. The script's input text
  *is* the ground truth.
- **Real meeting audio:** `scripts/fetch_ami_fixture.py` streams the AMI corpus
  (CC BY 4.0) and reconstructs a real multi-speaker meeting excerpt
  (EN2002a). Uses `datasets==2.21.0` + `librosa` (not `datasets` 5.x, which pulls
  torchcodec/torch).

> **Harness gotcha (documented in the baseline):** offline replay must lift the
> bounded input queue (`audio_input_queue_max_seconds` = 8 s), otherwise
> non-realtime playback floods the queue and silently drops the *start* of long
> clips, corrupting accuracy. The realtime latency pass keeps the production
> bound. The production realtime path is unaffected.

---

## What shipped

| # | Change | Commit | Measured result |
|---|---|---|---|
| #7 | ASR accuracy + latency baseline harness | `7682550` | Established the WER/CER + latency baseline below. |
| #2 | `large-v3-turbo` model option + GPU-aware recommendation | `f4b3ac6` | UI + recommendation only (the backend already passed the size string through). Turbo recommended on CUDA; CPU stays small/medium. |
| #5 | Vocabulary terms formatted as a Whisper glossary prompt | `4ad2484` | Proper-noun WER **0.190 → 0.095**; keyword recall **3/5 → 5/5**. |
| #10 | Opt-in neural speaker diarization (sherpa-onnx CAM++ embeddings) | `3a6420a` | Two-speaker synth: standalone 7/7 regions (energy 6/7), end-to-end 7/7 segments, 2 speakers. Off by default behind `MEETINGBRO_DIARIZATION_ENABLED`; falls back to energy diarizer if the model is missing. |
| #13 | Qwen3 preview lane enabled in `balanced` mode | `87c06b8` | Preview caption latency p50 **1.02 s** vs formal **5.17 s** (CPU, Qwen RTF 0.22). Degrades safely to formal-only if the optional ~700 MB Qwen model is absent. |

A real-audio validation pass on the AMI excerpt then **caught a production crash
in the shipped #10**: `NeuralDiarizer._assign_speaker` called
`manager.all_speakers()` as a method, but it is a list property — a crash
reachable at the speaker cap in any long meeting. Fixed in `02c6f64` with a
permissive embedding-search fallback.

---

## What was investigated and deliberately *not* shipped

The same measure-first discipline rejected several plausible changes because
they showed **no measured benefit**:

| # | Idea | Verdict | Why |
|---|---|---|---|
| #9 | zh/en code-switch tuning (language-lock, multilingual flag) | Deferred | The language-vote-lock is dead code under current wiring; flipping the formal adapter to `multilingual=True` produced a byte-identical transcript on the 6 s clip (per-segment detection needs ~30 s windows). Turn-based code-switch already scores CER 0.029. Needs a real zh/en fixture + per-segment language labeling. |
| #13 | Adaptive endpointing / fast-flush during active speech | Reverted | Root cause is pre-VAD batching up to `pre_vad_max_segment_seconds` (8 s) when gaps are short. Fast-flush doesn't fire while segments are pending. Safe tuning needs real meeting audio (natural pause distribution). The real latency win was **enabling the preview lane** — which is what shipped. |
| #18 | Overlap-aware pyannote segmentation for diarization | Closed, not planned | Decisive experiment on real single-channel AMI audio: energy diarizer = 8 spk / purity 0.80; full pyannote seg+clustering = 11/0.72, 7/0.54, forced-4 → 3/0.53. **Pyannote did not beat energy in any config** — premise unsupported, not worth the streaming-integration cost. |
| #8 | faster-whisper batched inference | Recommend skip | Batching helps bulk offline transcription, not per-window streaming — low value in this architecture. |
| #12 | Qwen3 preview confidence calibration | Deferred | Preview text quality is already user-validated as acceptable; the high-value action was enabling the lane, which is done. |

---

## Baseline numbers (balanced profile, formal lane, small/cpu/int8)

Full table and interpretation: [docs/benchmarks/baseline-2026-06.md](benchmarks/baseline-2026-06.md).

**Accuracy by language** (error rate; lower is better):

| lang | metric | mean | n |
|---|---|---|---|
| de | WER | 0.000 | 1 |
| en | WER | 0.621 | 2 |
| mixed | WER | 0.786 | 1 |
| zh | CER | 0.625 | 3 |

**End-to-end caption latency** (formal lane, `sample_en.wav`):

| p50 | p95 | max | RTF |
|---|---|---|---|
| 8.04 s | 13.81 s | 16.24 s | 0.36 |

> **Read these as relative, not absolute.** The fixtures are *adversarial stress
> clips* — background noise, rapped lyrics, singing, tongue-twisters,
> code-switch — chosen to expose regressions, not to represent clean meeting
> speech. The one clean sample (`de_de`, WER 0.00 here / ~0.14 on harder clean
> speech) is the realistic clean-speech reference. The baseline's value is
> before/after comparison.
>
> Multi-second formal-lane latency is expected: continuous speech makes pre-VAD
> hold a segment until its size cap. **Sub-second** captions come from the
> **preview lane** (p50 ≈ 1 s), enabled in balanced this round.

---

## Lessons

- **Measure-first paid off twice over** — it both quantified the wins (#5, #13)
  and killed three changes that *felt* right but moved no number (#9, #13-tuning,
  #18).
- **Synthetic fixtures miss bugs real audio finds.** The AMI excerpt surfaced a
  production crash that the SAPI fixtures never hit. Both fixture families are now
  in-tree and reusable.
- **Cheap param-flip wins are exhausted.** The clean wins this round were #2 and
  #5 (plus the #7 baseline, #10 diarization, and the #13 preview-lane flip).
  Remaining items need real meeting recordings (#13 latency, #9 deep
  code-switch), an English/far-field embedding model (#10 quality), or
  preview-lane rework (#12) — not another flag flip.

---

## Reproduce

Run with the MeetingBro conda environment's Python:

```bash
# Accuracy + formal-lane latency baseline (writes docs/benchmarks/baseline-2026-06.{md,json})
python scripts/benchmark_accuracy.py

# Preview-lane latency
python scripts/benchmark_accuracy.py --preview

# Score against synthetic fixtures (regenerate first with gen_synth_fixtures.py)
python scripts/benchmark_accuracy.py --manifest data/benchmark/synth/manifest.json

# Rebuild the real-audio (AMI) fixture
python scripts/fetch_ami_fixture.py        # AMI_CONFIG=sdm for single distant mic
```
