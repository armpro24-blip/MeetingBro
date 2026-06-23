# Latency × accuracy frontier — findings (2026-06, AMI EN2002a)

Analysis of the first sweep from `scripts/benchmark_latency_accuracy_frontier.py`
(raw numbers: [frontier-latency-accuracy.md](frontier-latency-accuracy.md) /
`.json`). Run: AMI EN2002a SDM, `small/cpu/int8`, balanced profile, formal lane
only, `forced_language=en`, sweeping `pre_vad_max_segment_seconds ∈ {8,6,4,3,2}`,
`asr_accumulation_seconds=2.0`, single run per point.

## TL;DR

**Do not act on this as a Pareto frontier yet.** The sweep is informative but
confounded — it changed an assumption rather than producing a knob setting to
ship.

## What the data actually shows

| pre_vad_max | WER | p50 | p95 | asr_rtf | seg |
|---|---|---|---|---|---|
| 8 (baseline) | 0.582 | **3.83** | 9.20 | 0.46 | 66 |
| 6 | 0.545 | 4.63 | 8.07 | 0.45 | 47 |
| 4 | 0.513 | 5.69 | 9.51 | 0.92 | 44 |
| 3 | 0.624 | 5.52 | 9.67 | 0.72 | 35 |
| 2 | 0.504 | 4.54 | 6.88 | **9.27** | 67 |

Three findings, in order of confidence:

1. **The 8 s cap is not "buying" accuracy.** WER is flat-to-slightly-better as
   the cap drops (0.582 → 0.545 → 0.513). Longer maximum segments do **not** help
   Whisper on far-field, overlapping meeting speech. The premise that a big cap
   protects accuracy is unsupported on real meeting audio.

2. **The 8 s cap is not the latency lever we assumed.** Latency did **not** fall
   as the cap dropped — the baseline (cap=8) actually had the *lowest* p50
   (3.83 s) and p50 was non-monotonic across the sweep. Why: on real
   conversational speech with natural pauses, segments end on the
   trailing-silence VAD (~0.6 s) **long before** they reach the cap, so the cap
   rarely binds. It is a worst-case guard for pause-free speech, not the
   typical-case driver. Typical-case caption latency is governed by the
   accumulation window + flush cadence + the resource governor — not this knob.

3. **The sweep is confounded by the resource governor / ASR safeguard.** At
   cap ≤ 4 the safeguard activated (`rtf>1` log lines) and `asr_rtf` went
   non-physical (9.27 at cap=2, vs 0.46 baseline), with non-monotonic segment
   counts (66/47/44/35/67). The governor *reacts* to short segments and changes
   pipeline behaviour, so per-point latency is not a clean function of the knob.
   The `★` Pareto mark on cap=2 is an artifact of this noise, not a real win.

## Implications

- **Methodology must be hardened before any frontier is trustworthy:**
  1. Hold the resource governor fixed (or disable the ASR safeguard) during the
     sweep so the knob's effect is isolated.
  2. Average ≥3 runs per point — realtime latency variance on a 120 s clip is
     high (and p50/p95 from ~40–66 samples is jumpy).
  3. Add the `asr_accumulation_seconds` axis (the more likely real latency lever).
  4. Consider a longer clip or several real clips.

- **Strategic:** tuning the existing *buffered* knobs shows limited, noisy
  headroom — which strengthens the case for the real prize, **streaming /
  incremental decode with stable-prefix (LocalAgreement) commit** (Step 2): get
  low latency from the same accurate model instead of trading along a shallow,
  governor-confounded curve.

## Reproduce

```bash
python scripts/benchmark_latency_accuracy_frontier.py --pre-vad-max 8,6,4,3,2 --accumulation 2.0
```
