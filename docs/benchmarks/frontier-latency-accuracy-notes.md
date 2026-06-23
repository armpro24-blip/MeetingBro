# Latency × accuracy frontier — findings (2026-06, AMI EN2002a)

Analysis of `scripts/benchmark_latency_accuracy_frontier.py` on the real AMI
meeting fixture. Two sweeps were run; the **hardened** one supersedes the first.

- Raw numbers: [frontier-latency-accuracy.md](frontier-latency-accuracy.md) / `.json`.
- Setup: AMI EN2002a SDM, `small/cpu/int8`, balanced profile, formal lane only,
  `forced_language=en`.

## Run 1 (first pass) — what it cost us, and why we re-ran

The first sweep (`pre_vad_max ∈ {8,6,4,3,2}`, single run, safeguard ON) was
**confounded**: the ASR safeguard fired non-deterministically on short segments
(`asr_rtf` spiked to a non-physical 9.27 at cap=2), latency was non-monotonic,
and segment counts jumped around. Its `★` marks were noise artifacts. It did,
however, hint that the 8s cap was buying neither accuracy nor latency — which the
hardened run then confirmed.

## Run 2 (hardened) — trustworthy result

`pre_vad_max ∈ {8,4}` × `asr_accumulation_seconds ∈ {2.0,1.0}`, **safeguard
disabled** (knob isolated), **3 runs/point with pooled latency samples**. The
per-run p50 range is tight (≈±0.06 s), so these are stable.

| pre_vad_max | accum | WER | p50 (s) | p50 range | p95 (s) |
|---|---|---|---|---|---|
| 8 (baseline) | 2.0 | 0.582 | 3.83 | 3.75–3.87 | 9.29 |
| 8 | 1.0 | 0.582 | 3.76 | 3.68–3.80 | 9.29 |
| 4 | 2.0 | 0.579 | 3.98 | 3.88–4.09 | **6.98** |
| 4 | 1.0 | 0.579 | 3.97 | 3.91–3.98 | **6.98** |

### Conclusions

1. **Accuracy is flat across every knob setting** (WER 0.582 → 0.579). The 8s cap
   buys *no* accuracy on real meeting speech, and the accumulation window doesn't
   move it either. The "big cap protects accuracy" premise is dead.

2. **The accumulation window is not a latency lever.** 2.0 → 1.0 moves p50 by
   ~0.07 s (within noise) and p95 not at all.

3. **`pre_vad_max_segment_seconds` is a TAIL-latency lever, not a median one.**
   8 → 4 cuts **p95 from 9.29 s → 6.98 s (−2.3 s, ~25%)** with p50 unchanged
   (~3.8–4.0 s) and WER flat-to-slightly-better. The cap only binds during long
   pause-free stretches (most segments end on the ~0.6 s trailing-silence VAD
   first), so lowering it trims the worst case without touching the typical case.

4. **The median (~3.8 s) is immovable by these knobs** — it is the structural cost
   of a buffered formal lane that waits for segment completion before decoding.

### So what

- **Cheap, data-backed win:** lower `pre_vad_max_segment_seconds` 8 → 4 in the
  balanced profile — ~25% tail-latency reduction, no accuracy cost.
  - **Caveat:** measured on one English fixture. Validate WER stays flat on the
    zh/de synthetic fixtures before changing the global default (accuracy-only,
    fast). The cap is a worst-case guard, so the downside risk is low (at worst,
    very long utterances split sooner), but breadth should be checked.
- **The real prize is the median, and no knob touches it.** Cutting the ~3.8 s
  median requires **Step 2: streaming / incremental decode with stable-prefix
  (LocalAgreement) commit** — get low latency from the same accurate model rather
  than waiting for whole segments. The frontier confirms there is no buffered-knob
  shortcut to a lower median.

## Reproduce

```bash
# hardened sweep (isolated knob, pooled multi-run latency)
python scripts/benchmark_latency_accuracy_frontier.py --pre-vad-max 8,4 --accumulation 2.0,1.0 --runs 3
```
