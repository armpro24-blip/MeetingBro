# MeetingBro 延迟 × 准确率前沿 (latency–accuracy frontier)

> 由 `scripts/benchmark_latency_accuracy_frontier.py` 生成。在**真实 AMI 会议片段**上
> 扫描分段旋钮,对每个配置在**相同音频**上分别测字幕延迟(实时回放)与 WER(离线评分),
> 给出可优化的 Pareto 前沿,而非两个孤立数字。`★` 标记 Pareto 最优点。

## 配置

- fixture: `ami_en2002a` (en, AMI EN2002a [300-420s], CC BY 4.0)
- Whisper: `small` / `cpu` / `int8` / beam `3`
- runtime profile: `balanced` (formal lane only, fast_preview disabled), forced_language=`en`
- swept: pre_vad_max_segment_seconds=8,6,4,3,2  asr_accumulation_seconds=2.0
- label: frontier-2026-06 AMI 1D pre_vad sweep

## 前沿点

| | pre_vad_max (s) | accum (s) | latency p50 (s) | p95 (s) | WER | seg(acc/lat) | asr_rtf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ★ | 8 (baseline) | 2 | 3.83 | 9.20 | 0.582 | 66/66 | 0.46 |
|  | 6 | 2 | 4.63 | 8.07 | 0.545 | 47/47 | 0.45 |
|  | 4 | 2 | 5.69 | 9.51 | 0.513 | 44/43 | 0.92 |
|  | 3 | 2 | 5.52 | 9.67 | 0.624 | 35/35 | 0.72 |
| ★ | 2 | 2 | 4.54 | 6.88 | 0.504 | 67/67 | 9.27 |

## 解读

- `★` = Pareto 最优(没有任何其它点同时在延迟和 WER 上都不差且至少一项更好)。
- 沿 `pre_vad_max_segment_seconds` 下降,延迟应下降;若 WER 不升或仅微升,即为**免费的延迟收益**,
  说明当前 8.0s 上限对真实会议语音偏保守。若 WER 明显恶化,则该点是 tradeoff 的拐点。
- 这是真实会议语音上的相对比较;绝对 WER 受 AMI 远场单麦(SDM)与重叠说话影响,看趋势而非绝对值。
