# MeetingBro ASR 基线报告 (baseline)

> 由 `scripts/benchmark_accuracy.py` 生成。仅正式通道(formal Whisper),已禁用预览(preview)与摘要,
> 确保结果可复现。后续优化 issue (#2-#9) 须引用此基线给出前/后对比。

## 配置

- Whisper model size: `small`
- device: `cpu`  compute type: `int8`  beam size: `3`
- runtime profile: `balanced`  (formal lane only, fast_preview disabled)
- manifest: `data\benchmark\manifest.json`
- label: baseline small/cpu/int8 (balanced profile)

## 准确率 (WER / CER)

| fixture | lang | metric | error_rate | kw recall | vocab | asr_rtf | dur (s) | seg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zh_fast1 | zh | cer | 0.500 | — | off | 0.47 | 11.4 | 7 |
| zh_raokouling | zh | cer | 0.586 | — | off | 0.37 | 20.8 | 7 |
| zh_qiqiu1 | zh | cer | 0.789 | — | off | 0.00 | 51.0 | 8 |
| en_f1_noise | en | wer | 0.290 | — | off | 0.97 | 19.0 | 6 |
| en_rap1 | en | wer | 0.952 | — | off | 0.76 | 29.0 | 1 |
| de_de | de | wer | 0.000 | — | off | 0.22 | 6.7 | 2 |
| mixed_codeswitch | mixed | wer | 0.786 | — | off | 0.49 | 6.2 | 4 |

### 按语言聚合

| lang | metric | mean | n |
| --- | --- | --- | --- |
| de | wer | 0.000 | 1 |
| en | wer | 0.621 | 2 |
| mixed | wer | 0.786 | 1 |
| zh | cer | 0.625 | 3 |

## 端到端字幕延迟 (formal lane)

源文件: `sample_en.wav` (循环铺到 ≥60s),实时回放。延迟 = 音频结束 → 字幕发出。

| p50 (s) | p95 (s) | max (s) | segments | asr_rtf |
| --- | --- | --- | --- | --- |
| 8.04 | 13.81 | 16.24 | 10 | 0.36 |

## 结果解读 (重要)

- **这些 fixture 是对抗性压力样本**(背景噪声 / 配乐说唱 / 歌唱 / 绕口令 / 多语 code-switch),
  **不代表干净会议语音**。绝对错误率偏高是预期的;唯一的干净样本 `de_de` (WER 0.14) 才是
  干净语音的现实参考。本基线的价值在于**相对比较**(优化前/后),而非绝对达标线。
- 离线评分关闭了音频输入队列上限(`audio_input_queue_max_seconds`),否则非实时回放会冲垮 8s
  队列、丢掉长片段的开头,污染准确率。生产实时路径不受影响。
- 正式通道字幕延迟为多秒级:连续无停顿语音会让 pre-VAD 把片段攒到上限才发出。**亚秒级**
  实时字幕由预览(preview)通道提供,需单独度量(见 #6/#13)。`sample_en.wav` 为循环铺长。

## 说明

- WER 用于 Latin 语言(en/de + code-switch),CER 用于 CJK(zh);见 `scripts/asr_metrics.py`。
- 此基线为**正式通道**质量;预览(preview)延迟与质量另行度量(见 #6/#13)。
- ground truth 提交于 `data/benchmark/manifest.json`;音频随 Qwen3 模型下载,gitignored。
- 复现: `python scripts/benchmark_accuracy.py`(MeetingBro conda 环境)。
