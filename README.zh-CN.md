<div align="right">
  <a href="README.md">English</a> | 简体中文
</div>

# MeetingBro

**开源、本地优先的 AI 会议助手 —— 实时转录、翻译与智能总结。**

MeetingBro 能够监听你的会议内容，实时将语音转录为文字，自动翻译多语言内容，并生成滚动式会议纪要 —— 所有功能无需任何平台插件或 API 集成即可运行。

它兼容 Zoom、Teams、Google Meet、BBB 等任意会议平台，因为它直接从电脑本地捕获音频，而非通过会议 API 连接。

支持平台：Windows、macOS、Linux。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/armpro24-blip/MeetingBro/actions/workflows/ci.yml/badge.svg)](https://github.com/armpro24-blip/MeetingBro/actions/workflows/ci.yml)

---

<video src="https://github.com/user-attachments/assets/54111a42-2e80-416f-9ee5-1402a96b3cd7" autoplay loop muted playsinline width="100%"></video>

---

## MeetingBro 能做什么？

| 功能 | 说明 |
|---|---|
| **实时转录** | 基于 Whisper 的本地实时语音转文字，数据不上传 |
| **实时字幕** | 可选中 / 英 / 德实时翻译字幕 |
| **实时翻译** | 中、英、德三语自动互译 |
| **滚动纪要** | AI 自动生成近 3–5 分钟内容的滚动摘要，实时刷新 |
| **会议看板** | 累积生成会议概览：议题、决议、待办事项、遗留问题 |
| **导出笔记** | 一键导出完整转录稿、摘要和会议笔记为 Markdown |
| **系统音频捕获** | 捕获任意会议平台音频（Windows 原生支持；macOS/Linux 需配置虚拟声卡） |
| **麦克风捕获** | 支持 Windows、macOS、Linux 现场会议 |
| **本地或云端大模型** | 支持 OpenAI、Groq、Ollama 等任意兼容 OpenAI API 的 LLM 生成摘要 |

> **无需 API Key 即可使用转录功能。** 若未配置 LLM，摘要功能将自动降级为本地关键词提取。

---

## 适合谁用？

- 🎓 **学生** —— 上网课需要自动生成课堂笔记
- 💼 **职场人士** —— 参加外语会议，需要实时翻译与纪要
- 🔬 **研究人员** —— 需要多语言访谈/会议转录
- 🔒 **隐私敏感用户** —— 希望数据完全本地处理，不上传云端
- 🛠️ **开发者** —— 想在本地会议助手基础上二次开发

---

## 安装前必读

**请务必在安装前阅读以下内容。**

- 安装过程需要在终端（命令行）中执行少量命令，下方提供了一步一步的指引。
- **Windows 是目前体验最完善的平台。** Windows 上可直接捕获系统音频（扬声器输出），因此能自动兼容任意会议软件。
- **macOS 和 Linux** 支持麦克风捕获。如需在线会议的系统音频捕获，需要额外配置：macOS 需安装 BlackHole 虚拟声卡并创建 Multi-Output Device；Linux 需配置 PulseAudio/PipeWire loopback。
- **Whisper 模型会在首次运行时自动下载。** `small` 模型约 460 MB，仅需下载一次，永久保存本地。
- **MeetingBro 提供三种运行模式**：`仅摘要`（适合低配设备）、`均衡`（推荐大多数用户）、`高性能`（适合高配设备，追求更强实时性）。
- **识别语言与运行模式相互独立。** 若会议为单一语言，可固定语言以获得更精准的识别；多语言会议建议设为 `自动检测`。
- **Qwen3 预览模型为可选高级配置**（约 700 MB）。建议普通用户先不安装，仅在需要独立快速预览通道的高配设备上启用。
- **LLM API Key 为可选项。** 没有 Key 的情况下，转录功能完全不受影响；仅 AI 摘要和翻译需要配置 Key。

---

## 快速启动

最快的方式是使用项目提供的安装/启动脚本。手动安装的等效步骤请参见下方的[分步安装](#分步安装)。

**Windows (PowerShell):**
```powershell
.\scripts\install.ps1
.\scripts\start.ps1
```

**macOS / Linux:**
```bash
chmod +x scripts/install.sh scripts/start.sh
./scripts/install.sh
./scripts/start.sh
```

脚本会自动完成以下操作：
- 检查 Python 3.12+ 和 Node.js 20+ 是否已安装
- 在 `app/backend/.venv` 创建 Python 虚拟环境
- 安装后端依赖（`pip install -e .`）
- 若不存在，将 `.env.example` 复制为 `app/backend/.env`
- 安装前端依赖（`npm install`）
- 启动后端（默认端口 8765，可通过 `BACKEND_PORT` 覆盖）
- 启动前端开发服务器（`npm run dev`）

在终端按 `Ctrl+C` 即可同时停止前后端服务。

> **端口说明：** MeetingBro 后端默认监听 **8765** 端口。脚本允许通过 `BACKEND_PORT` 环境变量修改，前端代码也默认连接 8765。除非你同时修改了前端配置，否则建议保持默认不变。

### Linux 额外配置

```bash
# 音频依赖
sudo apt install portaudio19-dev libportaudio2

# 可选：安装 Qwen3 ASR 预览后端
pip install sherpa-onnx
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"

# 部分 Linux 系统可能需要 Electron 沙箱权限
sudo chown root:root app/frontend/node_modules/electron/dist/chrome-sandbox
sudo chmod 4755 app/frontend/node_modules/electron/dist/chrome-sandbox

npm run dev
```

### macOS 额外配置

```bash
brew install portaudio
brew install --cask blackhole-2ch

# 可选：安装 Qwen3 ASR 预览后端
pip install sherpa-onnx
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"

python3 -m pip install sounddevice
```

然后打开 **音频 MIDI 设置**，创建 `多输出设备`：

1. 点击 `+` → `多输出设备`
2. 勾选 `BlackHole 2ch`
3. 勾选你的扬声器或耳机
4. 在 `系统设置 > 声音 > 输出` 中，选择新建的 `多输出设备`
5. 在 MeetingBro 中，选择 `系统音频` / `loopback`

如需验证 macOS 是否识别到虚拟输入设备，可运行：

```bash
python3 scripts/list_audio_devices.py
```

你应该能看到类似 `BlackHole 2ch` 且 `max_in > 0` 的设备。

会议结束后，如不再需要系统音频捕获，可将 macOS 输出切换回普通扬声器或耳机。


### 选择正确的运行模式

MeetingBro 的使用体验很大程度上取决于根据你的设备选择合适的运行模式，而非随意开关各项功能。

| 模式 | 适合场景 | 行为说明 |
|---|---|---|
| **仅摘要** | 低配或老旧设备 | 隐藏实时转录面板，最小化实时计算，专注于生成摘要和最终笔记 |
| **均衡** | 大多数用户 | 在响应速度、转录质量、字幕和摘要之间取得最佳平衡 |
| **高性能** | 高性能 CPU/GPU，以英文会议为主 | 采用更激进的、偏向质量的实时配置，预期有更多算力冗余 |

推荐起步方案：

- 若设备吃力或转录明显滞后，从 **仅摘要** 开始。
- 若设备性能中等，想要标准体验，使用 **均衡**。
- 若设备强劲，想要最激进的实时效果，尝试 **高性能**。

语言设置与运行模式独立：

- **语音识别 = 自动** —— 适合多语言混用会议
- **语音识别 = 英语 / 中文 / 德语** —— 适合单语言会议，可获得更精准的识别效果

---

## 分步安装

### 环境要求

| 工具 | 版本 | 获取方式 |
|---|---|---|
| **Python** | 3.12 或更高 | [python.org](https://www.python.org/downloads/) 或通过 Conda 安装 |
| **Conda** | 较新版本即可 | [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)（推荐 Miniconda） |
| **Node.js** | 18 或更高 | [nodejs.org](https://nodejs.org/) |
| **Git** | 任意版本 | [git-scm.com](https://git-scm.com/) |

### 步骤 1 —— 克隆仓库

```bash
git clone https://github.com/armpro24-blip/MeetingBro.git
cd MeetingBro
```

### 步骤 2 —— 创建 Python 环境

```bash
conda create -n MeetingBro python=3.12 -y
conda activate MeetingBro
```

每次新开终端时，都需要运行 `conda activate MeetingBro`。

### 步骤 3 —— 安装后端

```bash
cd app/backend
pip install -e "."
```

**Windows 用户额外安装 WASAPI 回环捕获库：**

```bash
pip install "soundcard>=0.4"
```

这让 MeetingBro 无需插件即可捕获 Zoom、Teams 等应用的音频。

返回项目根目录：

```bash
cd ../..
```

### 步骤 4 —— 配置环境变量

复制配置模板：

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (CMD)
copy .env.example .env
```

用任意文本编辑器打开 `.env` 文件。默认配置无需修改即可运行；如需 AI 摘要功能，可添加 LLM Key（参见下方的 [LLM 配置](#llm-配置-ai-摘要)）。

### 步骤 5 ——（可选高级）下载 Qwen3 预览模型

此步骤可选，普通使用无需执行。仅当你需要在高配设备上测试独立快速预览通道时才需要。

```bash
pip install sherpa-onnx
pip install huggingface_hub

huggingface-cli download csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 --local-dir models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
```

如果 `huggingface-cli` 不在 PATH 中，可使用：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"
```

模型约 700 MB，仅需下载一次。

如不需要此预览模型，可跳过本步骤。如需强制将预览通道回退到纯 Whisper 模式，可在 `.env` 中添加：

```env
MEETINGBRO_PREVIEW_ASR_BACKEND=whisper
```

### 步骤 6 —— 安装前端

```bash
cd app/frontend
npm install
cd ../..
```

---

## LLM 配置（AI 摘要）

MeetingBro 使用**可选**的云端或本地大模型来提供以下功能：

- AI 会议纪要（滚动摘要与会议看板）
- 中、英、德三语翻译

**没有 Key 的情况下：** 转录功能完全正常，摘要将降级为关键词提取。

**有 Key 的情况下：** 在 `.env` 文件中设置以下三行：

```env
MEETINGBRO_LLM_API_KEY=your_api_key_here
MEETINGBRO_LLM_BASE_URL=https://api.openai.com/v1
MEETINGBRO_LLM_MODEL=gpt-4o-mini
```

支持的提供商包括 **OpenAI、Groq、Mistral AI、OpenRouter、Together AI**，以及完全本地的 **Ollama**（无需 Key，数据不出本机）。

→ **各提供商的完整配置方法请参见 [docs/llm-providers.md](docs/llm-providers.md)。**

> **隐私说明：** 只有文本转录稿会被发送到你配置的 LLM 提供商。音频捕获和 Whisper 转录全程在本地运行。若使用 Ollama 等本地 LLM，数据完全不会离开你的设备。

---

## 运行 MeetingBro

始终先启动后端，再启动前端。

### 后端

```bash
conda activate MeetingBro
cd app/backend
meetingbro-backend
```

你应该能看到类似以下输出：

```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 前端

在**新终端窗口**中：

```bash
cd app/frontend
npm run dev
```

Electron 窗口会自动打开。若未自动打开，可在浏览器中访问 `http://localhost:5173`。

---

## 首次运行检查清单

开始会议前，请确认：

- [ ] 后端已启动（参见上方终端输出）
- [ ] Electron 窗口已打开
- [ ] 在 UI 中选择了正确的音频设备
- [ ] 在线会议：选择「系统音频」（macOS 需 BlackHole + 多输出设备；Linux 需 loopback monitor）
- [ ] 现场会议：选择你的麦克风

点击 **开始会议**，说几句话。几秒钟内你应该能在转录面板看到文字出现。

在第一次正式会议前，根据你的设备选择合适的模式：

- **仅摘要** —— 如果你更关心最终笔记而非实时转录
- **均衡** —— 如果你希望获得标准体验
- **高性能** —— 如果你的设备性能强劲，想要最激进的实时效果

若 Whisper 模型是首次使用，第一次转录可能需要 10–30 秒用于模型加载。后续转录会更快。

---

## 工作原理

MeetingBro 从你的电脑捕获音频，并通过本地流水线处理：

```
音频来源（麦克风或系统音频）
    ↓ 语音活动检测 (VAD)
    ↓ Whisper 语音识别
    ↓ 可选预览通道（仅高级配置）
    ↓ 说话人分离（可选）
    ↓ 翻译（可选，通过 LLM）
    ↓ 摘要生成（可选，通过 LLM）
    ↓ 实时 UI（转录 + 摘要）
    ↓ 导出（Markdown 文件）
```

默认配置下，Whisper 是主引擎。可选的 Qwen3 预览通道不再建议普通设备使用。

两种音频捕获模式：

- **在线模式**（Windows/macOS/Linux）：捕获系统音频（macOS 需 BlackHole 等虚拟声卡 + 多输出设备）
- **离线模式**（全平台）：捕获麦克风 —— 适用于现场会议

更多技术细节请参见 [docs/architecture.md](docs/architecture.md)。

---

## 常见问题排查

完整问题与解决方案请参见 **[docs/troubleshooting.md](docs/troubleshooting.md)**。

快速检查：

1. 运行 `python scripts/dep_check.py` 验证所有依赖是否安装正确。
2. 运行 `python scripts/list_audio_devices.py` 查看可用音频设备。
3. 确保先启动后端，再启动前端。

---

## 平台支持

| 平台 | 麦克风 | 系统音频 | 说明 |
|---|---|---|---|
| Windows 10/11 | ✅ | ✅ | 完整支持 |
| macOS | ✅ | ⚠️ | 系统音频需 BlackHole（或同类工具）+ 多输出设备 |
| Linux | ✅ | ⚠️ | 系统音频通过 PulseAudio/PipeWire loopback（需手动配置） |

各平台的详细配置方法请参见 [docs/platform-support.md](docs/platform-support.md)。

---

## 常见问题

请参见 **[docs/faq.md](docs/faq.md)** 获取常见问题的解答。

---

## 参与贡献

请参见 **[CONTRIBUTING.md](CONTRIBUTING.md)** 了解如何配置开发环境、运行测试和提交 Pull Request。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 本文件 —— 项目概览与快速开始 |
| [docs/llm-providers.md](docs/llm-providers.md) | LLM API 配置指南 |
| [docs/platform-support.md](docs/platform-support.md) | 各平台音频设置说明 |
| [docs/architecture.md](docs/architecture.md) | 技术架构与组件概览 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 常见问题与解决方案 |
| [docs/faq.md](docs/faq.md) | 常见问题 |
| [docs/roadmap.md](docs/roadmap.md) | 开发路线图 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开发环境与贡献指南 |
| [docs/product-principles.md](docs/product-principles.md) | 产品与工程原则 |

---

## License

MIT —— 详见 [LICENSE](LICENSE)。
