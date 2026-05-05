# MeetingBro

**Open-source local-first AI meeting assistant for live transcription, translation, and meeting summaries.**

MeetingBro listens to your meetings, transcribes speech in real time, translates between languages, and generates live summaries — all without requiring any platform plugin or integration.

It works with Zoom, Teams, Google Meet, BBB, and any other platform, because it captures audio directly from your computer rather than connecting to a meeting API.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/armpro24-blip/MeetingBro/actions/workflows/ci.yml/badge.svg)](https://github.com/armpro24-blip/MeetingBro/actions/workflows/ci.yml)

---

<video src="https://github.com/user-attachments/assets/54111a42-2e80-416f-9ee5-1402a68e3cd7" autoplay loop muted playsinline width="100%"></video>

---

## What can MeetingBro do?

| Feature | Details |
|---|---|
| **Live transcription** | Real-time speech-to-text using Whisper, running locally on your machine |
| **Live subtitles** | Optional translated subtitles for Chinese, English, and German |
| **Live translation** | Automatic translation between Chinese, English, and German |
| **Rolling summary** | AI-generated summary of the last 3–5 minutes, refreshes automatically |
| **Meeting Board** | Cumulative overview: topics, decisions, action items, open questions |
| **Exportable notes** | Export full transcript, summary, and notes to Markdown files |
| **System audio capture** | Capture any meeting platform's audio (Windows) |
| **Microphone capture** | Works on Windows, macOS, and Linux (in-person meetings) |
| **Local or cloud LLM** | Use OpenAI, Groq, Ollama, or any OpenAI-compatible API for summaries |

> **No API key is required to use transcription.** Summaries fall back to local heuristics if no LLM is configured.

---

## Who is this for?

- Students attending online lectures who want automatic notes
- Professionals who attend meetings in a second language
- Researchers who need multilingual transcripts
- Privacy-conscious users who want local-first AI tools
- Developers who want to build on or extend a local meeting assistant

---

## Before you start

**Please read this before installing.**

- You will need to type a few commands in a terminal (command prompt). The instructions are step-by-step.
- **Windows is the easiest platform** right now. On Windows, MeetingBro can capture system audio (everything your speakers play), so it works with any meeting platform automatically.
- On **macOS and Linux**, microphone capture works. For online meetings, system audio capture is supported with the right setup: macOS uses a virtual loopback device plus a Multi-Output Device, and Linux uses PulseAudio/PipeWire loopback.
- **Whisper models download automatically on first run.** The `small` model is about 460 MB. This happens once and is saved to your disk.
- **MeetingBro has three runtime modes**: `Summary only` for weaker devices, `Balanced` for most laptops/desktops, and `Performance` for stronger machines that can trade more compute for more aggressive realtime behavior.
- **Speech language is separate from runtime mode.** Set `Speech = Auto` for mixed-language meetings; set a specific speech language when the meeting stays in one language and you want tighter recognition.
- **The Qwen3 preview model is optional advanced setup** (about 700 MB). Most users should start without it. It is only useful if you explicitly want a separate fast-preview lane on stronger machines.
- **An LLM API key is optional.** Without one, transcription still works perfectly. You only need a key for AI-generated summaries and translation. Some providers such as Groq or OpenRouter may offer a free tier — check their current pricing pages.

---

## Quick Start

This is the shortest path to a working session. See [Step-by-step installation](#step-by-step-installation) below for more detail.

```bash
# 1. Clone the repo
git clone https://github.com/armpro24-blip/MeetingBro.git
cd MeetingBro

# 2. Create a Python environment
conda create -n MeetingBro python=3.12 -y
conda activate MeetingBro

# 3. Install the backend
cd app/backend
pip install -e "."
pip install "soundcard>=0.4"   # Windows only — skip on macOS/Linux
cd ../..

# 4. Copy the config template
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (Command Prompt)
copy .env.example .env

# 5. Install the frontend
cd app/frontend
npm install
cd ../..
```

Then open **two terminals** and run:

**Terminal 1:**
```bash
conda activate MeetingBro
cd app/backend
meetingbro-backend
```

**Terminal 2:**
```bash
cd app/frontend
npm run dev
```

### Linux setup

```bash
# Audio dependencies
sudo apt install portaudio19-dev libportaudio2

# Optional: install the Qwen3 ASR preview backend
pip install sherpa-onnx
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"

# Electron sandbox permissions may be required on some Linux systems
sudo chown root:root app/frontend/node_modules/electron/dist/chrome-sandbox
sudo chmod 4755 app/frontend/node_modules/electron/dist/chrome-sandbox

npm run dev
```

### macOS setup

```bash
brew install portaudio
brew install --cask blackhole-2ch

# Optional: install the Qwen3 ASR preview backend
pip install sherpa-onnx
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"

python3 -m pip install sounddevice
```

Then open **Audio MIDI Setup** and create a `Multi-Output Device`:

1. Click `+` → `Multi-Output Device`
2. Enable `BlackHole 2ch`
3. Enable your speakers or headphones
4. In `System Settings > Sound > Output`, select the new `Multi-Output Device`
5. In MeetingBro, choose `System Audio` / `loopback`

If you want to verify that macOS sees the virtual input, run:

```bash
python3 scripts/list_audio_devices.py
```

You should see a device like `BlackHole 2ch` with `max_in > 0`.

If you no longer need system audio after a meeting, switch the macOS output back to your normal speakers or headphones.


### Choose the right mode first

MeetingBro is easier to use if you pick a runtime mode based on your machine instead of turning random switches on and off.

| Mode | Best for | What it does |
|---|---|---|
| **Summary only** | Lower-end or older machines | Hides live transcript, minimizes realtime work, keeps transcript processing focused on generating summaries and final notes |
| **Balanced** | Most users | Best default tradeoff between responsiveness, transcript quality, subtitles, and summaries |
| **Performance** | Stronger CPUs/GPUs, mostly English meetings | Uses a more aggressive, quality-biased realtime configuration and expects more available compute |

Recommended starting point:

- If your machine struggles or the transcript falls behind, start with **Summary only**.
- If your machine is average and you want the normal MeetingBro experience, use **Balanced**.
- If your machine is strong and you want the most aggressive live behavior, try **Performance**.

Language choice is separate:

- Set **Speech = Auto** for multilingual meetings.
- Set **Speech = English / Chinese / German** for mostly single-language meetings.

---

## Step-by-step installation

### Requirements

| Tool | Version | How to get it |
|---|---|---|
| **Python** | 3.12 or later | [python.org](https://www.python.org/downloads/) or via Conda |
| **Conda** | any recent | [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) (Miniconda recommended) |
| **Node.js** | 18 or later | [nodejs.org](https://nodejs.org/) |
| **Git** | any | [git-scm.com](https://git-scm.com/) |

### Step 1 — Clone the repository

```bash
git clone https://github.com/armpro24-blip/MeetingBro.git
cd MeetingBro
```

### Step 2 — Create a Python environment

```bash
conda create -n MeetingBro python=3.12 -y
conda activate MeetingBro
```

You will need to run `conda activate MeetingBro` every time you open a new terminal to work on MeetingBro.

### Step 3 — Install the backend

```bash
cd app/backend
pip install -e "."
```

**Windows only** — install the WASAPI loopback helper library:

```bash
pip install "soundcard>=0.4"
```

This lets MeetingBro capture audio from Zoom, Teams, and other apps without any plugin.

Go back to the project root:

```bash
cd ../..
```

### Step 4 — Configure

Copy the configuration template:

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (Command Prompt)
copy .env.example .env
```

Open `.env` in any text editor. The defaults work without changes, but if you want AI summaries, add your LLM key (see [LLM setup](#llm-setup-for-ai-summaries) below).

### Step 5 — (Optional advanced) Download the Qwen3 preview model

This step is optional and not required for normal use. Start without it unless you are specifically testing the dedicated fast-preview path on a stronger machine.

```bash
pip install sherpa-onnx
pip install huggingface_hub

huggingface-cli download csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 --local-dir models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
```

If `huggingface-cli` is not on your PATH, use:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"
```

The model is about 700 MB and downloads once.

If you do not want the extra preview model, skip this step. To force the preview lane back to Whisper-only behavior, add this to your `.env`:

```env
MEETINGBRO_PREVIEW_ASR_BACKEND=whisper
```

### Step 6 — Install the frontend

```bash
cd app/frontend
npm install
cd ../..
```

---

## LLM setup for AI summaries

MeetingBro uses an **optional** cloud or local LLM for:

- AI meeting summaries (Rolling Summary and Meeting Board)
- Translation between Chinese, English, and German

**Without a key:** transcription still works. Summaries fall back to keyword extraction.

**With a key:** set these three lines in your `.env` file:

```env
MEETINGBRO_LLM_API_KEY=your_api_key_here
MEETINGBRO_LLM_BASE_URL=https://api.openai.com/v1
MEETINGBRO_LLM_MODEL=gpt-4o-mini
```

Supported providers include **OpenAI, Groq, Mistral AI, OpenRouter, Together AI**, and **Ollama** (fully local — no key, no data sent outside your machine).

→ **See [docs/llm-providers.md](docs/llm-providers.md) for full setup instructions for each provider.**

> **Privacy note:** Only the text transcript is sent to the LLM provider you configure. Audio capture and Whisper transcription run entirely on your machine. If you use a local LLM such as Ollama, no data leaves your device at all.

---

## Running MeetingBro

Always start the backend first, then the frontend.

### Backend

```bash
conda activate MeetingBro
cd app/backend
meetingbro-backend
```

You should see output like:

```
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Frontend

In a new terminal:

```bash
cd app/frontend
npm run dev
```

The Electron window opens automatically. If it does not, open your browser and go to `http://localhost:5173`.

---

## First successful run checklist

Before starting a session, verify:

- [ ] Backend is running (see terminal output above)
- [ ] Electron window has opened
- [ ] Audio device is selected in the UI
- [ ] For online meetings: select "System Audio" (macOS needs `BlackHole` + `Multi-Output Device`; Linux needs a loopback monitor)
- [ ] For in-person meetings: select your microphone

Click **Start Session** and speak a few words. You should see text appear in the transcript panel within a few seconds.

Before your first real meeting, choose a mode that matches your device:

- **Summary only** if you care more about final notes than live transcript.
- **Balanced** if you want the standard experience.
- **Performance** if your machine is strong and you want the heaviest realtime setup.

If the Whisper model hasn't been used before, the first transcription may take 10–30 seconds while the model loads. Subsequent transcriptions are faster.

---

## How it works

MeetingBro captures audio from your computer and processes it through a local pipeline:

```
Audio source (mic or system audio)
    ↓ Voice Activity Detection (VAD)
    ↓ Whisper ASR
    ↓ Optional preview lane (advanced setups only)
    ↓ Speaker diarization (optional)
    ↓ Translation (optional, via LLM)
    ↓ Summarization (optional, via LLM)
    ↓ Live UI (transcript + summaries)
    ↓ Export (Markdown files)
```

In the default setup, Whisper is the main engine. The optional Qwen3 preview path is no longer the normal recommendation for average devices.

Two audio capture modes:

- **Online mode** (Windows/macOS/Linux): captures system audio (macOS needs a virtual loopback device such as BlackHole plus a Multi-Output Device)
- **Offline mode** (all platforms): captures microphone — works for in-person meetings

See [docs/architecture.md](docs/architecture.md) for a full technical overview.

---

## Troubleshooting

Common issues and solutions are in **[docs/troubleshooting.md](docs/troubleshooting.md)**.

Quick checks:

1. Run `python scripts/dep_check.py` to verify all dependencies are installed.
2. Run `python scripts/list_audio_devices.py` to see available audio devices.
3. Make sure the backend is running before starting the frontend.

---

## Platform support

| Platform | Microphone | System audio | Notes |
|---|---|---|---|
| Windows 10/11 | ✅ | ✅ | Full support |
| macOS | ✅ | ⚠️ | System audio needs `BlackHole` (or similar) plus a `Multi-Output Device` |
| Linux | ✅ | ⚠️ | System audio via PulseAudio/PipeWire loopback (manual setup) |

See [docs/platform-support.md](docs/platform-support.md) for detailed instructions per platform.

---

## Frequently asked questions

See **[docs/faq.md](docs/faq.md)** for answers to common questions.

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to set up a development environment, run tests, and submit pull requests.

---

## Documentation index

| Document | Contents |
|---|---|
| [README.md](README.md) | This file — overview and quick start |
| [docs/llm-providers.md](docs/llm-providers.md) | How to configure an LLM API key |
| [docs/platform-support.md](docs/platform-support.md) | OS-specific audio setup |
| [docs/architecture.md](docs/architecture.md) | Technical architecture and component overview |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Common problems and solutions |
| [docs/faq.md](docs/faq.md) | Frequently asked questions |
| [docs/roadmap.md](docs/roadmap.md) | Development roadmap |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guide |
| [docs/product-principles.md](docs/product-principles.md) | Product and engineering principles |

---

## Suggested GitHub topics

Add these topics to the repository on GitHub for discoverability:

`meeting-assistant` `speech-to-text` `transcription` `whisper` `local-ai` `electron` `fastapi` `python` `multilingual` `privacy`

---

## License

MIT — see [LICENSE](LICENSE).
