# Troubleshooting

## First steps

Before anything else, run the dependency checker:

```bash
conda activate MeetingBro
python scripts/dep_check.py
```

This will tell you if a required library is missing or the wrong version.

---

## Backend won't start

**Error: `command not found: meetingbro-backend`**

The backend package is not installed, or the wrong Python environment is active.

```bash
conda activate MeetingBro
cd app/backend
pip install -e "."
meetingbro-backend
```

---

**Error: `ModuleNotFoundError: No module named 'soundcard'`**

On Windows, the system audio loopback library is missing:

```bash
pip install "soundcard>=0.4"
```

This is only needed on Windows for capturing system audio. On macOS/Linux you can ignore this error if you only use microphone input.

---

**Error: `Address already in use` on port 8000**

Another process is using port 8000. Either stop that process or change the port:

```bash
# Find what is using port 8000 (Windows)
netstat -ano | findstr :8000

# Find what is using port 8000 (macOS/Linux)
lsof -i :8000
```

---

## Frontend won't start or window doesn't open

**Error: `Cannot find module` or similar Node.js error**

Dependencies are not installed:

```bash
cd app/frontend
npm install
npm run dev
```

---

**Electron window does not open**

Try navigating to `http://localhost:5173` in your browser instead. If the page loads, Electron may have a display issue. Check the terminal for error messages.

---

## No transcription appearing

**Possible cause: Whisper model is loading for the first time**

The first time MeetingBro runs, it downloads the Whisper model (hundreds of MB). This can take 1–5 minutes depending on your internet speed. Wait for the backend terminal to show "model loaded" or similar output.

---

**Possible cause: wrong audio device selected**

Open a new terminal and run:

```bash
python scripts/list_audio_devices.py
```

This lists all available audio devices. Make sure you select the right one in the UI:
- For online meetings on Windows: look for a device with "loopback" or "stereo mix" in the name, or the name of your audio output device.
- For microphone input: select your microphone by name.

---

**Windows: loopback capture fails when headphones / headset are connected**

Some Bluetooth and USB headsets expose an audio format that the WASAPI loopback layer cannot open (the driver reports a non-float32 mix format). Symptoms: loopback works fine with built-in speakers, but as soon as you plug in the headset the backend throws `unsupported format` or falls back silently to microphone-only capture.

Work-arounds (pick one):

1. **Disconnect the headset** and use built-in speakers for the meeting, then re-connect it afterwards.
2. **Keep speakers as the default multimedia playback device** while setting the headset as the default *communication* device only (right-click the speaker icon → Sounds → Playback tab → right-click your headset → "Set as Default Communication Device", **not** "Set as Default Device").
3. **Update the headset audio driver** – newer drivers sometimes fix the WASAPI format negotiation bug.
4. **Use "mixed" audio source** (`?source=mixed` in the WebSocket URL or select it in the UI). This at least keeps microphone capture working while the loopback path is unavailable.

If you are still stuck, run `python scripts/list_audio_devices.py` and include the **soundcard** section (especially the loopback lines) in your bug report.

---

**Possible cause: microphone is muted or volume is too low**

Check your system volume settings. MeetingBro has a noise floor filter — very quiet audio will be ignored. Try speaking louder or moving closer to the microphone.

---

**Possible cause: VAD is filtering out speech**

If the room is noisy or the microphone signal is weak, the Voice Activity Detector (VAD) may not trigger. Try adding this to your `.env`:

```env
MEETINGBRO_PRE_VAD_THRESHOLD=0.3
MEETINGBRO_PRE_VAD_ENERGY_RMS_THRESHOLD=0.003
MEETINGBRO_WEAK_SPEECH_RESCUE_ENABLED=true
```

---

## Transcription is slow or falling behind

**Try a smaller Whisper model:**

```env
MEETINGBRO_WHISPER_SIZE=small
```

**Switch to a faster runtime profile:**

```env
MEETINGBRO_RUNTIME_PROFILE=low_latency
```

**Reduce the maximum segment length:**

```env
MEETINGBRO_PRE_VAD_MAX_SEGMENT_SECONDS=4
```

If the CPU is consistently overloaded, check the backend terminal for safeguard messages like `ASR safeguard triggered`.

---

## Transcription quality is poor

**Try a larger Whisper model:**

```env
MEETINGBRO_WHISPER_SIZE=medium
```

**For noisy environments, increase VAD sensitivity:**

```env
MEETINGBRO_WHISPER_VAD_THRESHOLD=0.35
MEETINGBRO_AUDIO_CONDITIONING_ENABLED=true
```

**For quiet speakers, lower the VAD threshold:**

```env
MEETINGBRO_PRE_VAD_THRESHOLD=0.3
MEETINGBRO_WHISPER_VAD_THRESHOLD=0.25
```

**Add vocabulary hints** for domain-specific words (names, product names, technical terms):

```env
MEETINGBRO_VOCABULARY_HINT=Alice, Bob, Kubernetes, FastAPI
```

---

## Qwen3 preview model fails to load

**Error: `sherpa_onnx` not installed**

```bash
pip install sherpa-onnx
```

**Error: model directory not found**

Check that the model directory exists under `models/`:

```bash
# Windows
dir models\

# macOS/Linux
ls models/
```

If the directory is missing, re-run the download:

```bash
pip install huggingface_hub
huggingface-cli download csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 --local-dir models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
```

If `huggingface-cli` is not on your PATH, use:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25', local_dir='models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25')"
```

**To disable Qwen3 and fall back to Whisper preview:**

```env
MEETINGBRO_PREVIEW_ASR_BACKEND=whisper
```

---

## LLM summaries are not appearing

**Check that the LLM key is set:**

Open `.env` and verify:

```env
MEETINGBRO_LLM_API_KEY=your_actual_key
MEETINGBRO_LLM_BASE_URL=https://api.openai.com/v1
MEETINGBRO_LLM_MODEL=gpt-4o-mini
```

**Check the backend logs** for errors like `AuthenticationError` or `ConnectionError`.

**Test that the API is reachable:**

```bash
# macOS / Linux
curl https://api.openai.com/v1/models -H "Authorization: Bearer your_actual_key"

# Windows PowerShell / Command Prompt
curl.exe https://api.openai.com/v1/models -H "Authorization: Bearer your_actual_key"
```

If you see a JSON response, the key works. If you see `Unauthorized`, the key is wrong or expired.

---

## Export fails or exports folder is empty

The `exports/` directory is created automatically. If export fails, check:

1. The backend is still running during export.
2. The `exports/` directory exists and is writable.
3. The backend terminal for any error messages.

---

## WebSocket connection errors in the UI

If the frontend shows "connection lost" or similar:

1. Make sure the backend is running (`meetingbro-backend`).
2. Check that nothing else is using port 8000.
3. Restart the backend, then refresh the frontend.

Run the WebSocket smoke test to verify the backend is healthy:

```bash
python scripts/ws_smoketest.py
```

---

## Still stuck?

Please open an issue and include:

- Your OS and Python version (`python --version`)
- Output of `python scripts/dep_check.py`
- The full error message from the backend terminal
- Steps to reproduce the problem
