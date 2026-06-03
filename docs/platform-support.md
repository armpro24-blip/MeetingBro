# Platform Support

## Summary

| Feature | Windows 10/11 | macOS | Linux |
|---|---|---|---|
| Microphone capture | ✅ | ✅ | ✅ |
| System audio (online meetings) | ✅ Native | ⚠️ Virtual cable required | ⚠️ Loopback device required |
| Electron desktop app | ✅ | ✅ | ✅ |
| Whisper ASR | ✅ | ✅ | ✅ |
| Qwen3 preview ASR | ✅ | ✅ | ✅ |
| GPU acceleration (CUDA) | ✅ NVIDIA | ✅ Metal (limited) | ✅ NVIDIA |

---

## Windows

Windows is the most fully supported platform.

### System audio capture

MeetingBro uses **Windows WASAPI loopback** via the `soundcard` library to capture system audio. This lets it hear everything your speakers play, regardless of which meeting app is running.

Install the loopback library:
```bash
pip install "soundcard>=0.4"
```

In the MeetingBro UI, select the audio output device (e.g., "Speakers (Realtek)" or "Headphones") as the capture source. MeetingBro will automatically select the loopback variant.

If no loopback device appears, check:
- Your audio drivers are up to date
- "Stereo Mix" is enabled in Windows Sound settings (right-click the speaker icon → Sounds → Recording tab → right-click to show disabled devices → enable Stereo Mix)

> **Headset / headphone caveat:** Some Bluetooth and USB headsets expose a PCM format that WASAPI loopback cannot initialise. If loopback works with built-in speakers but fails as soon as you connect a headset, see [Troubleshooting: Windows loopback capture fails when headphones are connected](troubleshooting.md#windows-loopback-capture-fails-when-headphones--headset-are-connected).

### Microphone capture

Works out of the box. Select your microphone in the UI.

---

## macOS

Microphone capture works natively. System audio capture requires a third-party virtual audio device.

### System audio capture (virtual cable)

The recommended free option is **BlackHole**:

1. Download and install [BlackHole 2ch](https://existential.audio/blackhole/)

2. Open **Audio MIDI Setup** (search in Spotlight)

3. Click the `+` button → **Create Multi-Output Device**
   - Add both **BlackHole 2ch** and your normal output (e.g., MacBook Pro Speakers or your headphones)
   - Check "Use" for both

4. Right-click the Multi-Output Device → **Use this device for sound output**
   (From now on, your system audio plays through both your speakers and BlackHole)

5. In MeetingBro, select **BlackHole 2ch** as the audio input device

6. Start your meeting — MeetingBro will now capture what the meeting app plays

> **Important:** when you are done, remember to switch your system output back to your regular speakers/headphones.

### Microphone + system audio simultaneously

Create a separate **Aggregate Device** in Audio MIDI Setup that combines your microphone and BlackHole as inputs. This lets MeetingBro capture both in one stream.

---

## Linux

Microphone capture works on most distributions. System audio capture depends on your audio server.

### System audio capture with PipeWire (recommended)

Most modern distributions (Ubuntu 22.04+, Fedora 34+, Arch) use PipeWire.

1. Install `pw-loopback`:
   ```bash
   # Ubuntu/Debian
   sudo apt install pipewire-audio-client-libraries

   # Arch
   sudo pacman -S pipewire
   ```

2. Create a loopback module:
   ```bash
   pw-loopback --capture-props='media.class=Audio/Sink' &
   ```
   This creates a virtual sink that also appears as a capture source.

3. Route your meeting app's audio to this sink using `pavucontrol` or `qpwgraph`.

4. In MeetingBro, select the loopback capture device.

### System audio capture with PulseAudio

1. Load the null sink module:
   ```bash
   pactl load-module module-null-sink sink_name=meetingbro_loopback
   pactl load-module module-loopback source=meetingbro_loopback.monitor
   ```

2. Route your meeting app to the null sink using `pavucontrol`.

3. In MeetingBro, select `meetingbro_loopback.monitor` as the input device.

### Listing audio devices

```bash
python scripts/list_audio_devices.py
```

This shows all devices that MeetingBro can see. Use the exact device name or index in the UI.

---

## GPU acceleration

Faster-whisper supports GPU inference, which significantly speeds up transcription.

### NVIDIA (CUDA)

```env
MEETINGBRO_WHISPER_DEVICE=cuda
MEETINGBRO_WHISPER_COMPUTE_TYPE=float16
```

Requirements:
- CUDA 11.8 or later
- cuDNN 8.x
- `pip install faster-whisper` (already included) — detects CUDA automatically

Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Apple Silicon (Metal)

Faster-whisper does not currently support Metal acceleration. Transcription runs on CPU, which is still fast on M-series chips due to unified memory.

### No GPU

The default `MEETINGBRO_WHISPER_DEVICE=auto` will use CPU if no GPU is found. On a modern CPU, the `small` model runs at approximately 2–5x real time.
