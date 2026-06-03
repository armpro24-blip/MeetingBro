"""Print WASAPI audio device info so we can target the right loopback device."""
from __future__ import annotations

import sys


def _sounddevice_section() -> None:
    import sounddevice as sd

    print("=== sounddevice ===")
    print("sounddevice version:", sd.__version__)
    hostapis = sd.query_hostapis()
    for i, h in enumerate(hostapis):
        print(f"\nhostapi[{i}] name={h['name']!r} default_in={h.get('default_input_device')} default_out={h.get('default_output_device')} keys={sorted(h.keys())}")

    print("\n--- devices ---")
    for i, d in enumerate(sd.query_devices()):
        print(
            f"[{i}] hostapi={d['hostapi']} name={d['name']!r} "
            f"max_in={d['max_input_channels']} max_out={d['max_output_channels']} "
            f"default_sr={d['default_samplerate']}"
        )

    wasapi_idx = next(
        (i for i, h in enumerate(hostapis) if "WASAPI" in h["name"].upper()), None
    )
    print("\nWASAPI hostapi index:", wasapi_idx)
    if wasapi_idx is not None:
        default_out = hostapis[wasapi_idx]["default_output_device"]
        print("WASAPI default output device index:", default_out)
        if default_out >= 0:
            print("WASAPI default output device info:", sd.query_devices(default_out))


def _soundcard_section() -> None:
    try:
        import soundcard as sc
    except Exception as exc:  # pragma: no cover
        print(f"\n=== soundcard (not installed: {exc}) ===")
        return

    print("\n=== soundcard ===")
    print("soundcard version:", getattr(sc, "__version__", "unknown"))

    try:
        default_speaker = sc.default_speaker()
        print(f"\ndefault speaker: {default_speaker.name!r} id={default_speaker.id!r} channels={default_speaker.channels}")
    except Exception as exc:
        print(f"\ndefault speaker: ERROR {exc}")

    try:
        default_mic = sc.default_microphone()
        print(f"default microphone: {default_mic.name!r} id={default_mic.id!r} channels={default_mic.channels}")
    except Exception as exc:
        print(f"default microphone: ERROR {exc}")

    print("\n--- all speakers ---")
    for sp in sc.all_speakers():
        print(f"  {sp.name!r} id={sp.id!r} channels={sp.channels}")

    print("\n--- all microphones (including loopback) ---")
    for mic in sc.all_microphones(include_loopback=True):
        loopback_flag = " [LOOPBACK]" if mic.isloopback else ""
        print(f"  {mic.name!r} id={mic.id!r} channels={mic.channels}{loopback_flag}")


def main() -> int:
    _sounddevice_section()
    _soundcard_section()
    return 0


if __name__ == "__main__":
    sys.exit(main())
