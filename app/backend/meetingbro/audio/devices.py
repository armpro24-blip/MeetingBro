"""Audio device enumeration for the in-app device picker.

Read-only helpers that list the input devices (microphones) and the system
output devices that can be captured via WASAPI loopback. Both are best-effort:
any backend that is unavailable on the current platform simply yields an empty
list rather than raising.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _enumerate_microphones() -> list[dict]:
    mics: list[dict] = []
    try:
        import sounddevice as sd
    except Exception as exc:  # noqa: BLE001
        logger.info("sounddevice unavailable for mic enumeration: %s", exc)
        return mics

    try:
        default = sd.default.device
        default_input = default[0] if isinstance(default, (list, tuple)) else None
    except Exception:
        default_input = None

    try:
        for idx, dev in enumerate(sd.query_devices()):
            if int(dev.get("max_input_channels", 0)) > 0:
                mics.append(
                    {
                        "id": str(idx),
                        "name": dev.get("name", f"Device {idx}"),
                        "default": idx == default_input,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("microphone enumeration failed: %s", exc)
    return mics


def _enumerate_loopbacks() -> tuple[list[dict], dict | None]:
    loopbacks: list[dict] = []
    system_default: dict | None = None
    try:
        import soundcard as sc  # Windows / WASAPI loopback
    except Exception as exc:  # noqa: BLE001
        logger.info("soundcard unavailable for loopback enumeration: %s", exc)
        return loopbacks, system_default

    try:
        default_speaker = sc.default_speaker()
    except Exception:
        default_speaker = None

    try:
        for spk in sc.all_speakers():
            is_default = default_speaker is not None and spk.name == default_speaker.name
            # The loopback source resolves speakers by name via
            # sc.get_microphone(name, include_loopback=True), so id == name.
            loopbacks.append({"id": spk.name, "name": spk.name, "default": is_default})
    except Exception as exc:  # noqa: BLE001
        logger.warning("loopback enumeration failed: %s", exc)

    if default_speaker is not None:
        system_default = {"id": default_speaker.name, "name": default_speaker.name}
    return loopbacks, system_default


def enumerate_audio_devices() -> dict:
    mics = _enumerate_microphones()
    loopbacks, system_default = _enumerate_loopbacks()
    return {"mics": mics, "loopbacks": loopbacks, "system_default": system_default}
