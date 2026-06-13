from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardwareProfile:
    cpu_count: int
    ctranslate2_cuda_available: bool
    ctranslate2_cuda_device_count: int = 0
    ctranslate2_cuda_error: str | None = None
    onnx_providers: tuple[str, ...] = field(default_factory=tuple)
    onnx_cuda_available: bool = False
    recommended_whisper_device: str = "cpu"
    recommended_whisper_compute_type: str = "int8"
    recommended_whisper_size: str = "medium"
    recommended_qwen_provider: str = "cpu"
    recommended_qwen_threads: int = 2
    recommended_runtime_profile: str = "balanced"

    @property
    def label(self) -> str:
        accel = "cuda" if self.ctranslate2_cuda_available else "cpu"
        return f"{accel}/{self.cpu_count}c"

    @property
    def summary(self) -> str:
        cuda = (
            f"CUDA x{self.ctranslate2_cuda_device_count}"
            if self.ctranslate2_cuda_available
            else "CUDA unavailable"
        )
        return (
            f"{cuda}; Whisper {self.recommended_whisper_device} "
            f"{self.recommended_whisper_compute_type}; "
            f"Qwen {self.recommended_qwen_provider} {self.recommended_qwen_threads} threads"
        )


def _detect_ctranslate2_cuda() -> tuple[bool, int, str | None]:
    try:
        import ctranslate2  # type: ignore

        get_count = getattr(ctranslate2, "get_cuda_device_count", None)
        if not callable(get_count):
            return False, 0, "ctranslate2 has no get_cuda_device_count()"
        count = int(get_count())
        if count <= 0:
            return False, 0, None
        return True, count, None
    except Exception as exc:
        return False, 0, str(exc)


def _detect_onnx_providers() -> tuple[tuple[str, ...], bool]:
    try:
        import onnxruntime as ort  # type: ignore

        providers = tuple(str(p) for p in ort.get_available_providers())
        return providers, "CUDAExecutionProvider" in providers
    except Exception:
        return (), False


def detect_hardware_profile() -> HardwareProfile:
    """Return a lightweight hardware/runtime capability profile.

    This intentionally detects import-level runtime capability, not just the
    physical device. A visible NVIDIA GPU is not useful to MeetingBro unless the
    current Python environment can actually initialize the relevant backend.
    """

    cpu_count = max(1, os.cpu_count() or 1)
    ct2_cuda, ct2_cuda_count, ct2_error = _detect_ctranslate2_cuda()
    onnx_providers, onnx_cuda = _detect_onnx_providers()

    if ct2_cuda:
        whisper_device = "cuda"
        whisper_compute = "float16"
        # large-v3-turbo decodes ~8x faster than large-v3 with near-SOTA accuracy;
        # on a GPU it is the best speed/accuracy trade-off for realtime captions.
        whisper_size = "large-v3-turbo"
        runtime_profile = "performance"
    else:
        whisper_device = "cpu"
        whisper_compute = "int8"
        whisper_size = "small" if cpu_count <= 4 else "medium"
        runtime_profile = "summary_only" if cpu_count <= 4 else "balanced"

    # Keep preview/Qwen on CPU by default. This separates Qwen preview from a
    # CUDA Whisper formal lane and avoids both models fighting for the same GPU.
    qwen_threads = 2 if cpu_count < 12 else 3

    return HardwareProfile(
        cpu_count=cpu_count,
        ctranslate2_cuda_available=ct2_cuda,
        ctranslate2_cuda_device_count=ct2_cuda_count,
        ctranslate2_cuda_error=ct2_error,
        onnx_providers=onnx_providers,
        onnx_cuda_available=onnx_cuda,
        recommended_whisper_device=whisper_device,
        recommended_whisper_compute_type=whisper_compute,
        recommended_whisper_size=whisper_size,
        recommended_qwen_provider="cpu",
        recommended_qwen_threads=qwen_threads,
        recommended_runtime_profile=runtime_profile,
    )

