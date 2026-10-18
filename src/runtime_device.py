"""Shared CUDA-first device selection for this project's model workloads."""
from __future__ import annotations

import gc
import re


def device_preference(settings: dict, component: str | None = None) -> str:
    """A component override wins; otherwise use runtime.device (default: auto)."""
    override = (settings.get(component) or {}).get("device") if component else None
    return str(override or (settings.get("runtime") or {}).get("device") or "auto")


def select_device(preference: str | None = "auto") -> str:
    """Prefer a working NVIDIA CUDA GPU, falling back to CPU with an explanation."""
    preference = (preference or "auto").strip().lower()
    if preference != "auto" and preference != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", preference):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>.")
    if preference == "cpu":
        print("Model device: CPU (explicit setting).", flush=True)
        return "cpu"

    import torch

    index = int(preference.split(":")[1]) if ":" in preference else 0
    device = f"cuda:{index}"
    try:
        if not torch.cuda.is_available():
            print("CUDA GPU unavailable; using CPU.", flush=True)
            return "cpu"
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist")
        # Detection alone does not prove the installed driver/build can run kernels.
        probe = torch.ones(1, device=device)
        probe.add_(1)
        torch.cuda.synchronize(device)
        name = torch.cuda.get_device_name(index)
    except (RuntimeError, AssertionError) as error:
        print(f"CUDA GPU is not usable ({error}); using CPU.", flush=True)
        clear_cuda_cache()
        return "cpu"
    print(f"Model device: {device} ({name}).", flush=True)
    return device


def is_cuda_resource_error(error: BaseException) -> bool:
    """Only classify memory/backend failures; do not hide ordinary model/code errors."""
    import torch

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return any(marker in message for marker in (
        "cuda out of memory", "cuda error: out of memory", "no kernel image is available",
        "not compiled with cuda", "cuda driver", "nvidia driver", "cuda initialization",
        "cuda error: initialization", "invalid device ordinal",
        "cublas_status_not_initialized", "cudnn_status_not_initialized",
        "all cuda-capable devices are busy", "no cuda-capable device",
    ))


def clear_cuda_cache() -> None:
    """Release unused allocations after a GPU failure, when the backend permits it."""
    import torch

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except (RuntimeError, AssertionError):
        # An unavailable/broken CUDA backend must not prevent a CPU retry.
        pass
