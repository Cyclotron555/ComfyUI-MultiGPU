"""Opt-in local device policy. No driver or power-management changes."""
import math
import os


def primary_device(devices, core_device):
    requested = os.environ.get("MGPU_PRIMARY_DEVICE")
    if requested:
        if requested not in devices:
            raise ValueError(f"MGPU_PRIMARY_DEVICE {requested!r} is unavailable: {devices}")
        return requested
    return str(core_device) if str(core_device) in devices else devices[0]


def memory_budget(device, total, free):
    """Conservative storage budget, not an allocator guarantee."""
    if os.environ.get("MGPU_SAFE_OFFLOAD") != "1":
        return total
    reserve = 8.0 if device == "cpu" else (2.0 if device == os.environ.get("MGPU_PRIMARY_DEVICE") else 1.0)
    value = float(os.environ.get("MGPU_RAM_RESERVE_GIB" if device == "cpu" else "MGPU_GPU_RESERVE_GIB", reserve))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Memory reservation must be finite and nonnegative")
    return max(0, min(total, free) - value * 1024**3)
