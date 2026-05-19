# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""Utility helpers: GPU detection, backend selection, layout conversion."""

import subprocess
import numpy as np
from .types import HIP, CPU, NHWC, NCHW


def is_gpu_available() -> bool:
    """Return True when a ROCm-capable GPU is accessible."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showbus"],
            capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


_default_backend: str | None = None


def get_default_backend() -> str:
    """Return 'hip' when a GPU is available, otherwise 'cpu'."""
    global _default_backend
    if _default_backend is None:
        _default_backend = HIP if is_gpu_available() else CPU
    return _default_backend


def detect_layout(images: np.ndarray) -> str:
    """
    Heuristic layout detection for a 4-D numpy array.

    Rules (mirrors the upstream rpp_pybind convention):
    - dim[-1] in {1, 3, 4} and dim[1] not in {1, 3, 4}  → NHWC
    - dim[1]  in {1, 3, 4} and dim[-1] not in {1, 3, 4} → NCHW
    - Both match → NHWC (uint8 images are almost always packed)
    - Neither matches → NHWC (assume packed)
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D array, got {images.ndim}-D")
    last = images.shape[-1]
    second = images.shape[1]
    last_ok   = last   in (1, 2, 3, 4)
    second_ok = second in (1, 2, 3, 4)
    if last_ok and not second_ok:
        return NHWC
    if second_ok and not last_ok:
        return NCHW
    return NHWC  # default; packed uint8 is by far the common case


def to_nhwc(images: np.ndarray, layout: str | None = None) -> np.ndarray:
    """
    Ensure images are in NHWC layout.  Transposes NCHW → NHWC if needed.
    Returns a C-contiguous uint8 array.  layout=None triggers auto-detection.
    """
    if layout is None:
        layout = detect_layout(images)
    if layout.lower() == NCHW:
        images = np.ascontiguousarray(images.transpose(0, 2, 3, 1))
    return np.ascontiguousarray(images)


def to_nchw(images: np.ndarray) -> np.ndarray:
    """Transpose an NHWC array to NCHW layout."""
    return np.ascontiguousarray(images.transpose(0, 3, 1, 2))


__all__ = [
    "is_gpu_available", "get_default_backend",
    "detect_layout", "to_nhwc", "to_nchw",
]
