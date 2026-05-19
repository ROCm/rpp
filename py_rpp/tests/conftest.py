"""
Shared pytest fixtures, marks, and helpers for py_rpp tests.

Test structure
--------------
* ``backend`` fixture — parametrized ["cpu", "hip"]; the "hip" variant is
  automatically skipped when no ROCm GPU is detected.
* ``make_batch`` / ``make_nchw`` / ``pix`` — concise image generators used
  across all test modules (imported directly, not fixtures, for clarity).

Running
-------
    pytest                          # CPU + HIP (HIP skipped if no GPU)
    pytest -m "not hip"             # CPU only
    pytest -m hip                   # HIP only (skip if no GPU)
    pytest -k test_brightness       # single test by keyword
    PY_RPP_DEBUG=1 pytest           # enable C++ call-trace logging
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

# ── Import: prefer the build tree over the source tree ───────────────────────
_BUILD_DIR = Path(__file__).parent.parent / "build"
if _BUILD_DIR.exists():
    sys.path.insert(0, str(_BUILD_DIR))

import py_rpp  # noqa: E402 — must follow sys.path manipulation


# ── GPU detection (run once at collection time) ───────────────────────────────

def _detect_gpu() -> bool:
    """Return True if at least one ROCm-capable GPU is visible."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showbus"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


GPU_AVAILABLE: bool = _detect_gpu()


# ── Custom marks ──────────────────────────────────────────────────────────────

# Convenience decorator: @pytest.mark.requires_hip
# Applied automatically by the ``backend`` fixture for the "hip" variant.
# Can also be used on entire test classes or modules that only make sense
# on GPU (e.g., timing benchmarks).
requires_hip = pytest.mark.skipif(
    not GPU_AVAILABLE,
    reason="No ROCm GPU detected — install ROCm or run on a GPU node",
)


# ── Backend fixture ───────────────────────────────────────────────────────────

@pytest.fixture(
    params=[
        "cpu",
        pytest.param("hip", marks=requires_hip),
    ],
    ids=["cpu", "hip"],
)
def backend(request: pytest.FixtureRequest) -> str:
    """Parametrised fixture: runs every consuming test on both CPU and HIP."""
    return request.param


# ── Image batch generators ────────────────────────────────────────────────────

def make_batch(
    n: int = 2,
    h: int = 32,
    w: int = 32,
    c: int = 3,
    *,
    fill: Optional[int] = None,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """Return a (N, H, W, C) image batch, uniform-filled or random uint8."""
    if fill is not None:
        return np.full((n, h, w, c), fill, dtype=dtype)
    return np.random.randint(0, 256, (n, h, w, c), dtype=dtype)


def make_nchw(
    n: int = 2,
    h: int = 32,
    w: int = 32,
    c: int = 3,
    *,
    fill: Optional[int] = None,
) -> np.ndarray:
    """Return a (N, C, H, W) image batch for layout-detection tests."""
    nhwc = make_batch(n, h, w, c, fill=fill)
    return np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))


def pix(arr: np.ndarray, n: int = 0) -> int:
    """Return pixel [n, 0, 0, 0] as int — handy for one-liner spot checks."""
    return int(arr[n, 0, 0, 0])
