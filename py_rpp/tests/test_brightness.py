"""
pytest tests for py_rpp.brightness (HIP backend).

Build py_rpp first, then run from the py_rpp/ root:
    source <AMD-important-scripts>/set_rocm_env/set_rocm_env.sh
    mkdir build && cd build && cmake .. && make -j$(nproc) && cd ..
    pytest tests/test_brightness.py -v
"""

import numpy as np
import pytest
import py_rpp


def make_batch(n, h, w, c, fill=None, dtype=np.uint8):
    if fill is not None:
        return np.full((n, h, w, c), fill, dtype=dtype)
    return np.random.randint(0, 200, (n, h, w, c), dtype=dtype)


def _pix(arr, n=0):
    """Return the value of pixel [n,0,0,0] for display."""
    return int(arr[n, 0, 0, 0])


# ─── passthrough ──────────────────────────────────────────────────────────────

def test_brightness_passthrough():
    """alpha=1, beta=0 must reproduce the input exactly."""
    src   = make_batch(2, 64, 64, 3)
    alpha = np.ones(2, dtype=np.float32)
    beta  = np.zeros(2, dtype=np.float32)
    print(f"\n  input shape : {src.shape}  dtype={src.dtype}")
    print(f"  alpha       : {alpha.tolist()}")
    print(f"  beta        : {beta.tolist()}")
    out = py_rpp.brightness(src, alpha, beta)
    print(f"  sample in   : img0[0,0,0]={_pix(src,0)}  img1[0,0,0]={_pix(src,1)}")
    print(f"  sample out  : img0[0,0,0]={_pix(out,0)}  img1[0,0,0]={_pix(out,1)}")
    print(f"  max abs diff: {np.abs(out.astype(int) - src.astype(int)).max()}")
    np.testing.assert_array_equal(out, src)


# ─── offset ───────────────────────────────────────────────────────────────────

def test_brightness_offset():
    """alpha=1, beta=50: output = clamp(pixel + 50, 0, 255)."""
    src      = make_batch(1, 8, 8, 3, fill=100)
    alpha    = np.array([1.0], dtype=np.float32)
    beta     = np.array([50.0], dtype=np.float32)
    expected = np.clip(src.astype(np.int32) + 50, 0, 255).astype(np.uint8)
    print(f"\n  input pixel : {_pix(src)}  (fill=100)")
    print(f"  alpha={alpha[0]}  beta={beta[0]}")
    print(f"  formula     : clamp({_pix(src)} * {alpha[0]} + {beta[0]}, 0, 255) = {_pix(expected)}")
    out = py_rpp.brightness(src, alpha, beta)
    print(f"  got         : {_pix(out)}  expected: {_pix(expected)}")
    np.testing.assert_array_equal(out, expected)


# ─── scale with clipping ──────────────────────────────────────────────────────

def test_brightness_scale_clipping():
    """alpha=2, beta=0 on pixel=200 must saturate to 255."""
    src   = make_batch(1, 4, 4, 3, fill=200)
    alpha = np.array([2.0], dtype=np.float32)
    beta  = np.array([0.0], dtype=np.float32)
    print(f"\n  input pixel : {_pix(src)}  (fill=200)")
    print(f"  alpha={alpha[0]}  beta={beta[0]}")
    print(f"  formula     : clamp({_pix(src)} * {alpha[0]} + {beta[0]}, 0, 255) → saturates at 255")
    out = py_rpp.brightness(src, alpha, beta)
    print(f"  got         : {_pix(out)}  expected: 255")
    np.testing.assert_array_equal(out, np.full_like(src, 255))


def test_brightness_scale_known():
    """alpha=2, beta=0 on pixel=100 → 200 (no clipping)."""
    src   = make_batch(1, 4, 4, 3, fill=100)
    alpha = np.array([2.0], dtype=np.float32)
    beta  = np.array([0.0], dtype=np.float32)
    print(f"\n  input pixel : {_pix(src)}  (fill=100)")
    print(f"  alpha={alpha[0]}  beta={beta[0]}")
    print(f"  formula     : clamp({_pix(src)} * {alpha[0]} + {beta[0]}, 0, 255) = 200")
    out = py_rpp.brightness(src, alpha, beta)
    print(f"  got         : {_pix(out)}  expected: 200")
    np.testing.assert_array_equal(out, np.full_like(src, 200))


# ─── batch: different params per image ───────────────────────────────────────

def test_brightness_batch_independence():
    """Each image in the batch gets its own alpha/beta."""
    src   = make_batch(2, 16, 16, 3, fill=100)
    alpha = np.array([1.0, 1.0], dtype=np.float32)
    beta  = np.array([0.0, 50.0], dtype=np.float32)
    print(f"\n  input shape : {src.shape}  fill=100")
    print(f"  img0: alpha={alpha[0]}  beta={beta[0]}  → expect 100")
    print(f"  img1: alpha={alpha[1]}  beta={beta[1]}  → expect 150")
    out = py_rpp.brightness(src, alpha, beta)
    print(f"  img0 got    : {_pix(out, 0)}  expected: 100")
    print(f"  img1 got    : {_pix(out, 1)}  expected: 150")
    np.testing.assert_array_equal(out[0], np.full((16, 16, 3), 100, dtype=np.uint8))
    np.testing.assert_array_equal(out[1], np.full((16, 16, 3), 150, dtype=np.uint8))


# ─── input validation ─────────────────────────────────────────────────────────

def test_brightness_wrong_ndim():
    """3-D input (missing batch dim) must raise."""
    bad = np.zeros((64, 64, 3), dtype=np.uint8)
    print(f"\n  input shape : {bad.shape}  (3-D, missing N dim)")
    print(f"  expect      : exception raised")
    with pytest.raises((ValueError, Exception)) as exc:
        py_rpp.brightness(bad, np.ones(1, dtype=np.float32), np.zeros(1, dtype=np.float32))
    print(f"  got         : {type(exc.value).__name__}: {exc.value}")


def test_brightness_alpha_wrong_length():
    """alpha length != batch size must raise."""
    src = make_batch(2, 8, 8, 3)
    print(f"\n  batch size  : {src.shape[0]}")
    print(f"  alpha length: 1  (mismatch — expect exception)")
    with pytest.raises((ValueError, Exception)) as exc:
        py_rpp.brightness(src, np.ones(1, dtype=np.float32), np.zeros(2, dtype=np.float32))
    print(f"  got         : {type(exc.value).__name__}: {exc.value}")
