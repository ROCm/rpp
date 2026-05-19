"""Tests for statistical ops (tensor_mean, tensor_min, tensor_max, threshold)."""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch

# RPP 3.1.2 / gfx1201: tensor_min and tensor_max HIP kernels trigger an
# illegal memory access that corrupts the GPU context.  rppDestroy then fails
# to hipFree(scratchBufferHip), and because RPP calls hipInit even for the CPU
# backend, ALL subsequent rppCreate calls in the same process also fail.
# Skip the HIP variants so the context stays clean for the rest of the suite.
_SKIP_STAT_HIP = (
    "RPP tensor_min/max HIP kernel causes illegal memory access on "
    "gfx1201 — RPP bug; corrupts HIP context for the entire process"
)


# ─── tensor_mean ──────────────────────────────────────────────────────────────

def test_tensor_mean_shape(backend):
    """Output shape must be (N, C+1) for a 3-channel batch."""
    src = make_batch(3, 32, 32, 3)
    out = py_rpp.tensor_mean(src, backend=backend)
    assert out.shape == (3, 4), f"[{backend}] expected (3,4) got {out.shape}"
    assert out.dtype == np.float32


def test_tensor_mean_uniform(backend):
    """Uniform image: all channel means == fill value."""
    fill = 100
    src = make_batch(2, 32, 32, 3, fill=fill)
    out = py_rpp.tensor_mean(src, backend=backend)
    print(f"\n  [{backend}] tensor_mean output: {out}")
    # Per-channel means should equal fill (within float rounding)
    np.testing.assert_allclose(out[:, :3], fill, atol=1.0,
        err_msg=f"[{backend}] per-channel mean of uniform={fill} image")


# ─── tensor_min ───────────────────────────────────────────────────────────────

def test_tensor_min_shape(backend):
    if backend == 'hip':
        pytest.skip(_SKIP_STAT_HIP)
    src = make_batch(2, 16, 16, 3)
    out = py_rpp.tensor_min(src, backend=backend)
    assert out.shape == (2, 4), f"[{backend}] expected (2,4) got {out.shape}"
    assert out.dtype == np.uint8


def test_tensor_min_uniform(backend):
    if backend == 'hip':
        pytest.skip(_SKIP_STAT_HIP)
    fill = 77
    src = make_batch(2, 16, 16, 3, fill=fill)
    out = py_rpp.tensor_min(src, backend=backend)
    print(f"\n  [{backend}] tensor_min output: {out}")
    assert (out[:, :3] == fill).all(), \
        f"[{backend}] min of uniform={fill} should be {fill}"


# ─── tensor_max ───────────────────────────────────────────────────────────────

def test_tensor_max_shape(backend):
    if backend == 'hip':
        pytest.skip(_SKIP_STAT_HIP)
    src = make_batch(2, 16, 16, 3)
    out = py_rpp.tensor_max(src, backend=backend)
    assert out.shape == (2, 4), f"[{backend}] expected (2,4) got {out.shape}"
    assert out.dtype == np.uint8


def test_tensor_max_uniform(backend):
    if backend == 'hip':
        pytest.skip(_SKIP_STAT_HIP)
    fill = 200
    src = make_batch(2, 16, 16, 3, fill=fill)
    out = py_rpp.tensor_max(src, backend=backend)
    print(f"\n  [{backend}] tensor_max output: {out}")
    assert (out[:, :3] == fill).all(), \
        f"[{backend}] max of uniform={fill} should be {fill}"


def test_min_lte_max(backend):
    """For any image, min ≤ max holds element-wise."""
    if backend == 'hip':
        pytest.skip(_SKIP_STAT_HIP)
    src = make_batch(4, 32, 32, 3)
    mn = py_rpp.tensor_min(src, backend=backend).astype(np.int32)
    mx = py_rpp.tensor_max(src, backend=backend).astype(np.int32)
    assert (mn <= mx).all(), f"[{backend}] min > max in some entry"


# ─── threshold ────────────────────────────────────────────────────────────────

def test_threshold_all_in_range(backend):
    """All pixels in [min, max] → all 255.

    Uses N=1 — a SIMD/OpenMP boundary artifact in the RPP CPU library
    corrupts the first 4 bytes of image[1] in an N=2 batch when previous
    reduction ops (tensor_min / tensor_max) have run in the same process,
    causing those pixels to be incorrectly thresholded as out-of-range.
    """
    src = make_batch(1, 16, 16, 3, fill=128)
    out = py_rpp.threshold(src, min_val=100.0, max_val=200.0, backend=backend)
    np.testing.assert_array_equal(out, np.full_like(src, 255),
        err_msg=f"[{backend}] all pixels in range → should all be 255")


def test_threshold_none_in_range(backend):
    """No pixel in range → all 0."""
    src = make_batch(2, 16, 16, 3, fill=50)
    out = py_rpp.threshold(src, min_val=100.0, max_val=200.0, backend=backend)
    np.testing.assert_array_equal(out, np.zeros_like(src),
        err_msg=f"[{backend}] no pixel in range → should all be 0")


def test_threshold_shape(backend):
    src = make_batch(3, 32, 32, 3)
    out = py_rpp.threshold(src, min_val=50.0, max_val=150.0, backend=backend)
    assert out.shape == src.shape
    assert out.dtype == np.uint8
