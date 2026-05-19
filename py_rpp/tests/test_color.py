"""
Tests for color augmentation ops.

Each test runs on both CPU and HIP via the ``backend`` fixture.
Tests use fn.py's scalar-broadcasting API — no manual numpy param arrays needed.
"""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch, make_nchw, pix


# ─── brightness ───────────────────────────────────────────────────────────────

def test_brightness_passthrough(backend):
    """alpha=1, beta=0 must be an exact identity."""
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.brightness(src, alpha=1.0, beta=0.0, backend=backend)
    assert out.shape == src.shape, f"shape mismatch: {out.shape}"
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] brightness(alpha=1, beta=0) should be identity")


def test_brightness_offset(backend):
    """alpha=1, beta=50: output = clamp(pixel + 50, 0, 255)."""
    src = make_batch(1, 8, 8, 3, fill=100)
    expected = np.clip(src.astype(np.int32) + 50, 0, 255).astype(np.uint8)
    out = py_rpp.brightness(src, alpha=1.0, beta=50.0, backend=backend)
    print(f"\n  [{backend}] in={pix(src)}  expected={pix(expected)}  got={pix(out)}")
    np.testing.assert_array_equal(out, expected)


def test_brightness_scale_and_clamp(backend):
    """alpha=2, beta=0: pixel=200 → saturates to 255."""
    src = make_batch(1, 4, 4, 3, fill=200)
    out = py_rpp.brightness(src, alpha=2.0, beta=0.0, backend=backend)
    np.testing.assert_array_equal(out, np.full_like(src, 255),
        err_msg=f"[{backend}] 200*2 should clamp to 255")


def test_brightness_batch_independence(backend):
    """Per-image alpha/beta are independent."""
    src = make_batch(2, 16, 16, 3, fill=100)
    out = py_rpp.brightness(src, alpha=[1.0, 1.0], beta=[0.0, 50.0], backend=backend)
    print(f"\n  [{backend}] img0={pix(out,0)} (exp 100)  img1={pix(out,1)} (exp 150)")
    np.testing.assert_array_equal(out[0], np.full((16, 16, 3), 100, dtype=np.uint8))
    np.testing.assert_array_equal(out[1], np.full((16, 16, 3), 150, dtype=np.uint8))


def test_brightness_nchw_input(backend):
    """NCHW input is auto-detected and transposed; result has same NHWC shape."""
    src_nchw = make_nchw(2, 32, 32, 3, fill=100)
    # fn.py detects NCHW and converts before calling C++
    out = py_rpp.brightness(src_nchw, alpha=1.0, beta=0.0, backend=backend)
    assert out.shape == (2, 32, 32, 3), f"Unexpected output shape: {out.shape}"


def test_brightness_wrong_ndim(backend):
    bad = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        py_rpp.brightness(bad, alpha=1.0, beta=0.0, backend=backend)


# ─── gamma_correction ─────────────────────────────────────────────────────────

def test_gamma_identity(backend):
    """gamma=1.0 should be approximately identity (LUT quantization ≤ 1/255)."""
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.gamma_correction(src, gamma=1.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 1, f"[{backend}] gamma=1.0 max diff: {diff.max()} > 1"


def test_gamma_zero_stays_zero(backend):
    """Pixel value 0: any gamma → 0."""
    src = make_batch(1, 8, 8, 3, fill=0)
    out = py_rpp.gamma_correction(src, gamma=2.5, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] 0^gamma should remain 0")


# ─── hue ──────────────────────────────────────────────────────────────────────

def test_hue_zero(backend):
    """hue_shift=0 should be approximately identity.

    Uses N=1 because some RPP CPU builds have a known limitation where a
    zero-valued per-image parameter causes images past the first to be skipped
    in the batch loop.  Shape/N>1 coverage lives in test_hue_shape_preserved.
    """
    src = make_batch(1, 32, 32, 3, fill=128)
    out = py_rpp.hue(src, hue_shift=0.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 2, f"[{backend}] hue=0 max diff {diff.max()}"


def test_hue_shape_preserved(backend):
    src = make_batch(3, 16, 16, 3)
    out = py_rpp.hue(src, hue_shift=45.0, backend=backend)
    assert out.shape == src.shape


# ─── saturation ───────────────────────────────────────────────────────────────

def test_saturation_identity(backend):
    """saturation=1.0 should be approximately identity.

    Uses N=1 — the RPP CPU saturation kernel uses (factor-1) as its internal
    delta; delta=0.0 triggers the same zero-param batch-skip limitation as
    other ops, so image[1+] is silently skipped in N>1 batches.
    """
    src = make_batch(1, 32, 32, 3)
    out = py_rpp.saturation(src, factor=1.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 2, f"[{backend}] saturation=1 max diff {diff.max()}"


def test_saturation_zero_grey(backend):
    """saturation=0.0 should convert to greyscale (R=G=B per pixel)."""
    src = make_batch(1, 16, 16, 3)
    out = py_rpp.saturation(src, factor=0.0, backend=backend)
    # All channels per pixel should be equal (greyscale)
    r, g, b = out[0, :, :, 0], out[0, :, :, 1], out[0, :, :, 2]
    np.testing.assert_array_equal(r, g, err_msg=f"[{backend}] sat=0: R≠G")
    np.testing.assert_array_equal(g, b, err_msg=f"[{backend}] sat=0: G≠B")


# ─── contrast ─────────────────────────────────────────────────────────────────

def test_contrast_identity(backend):
    """contrast_factor=1, center=128 → identity."""
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.contrast(src, contrast_factor=1.0, contrast_center=128.0,
                          backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 1, f"[{backend}] contrast identity max diff {diff.max()}"


# ─── exposure ─────────────────────────────────────────────────────────────────

def test_exposure_shape(backend):
    src = make_batch(2, 16, 16, 3)
    out = py_rpp.exposure(src, factor=0.0, backend=backend)
    assert out.shape == src.shape


# ─── blend ────────────────────────────────────────────────────────────────────

def test_blend_alpha_one(backend):
    """alpha=1.0: dst = src1 entirely."""
    a = make_batch(2, 16, 16, 3, fill=200)
    b = make_batch(2, 16, 16, 3, fill=50)
    out = py_rpp.blend(a, b, alpha=1.0, backend=backend)
    np.testing.assert_array_equal(out, a,
        err_msg=f"[{backend}] blend(alpha=1) should return src1")


def test_blend_alpha_zero(backend):
    """alpha=0.0: dst = src2 entirely."""
    a = make_batch(2, 16, 16, 3, fill=200)
    b = make_batch(2, 16, 16, 3, fill=50)
    out = py_rpp.blend(a, b, alpha=0.0, backend=backend)
    np.testing.assert_array_equal(out, b,
        err_msg=f"[{backend}] blend(alpha=0) should return src2")


# ─── color_twist ──────────────────────────────────────────────────────────────

def test_color_twist_identity(backend):
    """brightness=1, contrast=1, hue=0, saturation=1 → approximately identity.

    Uses N=1 — hue_shift=0.0 (zero-valued per-image parameter) triggers the
    RPP CPU batch-skip limitation where images past the first are not written.
    """
    src = make_batch(1, 32, 32, 3)
    out = py_rpp.color_twist(src, brightness=1.0, contrast=1.0,
                             hue_shift=0.0, saturation=1.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 2, f"[{backend}] color_twist identity max diff {diff.max()}"


# ─── histogram_equalize ───────────────────────────────────────────────────────

def test_histogram_equalize_shape(backend):
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.histogram_equalize(src, backend=backend)
    assert out.shape == src.shape, f"[{backend}] shape mismatch"
    assert out.dtype == np.uint8
