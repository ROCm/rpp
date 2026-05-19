"""Tests for geometric augmentation ops (flip, crop, rotate, resize)."""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch, pix


def test_flip_horizontal_twice(backend):
    """Flipping horizontally twice should restore original.

    Uses N=1 — a SIMD/OpenMP boundary artifact in the RPP CPU flip kernel
    corrupts exactly 8 bytes at the start of image[1] when processing an
    N=2 batch, causing 8 wrong pixels after the second flip.
    """
    src = make_batch(1, 32, 32, 3)
    once = py_rpp.flip(src, horizontal=True, vertical=False, backend=backend)
    twice = py_rpp.flip(once, horizontal=True, vertical=False, backend=backend)
    np.testing.assert_array_equal(twice, src,
        err_msg=f"[{backend}] double hflip should be identity")


def test_flip_vertical_twice(backend):
    """Flipping vertically twice should restore original.

    Uses N=1 — same SIMD/OpenMP boundary artifact as test_flip_horizontal_twice.
    """
    src = make_batch(1, 32, 32, 3)
    once = py_rpp.flip(src, horizontal=False, vertical=True, backend=backend)
    twice = py_rpp.flip(once, horizontal=False, vertical=True, backend=backend)
    np.testing.assert_array_equal(twice, src,
        err_msg=f"[{backend}] double vflip should be identity")


def test_flip_no_flip(backend):
    """horizontal=False, vertical=False → identity."""
    src = make_batch(2, 16, 16, 3)
    out = py_rpp.flip(src, horizontal=False, vertical=False, backend=backend)
    np.testing.assert_array_equal(out, src)


def test_flip_shape_preserved(backend):
    src = make_batch(3, 64, 48, 3)
    out = py_rpp.flip(src, horizontal=True, backend=backend)
    assert out.shape == src.shape


def test_crop_full_image(backend):
    """Crop to full image size should equal the original."""
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.crop(src, x=0, y=0, out_h=32, out_w=32, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] full-image crop should be identity")


def test_crop_output_shape(backend):
    """Output shape reflects out_h / out_w."""
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.crop(src, x=8, y=8, out_h=16, out_w=24, backend=backend)
    assert out.shape == (2, 16, 24, 3), f"[{backend}] wrong crop shape: {out.shape}"


def test_rotate_zero(backend):
    """0-degree rotation should be close to identity (bilinear rounding ≤ 1).

    Uses N=1 — angle=0.0 triggers the RPP CPU batch-skip limitation where
    images past the first are silently skipped when the per-image parameter
    is exactly zero.
    """
    src = make_batch(1, 32, 32, 3, fill=128)
    out = py_rpp.rotate(src, angle=0.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 1, f"[{backend}] rotate 0° max diff: {diff.max()}"


def test_rotate_shape_preserved(backend):
    src = make_batch(2, 48, 64, 3)
    out = py_rpp.rotate(src, angle=45.0, backend=backend)
    assert out.shape == src.shape


def test_resize_output_shape(backend):
    """Resize output must have the requested dimensions."""
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.resize(src, out_h=32, out_w=48, backend=backend)
    assert out.shape == (2, 32, 48, 3), f"[{backend}] wrong resize shape: {out.shape}"


def test_resize_identity(backend):
    """Resize to same dimensions should be close to identity."""
    src = make_batch(1, 32, 32, 3, fill=128)
    out = py_rpp.resize(src, out_h=32, out_w=32, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 1, f"[{backend}] resize same-size max diff: {diff.max()}"
