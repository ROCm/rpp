"""Tests for morphological ops (erode, dilate)."""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch


def test_erode_all_white(backend):
    """Eroding a fully white image should keep it white (no dark neighbours)."""
    src = make_batch(2, 32, 32, 3, fill=255)
    out = py_rpp.erode(src, kernel_size=3, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] erode on all-255 should stay 255")


def test_dilate_all_black(backend):
    """Dilating a fully black image should keep it black."""
    src = make_batch(2, 32, 32, 3, fill=0)
    out = py_rpp.dilate(src, kernel_size=3, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] dilate on all-0 should stay 0")


def test_erode_lowers_value(backend):
    """Erode reduces pixel values (or keeps them); never increases."""
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.erode(src, kernel_size=3, backend=backend)
    assert (out.astype(np.int32) <= src.astype(np.int32)).all(), \
        f"[{backend}] erode must not increase any pixel"


def test_dilate_raises_value(backend):
    """Dilate increases pixel values (or keeps them); never decreases."""
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.dilate(src, kernel_size=3, backend=backend)
    assert (out.astype(np.int32) >= src.astype(np.int32)).all(), \
        f"[{backend}] dilate must not decrease any pixel"


def test_erode_dilate_shape(backend):
    src = make_batch(3, 64, 48, 3)
    assert py_rpp.erode(src, kernel_size=5, backend=backend).shape == src.shape
    assert py_rpp.dilate(src, kernel_size=5, backend=backend).shape == src.shape


def test_even_kernel_raises(backend):
    src = make_batch(1, 16, 16, 3)
    with pytest.raises(Exception):
        py_rpp.erode(src, kernel_size=4, backend=backend)
