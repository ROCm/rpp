"""Tests for filter augmentation ops (box, gaussian, median)."""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch


def test_box_filter_uniform(backend):
    """A uniform image must survive a box filter unchanged."""
    src = make_batch(2, 32, 32, 3, fill=100)
    out = py_rpp.box_filter(src, kernel_size=3, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] box_filter on uniform should be identity")


def test_box_filter_shape(backend):
    src = make_batch(2, 64, 48, 3)
    out = py_rpp.box_filter(src, kernel_size=5, backend=backend)
    assert out.shape == src.shape


def test_box_filter_even_kernel_raises(backend):
    src = make_batch(1, 16, 16, 3)
    with pytest.raises(Exception):
        py_rpp.box_filter(src, kernel_size=4, backend=backend)


def test_gaussian_filter_uniform(backend):
    """A uniform image must survive a Gaussian filter unchanged."""
    src = make_batch(2, 32, 32, 3, fill=128)
    out = py_rpp.gaussian_filter(src, std_dev=1.0, kernel_size=3, backend=backend)
    # Interior pixels should be exact; edge pixels may differ due to border handling
    interior = out[:, 2:-2, 2:-2, :]
    expected = src[:, 2:-2, 2:-2, :]
    np.testing.assert_array_equal(interior, expected,
        err_msg=f"[{backend}] gaussian on uniform: interior should be identity")


def test_gaussian_filter_shape(backend):
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.gaussian_filter(src, std_dev=2.0, kernel_size=5, backend=backend)
    assert out.shape == src.shape


def test_median_filter_uniform(backend):
    """A uniform image must survive a median filter unchanged."""
    src = make_batch(2, 32, 32, 3, fill=150)
    out = py_rpp.median_filter(src, kernel_size=3, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] median on uniform should be identity")


def test_median_filter_shape(backend):
    src = make_batch(3, 48, 64, 1)
    out = py_rpp.median_filter(src, kernel_size=5, backend=backend)
    assert out.shape == src.shape
