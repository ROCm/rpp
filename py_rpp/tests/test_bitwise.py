"""Tests for bitwise ops (and, or, xor, not)."""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch


def test_bitwise_and_with_ff(backend):
    """AND with 0xFF is identity."""
    src = make_batch(2, 32, 32, 3)
    mask = np.full_like(src, 0xFF)
    out = py_rpp.bitwise_and(src, mask, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] AND with 0xFF should be identity")


def test_bitwise_and_with_zero(backend):
    """AND with 0x00 produces all zeros."""
    src = make_batch(2, 32, 32, 3)
    zero = np.zeros_like(src)
    out = py_rpp.bitwise_and(src, zero, backend=backend)
    np.testing.assert_array_equal(out, zero,
        err_msg=f"[{backend}] AND with 0 should be all-zero")


def test_bitwise_or_with_zero(backend):
    """OR with 0x00 is identity."""
    src = make_batch(2, 32, 32, 3)
    zero = np.zeros_like(src)
    out = py_rpp.bitwise_or(src, zero, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] OR with 0 should be identity")


def test_bitwise_or_with_ff(backend):
    """OR with 0xFF produces all 0xFF."""
    src = make_batch(2, 32, 32, 3)
    mask = np.full_like(src, 0xFF)
    out = py_rpp.bitwise_or(src, mask, backend=backend)
    np.testing.assert_array_equal(out, mask,
        err_msg=f"[{backend}] OR with 0xFF should be 0xFF")


def test_bitwise_not_twice(backend):
    """NOT applied twice returns the original."""
    src = make_batch(2, 32, 32, 3)
    once = py_rpp.bitwise_not(src, backend=backend)
    twice = py_rpp.bitwise_not(once, backend=backend)
    np.testing.assert_array_equal(twice, src,
        err_msg=f"[{backend}] double NOT should be identity")


def test_bitwise_not_value(backend):
    """NOT of 0 should be 255, NOT of 255 should be 0."""
    src = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    out = py_rpp.bitwise_not(src, backend=backend)
    np.testing.assert_array_equal(out, np.full_like(out, 255),
        err_msg=f"[{backend}] NOT(0) should be 255")


def test_bitwise_xor_self(backend):
    """XOR with itself produces all zeros."""
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.bitwise_xor(src, src.copy(), backend=backend)
    np.testing.assert_array_equal(out, np.zeros_like(src),
        err_msg=f"[{backend}] XOR with self should be 0")


def test_bitwise_xor_with_zero(backend):
    """XOR with 0 is identity."""
    src = make_batch(2, 32, 32, 3)
    zero = np.zeros_like(src)
    out = py_rpp.bitwise_xor(src, zero, backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] XOR with 0 should be identity")


def test_bitwise_shape_mismatch(backend):
    """Different shapes must raise."""
    a = make_batch(2, 16, 16, 3)
    b = make_batch(2, 32, 32, 3)
    with pytest.raises(Exception):
        py_rpp.bitwise_and(a, b, backend=backend)
