"""Tests for effect augmentation ops (gaussian_noise, salt_pepper, vignette, pixelate)."""

import numpy as np
import pytest
import py_rpp
from conftest import make_batch


# ─── gaussian_noise ───────────────────────────────────────────────────────────

def test_gaussian_noise_shape(backend):
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.gaussian_noise(src, mean=0.0, std_dev=10.0, backend=backend)
    assert out.shape == src.shape, f"[{backend}] shape mismatch"
    assert out.dtype == np.uint8


def test_gaussian_noise_zero_std(backend):
    """std_dev=0 should produce output very close to input.

    Uses N=1 — some RPP CPU builds silently skip images past the first when a
    per-image parameter (here std_dev) is exactly 0.0.
    """
    src = make_batch(1, 32, 32, 3, fill=100)
    out = py_rpp.gaussian_noise(src, mean=0.0, std_dev=0.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 1, f"[{backend}] std_dev=0 max diff: {diff.max()}"


def test_gaussian_noise_adds_variation(backend):
    """With large std_dev, output should differ from input in at least some pixels."""
    src = make_batch(2, 64, 64, 3, fill=128)
    out = py_rpp.gaussian_noise(src, mean=0.0, std_dev=30.0, seed=42, backend=backend)
    assert not np.array_equal(out, src), \
        f"[{backend}] gaussian noise with std=30 should change some pixels"


# ─── salt_and_pepper_noise ────────────────────────────────────────────────────

def test_salt_pepper_shape(backend):
    src = make_batch(2, 32, 32, 3)
    out = py_rpp.salt_and_pepper_noise(
        src, noise_prob=0.05, salt_prob=0.5,
        salt_value=255.0, pepper_value=0.0,    # fn.py normalises → [0,1] for RPP
        backend=backend)
    assert out.shape == src.shape, f"[{backend}] shape mismatch"
    assert out.dtype == np.uint8


def test_salt_pepper_zero_noise(backend):
    """noise_prob=0 → no noise, output equals input.

    Uses N=1 to avoid the zero-param batch-skip limitation in some RPP builds.
    """
    src = make_batch(1, 32, 32, 3, fill=128)
    out = py_rpp.salt_and_pepper_noise(
        src, noise_prob=0.0, salt_prob=0.5,
        salt_value=255.0, pepper_value=0.0,
        backend=backend)
    np.testing.assert_array_equal(out, src,
        err_msg=f"[{backend}] noise_prob=0 should leave image unchanged")


# ─── vignette ─────────────────────────────────────────────────────────────────

def test_vignette_shape(backend):
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.vignette(src, intensity=0.5, backend=backend)
    assert out.shape == src.shape, f"[{backend}] shape mismatch"
    assert out.dtype == np.uint8


def test_vignette_zero_intensity(backend):
    """intensity=0.0 should be approximately identity.

    Uses N=1 — some RPP CPU builds silently skip images past the first when
    the intensity parameter is exactly 0.0.
    """
    src = make_batch(1, 32, 32, 3, fill=128)
    out = py_rpp.vignette(src, intensity=0.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 2, f"[{backend}] vignette(0) max diff: {diff.max()}"


def test_vignette_darkens_edges(backend):
    """With intensity>0, edge pixels must be <= centre pixels."""
    src = make_batch(1, 64, 64, 3, fill=200)
    out = py_rpp.vignette(src, intensity=0.8, backend=backend)
    centre = int(out[0, 32, 32, 0])
    corner = int(out[0,  0,  0, 0])
    assert corner <= centre, \
        f"[{backend}] corner ({corner}) should be darker than centre ({centre})"


# ─── pixelate ─────────────────────────────────────────────────────────────────

def test_pixelate_shape(backend):
    src = make_batch(2, 64, 64, 3)
    out = py_rpp.pixelate(src, pixelation_percentage=50.0, backend=backend)
    assert out.shape == src.shape, f"[{backend}] shape mismatch"
    assert out.dtype == np.uint8


def test_pixelate_zero_percent(backend):
    """0% pixelation should be close to identity.

    Uses N=1 — pixelation_percentage=0.0 triggers the RPP CPU batch-skip
    limitation where images past the first are silently skipped.
    """
    src = make_batch(1, 32, 32, 3, fill=128)
    out = py_rpp.pixelate(src, pixelation_percentage=0.0, backend=backend)
    diff = np.abs(out.astype(np.int32) - src.astype(np.int32))
    assert diff.max() <= 2, f"[{backend}] pixelate(0%) max diff: {diff.max()}"


def test_pixelate_invalid_range(backend):
    src = make_batch(1, 16, 16, 3)
    with pytest.raises(Exception):
        py_rpp.pixelate(src, pixelation_percentage=150.0, backend=backend)
