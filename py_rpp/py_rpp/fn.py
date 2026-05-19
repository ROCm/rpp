# Copyright (c) 2026 Advanced Micro Devices, Inc.
"""
py_rpp.fn — ergonomic high-level wrappers around _py_rpp.

Improvements over the raw C++ bindings:
  - Scalar-to-batch broadcasting for all per-image parameters.
  - Layout auto-detection: accepts NHWC or NCHW input transparently.
  - Auto-backend: defaults to 'hip' when a GPU is available.
  - Optional roi_widths / roi_heights (planned; currently full-image ROI only).
  - try/finally handle lifecycle for advanced rpp_create/rpp_destroy callers.
"""

from __future__ import annotations

import numpy as np

from . import _py_rpp as _C
from .types import NHWC, NCHW
from .utils import get_default_backend, to_nhwc, detect_layout


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bk(backend: str | None) -> str:
    """Resolve None → default backend string."""
    return backend if backend is not None else get_default_backend()


def _param(value, n: int, dtype=np.float32) -> np.ndarray:
    """
    Broadcast a scalar or sequence to a 1-D numpy array of length n.

    Examples
    --------
    _param(1.5, 4)               → [1.5, 1.5, 1.5, 1.5]  float32
    _param([1.0, 2.0], 2)        → [1.0, 2.0]             float32
    _param(np.array([1,2]), 2)   → [1.0, 2.0]             float32
    """
    if np.isscalar(value):
        return np.full(n, value, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            f"Parameter must be a scalar or 1-D array of length {n}, "
            f"got shape {arr.shape}")
    return arr


def _flag(value, n: int) -> np.ndarray:
    """Broadcast a bool or sequence to uint32 (N,) array of 0/1."""
    if isinstance(value, bool):
        return np.full(n, int(value), dtype=np.uint32)
    return _param(value, n, dtype=np.uint32)


def _param_nc(value, n: int, c: int, dtype=np.float32) -> np.ndarray:
    """
    Broadcast to a flat 1-D array of length N*C for per-channel-per-image ops.

    Examples
    --------
    _param_nc(1.0, 2, 3)              → [1., 1., 1., 1., 1., 1.]  float32
    _param_nc([0.5, 0.8], 2, 3)       → [0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
    _param_nc(np.ones((2,3)), 2, 3)   → [1., 1., 1., 1., 1., 1.]
    """
    if np.isscalar(value):
        return np.full(n * c, value, dtype=dtype)
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 1 and arr.shape[0] == n:
        return np.repeat(arr, c)
    if arr.ndim == 2 and arr.shape == (n, c):
        return np.ascontiguousarray(arr).ravel()
    if arr.ndim == 1 and arr.shape[0] == n * c:
        return arr
    raise ValueError(
        f"threshold param must be scalar, ({n},), ({n},{c}), or ({n*c},); "
        f"got shape {arr.shape}"
    )


def _ensure_nhwc(images: np.ndarray, layout: str | None) -> np.ndarray:
    """Convert images to C-contiguous NHWC uint8 if needed."""
    return to_nhwc(images.astype(np.uint8, copy=False), layout)


# ─── Color augmentations ──────────────────────────────────────────────────────

def brightness(images, alpha=1.0, beta=0.0, *,
               input_layout=None, backend=None) -> np.ndarray:
    """
    Apply brightness: output = clamp(alpha * src + beta, 0, 255).

    Parameters
    ----------
    images : ndarray  uint8, shape (N, H, W, C) or (N, C, H, W)
    alpha  : scalar or (N,) float32  — per-image multiplier  [0, 20]
    beta   : scalar or (N,) float32  — per-image additive offset  [0, 255]
    input_layout : 'nhwc' | 'nchw' | None (auto-detect)
    backend      : 'hip' | 'cpu' | None (auto)
    """
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.brightness(imgs, _param(alpha, N), _param(beta, N), _bk(backend))


def gamma_correction(images, gamma=1.0, *,
                     input_layout=None, backend=None) -> np.ndarray:
    """Per-image gamma correction: output = src ^ gamma."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.gamma_correction(imgs, _param(gamma, N), _bk(backend))


def hue(images, hue_shift=0.0, *,
        input_layout=None, backend=None) -> np.ndarray:
    """Hue rotation in degrees [-180, 180] per image (3-channel RGB only)."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.hue(imgs, _param(hue_shift, N), _bk(backend))


def saturation(images, factor=1.0, *,
               input_layout=None, backend=None) -> np.ndarray:
    """Saturation scaling per image (1.0 = identity)."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.saturation(imgs, _param(factor, N), _bk(backend))


def contrast(images, contrast_factor=1.0, contrast_center=128.0, *,
             input_layout=None, backend=None) -> np.ndarray:
    """clamp(contrast_factor * (src - center) + center, 0, 255)."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.contrast(imgs, _param(contrast_factor, N),
                       _param(contrast_center, N), _bk(backend))


def exposure(images, factor=0.0, *,
             input_layout=None, backend=None) -> np.ndarray:
    """Exposure shift per image."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.exposure(imgs, _param(factor, N), _bk(backend))


def blend(images1, images2, alpha=0.5, *,
          input_layout=None, backend=None) -> np.ndarray:
    """dst = alpha * src1 + (1 - alpha) * src2."""
    i1 = _ensure_nhwc(images1, input_layout)
    i2 = _ensure_nhwc(images2, input_layout)
    N = i1.shape[0]
    return _C.blend(i1, i2, _param(alpha, N), _bk(backend))


def color_twist(images, brightness=1.0, contrast=1.0,
                hue_shift=0.0, saturation=1.0, *,
                input_layout=None, backend=None) -> np.ndarray:
    """Combined brightness + contrast + hue + saturation adjustment."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.color_twist(
        imgs,
        _param(brightness,  N), _param(contrast, N),
        _param(hue_shift,   N), _param(saturation, N),
        _bk(backend))


def histogram_equalize(images, *, input_layout=None, backend=None) -> np.ndarray:
    """Per-channel histogram equalization."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.histogram_equalize(imgs, _bk(backend))


# ─── Geometric augmentations ──────────────────────────────────────────────────

def flip(images, horizontal=False, vertical=False, *,
         input_layout=None, backend=None) -> np.ndarray:
    """
    Flip images.

    Parameters
    ----------
    horizontal : bool or (N,) uint32 — 1 = flip left↔right
    vertical   : bool or (N,) uint32 — 1 = flip top↔bottom
    """
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.flip(imgs, _flag(horizontal, N), _flag(vertical, N), _bk(backend))


def crop(images, x, y, out_h: int, out_w: int, *,
         input_layout=None, backend=None) -> np.ndarray:
    """
    Crop a uniform (out_h × out_w) region from each image.

    x, y : scalar or (N,) uint32 — top-left corner per image
    """
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.crop(imgs, _flag(x, N), _flag(y, N),
                   int(out_h), int(out_w), _bk(backend))


def rotate(images, angle=0.0, *, input_layout=None, backend=None) -> np.ndarray:
    """Rotate by angle (degrees) per image, bilinear, same output size."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.rotate(imgs, _param(angle, N), _bk(backend))


def resize(images, out_h: int, out_w: int, *,
           input_layout=None, backend=None) -> np.ndarray:
    """Resize all images to (out_h × out_w), bilinear interpolation."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.resize(imgs, int(out_h), int(out_w), _bk(backend))


# ─── Filter augmentations ─────────────────────────────────────────────────────

def box_filter(images, kernel_size: int = 3, *,
               input_layout=None, backend=None) -> np.ndarray:
    """Uniform (mean) box filter.  kernel_size must be odd."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.box_filter(imgs, int(kernel_size), _bk(backend))


def gaussian_filter(images, std_dev=1.0, kernel_size: int = 3, *,
                    input_layout=None, backend=None) -> np.ndarray:
    """Gaussian filter with per-image std_dev.  kernel_size must be odd."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.gaussian_filter(imgs, _param(std_dev, N),
                              int(kernel_size), _bk(backend))


def median_filter(images, kernel_size: int = 3, *,
                  input_layout=None, backend=None) -> np.ndarray:
    """Median filter (edge-preserving).  kernel_size must be odd."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.median_filter(imgs, int(kernel_size), _bk(backend))


# ─── Morphological operations ─────────────────────────────────────────────────

def erode(images, kernel_size: int = 3, *,
          input_layout=None, backend=None) -> np.ndarray:
    """Morphological erosion.  kernel_size must be odd."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.erode(imgs, int(kernel_size), _bk(backend))


def dilate(images, kernel_size: int = 3, *,
           input_layout=None, backend=None) -> np.ndarray:
    """Morphological dilation.  kernel_size must be odd."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.dilate(imgs, int(kernel_size), _bk(backend))


# ─── Bitwise operations ───────────────────────────────────────────────────────

def bitwise_and(images1, images2, *, input_layout=None, backend=None) -> np.ndarray:
    i1 = _ensure_nhwc(images1, input_layout); i2 = _ensure_nhwc(images2, input_layout)
    return _C.bitwise_and(i1, i2, _bk(backend))

def bitwise_or(images1, images2, *, input_layout=None, backend=None) -> np.ndarray:
    i1 = _ensure_nhwc(images1, input_layout); i2 = _ensure_nhwc(images2, input_layout)
    return _C.bitwise_or(i1, i2, _bk(backend))

def bitwise_xor(images1, images2, *, input_layout=None, backend=None) -> np.ndarray:
    i1 = _ensure_nhwc(images1, input_layout); i2 = _ensure_nhwc(images2, input_layout)
    return _C.bitwise_xor(i1, i2, _bk(backend))

def bitwise_not(images, *, input_layout=None, backend=None) -> np.ndarray:
    return _C.bitwise_not(_ensure_nhwc(images, input_layout), _bk(backend))


# ─── Statistical operations ───────────────────────────────────────────────────

def tensor_mean(images, *, input_layout=None, backend=None) -> np.ndarray:
    """Per-image per-channel mean.  Returns float32 (N, C+1)."""
    return _C.tensor_mean(_ensure_nhwc(images, input_layout), _bk(backend))

def tensor_min(images, *, input_layout=None, backend=None) -> np.ndarray:
    """Per-image per-channel min.  Returns uint8 (N, C+1)."""
    return _C.tensor_min(_ensure_nhwc(images, input_layout), _bk(backend))

def tensor_max(images, *, input_layout=None, backend=None) -> np.ndarray:
    """Per-image per-channel max.  Returns uint8 (N, C+1)."""
    return _C.tensor_max(_ensure_nhwc(images, input_layout), _bk(backend))

def threshold(images, min_val=0.0, max_val=255.0, *,
              input_layout=None, backend=None) -> np.ndarray:
    """Binary mask: 255 where min_val <= pixel <= max_val, else 0.

    min_val / max_val accept scalar, (N,), (N,C), or (N*C,) — the RPP kernel
    requires one threshold per channel per image (N*C total values).
    """
    imgs = _ensure_nhwc(images, input_layout)
    N, _, _, C = imgs.shape
    return _C.threshold(imgs,
                        _param_nc(min_val, N, C),
                        _param_nc(max_val, N, C),
                        _bk(backend))


# ─── Effects / augmentations ──────────────────────────────────────────────────

def gaussian_noise(images, mean=0.0, std_dev=10.0, seed: int = 0, *,
                   input_layout=None, backend=None) -> np.ndarray:
    """Additive Gaussian noise with per-image mean and std_dev."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.gaussian_noise(imgs, _param(mean, N), _param(std_dev, N),
                             int(seed), _bk(backend))


def salt_and_pepper_noise(images, noise_prob=0.05, salt_prob=0.5,
                          salt_value=255.0, pepper_value=0.0,
                          seed: int = 0, *,
                          input_layout=None, backend=None) -> np.ndarray:
    """Salt-and-pepper noise with per-image parameters.

    salt_value / pepper_value accept the natural uint8 scale [0, 255]; they
    are divided by 255 internally before forwarding to RPP, which requires
    normalised floats in [0, 1].
    """
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    # Public API accepts [0, 255] (natural uint8 scale); RPP requires [0, 1].
    sv = _param(salt_value, N) / 255.0
    pv = _param(pepper_value, N) / 255.0
    return _C.salt_and_pepper_noise(
        imgs,
        _param(noise_prob, N), _param(salt_prob, N),
        sv, pv,
        int(seed), _bk(backend))


def vignette(images, intensity=0.5, *, input_layout=None, backend=None) -> np.ndarray:
    """Vignette effect: darkens image edges.  intensity per image [0, 1]."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.vignette(imgs, _param(intensity, N), _bk(backend))


def pixelate(images, pixelation_percentage=50.0, *,
             input_layout=None, backend=None) -> np.ndarray:
    """Pixelate effect.  pixelation_percentage in [0, 100]."""
    imgs = _ensure_nhwc(images, input_layout)
    return _C.pixelate(imgs, float(pixelation_percentage), _bk(backend))


# ─── Public surface ───────────────────────────────────────────────────────────
__all__ = [
    # color
    "brightness", "gamma_correction", "hue", "saturation", "contrast",
    "exposure", "blend", "color_twist", "histogram_equalize",
    # geometric
    "flip", "crop", "rotate", "resize",
    # filter
    "box_filter", "gaussian_filter", "median_filter",
    # morphological
    "erode", "dilate",
    # bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    # statistical
    "tensor_mean", "tensor_min", "tensor_max", "threshold",
    # effects
    "gaussian_noise", "salt_and_pepper_noise", "vignette", "pixelate",
]
