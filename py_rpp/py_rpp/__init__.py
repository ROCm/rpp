# py_rpp — Python bindings for RPP (ROCm Performance Primitives)
#
# Layered design:
#   _py_rpp        — C++ pybind11 extension (low-level, NHWC arrays only)
#   py_rpp.fn      — ergonomic wrappers (scalar broadcasting, layout detection)
#   py_rpp.utils   — GPU detection, layout conversion helpers
#   py_rpp.types   — Backend / layout string constants
#
# Importing py_rpp re-exports the fn.py layer, so callers can write:
#   import py_rpp
#   out = py_rpp.brightness(images, alpha=1.5, beta=10)   # scalar works!

from . import _py_rpp  # noqa: F401 — expose low-level module as py_rpp._py_rpp
from . import types, utils, fn

from .fn import (
    brightness, gamma_correction, hue, saturation, contrast,
    exposure, blend, color_twist, histogram_equalize,
    flip, crop, rotate, resize,
    box_filter, gaussian_filter, median_filter,
    erode, dilate,
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not,
    tensor_mean, tensor_min, tensor_max, threshold,
    gaussian_noise, salt_and_pepper_noise, vignette, pixelate,
)

__version__ = "0.2.0"
