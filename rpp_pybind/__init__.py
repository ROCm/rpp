"""
PyRPP - Python bindings for AMD ROCm Performance Primitives.

Package layout:
- rpp_pybind/            (this Python package)
- rpp_pybind/_rpp_pybind (pybind11 extension module)

The extension is intentionally named `_rpp_pybind` to avoid a name collision
between the Python package (`rpp_pybind`) and the compiled module, which can
cause pybind11 to register types twice (e.g. "type ... is already registered").
"""

from __future__ import annotations

from . import _rpp_pybind as _C  # compiled extension

__version__ = getattr(_C, "__version__", "0.0.0")

# Re-export the C++ `types` submodule at the package level so callers can do
# `import rpp_pybind; rpp_pybind.types.RppBackend`.
types = _C.types

# Core functions
brightness = _C.brightness
gamma_correction = _C.gamma_correction
contrast = _C.contrast
hue = _C.hue

resize = _C.resize
flip = _C.flip
rotate = _C.rotate
crop = _C.crop

vignette = _C.vignette
pixelate = _C.pixelate

rppCreate = _C.rppCreate
rppDestroy = _C.rppDestroy

__all__ = [
    "types",
    "__version__",
    # Color augmentations
    "brightness",
    "gamma_correction",
    "contrast",
    "hue",
    # Geometric augmentations
    "resize",
    "flip",
    "rotate",
    "crop",
    # Effects augmentations
    "vignette",
    "pixelate",
    # Core handle functions
    "rppCreate",
    "rppDestroy",
]
