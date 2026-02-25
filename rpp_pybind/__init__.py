# MIT License

# Copyright (c) 2026 Advanced Micro Devices, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
