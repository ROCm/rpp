"""
PyRPP - Python bindings for AMD ROCm Performance Primitives
Simple structure like rocAL
"""

# Import the C++ module
try:
    from . import rpp_pybind
except ImportError:
    import rpp_pybind

# Make C++ functions available at package level
brightness = rpp_pybind.brightness
gamma_correction = rpp_pybind.gamma_correction
contrast = rpp_pybind.contrast
hue = rpp_pybind.hue
resize = rpp_pybind.resize
flip = rpp_pybind.flip
rotate = rpp_pybind.rotate
crop = rpp_pybind.crop
vignette = rpp_pybind.vignette
pixelate = rpp_pybind.pixelate
rppCreate = rpp_pybind.rppCreate
rppDestroy = rpp_pybind.rppDestroy

# Import types module
from . import types

# Import submodules
from . import fn
from . import utils

__version__ = "1.0.0"

__all__ = [
    # Color augmentations
    'brightness', 'gamma_correction', 'contrast', 'hue',
    # Geometric augmentations  
    'resize', 'flip', 'rotate', 'crop',
    # Effects augmentations
    'vignette', 'pixelate',
    # Core functions
    'rppCreate', 'rppDestroy',
    # Modules
    'types', 'fn', 'utils'
]
