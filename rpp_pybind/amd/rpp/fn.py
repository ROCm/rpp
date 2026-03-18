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
RPP Augmentation Functions
===========================

High-level wrapper functions for RPP augmentations.
"""

import rpp_pybind
from .rpp_types import get_default_backend, HOST, HIP
import ctypes
import torch

# Direct access to C++ functions
_brightness = rpp_pybind.brightness
_gamma_correction = rpp_pybind.gamma_correction
_contrast = rpp_pybind.contrast
_hue = rpp_pybind.hue
_flip = rpp_pybind.flip
_resize = rpp_pybind.resize
_rotate = rpp_pybind.rotate
_crop = rpp_pybind.crop
_vignette = rpp_pybind.vignette
_pixelate = rpp_pybind.pixelate
rppCreate = rpp_pybind.rppCreate
rppDestroy = rpp_pybind.rppDestroy

def _resolve_layout(images, input_layout=None):
    """
    Returns: layout_str ('NCHW' or 'NHWC')
    """

    if input_layout is not None:
        return input_layout.upper()

    if images.ndim != 4:
        raise ValueError("Expected 4D tensor (B, C, H, W) or (B, H, W, C)")

    # Check for typical channel dimensions (1 for grayscale, 3 for RGB)
    # Only use 1 and 3 to avoid ambiguity with width=4
    last_dim = images.shape[-1]
    second_dim = images.shape[1]
    
    # Unambiguous cases: channel dim is 1 or 3
    if last_dim in (1, 3) and second_dim not in (1, 3):
        return "NHWC"
    
    if second_dim in (1, 3) and last_dim not in (1, 3):
        return "NCHW"
    
    # Both dimensions could be channels - ambiguous case
    if last_dim in (1, 3, 4) and second_dim in (1, 3, 4):
        raise ValueError(
            f"Ambiguous tensor shape {tuple(images.shape)}: both dim[1]={second_dim} and "
            f"dim[-1]={last_dim} could be channels. Please specify input_layout explicitly."
        )

    raise ValueError(
        f"Unable to infer layout from shape {tuple(images.shape)}. "
        "Please specify input_layout explicitly."
    )

def _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights):
    """
    Common preparation for all augmentation functions.
    
    Returns:
        (images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle)
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
    
    if backend == HIP:
        if not images.is_cuda:
            images = images.cuda()
    else:  # HOST
        if images.is_cuda:
            images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()
    layout = _resolve_layout(images, input_layout)

    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if layout == "NCHW" else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if layout == "NCHW" else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    return images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle

# Wrapper functions - 10 augmentations from different groups

# Color Augmentations (4)
def brightness(images, alpha=1.0, beta=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image brightness.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        alpha: Brightness multiplier (default 1.0)
        beta: Brightness offset (default 0.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Augmented images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _brightness(images, output, [alpha] * batch_size, [beta] * batch_size, 
                    roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP brightness failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def gamma_correction(images, gamma=1.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply gamma correction.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        gamma: Gamma value (default 1.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Gamma-corrected images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _gamma_correction(images, output, [gamma] * batch_size, 
                         roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP gamma_correction failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def contrast(images, contrast_factor=1.0, contrast_center=128.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image contrast.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        contrast_factor: Contrast factor (default 1.0)
        contrast_center: Center value for contrast (default 128.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Contrast-adjusted images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _contrast(images, output, [contrast_factor] * batch_size, [contrast_center] * batch_size,
                  roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP contrast failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def hue(images, hue_shift=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image hue (for RGB images only).
    
    Args:
        images: Input tensor (B, 3, H, W) or (B, H, W, 3) - RGB images only
        hue_shift: Hue shift in degrees (default 0.0, range 0-359)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Hue-adjusted images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _hue(images, output, [hue_shift] * batch_size, roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP hue failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


# Geometric Augmentations (4)
def flip(images, horizontal=False, vertical=False, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Flip images horizontally and/or vertically.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        horizontal: Flip horizontally (bool or list, default False)
        vertical: Flip vertically (bool or list, default False)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Flipped images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        # Convert bool to list if needed
        horizontal = [int(horizontal)] * batch_size if isinstance(horizontal, bool) else horizontal
        vertical = [int(vertical)] * batch_size if isinstance(vertical, bool) else vertical
        
        _flip(images, output, horizontal, vertical, roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP flip failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def resize(images, width, height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Resize images.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        width: Target width
        height: Target height
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Resized images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _resize(images, output, [width] * batch_size, [height] * batch_size,
                roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP resize failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def rotate(images, angle=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Rotate images by given angle.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        angle: Rotation angle in degrees (default 0.0, positive = counter-clockwise)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Rotated images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _rotate(images, output, [angle] * batch_size, roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP rotate failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def crop(images, x1, y1, crop_width, crop_height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Crop images to specified region.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        x1: Top-left x coordinate
        y1: Top-left y coordinate
        crop_width: Width of crop region
        crop_height: Height of crop region
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Cropped images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        # Convert scalars to lists
        x1_array = [x1] * batch_size if isinstance(x1, (int, float)) else x1
        y1_array = [y1] * batch_size if isinstance(y1, (int, float)) else y1
        width_array = [crop_width] * batch_size if isinstance(crop_width, (int, float)) else crop_width
        height_array = [crop_height] * batch_size if isinstance(crop_height, (int, float)) else crop_height
        
        _crop(images, output, x1_array, y1_array, width_array, height_array, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP crop failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


# Effects Augmentations (2)
def vignette(images, intensity=0.5, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply vignette effect to images.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        intensity: Vignette intensity (default 0.5, range 0.0-1.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Images with vignette effect
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        _vignette(images, output, [intensity] * batch_size, roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP vignette failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


def pixelate(images, pixelation_percentage=50.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply pixelate effect to images.
    
    Args:
        images: Input tensor (B, C, H, W) or (B, H, W, C)
        pixelation_percentage: Pixelation level (default 50.0, range 0-100)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        input_layout: Input tensor layout ('NCHW' or 'NHWC', None = auto-detect)
        output_layout: Output tensor layout (currently unused, reserved for future use)
        backend: RppBackend (None = auto-detect, HOST or HIP)
    
    Returns:
        Pixelated images tensor
    """
    images, output, layout, roi_widths, roi_heights, batch_size, backend_int, handle = \
        _prepare_tensors(images, backend, input_layout, roi_widths, roi_heights)
    
    try:
        # Create scratch buffer (size is same for both NCHW and NHWC layouts)
        scratch_size = batch_size * images.shape[1] * images.shape[2] * images.shape[3]
        scratch = torch.empty(scratch_size, dtype=torch.float32, device=images.device)
        
        _pixelate(images, output, scratch, pixelation_percentage, roi_widths, roi_heights, handle, backend_int)
        return output
    except Exception as e:
        raise RuntimeError(f"RPP pixelate failed: {e}") from e
    finally:
        rppDestroy(handle, backend_int)


__all__ = [
    # Color augmentations
    'brightness',
    'gamma_correction', 
    'contrast',
    'hue',
    # Geometric augmentations
    'flip',
    'resize',
    'rotate',
    'crop',
    # Effects augmentations
    'vignette',
    'pixelate'
]
