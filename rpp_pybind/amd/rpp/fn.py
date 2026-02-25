# MIT License

# Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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
Similar to rocAL's fn.py pattern.
"""

# Simple imports - like rocAL
import rpp_pybind
from rpp_pybind.amd.rpp.rpp_types import get_default_backend, HOST, HIP
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

def _prepare_tensor_and_backend(tensor, backend):
    """Prepare tensor for RPP operations"""   
    if backend is None:
        backend = get_default_backend()

    # Get backend integer consistently
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)

    # Validate tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {tensor.dim()}D")

    # Ensure contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Ensure correct device
    if backend_int == 1:
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        torch.cuda.synchronize()
    else:  # HOST
        if tensor.is_cuda:
            tensor = tensor.cpu()
    
    return tensor, backend_int

# Wrapper functions - 10 augmentations from different groups

# Color Augmentations (4)
def brightness(images, alpha=1.0, beta=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image brightness.
    
    Args:
        images: Input tensor (B, C, H, W) - PyTorch tensor
        alpha: Brightness multiplier (default 1.0)
        beta: Brightness offset (default 0.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Augmented images tensor
    """
    if backend is None:
        backend = get_default_backend()

    # Convert backend to integer
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    # Ensure tensor is contiguous and on correct device  
    if not images.is_contiguous():
        images = images.contiguous()
    
    # Move to correct device
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()

    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)

    alpha_array = [alpha] * batch_size
    beta_array = [beta] * batch_size
    
    _brightness(images, output, alpha_array, beta_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def gamma_correction(images, gamma=1.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply gamma correction.
    
    Args:
        images: Input tensor (B, C, H, W)
        gamma: Gamma value (default 1.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Gamma-corrected images
    """
    if backend is None:
        backend = get_default_backend()

    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()

    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()

    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size

    handle = rppCreate(batch_size, backend_int)
    gamma_array = [gamma] * batch_size
    
    _gamma_correction(images, output, gamma_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output

def contrast(images, contrast_factor=1.0, contrast_center=128.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image contrast.
    
    Args:
        images: Input tensor (B, C, H, W)
        contrast_factor: Contrast factor
        contrast_center: Center value for contrast
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Contrast-adjusted images
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    contrast_factor_array = [contrast_factor] * batch_size
    contrast_center_array = [contrast_center] * batch_size
    
    _contrast(images, output, contrast_factor_array, contrast_center_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def hue(images, hue_shift=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Adjust image hue (for RGB images only).
    
    Args:
        images: Input tensor (B, 3, H, W) - RGB images only
        hue_shift: Hue shift in degrees (0-359)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Hue-adjusted images
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    hue_shift_array = [hue_shift] * batch_size
    
    _hue(images, output, hue_shift_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


# Geometric Augmentations (4)
def flip(images, horizontal=False, vertical=False, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Flip images horizontally and/or vertically.
    
    Args:
        images: Input tensor (B, C, H, W)
        horizontal: Flip horizontally (bool or list)
        vertical: Flip vertically (bool or list)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Flipped images tensor
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    # Convert bool to list if needed
    if isinstance(horizontal, bool):
        horizontal = [int(horizontal)] * batch_size
    if isinstance(vertical, bool):
        vertical = [int(vertical)] * batch_size
    
    _flip(images, output, horizontal, vertical, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def resize(images, width, height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Resize images.
    
    Args:
        images: Input tensor (B, C, H, W)
        width: Target width
        height: Target height
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Resized images tensor
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()

    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    
    width_array = [width] * batch_size
    height_array = [height] * batch_size
    
    _resize(images, output, width_array, height_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def rotate(images, angle=0.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Rotate images by given angle.
    
    Args:
        images: Input tensor (B, C, H, W)
        angle: Rotation angle in degrees (positive = counter-clockwise)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Rotated images tensor
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    angle_array = [angle] * batch_size
    
    _rotate(images, output, angle_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def crop(images, x1, y1, crop_width, crop_height, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Crop images to specified region.
    
    Args:
        images: Input tensor (B, C, H, W)
        x1: Top-left x coordinate
        y1: Top-left y coordinate  
        crop_width: Width of crop region
        crop_height: Height of crop region
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Cropped images tensor
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()

    handle = rppCreate(batch_size, backend_int)
    
    # Convert scalars to lists
    x1_array = [x1] * batch_size if isinstance(x1, (int, float)) else x1
    y1_array = [y1] * batch_size if isinstance(y1, (int, float)) else y1
    width_array = [crop_width] * batch_size if isinstance(crop_width, (int, float)) else crop_width
    height_array = [crop_height] * batch_size if isinstance(crop_height, (int, float)) else crop_height
    
    _crop(images, output, x1_array, y1_array, width_array, height_array, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


# Effects Augmentations (2)
def vignette(images, intensity=0.5, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply vignette effect to images.
    
    Args:
        images: Input tensor (B, C, H, W)
        intensity: Vignette intensity (0.0 to 1.0)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Images with vignette effect
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.zeros_like(images).contiguous()
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if images.shape[1] <= 3 else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if images.shape[1] <= 3 else images.shape[1]] * batch_size
    
    handle = rppCreate(batch_size, backend_int)
    intensity_array = [intensity] * batch_size
    _vignette(images, output, intensity_array, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


def pixelate(images, pixelation_percentage=50.0, roi_widths=None, roi_heights=None, input_layout=None, output_layout=None, backend=None):
    """
    Apply pixelate effect to images.
    
    Args:
        images: Input tensor (B, C, H, W)
        pixelation_percentage: Pixelation level (0-100)
        roi_widths: List of actual image widths (None = use full tensor width)
        roi_heights: List of actual image heights (None = use full tensor height)
        backend: RppBackend (None = auto-detect)
    
    Returns:
        Pixelated images
    """
    if backend is None:
        backend = get_default_backend()
    
    backend_int = backend.value if hasattr(backend, 'value') else int(backend)
    
    if not images.is_contiguous():
        images = images.contiguous()
        
    if backend == HIP and not images.is_cuda:
        images = images.cuda()
    elif backend == HOST and images.is_cuda:
        images = images.cpu()
    
    batch_size = images.shape[0]
    output = torch.empty_like(images)
    
    # Set ROI dimensions
    if roi_widths is None:
        roi_widths = [images.shape[3] if input_layout == 'NCHW' else images.shape[2]] * batch_size
    if roi_heights is None:
        roi_heights = [images.shape[2] if input_layout == 'NCHW' else images.shape[1]] * batch_size

    # Create scratch buffer
    if input_layout == 'NCHW':
        scratch_size = batch_size * images.shape[1] * images.shape[2] * images.shape[3]
    else:  # NHWC
        scratch_size = batch_size * images.shape[1] * images.shape[2] * images.shape[3]
    scratch = torch.empty(scratch_size, dtype=torch.float32, device=images.device)
    
    handle = rppCreate(batch_size, backend_int)
    _pixelate(images, output, scratch, pixelation_percentage, roi_widths, roi_heights, handle, backend_int)
    rppDestroy(handle, backend_int)
    
    return output


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
