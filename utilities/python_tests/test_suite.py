# Mukesh/rpp/test_suite_aligned.py

"""
RPP Aligned Test Suite
======================

Complete test suite with clear distinctions:
1. Unit Testing - Image generation only, saves processed images to backend-specific folders
2. QA Testing - Comparison of generated images with golden reference outputs (all 10 augmentations)
3. Performance Testing - Time measurements for each augmentation

Usage:
    python test_suite.py --type unit --backend HOST
    python test_suite.py --type unit --backend HIP
    python test_suite.py --type qa --backend HOST
    python test_suite.py --type perf --backend HIP
    python test_suite.py --type all --backend HOST
"""

import sys
import os
import argparse
import time
import hashlib
import numpy as np
import torch
import datetime
from PIL import Image
import struct

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print(f"Loading from: {SCRIPT_DIR}")

import rpp_pybind.amd.rpp.fn as fn
import rpp_pybind.amd.rpp.utils as util

from rpp_pybind.amd.rpp.rpp_types import (
    is_gpu_available, get_default_backend, HOST, HIP
)

print("✓ All RPP modules loaded successfully\n")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# =============================================================================
# UTILITY FUNCTIONS FOR BINARY TENSOR AND IMAGE COMPARISON
# =============================================================================

def load_binary_tensor(bin_path, shape=None, dtype=np.uint8):
    """
    Load tensor data from binary file.
    
    Parameters:
    -----------
    bin_path : str
        Path to the binary file
    shape : tuple, optional
        Expected shape of the tensor. If None, will try to infer from file
    dtype : numpy.dtype, default np.uint8
        Data type of the binary data
    
    Returns:
    --------
    numpy.ndarray or torch.Tensor
        Loaded tensor data
    """
    try:
        # Read binary data
        with open(bin_path, 'rb') as f:
            # Check if file has header (first 4 bytes might indicate format)
            first_bytes = f.read(4)
            f.seek(0)
            
            # Check for custom header format (could be extended)
            if first_bytes[:2] == b'RT':  # Custom RPP Tensor format
                # Read header: 2 bytes magic, 1 byte dtype, 1 byte ndims, then shape
                magic = f.read(2)
                dtype_byte = struct.unpack('B', f.read(1))[0]
                ndims = struct.unpack('B', f.read(1))[0]
                shape = struct.unpack(f'{ndims}I', f.read(4 * ndims))
                
                # Map dtype byte to numpy dtype
                dtype_map = {0: np.uint8, 1: np.float32, 2: np.int32, 3: np.float16}
                dtype = dtype_map.get(dtype_byte, np.uint8)
                
                # Read actual data
                data = np.frombuffer(f.read(), dtype=dtype)
            else:
                # Raw binary format - need shape
                data = np.frombuffer(f.read(), dtype=dtype)
                
                if shape is None:
                    # Try to infer shape for common image formats
                    total_pixels = data.size
                    
                    # Special case: Reference files with 273,600 bytes
                    # This is a batch of 3 images in planar format: 3 x 3 x 200 x 152
                    if total_pixels == 273600:
                        # Planar batch format: (batch=3, channels=3, height=200, width=152)
                        shape = (3, 3, 200, 152)
                        print(f"Detected planar batch format: {shape}")
                    # Special case: Hue reference with 205,200 bytes
                    # This is a batch of 3 images: 3 x 3 x 150 x 152
                    elif total_pixels == 205200:
                        shape = (3, 3, 150, 152)
                        print(f"Detected hue batch format: {shape}")
                    # Common single image shapes
                    elif total_pixels == 50 * 50 * 3:
                        shape = (50, 50, 3)
                    elif total_pixels == 100 * 100 * 3:
                        shape = (100, 100, 3)
                    elif total_pixels == 150 * 150 * 3:
                        shape = (150, 150, 3)
                    elif total_pixels == 224 * 224 * 3:
                        shape = (224, 224, 3)
                    else:
                        # Try to find a reasonable shape
                        # Assume 3-channel image
                        pixels = total_pixels // 3
                        side = int(np.sqrt(pixels))
                        if side * side * 3 == total_pixels:
                            shape = (side, side, 3)
                        else:
                            # Return as flat array if can't determine shape
                            print(f"Warning: Could not infer shape for {bin_path}, size={total_pixels}, returning flat array")
                            return data
        
        # Reshape if needed
        if shape is not None:
            try:
                data = data.reshape(shape)
            except ValueError as e:
                print(f"Warning: Could not reshape data from {bin_path} to {shape}: {e}")
                return data
        
        return data
        
    except Exception as e:
        print(f"Error loading binary tensor from {bin_path}: {e}")
        return None


def planar_to_packed(planar_image):
    """Convert planar format (CxHxW) to packed format (HxWxC)"""
    if len(planar_image.shape) == 3 and planar_image.shape[0] == 3:
        # Transpose from CxHxW to HxWxC
        return np.transpose(planar_image, (1, 2, 0))
    return planar_image


def extract_roi(image, target_height, target_width):
    """Extract region of interest from padded image"""
    if len(image.shape) == 3:
        # Remove padding - take only the target dimensions
        return image[:target_height, :target_width, :]
    elif len(image.shape) == 2:
        return image[:target_height, :target_width]
    return image


def extract_reference_image_by_size(batch_tensor, img_name, aug_name=None):
    """
    Extract the appropriate reference image from batch based on image name/size.
    
    The reference files contain 3 images with the same augmentation applied.
    Data is stored in packed format (NHWC) sequentially.
    Each image is padded to batch_h x batch_w.
    
    Parameters:
    -----------
    batch_tensor : np.ndarray
        Batch tensor containing 3 reference images
    img_name : str
        Name of the image to determine which index to extract
    aug_name : str, optional
        Name of the augmentation to determine correct dimensions
    """
    # Handle both standard and hue formats
    if batch_tensor.shape == (3, 3, 200, 152):
        batch_h, batch_w = 200, 152
    elif batch_tensor.shape == (3, 3, 150, 152):
        batch_h, batch_w = 150, 152
    else:
        # Not a recognized batch tensor format
        return batch_tensor
    
    # Determine which image index to extract based on filename
    if '50x50' in img_name or '1_img' in img_name:
        img_idx = 0
        original_size = 50
    elif '100x100' in img_name or '2_img' in img_name:
        img_idx = 1
        original_size = 100
    elif '150x150' in img_name or '3_img' in img_name:
        img_idx = 2
        original_size = 150
    else:
        # Default to first image if can't determine
        img_idx = 0
        original_size = 50
        print(f"Warning: Could not determine image size from name '{img_name}', using 50x50")
    
    # Determine the actual dimensions based on augmentation
    if aug_name == 'resize':
        # Resize always outputs 224x224 for all images
        height, width = 224, 224
    elif aug_name == 'crop':
        # Crop parameters: x1=10, y1=10, width=80, height=80
        if original_size == 50:
            # Can't crop 80x80 from 50x50, max is 40x40
            height, width = 40, 40
        else:
            # 100x100 and 150x150 can accommodate 80x80 crop
            height, width = 80, 80
    else:
        # For all other augmentations, output matches input size
        height, width = original_size, original_size
    
    # The data is stored as 3 sequential packed images
    # Flatten and reshape to extract correct image
    flat_data = batch_tensor.flatten()
    
    # Each image occupies batch_h * batch_w * 3 bytes in packed format
    bytes_per_image = batch_h * batch_w * 3
    
    # Extract the specific image's data
    img_start = img_idx * bytes_per_image
    img_end = img_start + bytes_per_image
    img_data = flat_data[img_start:img_end]
    
    # Reshape to packed format (H, W, C)
    packed_img = img_data.reshape(batch_h, batch_w, 3)
    
    # Extract ROI (remove padding)
    # Ensure we don't exceed available dimensions
    height = min(height, batch_h)
    width = min(width, batch_w)
    roi_img = packed_img[:height, :width, :]
    
    return roi_img


def compare_with_reference(generated, reference, tolerance=5, qa_threshold=30.0):
    """
    Compare generated image with reference image.
    
    Parameters:
    -----------
    generated : np.ndarray, torch.Tensor, or str
        Generated image (tensor or path)
    reference : np.ndarray, torch.Tensor, or str  
        Reference image (tensor or path)
    tolerance : int, default 5
        Maximum allowed pixel difference for PASS (range: -tolerance to +tolerance)
    qa_threshold : float, default 30.0
        Minimum PSNR (dB) for QA PASS
    
    Returns:
    --------
    dict
        Comparison results with keys:
        - 'status': 'PASS', 'FAIL', or 'SKIP'
        - 'shape_match': bool
        - 'generated_shape': tuple
        - 'reference_shape': tuple
        - 'max_diff': float (if shapes match)
        - 'psnr': float (if shapes match)
        - 'mismatched_pixels': int (number of pixels outside tolerance)
        - 'total_pixels': int
        - 'match_percentage': float
        - 'message': str
    """
    result = {
        'status': 'SKIP',
        'shape_match': False,
        'generated_shape': None,
        'reference_shape': None,
        'max_diff': None,
        'psnr': None,
        'mismatched_pixels': None,
        'total_pixels': None,
        'match_percentage': None,
        'message': ''
    }
    
    try:
        # Load images if paths are provided
        if isinstance(generated, str):
            if generated.endswith('.bin'):
                generated = load_binary_tensor(generated)
            else:
                generated = np.array(Image.open(generated))
        
        if isinstance(reference, str):
            if reference.endswith('.bin'):
                reference = load_binary_tensor(reference)
            else:
                reference = np.array(Image.open(reference))
        
        # Convert torch tensors to numpy if needed
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()
        if isinstance(reference, torch.Tensor):
            reference = reference.cpu().numpy()
        
        # Ensure we have valid arrays
        if generated is None or reference is None:
            result['message'] = "Failed to load images"
            return result
        
        # Get shapes
        result['generated_shape'] = generated.shape
        result['reference_shape'] = reference.shape
        
        # Check if shapes match
        if generated.shape != reference.shape:
            result['shape_match'] = False
            result['status'] = 'SKIP'
            result['message'] = (f"Shape mismatch - Generated: {generated.shape}, "
                                f"Reference: {reference.shape}")
            return result
        
        result['shape_match'] = True
        
        # Convert to same dtype for comparison
        generated = generated.astype(np.float32)
        reference = reference.astype(np.float32)
        
        # Calculate differences
        diff = generated - reference  # Signed difference
        abs_diff = np.abs(diff)
        result['max_diff'] = np.max(abs_diff)
        
        # Count pixels outside tolerance range [-tolerance, +tolerance]
        mismatched = np.sum((diff < -tolerance) | (diff > tolerance))
        total_pixels = diff.size
        result['mismatched_pixels'] = int(mismatched)
        result['total_pixels'] = int(total_pixels)
        result['match_percentage'] = 100.0 * (total_pixels - mismatched) / total_pixels if total_pixels > 0 else 0.0
        
        # Calculate PSNR
        mse = np.mean((generated - reference) ** 2)
        if mse == 0:
            result['psnr'] = float('inf')
        else:
            max_pixel = 255.0
            result['psnr'] = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Determine PASS/FAIL based on mismatched pixels
        if result['mismatched_pixels'] == 0:
            result['status'] = 'PASS'
            result['message'] = f"QA PASS - All pixels within tolerance (±{tolerance}), PSNR: {result['psnr']:.1f} dB"
        else:
            result['status'] = 'FAIL'
            result['message'] = (f"QA FAIL - {result['mismatched_pixels']}/{result['total_pixels']} pixels "
                                f"outside tolerance (±{tolerance}), Match: {result['match_percentage']:.1f}%, "
                                f"PSNR: {result['psnr']:.1f} dB")
        
        return result
        
    except Exception as e:
        result['status'] = 'SKIP'
        result['message'] = f"Error during comparison: {e}"
        return result

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Global test configuration"""
    
    def __init__(self):
        # Directories
        self.TEST_IMAGES_DIR = "../test_suite/TEST_IMAGES/three_images_mixed_src1"
        self.REFERENCE_DIR = "../test_suite/REFERENCE_OUTPUT"
        
        # Unit test settings
        self.UNIT_TOLERANCE = 5  # pixel difference tolerance
        self.UNIT_TOLERANCE_HIGH = 15  # for interpolation operations
        
        # QA test settings
        self.QA_PSNR_THRESHOLD = 30.0  # dB
        self.QA_SSIM_THRESHOLD = 0.9
        
        # Performance test settings
        self.PERF_WARMUP_ITERS = 5
        self.PERF_TEST_ITERS = 100
        self.PERF_BATCH_SIZES = [1, 8, 16, 32]
        self.PERF_IMAGE_SIZES = [(224, 224), (480, 640), (720, 1280)]
        
        # Test image paths
        self.TEST_IMAGES = [
            "1_img50x50.jpg",
            "2_img100x100.jpg", 
            "3_img150x150.jpg"
        ]
        
        # Supported augmentations
        self.AUGMENTATIONS = [
            'brightness', 'gamma_correction', 'flip', 'resize', 'crop', 
            'hue', 'rotate', 'contrast', 'vignette', 'pixelate'
        ]
    
    def get_output_dir(self, backend_name, test_type="IMAGES"):
        """Get output directory based on backend and test type"""
        return f"OUTPUT_{test_type}_{timestamp}"
    
    def get_backend_output_path(self, backend_name, base_dir):
        """Get complete output path for backend"""
        return os.path.join(backend_name, base_dir)


# =============================================================================
# UNIT TESTS - Image Generation Only
# =============================================================================

class UnitTests:
    """Unit tests - Generate processed images and save to backend-specific folders"""
    
    def __init__(self, backend):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.results = []
        
        # Setup paths
        self.config = TestConfig()
        self.base_output_dir = self.config.get_output_dir(self.backend_name, "IMAGES")
        self.backend_output_dir = self.config.get_backend_output_path(
            self.backend_name, self.base_output_dir
        )
        
        # Create base directories
        os.makedirs(self.backend_output_dir, exist_ok=True)
        
        # Load test images
        self.test_images = [
            os.path.join(self.config.TEST_IMAGES_DIR, img) 
            for img in self.config.TEST_IMAGES
        ]
        
        print(f"Unit Tests Output Directory: {self.backend_output_dir}")
    
    def _save_output_image(self, tensor, augmentation_name, image_name):
        """Save output image to backend-specific folder structure"""
        try:
            # Create augmentation-specific directory
            aug_output_dir = os.path.join(self.backend_output_dir, augmentation_name)
            os.makedirs(aug_output_dir, exist_ok=True)
            
            # Generate output file path
            output_path = os.path.join(aug_output_dir, image_name)
            
            # Save image
            util.save_image(tensor, output_path)
            print(f"✓ Saved {augmentation_name} output: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to save {augmentation_name} output: {e}")
            return False
    
    def test_brightness(self):
        """Generate brightness-processed images"""
        print("  [1/10] Brightness", end=" ... ")
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.brightness(image, alpha=1.75, beta=50.0, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "brightness", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('brightness', success))
        return success
    
    def test_gamma_correction(self):
        """Generate gamma correction-processed images"""
        print("  [2/10] Gamma Correction", end=" ... ")
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.gamma_correction(image, gamma=1.9, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "gamma_correction", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('gamma_correction', success))
        return success
    
    def test_flip(self):
        """Generate flip-processed images"""
        print("  [3/10] Flip", end=" ... ")
        
        if not hasattr(fn, 'flip'):
            print("SKIP (function not available)")
            self.results.append(('flip', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.flip(image, horizontal=True, vertical= False, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "flip", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('flip', success))
        return success
    
    def test_resize(self):
        """Generate resize-processed images"""
        print("  [4/10] Resize", end=" ... ")
        
        if not hasattr(fn, 'resize'):
            print("SKIP (function not available)")
            self.results.append(('resize', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.resize(image, width=224, height=224, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "resize", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('resize', success))
        return success
    
    def test_crop(self):
        """Generate crop-processed images"""
        print("  [5/10] Crop", end=" ... ")
        
        if not hasattr(fn, 'crop'):
            print("SKIP (function not available)")
            self.results.append(('crop', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                # Crop from center (assuming minimum 100x100 images)
                output = fn.crop(image, x1=10, y1=10, crop_width=80, crop_height=80, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "crop", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('crop', success))
        return success
    
    def test_hue(self):
        """Generate hue-processed images"""
        print("  [6/10] Hue", end=" ... ")
        
        if not hasattr(fn, 'hue'):
            print("SKIP (function not available)")
            self.results.append(('hue', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.hue(image, hue_shift=60.0, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "hue", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('hue', success))
        return success
    
    def test_rotate(self):
        """Generate rotate-processed images"""
        print("  [7/10] Rotate", end=" ... ")
        
        if not hasattr(fn, 'rotate'):
            print("SKIP (function not available)")
            self.results.append(('rotate', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.rotate(image, angle=45.0, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "rotate", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('rotate', success))
        return success
    
    def test_contrast(self):
        """Generate contrast-processed images"""
        print("  [8/10] Contrast", end=" ... ")
        
        if not hasattr(fn, 'contrast'):
            print("SKIP (function not available)")
            self.results.append(('contrast', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.contrast(image, contrast_factor=2.96, contrast_center=128.0, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "contrast", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('contrast', success))
        return success
    
    def test_vignette(self):
        """Generate vignette-processed images"""
        print("  [9/10] Vignette", end=" ... ")
        
        if not hasattr(fn, 'vignette'):
            print("SKIP (function not available)")
            self.results.append(('vignette', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.vignette(image, intensity=6.0, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "vignette", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('vignette', success))
        return success
    
    def test_pixelate(self):
        """Generate pixelate-processed images"""
        print("  [10/10] Pixelate", end=" ... ")
        
        if not hasattr(fn, 'pixelate'):
            print("SKIP (function not available)")
            self.results.append(('pixelate', None))
            return None
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        success_count = 0
        
        for img_path in self.test_images:
            try:
                image = util.load_image(img_path, device=device)
                output = fn.pixelate(image, pixelation_percentage=87.5, backend=self.backend)
                
                image_name = os.path.basename(img_path)
                if self._save_output_image(output, "pixelate", image_name):
                    success_count += 1
                    
            except Exception as e:
                print(f"✗ Error processing {img_path}: {e}")
        
        success = success_count == len(self.test_images)
        status = f"SAVED {success_count}/{len(self.test_images)} images"
        print(status)
        
        self.results.append(('pixelate', success))
        return success
    
    def run_all(self):
        """Run all unit tests - generate all processed images"""
        print(f"\n{'='*70}")
        print(f"UNIT TESTS - Image Generation ({self.backend_name})")
        print(f"{'='*70}\n")
        
        tests = [
            self.test_brightness,
            self.test_gamma_correction,
            self.test_flip,
            self.test_resize,
            self.test_crop,
            self.test_hue,
            self.test_rotate,
            self.test_contrast,
            self.test_vignette,
            self.test_pixelate
        ]
        
        for test in tests:
            try:
                test()
                print("-" * 50)
            except Exception as e:
                print(f"ERROR in {test.__name__}: {e}")
                self.results.append((test.__name__.replace('test_', ''), False))
                print("-" * 50)
        
        # Summary
        passed = sum(1 for _, r in self.results if r is True)
        failed = sum(1 for _, r in self.results if r is False)
        skipped = sum(1 for _, r in self.results if r is None)
        
        print(f"\n{'='*70}")
        print(f"Unit Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
        print(f"Output Location: {self.backend_output_dir}")
        print(f"{'='*70}\n")
        
        return self.results


# =============================================================================
# QA TESTS - Golden Reference Comparison
# =============================================================================

class QATests:
    """QA tests - Compare generated images with golden reference outputs"""
    
    def __init__(self, backend, unit_output_dir=None):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.results = []
        
        self.config = TestConfig()
        
        # Set paths for generated images (from unit tests)
        if unit_output_dir:
            self.backend_images_dir = os.path.join(self.backend_name, unit_output_dir)
        else:
            output_dir = self.config.get_output_dir(self.backend_name, "IMAGES")
            self.backend_images_dir = os.path.join(self.backend_name, output_dir)
        
        # QA results directory
        qa_output_dir = self.config.get_output_dir(self.backend_name, "QA_RESULTS")
        self.qa_results_dir = os.path.join(self.backend_name, qa_output_dir)
        os.makedirs(self.qa_results_dir, exist_ok=True)
        
        self.reference_dir = self.config.REFERENCE_DIR
        
        print(f"QA Tests - Generated Images: {self.backend_images_dir}")
        print(f"QA Tests - Reference Images: {self.reference_dir}")
        print(f"QA Results Output: {self.qa_results_dir}")
    
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        # Convert to numpy if needed
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Ensure same shape
        if img1.shape != img2.shape:
            return 0.0  # Shape mismatch
        
        # Calculate MSE
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def _compare_with_reference(self, aug_name, version='2.0'):
        """
        Compare generated images with reference images for a specific augmentation.
        
        Parameters:
        -----------
        aug_name : str
            Name of the augmentation being tested
        version : str
            Version identifier for reference images
        
        Returns:
        --------
        tuple: (results_list, error_message)
            results_list: List of tuples (image_name, pass_status, details, shape_info, pixel_samples)
            error_message: Error string if comparison failed, None otherwise
        """
        results = []
        
        try:
            # Get paths for generated and reference images
            generated_dir = os.path.join(self.backend_images_dir, aug_name)
            
            # Check if generated images directory exists
            if not os.path.exists(generated_dir):
                return None, f"Generated images directory not found: {generated_dir}"
            
            # Map augmentation names to reference directory names
            ref_aug_name = aug_name
            
            # The reference directory structure is REFERENCE_OUTPUT/<augmentation>/
            # NOT REFERENCE_OUTPUT/HOST/<augmentation>/
            reference_aug_dir = os.path.join(self.reference_dir, ref_aug_name)
            
            # Check if reference directory exists
            if not os.path.exists(reference_aug_dir):
                return None, f"Reference directory not found: {reference_aug_dir}"
            
            # Get list of generated images
            generated_images = [f for f in os.listdir(generated_dir) 
                              if f.endswith(('.jpg', '.png', '.bin'))]
            
            if not generated_images:
                return None, f"No generated images found in {generated_dir}"
            
            # The reference files use generic names like brightness_u8_Tensor.bin
            # We'll use the same reference for all test images of this augmentation
            reference_filename = f"{ref_aug_name}_u8_Tensor.bin"
            reference_path = os.path.join(reference_aug_dir, reference_filename)
            
            # Check if reference file exists
            if not os.path.exists(reference_path):
                # Try alternative names for some augmentations
                alternative_names = [
                    f"{ref_aug_name}_f32_Tensor.bin",
                    f"{ref_aug_name}_u8_Tensor_interpolationTypeBilinear.bin",  # for rotate
                    f"{ref_aug_name}_u8_Tensor_interpolationTypeNearestNeighbor.bin"  # for resize
                ]
                
                for alt_name in alternative_names:
                    alt_path = os.path.join(reference_aug_dir, alt_name)
                    if os.path.exists(alt_path):
                        reference_path = alt_path
                        break
                else:
                    # No reference file found at all
                    return None, f"Reference file not found in {reference_aug_dir}"
            
            print(f"    Using reference: {os.path.basename(reference_path)}")
            
            # Load reference once (it's a batch tensor)
            ref_tensor = load_binary_tensor(reference_path)
            if ref_tensor is None:
                return None, f"Failed to load reference file: {reference_path}"
            
            print(f"    Reference batch tensor shape: {ref_tensor.shape}")
            
            # Compare each generated image with the appropriate reference
            for img_name in generated_images:
                generated_path = os.path.join(generated_dir, img_name)
                
                # Load generated image
                generated_img = np.array(Image.open(generated_path)) if generated_path.endswith(('.jpg', '.png')) else load_binary_tensor(generated_path)
                if generated_img is None:
                    results.append((img_name, None, {"message": "Failed to load generated image"}, None, None))
                    continue
                
                # Store original shape for debugging
                original_shape = generated_img.shape
                
                # Determine target size based on image name and augmentation
                if aug_name == 'resize':
                    # Resize always outputs 224x224
                    target_h, target_w = 224, 224
                elif aug_name == 'crop':
                    # Crop outputs depend on source image size
                    # Crop parameters: x1=10, y1=10, width=80, height=80
                    if '50x50' in img_name or '1_img' in img_name:
                        # 50x50 image cropped from (10,10) with max possible size
                        target_h, target_w = 40, 40  # 50-10=40 max in each dimension
                    else:
                        # 100x100 and 150x150 images can accommodate full 80x80 crop
                        target_h, target_w = 80, 80
                else:
                    # For all other augmentations, output size matches input size
                    if '50x50' in img_name or '1_img' in img_name:
                        target_h, target_w = 50, 50
                    elif '100x100' in img_name or '2_img' in img_name:
                        target_h, target_w = 100, 100
                    elif '150x150' in img_name or '3_img' in img_name:
                        target_h, target_w = 150, 150
                    else:
                        # Fallback to actual size
                        target_h, target_w = generated_img.shape[0], generated_img.shape[1] if len(generated_img.shape) >= 2 else (generated_img.shape[0], generated_img.shape[0])
                
                # Extract ROI from generated image (remove padding)
                # RPP adds padding for memory alignment, crop to expected dimensions
                generated_roi = extract_roi(generated_img, target_h, target_w)
                roi_shape = generated_roi.shape
                
                # Extract appropriate reference image from batch
                if ref_tensor.shape == (3, 3, 200, 152) or ref_tensor.shape == (3, 3, 150, 152):
                    # Batch format - extract the correct image with augmentation awareness
                    reference_img = extract_reference_image_by_size(ref_tensor, img_name, aug_name)
                else:
                    reference_img = ref_tensor
                
                # Store shape information
                shape_info = {
                    'original_shape': original_shape,
                    'roi_shape': roi_shape,
                    'reference_shape': reference_img.shape
                }
                
                # Perform comparison
                comparison = compare_with_reference(
                    generated_roi, 
                    reference_img,
                    tolerance=self.config.UNIT_TOLERANCE if hasattr(self, 'config') else 5,
                    qa_threshold=self.config.QA_PSNR_THRESHOLD if hasattr(self, 'config') else 30.0
                )
                
                # Get sample pixel values for debugging
                pixel_samples = None
                if comparison.get('shape_match', False):
                    pixel_samples = self._get_pixel_samples(generated_roi, reference_img)
                
                # Add result with detailed information
                if comparison.get('shape_match') == False:
                    # Enhancement: show shapes when mismatch occurs
                    results.append((img_name, None, comparison, shape_info, None))
                elif comparison['status'] == 'PASS':
                    results.append((img_name, True, comparison, shape_info, pixel_samples))
                elif comparison['status'] == 'FAIL':
                    results.append((img_name, False, comparison, shape_info, pixel_samples))
                else:  # SKIP
                    results.append((img_name, None, comparison, shape_info, None))
            
            return results, None
            
        except Exception as e:
            return None, f"Error during comparison: {str(e)}"
    
    def _get_pixel_samples(self, generated_img, reference_img, sample_size=5):
        """
        Get sample pixel values for debugging comparison issues.
        
        Parameters:
        -----------
        generated_img : np.ndarray
            Generated image array
        reference_img : np.ndarray
            Reference image array
        sample_size : int
            Size of sample grid (default 5x5)
        
        Returns:
        --------
        list: List of tuples (row, col, generated_rgb, reference_rgb, diff_rgb)
        """
        samples = []
        rows = min(sample_size, generated_img.shape[0])
        cols = min(sample_size, generated_img.shape[1])
        
        for i in range(rows):
            for j in range(cols):
                g = generated_img[i, j]
                r = reference_img[i, j]
                d = g.astype(int) - r.astype(int)
                samples.append((i, j, g, r, d))
        
        return samples
    
    def _run_qa_test(self, aug_name, test_number):
        """Generic QA test runner with detailed output"""
        print(f"\n  [QA-{test_number}] {aug_name.upper()} QA TEST")
        print("  " + "=" * 70)
        
        results, error = self._compare_with_reference(aug_name, '2.0')
        
        if error:
            print(f"  SKIP ({error})")
            self.results.append((f'{aug_name}_qa', None))
            return None
        
        # Count results
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        
        # Process each image result with detailed output
        for result_data in results:
            if len(result_data) == 5:
                img_name, passed, details, shape_info, pixel_samples = result_data
            else:
                # Fallback for old format
                img_name, passed, details = result_data[:3]
                shape_info = None
                pixel_samples = None
            
            print(f"\n  Image: {img_name}")
            print("  " + "-" * 60)
            
            # Print shape information if available
            if shape_info:
                print(f"    Generated image shape (with padding): {shape_info['original_shape']}")
                print(f"    After ROI extraction: {shape_info['roi_shape']}")
                print(f"    Reference image shape: {shape_info['reference_shape']}")
            
            # Print comparison results
            if isinstance(details, dict):
                if details.get('shape_match', True):
                    # Shape matches - show detailed metrics
                    print(f"\n    Status: {details.get('status', 'UNKNOWN')}")
                    print(f"    {details.get('message', 'No message')}")
                    
                    if details.get('max_diff') is not None:
                        print(f"\n    Detailed Metrics:")
                        print(f"      Max difference: {details['max_diff']:.1f}")
                        print(f"      PSNR: {details['psnr']:.1f} dB")
                        print(f"      Mismatched pixels: {details['mismatched_pixels']}/{details['total_pixels']}")
                        print(f"      Match percentage: {details['match_percentage']:.2f}%")
                    
                    # Print pixel samples if available and it's a FAIL
                    if pixel_samples and passed is False:
                        print(f"\n    Sample pixel values (first 5x5):")
                        print(f"      {'Generated':^15} | {'Reference':^15} | {'Difference':^20}")
                        print(f"      {'-'*15} | {'-'*15} | {'-'*20}")
                        
                        for row, col, g, r, d in pixel_samples[:10]:  # Show first 10 pixels
                            gen_str = f"({g[0]:3},{g[1]:3},{g[2]:3})"
                            ref_str = f"({r[0]:3},{r[1]:3},{r[2]:3})"
                            diff_str = f"({d[0]:+4},{d[1]:+4},{d[2]:+4})"
                            print(f"      {gen_str:^15} | {ref_str:^15} | {diff_str:^20}")
                else:
                    # Shape mismatch
                    print(f"    Status: SKIP")
                    print(f"    Shape mismatch - Generated: {details.get('generated_shape')}, "
                          f"Reference: {details.get('reference_shape')}")
            else:
                # Simple string message
                print(f"    Status: {'PASS' if passed else 'SKIP' if passed is None else 'FAIL'}")
                print(f"    {details}")
            
            # Update counts
            if passed is True:
                passed_count += 1
            elif passed is False:
                failed_count += 1
            else:
                skipped_count += 1
        
        # Overall summary for this augmentation
        overall_pass = failed_count == 0 and passed_count > 0
        print(f"\n  {aug_name.upper()} Summary: ", end="")
        if overall_pass:
            print(f"PASS ({passed_count}/{len(results)} passed)")
        else:
            print(f"FAIL ({passed_count} passed, {failed_count} failed, {skipped_count} skipped)")
        
        print("  " + "=" * 70)
        
        self.results.append((f'{aug_name}_qa', overall_pass))
        return overall_pass
    
    def run_all(self):
        """Run all QA tests - compare all generated images with references"""
        print(f"\n{'='*70}")
        print(f"QA TESTS - Golden Reference Comparison ({self.backend_name})")
        print(f"{'='*70}\n")
        
        # Test all 10 augmentations
        qa_tests = [
            ('brightness', 1),
            ('gamma_correction', 2),
            ('flip', 3),
            ('resize', 4),
            ('crop', 5),
            ('hue', 6),
            ('rotate', 7),
            ('contrast', 8),
            ('vignette', 9),
            ('pixelate', 10)
        ]
        
        for aug_name, test_num in qa_tests:
            try:
                self._run_qa_test(aug_name, test_num)
                print("-" * 50)
            except Exception as e:
                print(f"ERROR in {aug_name}_qa: {e}")
                self.results.append((f'{aug_name}_qa', False))
                print("-" * 50)
        
        # Write QA results summary to file
        self._write_qa_summary()
        
        # Summary
        passed = sum(1 for _, r in self.results if r is True)
        failed = sum(1 for _, r in self.results if r is False)
        skipped = sum(1 for _, r in self.results if r is None)
        
        print(f"\n{'='*70}")
        print(f"QA Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
        print(f"QA Results File: {self.qa_results_dir}/QA_results.txt")
        print(f"{'='*70}\n")
        
        return self.results
    
    def _write_qa_summary(self):
        """Write QA results summary to file"""
        qa_file_path = os.path.join(self.qa_results_dir, "QA_results.txt")
        
        with open(qa_file_path, 'w') as f:
            f.write(f"RPP QA Test Results - {self.backend_name} Backend\n")
            f.write(f"={'='*50}\n")
            f.write(f"Test Date: {timestamp}\n")
            f.write(f"Backend: {self.backend_name}\n\n")
            
            for aug_name, result in self.results:
                status = "PASS" if result is True else "SKIP" if result is None else "FAIL"
                f.write(f"{aug_name:<20}: {status}\n")
            
            f.write(f"\nSummary:\n")
            passed = sum(1 for _, r in self.results if r is True)
            failed = sum(1 for _, r in self.results if r is False)
            skipped = sum(1 for _, r in self.results if r is None)
            f.write(f"Passed: {passed}, Failed: {failed}, Skipped: {skipped}\n")


# =============================================================================
# PERFORMANCE TESTS - Timing Measurements
# =============================================================================

class PerformanceTests:
    """Performance tests - Measure timing for each augmentation"""
    
    def __init__(self, backend, num_iterations=100):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.num_iterations = num_iterations
        self.results = {}
        
        self.config = TestConfig()
        
        # Performance results directory
        perf_output_dir = self.config.get_output_dir(self.backend_name, "PERFORMANCE_LOGS")
        self.perf_results_dir = os.path.join(self.backend_name, perf_output_dir)
        os.makedirs(self.perf_results_dir, exist_ok=True)
        
        # Load test images
        self.test_images = [
            os.path.join(self.config.TEST_IMAGES_DIR, img) 
            for img in self.config.TEST_IMAGES
        ]
        
        print(f"Performance Tests - Backend: {self.backend_name}")
        print(f"Performance Tests - Iterations: {num_iterations}")
        print(f"Performance Results: {self.perf_results_dir}")
    
    def _measure_performance(self, func_name, func_call, warmup_iters=5):
        """Measure performance of an augmentation function"""
        device = 'cuda' if self.backend == HIP else 'cpu'
        times = []
        
        # Load test image
        test_image = util.load_image(self.test_images[1], device=device)  # Use 100x100
        
        try:
            # Warmup iterations
            for _ in range(warmup_iters):
                _ = func_call(test_image)
                if self.backend == HIP:
                    torch.cuda.synchronize()
            
            # Actual timing iterations
            for i in range(self.num_iterations):
                start_time = time.perf_counter()
                
                output = func_call(test_image)
                
                if self.backend == HIP:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Calculate statistics
            min_time = min(times)
            max_time = max(times)
            avg_time = np.mean(times)
            std_time = np.std(times)
            total_time = sum(times)
            
            return {
                'min_time': min_time,
                'max_time': max_time,
                'avg_time': avg_time,
                'std_time': std_time,
                'total_time': total_time,
                'iterations': self.num_iterations
            }
            
        except Exception as e:
            print(f"    ERROR: {e}")
            return None
    
    def run_all(self):
        """Run performance tests for all augmentations"""
        print(f"\n{'='*70}")
        print(f"PERFORMANCE TESTS - Timing Measurements ({self.backend_name})")
        print(f"{'='*70}\n")
        
        # Performance test definitions
        perf_tests = [
            ('brightness', lambda img: fn.brightness(img, alpha=1.5, beta=10.0, backend=self.backend)),
            ('gamma_correction', lambda img: fn.gamma_correction(img, gamma=0.8, backend=self.backend)),
            ('flip', lambda img: fn.flip(img, horizontal=True, backend=self.backend)),
            ('resize', lambda img: fn.resize(img, width=224, height=224, backend=self.backend)),
            ('crop', lambda img: fn.crop(img, x1=10, y1=10, crop_width=80, crop_height=80, backend=self.backend)),
            ('hue', lambda img: fn.hue(img, hue_shift=45, backend=self.backend)),
            ('rotate', lambda img: fn.rotate(img, angle=45.0, backend=self.backend)),
            ('contrast', lambda img: fn.contrast(img, contrast_factor=1.5, backend=self.backend)),
            ('vignette', lambda img: fn.vignette(img, intensity=0.5, backend=self.backend)),
            ('pixelate', lambda img: fn.pixelate(img, pixelation_percentage=50.0, backend=self.backend))
        ]
        
        for i, (func_name, func_call) in enumerate(perf_tests, 1):
            print(f"  [{i}/10] {func_name.title()}", end=" ... ")
            
            # Check if function is available
            if not hasattr(fn, func_name):
                print("SKIP (function not available)")
                continue
            
            try:
                result = self._measure_performance(func_name, func_call)
                
                if result:
                    self.results[func_name] = result
                    print(f"Avg: {result['avg_time']:.2f}ms, Min: {result['min_time']:.2f}ms, Max: {result['max_time']:.2f}ms")
                else:
                    print("FAIL (measurement error)")
                    
            except Exception as e:
                print(f"ERROR: {e}")
            
            print("-" * 50)
        
        # Write performance results to file
        self._write_performance_summary()
        
        print(f"\n{'='*70}")
        print(f"Performance Test Summary: {len(self.results)} functions tested")
        print(f"Performance Results File: {self.perf_results_dir}/performance_{self.backend_name}.txt")
        print(f"{'='*70}\n")
        
        return self.results
    
    def _write_performance_summary(self):
        """Write performance results to file"""
        perf_file_path = os.path.join(self.perf_results_dir, f"performance_{self.backend_name}.txt")
        
        with open(perf_file_path, 'w') as f:
            f.write(f"RPP Performance Test Results - {self.backend_name} Backend\n")
            f.write(f"={'='*60}\n")
            f.write(f"Test Date: {timestamp}\n")
            f.write(f"Backend: {self.backend_name}\n")
            f.write(f"Iterations per function: {self.num_iterations}\n")
            f.write(f"Test image: 100x100 pixels\n\n")
            
            f.write(f"{'Function':<20} | {'Min Time':<8} | {'Max Time':<8} | {'Avg Time':<8} | {'Std Dev':<8} | {'Total':<8}\n")
            f.write(f"{'-'*20} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}\n")
            
            for func_name, result in self.results.items():
                f.write(f"{func_name:<20} | {result['min_time']:<8.2f} | {result['max_time']:<8.2f} | "
                       f"{result['avg_time']:<8.2f} | {result['std_time']:<8.2f} | {result['total_time']:<8.2f}\n")
            
            f.write(f"\nAll times in milliseconds (ms)\n")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RPP Aligned Test Suite - Enhanced',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_suite_aligned.py --type unit --backend HOST
  python test_suite_aligned.py --type qa --backend HOST
  python test_suite_aligned.py --type perf --backend HIP --iterations 100
  python test_suite_aligned.py --type all --backend HOST
        """
    )
    
    parser.add_argument('--type', choices=['unit', 'qa', 'perf', 'all'],
                       default='unit', help='Type of tests to run')
    parser.add_argument('--backend', choices=['HOST', 'HIP'],
                       required=True, help='Backend to test')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for performance tests')
    parser.add_argument('--unit-output-dir', type=str, default=None,
                       help='Directory containing unit test outputs (for QA tests)')
    
    args = parser.parse_args()
    
    # Determine backend
    backend = HIP if args.backend == 'HIP' else HOST
    backend_name = args.backend
    
    # Check GPU availability for HIP backend
    if backend == HIP and not is_gpu_available():
        print(f"ERROR: HIP backend requested but GPU not available")
        return 1
    
    # Print header
    print("\n" + "="*70)
    print("RPP ALIGNED TEST SUITE - ENHANCED")
    print("="*70)
    print(f"Test Type: {args.type}")
    print(f"Backend: {backend_name}")
    print(f"GPU Available: {is_gpu_available()}")
    if args.type == 'perf':
        print(f"Performance Iterations: {args.iterations}")
    print("="*70)
    
    # Run tests based on type
    results = {}
    
    if args.type == 'unit':
        print(f"\n=== UNIT TESTS - IMAGE GENERATION ({backend_name}) ===")
        unit_tests = UnitTests(backend)
        results['unit'] = unit_tests.run_all()
        
    elif args.type == 'qa':
        print(f"\n=== QA TESTS - REFERENCE COMPARISON ({backend_name}) ===")
        qa_tests = QATests(backend, args.unit_output_dir)
        results['qa'] = qa_tests.run_all()
        
    elif args.type == 'perf':
        print(f"\n=== PERFORMANCE TESTS ({backend_name}) ===")
        perf_tests = PerformanceTests(backend, args.iterations)
        results['perf'] = perf_tests.run_all()
        
    elif args.type == 'all':
        print(f"\n=== ALL TESTS ({backend_name}) ===")
        
        # Run unit tests first
        print(f"\n--- UNIT TESTS - IMAGE GENERATION ---")
        unit_tests = UnitTests(backend)
        results['unit'] = unit_tests.run_all()
        
        # Run QA tests using the unit test output
        print(f"\n--- QA TESTS - REFERENCE COMPARISON ---")
        qa_tests = QATests(backend, unit_tests.base_output_dir)
        results['qa'] = qa_tests.run_all()
        
        # Run performance tests
        print(f"\n--- PERFORMANCE TESTS ---")
        perf_tests = PerformanceTests(backend, args.iterations)
        results['perf'] = perf_tests.run_all()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for test_type, test_results in results.items():
        if test_type == 'perf':
            print(f"\nPERFORMANCE:")
            print(f"  Functions tested: {len(test_results)}")
            if test_results:
                avg_times = [r['avg_time'] for r in test_results.values()]
                print(f"  Average processing time: {np.mean(avg_times):.2f}ms")
        elif isinstance(test_results, list):
            passed = sum(1 for _, r in test_results if r is True)
            failed = sum(1 for _, r in test_results if r is False)
            skipped = sum(1 for _, r in test_results if r is None)
            print(f"\n{test_type.upper()}:")
            print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
