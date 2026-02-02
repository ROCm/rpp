# Mukesh/rpp/test_suite_aligned.py

"""
RPP Aligned Test Suite
======================

Complete test suite with clear distinctions:
1. Unit Testing - Image generation only, saves processed images to backend-specific folders
2. QA Testing - Comparison of generated images with golden reference outputs (all 10 augmentations)
3. Performance Testing - Time measurements for each augmentation

Usage:
    python test_suite_aligned.py --type unit --backend HOST
    python test_suite_aligned.py --type unit --backend HIP
    python test_suite_aligned.py --type qa --backend HOST
    python test_suite_aligned.py --type perf --backend HIP
    python test_suite_aligned.py --type all --backend HOST
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
                # image = util.load_image(img_path, device=device, apply_padding=False)
                output = fn.brightness(image, alpha=1.5, beta=10.0, backend=self.backend)
                
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
                output = fn.gamma_correction(image, gamma=0.8, backend=self.backend)
                
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
                output = fn.flip(image, horizontal=True, backend=self.backend)
                
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
                output = fn.hue(image, hue_shift=45, backend=self.backend)
                
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
                output = fn.contrast(image, contrast_factor=1.5, backend=self.backend)
                
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
                output = fn.vignette(image, intensity=0.5, backend=self.backend)
                
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
                output = fn.pixelate(image, pixelation_percentage=50.0, backend=self.backend)
                
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
    
    def _load_binary_tensor(self, bin_path):
        """Load binary tensor reference file and convert to numpy arrays for each test image"""
        try:
            print(f"    Loading binary reference: {bin_path}")
            
            # Load binary tensor file
            with open(bin_path, 'rb') as f:
                data = f.read()
            
            # Convert to numpy array (uint8 format)
            tensor_data = np.frombuffer(data, dtype=np.uint8)
            print(f"    Binary data size: {len(tensor_data)} bytes")
            
            # The binary file contains concatenated data for all test images
            # We need to split it into individual images based on the test image sizes
            images = []
            offset = 0
            
            for img_name in self.config.TEST_IMAGES:
                # Parse image dimensions from filename
                if "50x50" in img_name:
                    h, w = 50, 50
                elif "100x100" in img_name:
                    h, w = 100, 100
                elif "150x150" in img_name:
                    h, w = 150, 150
                else:
                    raise ValueError(f"Cannot determine size for image: {img_name}")
                
                # Calculate expected size for this image (HWC format)
                expected_size = h * w * 3
                
                if offset + expected_size > len(tensor_data):
                    print(f"    Warning: Not enough data for {img_name}, skipping")
                    images.append(None)
                    continue
                
                # Extract data for this image
                img_data = tensor_data[offset:offset + expected_size]
                
                # Reshape to (H, W, C) format
                img_array = img_data.reshape(h, w, 3)
                images.append(img_array)
                
                offset += expected_size
                print(f"    Extracted {img_name}: {img_array.shape}")
            
            return images
            
        except Exception as e:
            raise RuntimeError(f"Failed to load binary tensor from {bin_path}: {e}")
    
    def _compare_with_reference(self, aug_name, tolerance=None):
        """Compare generated images with binary reference tensor for a specific augmentation"""
        if tolerance is None:
            tolerance = self.config.UNIT_TOLERANCE
        
        aug_images_dir = os.path.join(self.backend_images_dir, aug_name)
        aug_ref_dir = os.path.join(self.reference_dir, aug_name)
        
        if not os.path.exists(aug_images_dir):
            return None, f"Generated images not found: {aug_images_dir}"
        
        if not os.path.exists(aug_ref_dir):
            return None, f"Reference directory not found: {aug_ref_dir}"
        
        # Look for binary reference file
        ref_bin_path = os.path.join(aug_ref_dir, f"{aug_name}_u8_Tensor.bin")
        if not os.path.exists(ref_bin_path):
            return None, f"Binary reference file not found: {ref_bin_path}"
        
        try:
            # Load binary reference tensor and split into individual images
            reference_images = self._load_binary_tensor(ref_bin_path)
            
        except Exception as e:
            return None, f"Failed to load binary reference: {e}"
        
        results = []
        for i, img_name in enumerate(self.config.TEST_IMAGES):
            img_path = os.path.join(aug_images_dir, img_name)
            
            if not os.path.exists(img_path):
                results.append((img_name, False, f"Generated image missing: {img_path}"))
                continue
            
            if i >= len(reference_images) or reference_images[i] is None:
                results.append((img_name, None, f"Reference data missing for image {i}"))
                continue
            
            try:
                # Load generated image
                generated = np.array(Image.open(img_path))
                reference = reference_images[i]
                
                print(f"    Comparing {img_name}: gen={generated.shape} vs ref={reference.shape}")
                
                # Handle potential shape mismatches
                if generated.shape != reference.shape:
                    # Try to match shapes if possible
                    if len(generated.shape) == 3 and len(reference.shape) == 3:
                        min_h = min(generated.shape[0], reference.shape[0])
                        min_w = min(generated.shape[1], reference.shape[1])
                        min_c = min(generated.shape[2], reference.shape[2])
                        generated = generated[:min_h, :min_w, :min_c]
                        reference = reference[:min_h, :min_w, :min_c]
                        print(f"    Shape adjusted to: {generated.shape}")
                    else:
                        results.append((img_name, False, f"Shape mismatch: gen={generated.shape} vs ref={reference.shape}"))
                        continue
                
                # Calculate differences
                diff = np.abs(generated.astype(np.float32) - reference.astype(np.float32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                
                # Calculate PSNR
                psnr = self.calculate_psnr(generated, reference)
                
                # Determine if passed
                passed = max_diff <= tolerance and psnr >= self.config.QA_PSNR_THRESHOLD
                
                results.append((img_name, passed, {
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'psnr': psnr,
                    'tolerance': tolerance,
                    'psnr_threshold': self.config.QA_PSNR_THRESHOLD
                }))
                
            except Exception as e:
                results.append((img_name, False, f"Comparison error: {e}"))
        
        return results, None
    
    def _run_qa_test(self, aug_name, test_number):
        """Generic QA test runner"""
        print(f"  [QA-{test_number}] {aug_name.title()} Comparison", end=" ... ")
        
        results, error = self._compare_with_reference(aug_name)
        
        if error:
            print(f"SKIP ({error})")
            self.results.append((f'{aug_name}_qa', None))
            return None
        
        passed_count = sum(1 for _, passed, _ in results if passed is True)
        failed_count = sum(1 for _, passed, _ in results if passed is False)
        skipped_count = sum(1 for _, passed, _ in results if passed is None)
        
        overall_pass = failed_count == 0 and passed_count > 0
        status = f"PASS ({passed_count}/{len(results)})" if overall_pass else f"FAIL ({passed_count}/{len(results)})"
        print(status)
        
        # Print detailed results
        for img_name, passed, details in results:
            if isinstance(details, dict):
                print(f"    {img_name}: {'PASS' if passed else 'FAIL'} "
                      f"(max_diff={details['max_diff']:.1f}, psnr={details['psnr']:.1f}dB)")
            else:
                print(f"    {img_name}: {'PASS' if passed else 'SKIP' if passed is None else 'FAIL'} ({details})")
        
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
