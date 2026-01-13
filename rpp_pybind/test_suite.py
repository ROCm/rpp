# Mukesh/rpp/test_suite.py

"""
RPP Comprehensive Test Suite
============================

Single unified test suite combining:
1. Unit Testing - Golden reference comparison with C++ test outputs
2. QA Testing - Quality assurance, edge cases, and correctness validation
3. Performance Testing - Throughput and latency benchmarks

Tests all 10 augmentations on both HOST and HIP backends.

LOCATION: Save as /media/rpp1/rpp_pybind/test_suite.py

Usage:
    cd /media/rpp1/rpp_pybind
    python test_suite.py --type all --backend HOST
    python test_suite.py --type unit --backend HIP
    python test_suite.py --type qa
    python test_suite.py --type performance --batch-size 8
"""

import sys
import os
import argparse
import time
import hashlib
import numpy as np
import torch
from PIL import Image

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print(f"Loading from: {SCRIPT_DIR}")

# import rpp_pybind
# Import RPP modules
# try:
from amd.rpp.fn import (
    brightness, gamma_correction, contrast, hue,
    flip, resize, rotate, crop, vignette, pixelate
)
from amd.rpp.rpp_types import (
    is_gpu_available, get_default_backend, HOST, HIP
)
from amd.rpp.utils import create_test_batch, load_image, tensor_to_numpy

print("✓ All RPP modules loaded successfully\n")
    
# except ImportError as e:
#     print(f"✗ Import error: {e}")
#     print("\nMake sure you're running from /media/rpp1/rpp_pybind/")
#     sys.exit(1)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Global test configuration"""
    
    # Directories
    TEST_IMAGES_DIR = "../utilities/test_suite/TEST_IMAGES/three_images_mixed_src1"
    REFERENCE_DIR = "../utilities/test_suite/REFERENCE_OUTPUT"
    
    # Unit test settings
    UNIT_TOLERANCE = 5  # pixel difference tolerance
    UNIT_TOLERANCE_HIGH = 15  # for interpolation operations
    
    # QA test settings
    QA_PSNR_THRESHOLD = 30.0  # dB
    QA_SSIM_THRESHOLD = 0.9
    
    # Performance test settings
    PERF_WARMUP_ITERS = 10
    PERF_TEST_ITERS = 100
    PERF_BATCH_SIZES = [1, 8, 16, 32]
    PERF_IMAGE_SIZES = [(224, 224), (480, 640), (720, 1280)]
    
    # Test image paths
    TEST_IMAGES = [
        "img1.jpg",
        "img2.jpg", 
        "img3.jpg"
    ]


# =============================================================================
# UNIT TESTS - Golden Reference Comparison
# =============================================================================

class UnitTests:
    """Unit tests comparing against C++ golden reference outputs"""
    
    def __init__(self, backend):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.results = []
        self.test_images = [
            os.path.join(TestConfig.TEST_IMAGES_DIR, img) 
            for img in TestConfig.TEST_IMAGES
        ]
    
    def _compare_with_reference(self, output, ref_path, tolerance):
        """Compare output with golden reference"""
        if not os.path.exists(ref_path):
            return None, 0, 0  # Reference not found
        
        # Load reference
        reference = np.array(Image.open(ref_path))
        
        # Convert output to numpy
        if isinstance(output, torch.Tensor):
            output_np = tensor_to_numpy(output[0])  # Remove batch dim
        else:
            output_np = output
        
        # Calculate differences
        diff = np.abs(output_np.astype(np.float32) - reference.astype(np.float32))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        passed = max_diff <= tolerance
        return passed, max_diff, mean_diff
    
    def test_brightness(self):
        """Test brightness against golden reference"""
        print("  [1/10] Brightness", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = brightness(image, alpha=1.5, beta=10.0, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR, 
            f"brightness/brightness_u8_Tensor_{self.backend_name}_three_images_mixed_src1_interpolationType0_noiseType0_alpha1.5_beta10.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f}, mean={mean_diff:.1f})")
        
        self.results.append(('brightness', passed))
        return output
    
    def test_gamma_correction(self):
        """Test gamma correction"""
        print("  [2/10] Gamma Correction", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = gamma_correction(image, gamma=0.8, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"gamma_correction/gamma_correction_u8_Tensor_{self.backend_name}_three_images_mixed_src1_gamma0.8.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('gamma_correction', passed))
        return output
    
    def test_contrast(self):
        """Test contrast"""
        print("  [3/10] Contrast", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = contrast(image, contrast_factor=2.0, contrast_center=128.0, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"contrast/contrast_u8_Tensor_{self.backend_name}_three_images_mixed_src1_contrastFactor2_contrastCenter128.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('contrast', passed))
        return output
    
    def test_hue(self):
        """Test hue"""
        print("  [4/10] Hue", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = hue(image, hue_shift=45.0, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"hue/hue_u8_Tensor_{self.backend_name}_three_images_mixed_src1_hue45.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE_HIGH)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('hue', passed))
        return output
    
    def test_flip(self):
        """Test flip"""
        print("  [5/10] Flip", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = flip(image, horizontal=True, vertical=False, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"flip/flip_u8_Tensor_{self.backend_name}_three_images_mixed_src1_horizontal1_vertical0.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('flip', passed))
        return output
    
    def test_resize(self):
        """Test resize"""
        print("  [6/10] Resize", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = resize(image, width=224, height=224, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"resize/resize_u8_Tensor_{self.backend_name}_three_images_mixed_src1_224x224_interpolationType1.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE_HIGH)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('resize', passed))
        return output
    
    def test_rotate(self):
        """Test rotate"""
        print("  [7/10] Rotate", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = rotate(image, angle=30.0, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"rotate/rotate_u8_Tensor_{self.backend_name}_three_images_mixed_src1_angle30_interpolationType1.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE_HIGH)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('rotate', passed))
        return output
    
    def test_crop(self):
        """Test crop"""
        print("  [8/10] Crop", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = crop(image, x1=100, y1=100, crop_width=200, crop_height=200, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"crop/crop_u8_Tensor_{self.backend_name}_three_images_mixed_src1_100_100_200_200.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('crop', passed))
        return output
    
    def test_vignette(self):
        """Test vignette"""
        print("  [9/10] Vignette", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = vignette(image, intensity=0.7, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"vignette/vignette_u8_Tensor_{self.backend_name}_three_images_mixed_src1_intensity0.7.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE_HIGH)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('vignette', passed))
        return output
    
    def test_pixelate(self):
        """Test pixelate"""
        print("  [10/10] Pixelate", end=" ... ")
        
        image = load_image(self.test_images[0])
        output = pixelate(image, pixelation_percentage=70.0, backend=self.backend)
        
        ref_path = os.path.join(TestConfig.REFERENCE_DIR,
            f"pixelate/pixelate_u8_Tensor_{self.backend_name}_three_images_mixed_src1_pixelationPercentage70.jpg")
        
        passed, max_diff, mean_diff = self._compare_with_reference(
            output, ref_path, TestConfig.UNIT_TOLERANCE_HIGH)
        
        if passed is None:
            print(f"SKIP (no reference)")
        elif passed:
            print(f"PASS (max_diff={max_diff:.1f})")
        else:
            print(f"FAIL (max_diff={max_diff:.1f})")
        
        self.results.append(('pixelate', passed))
        return output
    
    def run_all(self):
        """Run all unit tests"""
        print(f"\n{'='*70}")
        print(f"UNIT TESTS - Golden Reference Comparison ({self.backend_name})")
        print(f"{'='*70}\n")
        
        tests = [
            self.test_brightness,
            self.test_gamma_correction,
            self.test_contrast,
            self.test_hue,
            self.test_flip,
            self.test_resize,
            self.test_rotate,
            self.test_crop,
            self.test_vignette,
            self.test_pixelate
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"ERROR: {e}")
                self.results.append((test.__name__.replace('test_', ''), False))
        
        # Summary
        passed = sum(1 for _, r in self.results if r is True)
        failed = sum(1 for _, r in self.results if r is False)
        skipped = sum(1 for _, r in self.results if r is None)
        
        print(f"\n{'='*70}")
        print(f"Unit Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
        print(f"{'='*70}\n")
        
        return self.results


# =============================================================================
# QA TESTS - Quality Assurance
# =============================================================================

class QATests:
    """Quality assurance tests for correctness validation"""
    
    def __init__(self, backend):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.results = []
    
    def _validate_range(self, tensor):
        """Validate tensor values are in valid range"""
        return (tensor >= 0).all() and (tensor <= 255).all()
    
    def test_parameter_validation(self):
        """Test extreme and edge case parameters"""
        print("\n  [QA-1] Parameter Validation")
        
        test_img = create_test_batch(2, 100, 100, 3, 'cpu')
        
        # Brightness extreme values
        try:
            out = brightness(test_img, alpha=5.0, beta=100, backend=self.backend)
            valid = self._validate_range(out)
            print(f"    Brightness extreme: {'PASS' if valid else 'FAIL'}")
            self.results.append(('brightness_extreme', valid))
        except Exception as e:
            print(f"    Brightness extreme: FAIL ({e})")
            self.results.append(('brightness_extreme', False))
        
        # Gamma edge cases
        for gamma_val in [0.1, 5.0]:
            try:
                out = gamma_correction(test_img, gamma=gamma_val, backend=self.backend)
                valid = self._validate_range(out)
                print(f"    Gamma {gamma_val}: {'PASS' if valid else 'FAIL'}")
                self.results.append((f'gamma_{gamma_val}', valid))
            except Exception as e:
                print(f"    Gamma {gamma_val}: FAIL ({e})")
                self.results.append((f'gamma_{gamma_val}', False))
    
    def test_geometric_consistency(self):
        """Test geometric transformation consistency"""
        print("\n  [QA-2] Geometric Consistency")
        
        # Create checkerboard pattern
        test_img = torch.zeros(1, 3, 256, 256)
        for i in range(0, 256, 64):
            for j in range(0, 256, 64):
                if (i//64 + j//64) % 2 == 0:
                    test_img[:, :, i:i+64, j:j+64] = 255
        
        # Double flip should return to original
        flipped = flip(test_img, horizontal=True, backend=self.backend)
        double_flip = flip(flipped, horizontal=True, backend=self.backend)
        consistent = torch.allclose(test_img, double_flip, atol=1)
        print(f"    Double flip: {'PASS' if consistent else 'FAIL'}")
        self.results.append(('double_flip', consistent))
        
        # 360° rotation should return to original
        rotated = rotate(test_img, angle=360.0, backend=self.backend)
        rotate_consistent = torch.allclose(test_img, rotated, atol=5)
        print(f"    360° rotation: {'PASS' if rotate_consistent else 'FAIL'}")
        self.results.append(('rotate_360', rotate_consistent))
    
    def test_batch_consistency(self):
        """Test batch vs individual processing consistency"""
        print("\n  [QA-3] Batch Processing Consistency")
        
        # Create batch of 3 images
        images = [create_test_batch(1, 100, 100, 3, 'cpu') for _ in range(3)]
        batch = torch.cat(images, dim=0)
        
        # Test brightness
        batch_out = brightness(batch, alpha=1.5, beta=10, backend=self.backend)
        individual_outs = [brightness(img, alpha=1.5, beta=10, backend=self.backend) for img in images]
        
        consistent = True
        for i in range(3):
            if not torch.allclose(batch_out[i:i+1], individual_outs[i], atol=1):
                consistent = False
                break
        
        print(f"    Batch consistency: {'PASS' if consistent else 'FAIL'}")
        self.results.append(('batch_consistency', consistent))
    
    def test_resize_quality(self):
        """Test resize with various sizes"""
        print("\n  [QA-4] Resize Quality")
        
        img = create_test_batch(1, 480, 640, 3, 'cpu')
        
        test_sizes = [(224, 224), (256, 256), (512, 512)]
        all_passed = True
        
        for h, w in test_sizes:
            out = resize(img, width=w, height=h, backend=self.backend)
            dims_ok = out.shape[2] == h and out.shape[3] == w
            if not dims_ok:
                all_passed = False
            print(f"    Resize {w}x{h}: {'PASS' if dims_ok else 'FAIL'}")
        
        self.results.append(('resize_quality', all_passed))
    
    def test_crop_accuracy(self):
        """Test crop boundary conditions"""
        print("\n  [QA-5] Crop Accuracy")
        
        # Create image with known pattern
        img = torch.zeros(1, 3, 400, 600)
        for i in range(400):
            img[:, 0, i, :] = int(i * 255 / 400)
        
        # Test corner crops
        crops = [
            (0, 0, 100, 100, "top-left"),
            (500, 300, 100, 100, "bottom-right")
        ]
        
        all_passed = True
        for x, y, w, h, desc in crops:
            out = crop(img, x1=x, y1=y, crop_width=w, crop_height=h, backend=self.backend)
            dims_ok = out.shape[2] == h and out.shape[3] == w
            if not dims_ok:
                all_passed = False
            print(f"    Crop {desc}: {'PASS' if dims_ok else 'FAIL'}")
        
        self.results.append(('crop_accuracy', all_passed))
    
    def test_color_transform_validity(self):
        """Test color transforms on different channel counts"""
        print("\n  [QA-6] Color Transform Validity")
        
        rgb_img = create_test_batch(1, 100, 100, 3, 'cpu')
        gray_img = create_test_batch(1, 100, 100, 1, 'cpu')
        
        # Hue should only work on RGB
        try:
            hue(rgb_img, hue_shift=45, backend=self.backend)
            rgb_pass = True
        except:
            rgb_pass = False
        
        try:
            hue(gray_img, hue_shift=45, backend=self.backend)
            gray_fail = False  # Should have failed
        except ValueError:
            gray_fail = True  # Correctly rejected
        
        hue_valid = rgb_pass and gray_fail
        print(f"    Hue channel validation: {'PASS' if hue_valid else 'FAIL'}")
        self.results.append(('hue_channel_validation', hue_valid))
    
    def run_all(self):
        """Run all QA tests"""
        print(f"\n{'='*70}")
        print(f"QA TESTS - Quality Assurance ({self.backend_name})")
        print(f"{'='*70}")
        
        tests = [
            self.test_parameter_validation,
            self.test_geometric_consistency,
            self.test_batch_consistency,
            self.test_resize_quality,
            self.test_crop_accuracy,
            self.test_color_transform_validity
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n  ERROR in {test.__name__}: {e}")
        
        # Summary
        passed = sum(1 for _, r in self.results if r is True)
        failed = sum(1 for _, r in self.results if r is False)
        
        print(f"\n{'='*70}")
        print(f"QA Test Summary: {passed} passed, {failed} failed")
        print(f"{'='*70}\n")
        
        return self.results


# =============================================================================
# PERFORMANCE TESTS - Benchmarking
# =============================================================================

class PerformanceTests:
    """Performance benchmarking tests"""
    
    def __init__(self, backend):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.results = {}
    
    def _measure(self, func, *args, **kwargs):
        """Measure execution time"""
        # Warmup
        for _ in range(TestConfig.PERF_WARMUP_ITERS):
            _ = func(*args, **kwargs)
        
        if self.backend == HIP:
            torch.cuda.synchronize()
        
        # Measure
        start = time.perf_counter()
        for _ in range(TestConfig.PERF_TEST_ITERS):
            _ = func(*args, **kwargs)
        
        if self.backend == HIP:
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        return ((end - start) / TestConfig.PERF_TEST_ITERS) * 1000  # ms
    
    def benchmark_all_augmentations(self):
        """Benchmark all 10 augmentations"""
        print(f"\n{'='*70}")
        print(f"PERFORMANCE TESTS - Benchmark All Augmentations ({self.backend_name})")
        print(f"{'='*70}\n")
        
        print(f"{'Augmentation':<20} {'Batch=8':<12} {'224x224':<12} {'Time(ms)':<12} {'Img/sec':<12}")
        print("-" * 70)
        
        batch_size = 8
        device = 'cuda' if self.backend == HIP else 'cpu'
        img = create_test_batch(batch_size, 224, 224, 3, device)
        
        augmentations = [
            ('brightness', lambda: brightness(img, alpha=1.5, beta=10.0, backend=self.backend)),
            ('gamma_correction', lambda: gamma_correction(img, gamma=0.8, backend=self.backend)),
            ('contrast', lambda: contrast(img, contrast_factor=2.0, backend=self.backend)),
            ('hue', lambda: hue(img, hue_shift=45.0, backend=self.backend)),
            ('flip', lambda: flip(img, horizontal=True, backend=self.backend)),
            ('resize', lambda: resize(img, width=256, height=256, backend=self.backend)),
            ('rotate', lambda: rotate(img, angle=30.0, backend=self.backend)),
            ('crop', lambda: crop(img, x1=50, y1=50, crop_width=150, crop_height=150, backend=self.backend)),
            ('vignette', lambda: vignette(img, intensity=0.7, backend=self.backend)),
            ('pixelate', lambda: pixelate(img, pixelation_percentage=70.0, backend=self.backend))
        ]
        
        results = []
        for name, func in augmentations:
            try:
                avg_time = self._measure(func)
                throughput = (batch_size * 1000.0) / avg_time
                
                print(f"{name:<20} {batch_size:<12} {'224x224':<12} {avg_time:<12.2f} {throughput:<12.1f}")
                
                results.append({
                    'augmentation': name,
                    'time_ms': avg_time,
                    'throughput': throughput
                })
            except Exception as e:
                print(f"{name:<20} ERROR: {e}")
        
        self.results['comparison'] = results
        
        # Print ranking
        print(f"\n{'='*70}")
        print("Performance Ranking (by throughput):")
        print(f"{'='*70}")
        
        sorted_results = sorted(results, key=lambda x: x['throughput'], reverse=True)
        for i, r in enumerate(sorted_results, 1):
            print(f"{i:2d}. {r['augmentation']:20s}: {r['throughput']:8.1f} img/sec")
        
        return results
    
    def benchmark_batch_scaling(self):
        """Benchmark batch size scaling"""
        print(f"\n{'='*70}")
        print(f"Batch Size Scaling Test - Brightness ({self.backend_name})")
        print(f"{'='*70}\n")
        
        print(f"{'Batch Size':<15} {'Time (ms)':<15} {'Throughput (img/sec)':<25}")
        print("-" * 55)
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        
        for batch_size in TestConfig.PERF_BATCH_SIZES:
            img = create_test_batch(batch_size, 224, 224, 3, device)
            
            avg_time = self._measure(
                brightness, img, alpha=1.5, beta=10.0, backend=self.backend
            )
            throughput = (batch_size * 1000.0) / avg_time
            
            print(f"{batch_size:<15} {avg_time:<15.2f} {throughput:<25.1f}")
        
        print()
    
    def run_all(self):
        """Run all performance tests"""
        self.benchmark_all_augmentations()
        self.benchmark_batch_scaling()
        
        return self.results


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RPP Comprehensive Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_suite.py --type all --backend HOST
  python test_suite.py --type unit --backend HIP
  python test_suite.py --type qa
  python test_suite.py --type performance --backend HIP
        """
    )
    
    parser.add_argument('--type', choices=['unit', 'qa', 'performance', 'all'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--backend', choices=['HOST', 'HIP'],
                       default=None, help='Backend to test (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for performance tests')
    
    args = parser.parse_args()
    
    # Determine backend(s) to test
    if args.backend:
        backends = [HIP if args.backend == 'HIP' else HOST]
    else:
        # Auto-detect: use both if GPU available, else just HOST
        backends = [HOST, HIP] if is_gpu_available() else [HOST]
    
    # Update config
    if args.batch_size:
        TestConfig.PERF_BATCH_SIZES = [1, args.batch_size, args.batch_size * 2]
    
    # Print header
    print("\n" + "="*70)
    print("RPP COMPREHENSIVE TEST SUITE")
    print("="*70)
    print(f"Test Type: {args.type}")
    print(f"Backends: {['HIP' if b == HIP else 'HOST' for b in backends]}")
    print(f"GPU Available: {is_gpu_available()}")
    print("="*70)
    
    # Run tests for each backend
    all_results = {}
    
    for backend in backends:
        backend_name = "HIP" if backend == HIP else "HOST"
        
        # Skip HIP if GPU not available
        if backend == HIP and not is_gpu_available():
            print(f"\nSkipping HIP backend (GPU not available)")
            continue
        
        if args.type in ['unit', 'all']:
            unit_tests = UnitTests(backend)
            all_results[f'unit_{backend_name}'] = unit_tests.run_all()
        
        if args.type in ['qa', 'all']:
            qa_tests = QATests(backend)
            all_results[f'qa_{backend_name}'] = qa_tests.run_all()
        
        if args.type in ['performance', 'all']:
            perf_tests = PerformanceTests(backend)
            all_results[f'perf_{backend_name}'] = perf_tests.run_all()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for key, results in all_results.items():
        if isinstance(results, list):
            if 'unit' in key or 'qa' in key:
                passed = sum(1 for _, r in results if r is True)
                failed = sum(1 for _, r in results if r is False)
                skipped = sum(1 for _, r in results if r is None)
                print(f"\n{key.upper()}:")
                print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())