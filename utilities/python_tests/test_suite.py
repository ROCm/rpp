# Mukesh/rpp/test_suite.py

"""
RPP Comprehensive Test Suite
============================

Single unified test suite combining:
1. Unit Testing - Golden reference comparison with C++ test outputs
2. QA Testing - Quality assurance, edge cases, and correctness validation
3. Performance Testing - Throughput and latency benchmarks

Tests all 10 augmentations on both HOST and HIP backends.

Usage:
    python test_suite.py --type all --backend HOST
    python test_suite.py --type unit --backend HIP
    python test_suite.py --type qa[]
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

from rpp_pybind.fn import (
    brightness, gamma_correction
    # , contrast, hue,
    # flip, resize, rotate, crop, vignette, pixelate
)
from rpp_pybind.amd.rpp.rpp_types import (
    is_gpu_available, get_default_backend, HOST, HIP
)
from rpp_pybind.amd.rpp.utils import create_test_batch, load_image, tensor_to_numpy

print("✓ All RPP modules loaded successfully\n")

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Global test configuration"""
    
    # Directories
    TEST_IMAGES_DIR = "../test_suite/TEST_IMAGES/three_images_mixed_src1"
    REFERENCE_DIR = "../test_suite/REFERENCE_OUTPUT"
    
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
        "1_img50x50.jpg",
        "2_img100x100.jpg", 
        "3_img150x150.jpg"
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
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        image = load_image(self.test_images[0], device=device)
        # image = load_image(self.test_images[0])
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
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        image = load_image(self.test_images[0], device=device)
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
    
    def run_all(self):
        """Run all unit tests"""
        print(f"\n{'='*70}")
        print(f"UNIT TESTS - Golden Reference Comparison ({self.backend_name})")
        print(f"{'='*70}\n")
        
        tests = [
            self.test_brightness,
            self.test_gamma_correction
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
        
        # Test image directories (updated paths)
        self.test_images_dir = "../test_suite/TEST_IMAGES"
        self.reference_dir = "../test_suite/REFERENCE_OUTPUT"
        
        # QA thresholds
        self.psnr_threshold = 30.0  # dB - minimum acceptable PSNR
        self.ssim_threshold = 0.9   # structural similarity threshold
        
        # Check function availability
        self.available_functions, self.missing_functions = self.check_function_availability()
        if self.missing_functions:
            print(f"Warning: Missing functions: {', '.join(self.missing_functions)}")
    
    def check_function_availability(self):
        """Check which RPP functions are available"""
        available_functions = []
        missing_functions = []
        
        # Import the module to check available functions
        from rpp_pybind import fn
        
        functions_to_check = ['brightness', 'gamma_correction', 'flip', 'resize', 'crop', 'hue', 
                             'rotate', 'contrast', 'vignette', 'pixelate']
        
        for func_name in functions_to_check:
            try:
                func = getattr(fn, func_name)
                available_functions.append(func_name)
            except AttributeError:
                missing_functions.append(func_name)
        
        return available_functions, missing_functions
    
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        # Convert to numpy if needed
        if isinstance(img1, torch.Tensor):
            img1 = tensor_to_numpy(img1)
        if isinstance(img2, torch.Tensor):
            img2 = tensor_to_numpy(img2)
        
        # Calculate MSE
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_checksum(self, tensor):
        """Calculate checksum for deterministic tests"""
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
        else:
            arr = tensor
        
        # Create hash
        return hashlib.md5(arr.tobytes()).hexdigest()
    
    def validate_output_range(self, output, dtype=None):
        """Validate output is within expected range"""
        if dtype is None:
            dtype = output.dtype
            
        if dtype == torch.uint8:
            return (output >= 0).all() and (output <= 255).all()
        elif dtype in [torch.float16, torch.float32]:
            # For normalized float, expect 0-1 range
            return (output >= 0).all() and (output <= 1.0).all()
        elif dtype == torch.int8:
            return (output >= -128).all() and (output <= 127).all()
        return False
    
    def _validate_range(self, tensor):
        """Validate tensor values are in valid range (legacy method)"""
        return self.validate_output_range(tensor)
    
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
        
        # Test identity operations (no change expected)
        try:
            identity_out = brightness(test_img, alpha=1.0, beta=0.0, backend=self.backend)
            checksum_match = self.calculate_checksum(test_img) == self.calculate_checksum(identity_out)
            print(f"    Brightness identity: {'PASS' if checksum_match else 'FAIL'}")
            self.results.append(('brightness_identity', checksum_match))
        except Exception as e:
            print(f"    Brightness identity: FAIL ({e})")
            self.results.append(('brightness_identity', False))
    
    def test_brightness_quality(self):
        """Test brightness quality across different parameters"""
        print("\n  [QA-1.5] Brightness Quality Analysis")
        
        # Create test image with known pattern
        test_img = create_test_batch(1, 100, 100, 3, 'cpu')
        
        # Test multiple parameter combinations
        test_params = [
            (1.0, 0.0, "Identity"),
            (1.5, 10.0, "Standard bright"),
            (0.5, 0.0, "Darken"),
            (2.0, 50.0, "Very bright"),
            (0.1, 0.0, "Very dark")
        ]
        
        for alpha, beta, desc in test_params:
            try:
                output = brightness(test_img, alpha=alpha, beta=beta, backend=self.backend)
                
                # Validate output range
                range_valid = self.validate_output_range(output)
                
                # Check shape preservation
                shape_preserved = output.shape == test_img.shape
                
                # For identity case, check PSNR
                if alpha == 1.0 and beta == 0.0:
                    psnr = self.calculate_psnr(test_img[0], output[0])
                    identity_preserved = psnr > 50  # Should be very high for identity
                else:
                    identity_preserved = None
                    psnr = None
                
                status = "PASS" if range_valid and shape_preserved else "FAIL"
                print(f"    {desc} (α={alpha}, β={beta}): {status}")
                if psnr is not None:
                    print(f"      Identity PSNR: {psnr:.1f}dB")
                
                self.results.append({
                    'test': 'brightness_quality',
                    'params': desc,
                    'range_valid': range_valid,
                    'shape_preserved': shape_preserved,
                    'identity_preserved': identity_preserved
                })
                
            except Exception as e:
                print(f"    {desc}: FAIL ({e})")
                self.results.append({
                    'test': 'brightness_quality',
                    'params': desc,
                    'error': str(e)
                })
    
    def test_geometric_consistency(self):
        """Test geometric transformation consistency"""
        print("\n  [QA-2] Geometric Consistency")
        
        if not any(func in self.available_functions for func in ['flip', 'rotate']):
            print("    SKIP: No geometric functions available (flip, rotate)")
            return
        
        # Create checkerboard pattern
        test_img = torch.zeros(1, 3, 256, 256, dtype=torch.uint8)
        for i in range(0, 256, 64):
            for j in range(0, 256, 64):
                if (i//64 + j//64) % 2 == 0:
                    test_img[:, :, i:i+64, j:j+64] = 255
        
        # Test flip consistency if available
        if 'flip' in self.available_functions:
            try:
                from rpp_pybind.fn import flip
                flipped = flip(test_img, horizontal=True, backend=self.backend)
                double_flip = flip(flipped, horizontal=True, backend=self.backend)
                consistent = torch.allclose(test_img, double_flip, atol=1)
                print(f"    Double flip: {'PASS' if consistent else 'FAIL'}")
                self.results.append(('double_flip', consistent))
            except Exception as e:
                print(f"    Double flip: FAIL ({e})")
                self.results.append(('double_flip', False))
        else:
            print("    Double flip: SKIP (function not available)")
        
        # Test rotation consistency if available
        if 'rotate' in self.available_functions:
            try:
                from rpp_pybind.fn import rotate
                rotated = rotate(test_img, angle=360.0, backend=self.backend)
                rotate_consistent = torch.allclose(test_img, rotated, atol=5)
                print(f"    360° rotation: {'PASS' if rotate_consistent else 'FAIL'}")
                self.results.append(('rotate_360', rotate_consistent))
            except Exception as e:
                print(f"    360° rotation: FAIL ({e})")
                self.results.append(('rotate_360', False))
        else:
            print("    360° rotation: SKIP (function not available)")
    
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
        
        if 'resize' not in self.available_functions:
            print("    SKIP: Resize function not available")
            return
        
        try:
            from rpp_pybind.fn import resize
            img = create_test_batch(1, 480, 640, 3, 'cpu')
            
            test_sizes = [(224, 224), (256, 256), (512, 512)]
            all_passed = True
            
            for h, w in test_sizes:
                try:
                    out = resize(img, width=w, height=h, backend=self.backend)
                    dims_ok = out.shape[2] == h and out.shape[3] == w
                    range_ok = self.validate_output_range(out)
                    
                    if not dims_ok or not range_ok:
                        all_passed = False
                    
                    status = "PASS" if (dims_ok and range_ok) else "FAIL"
                    print(f"    Resize {w}x{h}: {status}")
                    
                    self.results.append({
                        'test': 'resize_quality', 
                        'size': f'{w}x{h}',
                        'dims_ok': dims_ok,
                        'range_ok': range_ok
                    })
                except Exception as e:
                    print(f"    Resize {w}x{h}: FAIL ({e})")
                    all_passed = False
                    self.results.append({
                        'test': 'resize_quality', 
                        'size': f'{w}x{h}',
                        'error': str(e)
                    })
        except Exception as e:
            print(f"    Resize tests: FAIL ({e})")
    
    def test_crop_accuracy(self):
        """Test crop boundary conditions"""
        print("\n  [QA-5] Crop Accuracy")
        
        if 'crop' not in self.available_functions:
            print("    SKIP: Crop function not available")
            return
        
        try:
            from rpp_pybind.fn import crop
            
            # Create image with known pattern
            img = torch.zeros(1, 3, 400, 600, dtype=torch.uint8)
            for i in range(400):
                img[:, 0, i, :] = int(i * 255 / 400)
            
            # Test corner crops
            crops = [
                (0, 0, 100, 100, "top-left"),
                (500, 300, 100, 100, "bottom-right"),
                (200, 150, 200, 150, "center")
            ]
            
            all_passed = True
            for x, y, w, h, desc in crops:
                # Skip invalid crops
                if x + w > 600 or y + h > 400:
                    continue
                    
                try:
                    out = crop(img, x1=x, y1=y, crop_width=w, crop_height=h, backend=self.backend)
                    dims_ok = out.shape[2] == h and out.shape[3] == w
                    range_ok = self.validate_output_range(out)
                    
                    # Verify content (check gradient pattern)
                    if dims_ok and y < 400:
                        expected_red = int(y * 255 / 400)
                        actual_red = out[0, 0, 0, 0].item()
                        content_ok = abs(actual_red - expected_red) <= 2
                    else:
                        content_ok = True
                    
                    if not dims_ok or not range_ok or not content_ok:
                        all_passed = False
                    
                    status = "PASS" if (dims_ok and range_ok and content_ok) else "FAIL"
                    print(f"    Crop {desc}: {status}")
                    
                    self.results.append({
                        'test': 'crop_accuracy',
                        'scenario': desc,
                        'dims_ok': dims_ok,
                        'range_ok': range_ok,
                        'content_ok': content_ok
                    })
                except Exception as e:
                    print(f"    Crop {desc}: FAIL ({e})")
                    all_passed = False
                    self.results.append({
                        'test': 'crop_accuracy',
                        'scenario': desc,
                        'error': str(e)
                    })
        except Exception as e:
            print(f"    Crop tests: FAIL ({e})")
    
    def test_color_transform_validity(self):
        """Test color transforms on different channel counts"""
        print("\n  [QA-6] Color Transform Validity")
        
        if not any(func in self.available_functions for func in ['hue', 'contrast']):
            print("    SKIP: No color transform functions available (hue, contrast)")
            return
        
        rgb_img = create_test_batch(1, 100, 100, 3, 'cpu')
        gray_img = create_test_batch(1, 100, 100, 1, 'cpu')
        
        # Test Hue function if available
        if 'hue' in self.available_functions:
            try:
                from rpp_pybind.fn import hue
                
                # RGB should work
                try:
                    hue_out = hue(rgb_img, hue_shift=45, backend=self.backend)
                    rgb_pass = self.validate_output_range(hue_out)
                    print(f"    Hue RGB: {'PASS' if rgb_pass else 'FAIL'}")
                except Exception as e:
                    rgb_pass = False
                    print(f"    Hue RGB: FAIL ({e})")
                
                # Grayscale should fail or be rejected
                try:
                    hue_gray_out = hue(gray_img, hue_shift=45, backend=self.backend)
                    gray_properly_handled = False  # Should not succeed for grayscale
                    print("    Hue Grayscale: FAIL (should reject grayscale)")
                except (ValueError, RuntimeError):
                    gray_properly_handled = True  # Correctly rejected
                    print("    Hue Grayscale: PASS (correctly rejected)")
                except Exception as e:
                    gray_properly_handled = False
                    print(f"    Hue Grayscale: FAIL ({e})")
                
                self.results.append({
                    'test': 'color_transform_validity',
                    'function': 'hue',
                    'rgb_pass': rgb_pass,
                    'gray_properly_handled': gray_properly_handled
                })
                
            except Exception as e:
                print(f"    Hue tests: FAIL ({e})")
        else:
            print("    Hue: SKIP (function not available)")
        
        # Test Contrast function if available
        if 'contrast' in self.available_functions:
            try:
                from rpp_pybind.fn import contrast
                
                for channels, desc, img in [(3, "RGB", rgb_img), (1, "Grayscale", gray_img)]:
                    try:
                        contrast_out = contrast(img, contrast_factor=1.5, backend=self.backend)
                        valid = self.validate_output_range(contrast_out)
                        shape_ok = contrast_out.shape == img.shape
                        
                        status = "PASS" if (valid and shape_ok) else "FAIL"
                        print(f"    Contrast {desc}: {status}")
                        
                        self.results.append({
                            'test': 'color_transform_validity',
                            'function': 'contrast',
                            'channels': channels,
                            'valid': valid,
                            'shape_ok': shape_ok
                        })
                    except Exception as e:
                        print(f"    Contrast {desc}: FAIL ({e})")
                        self.results.append({
                            'test': 'color_transform_validity',
                            'function': 'contrast',
                            'channels': channels,
                            'error': str(e)
                        })
                        
            except Exception as e:
                print(f"    Contrast tests: FAIL ({e})")
        else:
            print("    Contrast: SKIP (function not available)")
    
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
        
        # if args.type in ['performance', 'all']:
        #     perf_tests = PerformanceTests(backend)
        #     all_results[f'perf_{backend_name}'] = perf_tests.run_all()
    
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
