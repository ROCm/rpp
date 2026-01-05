#!/usr/bin/env python3
"""
PyRPP Unit Tests with Golden Reference Comparison
Tests against existing C++ test suite golden outputs
"""

import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import rpp_pybind
from rpp_pybind import fn, utils, types

# Test configuration matching C++ test suite
TEST_IMAGES_DIR = "../utilities/test_suite/TEST_IMAGES/three_images_mixed_src1"
REFERENCE_DIR = "../utilities/test_suite/REFERENCE_OUTPUT"

def load_reference_image(ref_path):
    """Load reference image as numpy array"""
    ref_img = Image.open(ref_path)
    return np.array(ref_img)

def compare_images(output, reference, tolerance=5):
    """
    Compare output with reference image
    tolerance: allowed pixel difference (matching C++ suite)
    """
    # Convert tensor to numpy
    if isinstance(output, torch.Tensor):
        output_np = utils.tensor_to_numpy(output)
    else:
        output_np = output
    
    # Calculate difference
    diff = np.abs(output_np.astype(np.float32) - reference.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Check if within tolerance
    passed = max_diff <= tolerance
    
    return passed, max_diff, mean_diff

class UnitTests:
    def __init__(self, backend=None):
        self.backend = backend or types.get_default_backend()
        self.test_images = [
            os.path.join(TEST_IMAGES_DIR, "img1.jpg"),
            os.path.join(TEST_IMAGES_DIR, "img2.jpg"),
            os.path.join(TEST_IMAGES_DIR, "img3.jpg")
        ]
        self.results = []
    
    def test_brightness(self):
        """Test brightness against golden reference"""
        print("\n[TEST] Brightness")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply brightness (matching C++ params: alpha=1.5, beta=10)
        output = fn.brightness(image, alpha=1.5, beta=10.0, backend=self.backend)
        
        # Load reference
        ref_path = os.path.join(REFERENCE_DIR, "brightness/brightness_u8_Tensor_HOST_three_images_mixed_src1_interpolationType0_noiseType0_alpha1.5_beta10.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            # Remove batch dimension and convert
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('brightness', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('brightness', None))
        
        return output
    
    def test_gamma_correction(self):
        """Test gamma correction against golden reference"""
        print("\n[TEST] Gamma Correction")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply gamma (matching C++ params: gamma=0.8)
        output = fn.gamma_correction(image, gamma=0.8, backend=self.backend)
        
        # Load reference
        ref_path = os.path.join(REFERENCE_DIR, "gamma_correction/gamma_correction_u8_Tensor_HOST_three_images_mixed_src1_gamma0.8.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('gamma_correction', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('gamma_correction', None))
        
        return output
    
    def test_contrast(self):
        """Test contrast against golden reference"""
        print("\n[TEST] Contrast")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply contrast (typical C++ params)
        output = fn.contrast(image, contrast_factor=2.0, contrast_center=128.0, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "contrast/contrast_u8_Tensor_HOST_three_images_mixed_src1_contrastFactor2_contrastCenter128.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('contrast', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('contrast', None))
        
        return output
    
    def test_hue(self):
        """Test hue against golden reference"""
        print("\n[TEST] Hue")
        
        # Load RGB test image
        image = utils.load_image(self.test_images[0])
        
        # Apply hue shift (typical C++ param: 45 degrees)
        output = fn.hue(image, hue_shift=45.0, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "hue/hue_u8_Tensor_HOST_three_images_mixed_src1_hue45.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference, tolerance=10)  # Higher tolerance for color transforms
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('hue', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('hue', None))
        
        return output
    
    def test_flip(self):
        """Test flip against golden reference"""
        print("\n[TEST] Flip")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply horizontal flip
        output = fn.flip(image, horizontal=True, vertical=False, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "flip/flip_u8_Tensor_HOST_three_images_mixed_src1_horizontal1_vertical0.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('flip', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('flip', None))
        
        return output
    
    def test_resize(self):
        """Test resize against golden reference"""
        print("\n[TEST] Resize")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply resize (to 224x224)
        output = fn.resize(image, width=224, height=224, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "resize/resize_u8_Tensor_HOST_three_images_mixed_src1_224x224_interpolationType1.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference, tolerance=10)  # Higher tolerance for interpolation
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('resize', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('resize', None))
        
        return output
    
    def test_rotate(self):
        """Test rotate against golden reference"""
        print("\n[TEST] Rotate")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply rotation (30 degrees)
        output = fn.rotate(image, angle=30.0, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "rotate/rotate_u8_Tensor_HOST_three_images_mixed_src1_angle30_interpolationType1.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference, tolerance=15)  # Higher tolerance for rotation
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('rotate', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('rotate', None))
        
        return output
    
    def test_crop(self):
        """Test crop against golden reference"""
        print("\n[TEST] Crop")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply crop (100,100,200,200)
        output = fn.crop(image, x1=100, y1=100, crop_width=200, crop_height=200, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "crop/crop_u8_Tensor_HOST_three_images_mixed_src1_100_100_200_200.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('crop', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('crop', None))
        
        return output
    
    def test_vignette(self):
        """Test vignette against golden reference"""
        print("\n[TEST] Vignette")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply vignette
        output = fn.vignette(image, intensity=0.7, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "vignette/vignette_u8_Tensor_HOST_three_images_mixed_src1_intensity0.7.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference, tolerance=10)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('vignette', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('vignette', None))
        
        return output
    
    def test_pixelate(self):
        """Test pixelate against golden reference"""
        print("\n[TEST] Pixelate")
        
        # Load test image
        image = utils.load_image(self.test_images[0])
        
        # Apply pixelate
        output = fn.pixelate(image, pixelation_percentage=70.0, backend=self.backend)
        
        # Look for reference
        ref_path = os.path.join(REFERENCE_DIR, "pixelate/pixelate_u8_Tensor_HOST_three_images_mixed_src1_pixelationPercentage70.jpg")
        if os.path.exists(ref_path):
            reference = load_reference_image(ref_path)
            output_np = utils.tensor_to_numpy(output[0])
            
            passed, max_diff, mean_diff = compare_images(output_np, reference, tolerance=10)
            print(f"  Max diff: {max_diff:.2f}, Mean diff: {mean_diff:.2f}")
            print(f"  Result: {'PASS' if passed else 'FAIL'}")
            self.results.append(('pixelate', passed))
        else:
            print("  Reference image not found - generating output only")
            self.results.append(('pixelate', None))
        
        return output
    
    def run_all_tests(self):
        """Run all unit tests"""
        print("\n" + "="*60)
        print("PyRPP Unit Tests - Golden Reference Comparison")
        print("="*60)
        
        test_methods = [
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
        
        for test in test_methods:
            try:
                test()
            except Exception as e:
                print(f"  Error: {e}")
                self.results.append((test.__name__.replace('test_', ''), False))
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary:")
        print("="*60)
        
        passed = sum(1 for _, result in self.results if result is True)
        failed = sum(1 for _, result in self.results if result is False)
        skipped = sum(1 for _, result in self.results if result is None)
        
        for name, result in self.results:
            status = "PASS" if result is True else "FAIL" if result is False else "SKIP"
            print(f"  {name:20s}: {status}")
        
        print(f"\nTotal: {len(self.results)}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        return failed == 0

def main():
    parser = argparse.ArgumentParser(description='PyRPP Unit Tests')
    parser.add_argument('--backend', choices=['HOST', 'HIP'], default='HOST',
                        help='Backend to use (default: HOST)')
    parser.add_argument('--test', help='Run specific test')
    
    args = parser.parse_args()
    
    backend = types.HIP if args.backend == 'HIP' else types.HOST
    tests = UnitTests(backend)
    
    if args.test:
        # Run specific test
        test_method = getattr(tests, f'test_{args.test}', None)
        if test_method:
            test_method()
        else:
            print(f"Test '{args.test}' not found")
            return 1
    else:
        # Run all tests
        success = tests.run_all_tests()
        return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
