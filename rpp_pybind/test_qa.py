#!/usr/bin/env python3
"""
PyRPP QA Tests
Quality assurance tests for validating correctness and output quality
"""

import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image
import hashlib

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import rpp_pybind
from rpp_pybind import fn, utils, types

class QATests:
    def __init__(self, backend=None):
        self.backend = backend or types.get_default_backend()
        self.results = []
        
        # Test image directories
        self.test_images_dir = "../utilities/test_suite/TEST_IMAGES"
        self.reference_dir = "../utilities/test_suite/REFERENCE_OUTPUT"
        
        # QA thresholds
        self.psnr_threshold = 30.0  # dB - minimum acceptable PSNR
        self.ssim_threshold = 0.9   # structural similarity threshold
        
    def calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        # Convert to numpy if needed
        if isinstance(img1, torch.Tensor):
            img1 = utils.tensor_to_numpy(img1)
        if isinstance(img2, torch.Tensor):
            img2 = utils.tensor_to_numpy(img2)
        
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
    
    def validate_output_range(self, output, dtype):
        """Validate output is within expected range"""
        if dtype == torch.uint8:
            return (output >= 0).all() and (output <= 255).all()
        elif dtype in [torch.float16, torch.float32]:
            # For normalized float, expect 0-1 range
            return (output >= 0).all() and (output <= 1.0).all()
        elif dtype == torch.int8:
            return (output >= -128).all() and (output <= 127).all()
        return False
    
    def test_brightness_quality(self):
        """Test brightness quality and correctness"""
        print("\n[QA] Brightness Quality Test")
        
        # Load test images
        test_sets = [
            ("three_images_mixed_src1", ["img1.jpg", "img2.jpg", "img3.jpg"]),
            ("three_images_150x150_src1", ["cows_150x150.jpg", "castle_150x150.jpg", "cat_150x150.jpg"])
        ]
        
        for set_name, images in test_sets:
            print(f"\n  Testing set: {set_name}")
            
            for img_name in images:
                img_path = os.path.join(self.test_images_dir, set_name, img_name)
                if not os.path.exists(img_path):
                    continue
                
                # Load image
                image = utils.load_image(img_path)
                
                # Test different alpha/beta values
                test_params = [
                    (1.0, 0.0),    # No change
                    (1.5, 10.0),   # Standard brightening
                    (0.5, 0.0),    # Darkening
                    (2.0, 50.0),   # Strong brightening
                ]
                
                for alpha, beta in test_params:
                    output = fn.brightness(image, alpha=alpha, beta=beta, backend=self.backend)
                    
                    # Validate output
                    valid_range = self.validate_output_range(output, output.dtype)
                    
                    # Check shape preservation
                    shape_preserved = output.shape == image.shape
                    
                    # Special case: alpha=1, beta=0 should be identity
                    if alpha == 1.0 and beta == 0.0:
                        checksum_match = self.calculate_checksum(output) == self.calculate_checksum(image)
                    else:
                        checksum_match = None
                    
                    print(f"    {img_name} (α={alpha}, β={beta}): Range={'OK' if valid_range else 'FAIL'}, Shape={'OK' if shape_preserved else 'FAIL'}")
                    
                    self.results.append({
                        'test': 'brightness_quality',
                        'image': img_name,
                        'params': f'alpha={alpha}, beta={beta}',
                        'range_valid': valid_range,
                        'shape_preserved': shape_preserved,
                        'identity_check': checksum_match
                    })
    
    def test_geometric_consistency(self):
        """Test geometric transformations for consistency"""
        print("\n[QA] Geometric Consistency Tests")
        
        # Create test pattern
        height, width = 256, 256
        test_pattern = torch.zeros(1, 3, height, width, dtype=torch.uint8)
        
        # Create checkerboard pattern
        checker_size = 32
        for i in range(0, height, checker_size * 2):
            for j in range(0, width, checker_size * 2):
                test_pattern[:, :, i:i+checker_size, j:j+checker_size] = 255
                test_pattern[:, :, i+checker_size:i+2*checker_size, j+checker_size:j+2*checker_size] = 255
        
        # Test flip consistency
        print("\n  Flip Consistency:")
        h_flip = fn.flip(test_pattern, horizontal=True, backend=self.backend)
        h_flip_twice = fn.flip(h_flip, horizontal=True, backend=self.backend)
        flip_consistent = torch.allclose(test_pattern, h_flip_twice, atol=1)
        print(f"    Double horizontal flip: {'PASS' if flip_consistent else 'FAIL'}")
        
        # Test rotate consistency
        print("\n  Rotate Consistency:")
        rotations = [90, 180, 270, 360]
        for angle in rotations:
            rotated = fn.rotate(test_pattern, angle=float(angle), backend=self.backend)
            if angle == 360:
                # 360 degree rotation should return to original
                rotate_consistent = torch.allclose(test_pattern, rotated, atol=5)
                print(f"    {angle}° rotation: {'PASS' if rotate_consistent else 'FAIL'}")
            
        self.results.append({
            'test': 'geometric_consistency',
            'flip_consistent': flip_consistent,
            'rotate_360_consistent': rotate_consistent
        })
    
    def test_resize_quality(self):
        """Test resize quality with different interpolations"""
        print("\n[QA] Resize Quality Tests")
        
        # Load a detailed test image
        img_path = os.path.join(self.test_images_dir, "three_images_mixed_src1/img1.jpg")
        if os.path.exists(img_path):
            image = utils.load_image(img_path)
            original_h, original_w = image.shape[2], image.shape[3]
            
            # Test different resize scenarios
            resize_tests = [
                (original_w//2, original_h//2, "Downscale 2x"),
                (original_w*2, original_h*2, "Upscale 2x"),
                (224, 224, "Fixed size"),
                (original_w, original_h, "Same size")
            ]
            
            for new_w, new_h, desc in resize_tests:
                resized = fn.resize(image, width=new_w, height=new_h, backend=self.backend)
                
                # Check dimensions
                dims_correct = resized.shape[2] == new_h and resized.shape[3] == new_w
                
                # For same-size resize, check if content is preserved
                if new_w == original_w and new_h == original_h:
                    psnr = self.calculate_psnr(image[0], resized[0])
                    content_preserved = psnr > 40  # High PSNR expected
                else:
                    content_preserved = None
                    psnr = None
                
                print(f"  {desc}: Dims={'OK' if dims_correct else 'FAIL'}", end='')
                if psnr is not None:
                    print(f", PSNR={psnr:.1f}dB")
                else:
                    print()
                
                self.results.append({
                    'test': 'resize_quality',
                    'scenario': desc,
                    'dims_correct': dims_correct,
                    'psnr': psnr
                })
    
    def test_crop_accuracy(self):
        """Test crop accuracy and boundary conditions"""
        print("\n[QA] Crop Accuracy Tests")
        
        # Create test image with known pattern
        height, width = 400, 600
        test_img = torch.zeros(1, 3, height, width, dtype=torch.uint8)
        
        # Fill with gradient pattern for easy verification
        for i in range(height):
            test_img[:, 0, i, :] = int(i * 255 / height)  # Red channel: vertical gradient
        for j in range(width):
            test_img[:, 1, :, j] = int(j * 255 / width)   # Green channel: horizontal gradient
        test_img[:, 2, :, :] = 128  # Blue channel: constant
        
        # Test various crop scenarios
        crop_tests = [
            (0, 0, 100, 100, "Top-left corner"),
            (width-100, height-100, 100, 100, "Bottom-right corner"),
            (100, 100, 200, 200, "Center region"),
            (0, 0, width, height, "Full image"),
        ]
        
        for x1, y1, crop_w, crop_h, desc in crop_tests:
            # Skip invalid crops
            if x1 + crop_w > width or y1 + crop_h > height:
                continue
            
            cropped = fn.crop(test_img, x1=x1, y1=y1, crop_width=crop_w, crop_height=crop_h, backend=self.backend)
            
            # Verify dimensions
            dims_correct = cropped.shape[2] == crop_h and cropped.shape[3] == crop_w
            
            # Verify content (check corners)
            content_correct = True
            if dims_correct:
                # Check if gradient values match expected
                expected_red_tl = int(y1 * 255 / height)
                actual_red_tl = cropped[0, 0, 0, 0].item()
                content_correct = abs(actual_red_tl - expected_red_tl) <= 2
            
            print(f"  {desc}: Dims={'OK' if dims_correct else 'FAIL'}, Content={'OK' if content_correct else 'FAIL'}")
            
            self.results.append({
                'test': 'crop_accuracy',
                'scenario': desc,
                'dims_correct': dims_correct,
                'content_correct': content_correct
            })
    
    def test_color_transforms_validity(self):
        """Test color transform validity"""
        print("\n[QA] Color Transform Validity Tests")
        
        # Create test images
        test_images = [
            utils.create_test_batch(1, 224, 224, channels=3),  # RGB
            utils.create_test_batch(1, 224, 224, channels=1),  # Grayscale
        ]
        
        for i, test_img in enumerate(test_images):
            channels = test_img.shape[1]
            print(f"\n  Testing {channels}-channel image:")
            
            # Gamma correction - should work for both
            try:
                gamma_out = fn.gamma_correction(test_img, gamma=0.8, backend=self.backend)
                gamma_valid = self.validate_output_range(gamma_out, gamma_out.dtype)
                print(f"    Gamma correction: {'PASS' if gamma_valid else 'FAIL'}")
            except Exception as e:
                gamma_valid = False
                print(f"    Gamma correction: FAIL ({str(e)})")
            
            # Hue - should only work for RGB
            try:
                hue_out = fn.hue(test_img, hue_shift=45.0, backend=self.backend)
                hue_valid = channels == 3  # Should succeed only for RGB
                print(f"    Hue shift: {'PASS' if hue_valid else 'FAIL'}")
            except ValueError as e:
                hue_valid = channels != 3  # Should fail for non-RGB
                print(f"    Hue shift: {'PASS (correctly rejected)' if hue_valid else 'FAIL'}")
            except Exception as e:
                hue_valid = False
                print(f"    Hue shift: FAIL ({str(e)})")
            
            self.results.append({
                'test': f'color_transforms_{channels}ch',
                'gamma_valid': gamma_valid,
                'hue_valid': hue_valid
            })
    
    def test_effects_quality(self):
        """Test effects augmentations quality"""
        print("\n[QA] Effects Quality Tests")
        
        # Use same test image as other tests
        img_path = os.path.join(self.test_images_dir, "three_images_mixed_src1/img1.jpg")
        if os.path.exists(img_path):
            image = utils.load_image(img_path)
            
            # Test vignette
            print("\n  Vignette Effect:")
            for intensity in [0.0, 0.5, 1.0]:
                vignette_out = fn.vignette(image, intensity=intensity, backend=self.backend)
                valid_range = self.validate_output_range(vignette_out, vignette_out.dtype)
                
                # Intensity 0 should be identity
                if intensity == 0.0:
                    identity = torch.allclose(image, vignette_out, atol=1)
                    print(f"    Intensity {intensity}: Range={'OK' if valid_range else 'FAIL'}, Identity={'OK' if identity else 'FAIL'}")
                else:
                    # Check center vs edges brightness
                    center = vignette_out[0, :, image.shape[2]//2, image.shape[3]//2].float().mean()
                    corner = vignette_out[0, :, 0, 0].float().mean()
                    darker_edges = corner < center
                    print(f"    Intensity {intensity}: Range={'OK' if valid_range else 'FAIL'}, Darker edges={'OK' if darker_edges else 'FAIL'}")
            
            # Test pixelate
            print("\n  Pixelate Effect:")
            for percentage in [0.0, 50.0, 100.0]:
                pixelate_out = fn.pixelate(image, pixelation_percentage=percentage, backend=self.backend)
                valid_range = self.validate_output_range(pixelate_out, pixelate_out.dtype)
                print(f"    Percentage {percentage}%: Range={'OK' if valid_range else 'FAIL'}")
    
    def test_batch_consistency(self):
        """Test batch processing consistency"""
        print("\n[QA] Batch Processing Consistency")
        
        # Load multiple test images
        test_images = []
        for i in range(1, 4):
            img_path = os.path.join(self.test_images_dir, f"three_images_mixed_src1/img{i}.jpg")
            if os.path.exists(img_path):
                test_images.append(utils.load_image(img_path))
        
        if len(test_images) >= 3:
            # Create batch
            batch = torch.cat(test_images, dim=0)
            
            # Test each augmentation
            augmentations = [
                ('brightness', lambda x: fn.brightness(x, alpha=1.5, beta=10.0, backend=self.backend)),
                ('gamma_correction', lambda x: fn.gamma_correction(x, gamma=0.8, backend=self.backend)),
                ('contrast', lambda x: fn.contrast(x, contrast_factor=2.0, backend=self.backend)),
                ('flip', lambda x: fn.flip(x, horizontal=True, backend=self.backend)),
                ('resize', lambda x: fn.resize(x, width=256, height=256, backend=self.backend)),
                ('rotate', lambda x: fn.rotate(x, angle=30.0, backend=self.backend)),
                ('vignette', lambda x: fn.vignette(x, intensity=0.7, backend=self.backend)),
            ]
            
            for aug_name, aug_func in augmentations:
                # Process batch
                batch_output = aug_func(batch)
                
                # Process individually and compare
                individual_outputs = []
                for img in test_images:
                    individual_outputs.append(aug_func(img))
                
                # Compare batch vs individual processing
                consistent = True
                for i in range(len(test_images)):
                    if not torch.allclose(batch_output[i:i+1], individual_outputs[i], atol=1):
                        consistent = False
                        break
                
                print(f"  {aug_name}: Batch consistency {'PASS' if consistent else 'FAIL'}")
                
                self.results.append({
                    'test': 'batch_consistency',
                    'augmentation': aug_name,
                    'consistent': consistent
                })
    
    def run_all_tests(self):
        """Run all QA tests"""
        print("\n" + "="*60)
        print("PyRPP QA Tests")
        print("="*60)
        
        test_methods = [
            self.test_brightness_quality,
            self.test_geometric_consistency,
            self.test_resize_quality,
            self.test_crop_accuracy,
            self.test_color_transforms_validity,
            self.test_effects_quality,
            self.test_batch_consistency
        ]
        
        for test in test_methods:
            try:
                test()
            except Exception as e:
                print(f"\n  Error in {test.__name__}: {e}")
                self.results.append({
                    'test': test.__name__,
                    'error': str(e)
                })
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print QA summary"""
        print("\n" + "="*60)
        print("QA Test Summary")
        print("="*60)
        
        # Count results
        total_tests = len(self.results)
        passed_tests = 0
        failed_tests = 0
        
        for result in self.results:
            if 'error' in result:
                failed_tests += 1
            else:
                # Check various pass conditions
                passed = True
                if 'range_valid' in result and not result['range_valid']:
                    passed = False
                if 'dims_correct' in result and not result['dims_correct']:
                    passed = False
                if 'consistent' in result and not result['consistent']:
                    passed = False
                if 'identity_check' in result and result['identity_check'] is False:
                    passed = False
                
                if passed:
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        print(f"\nTotal QA checks: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        
        # Detailed failures
        if failed_tests > 0:
            print("\nFailed checks:")
            for result in self.results:
                if 'error' in result:
                    print(f"  - {result['test']}: {result['error']}")
                else:
                    for key, value in result.items():
                        if key.endswith('_valid') or key.endswith('_correct') or key == 'consistent':
                            if not value:
                                print(f"  - {result.get('test', 'unknown')}: {key} = {value}")

def main():
    parser = argparse.ArgumentParser(description='PyRPP QA Tests')
    parser.add_argument('--backend', choices=['HOST', 'HIP'], default='HOST',
                        help='Backend to use (default: HOST)')
    parser.add_argument('--test', help='Run specific test')
    
    args = parser.parse_args()
    
    backend = types.HIP if args.backend == 'HIP' else types.HOST
    qa_tests = QATests(backend)
    
    if args.test:
        # Run specific test
        test_method = getattr(qa_tests, f'test_{args.test}', None)
        if test_method:
            test_method()
        else:
            print(f"Test '{args.test}' not found")
            return 1
    else:
        # Run all tests
        qa_tests.run_all_tests()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
