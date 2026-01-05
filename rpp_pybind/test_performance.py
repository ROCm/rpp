#!/usr/bin/env python3
"""
PyRPP Performance Tests
Measures execution time and throughput, comparing with C++ benchmarks
"""

import sys
import os
import argparse
import time
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

import rpp_pybind
from rpp_pybind import fn, utils, types

class PerformanceTests:
    def __init__(self, backend=None):
        self.backend = backend or types.get_default_backend()
        self.results = {}
        
        # Test configurations matching C++ performance tests
        self.batch_sizes = [1, 8, 16, 32, 64]
        self.image_sizes = [(224, 224), (480, 640), (720, 1280), (1080, 1920)]
        self.warmup_iterations = 10
        self.test_iterations = 100
        
        # Use same test images as other test files
        self.test_images_dir = "../utilities/test_suite/TEST_IMAGES/three_images_mixed_src1"
        self.test_images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        
    def create_test_batch(self, batch_size, height, width, use_real_images=False):
        """Create test batch for performance testing"""
        if use_real_images:
            # Try to use real images for more realistic performance testing
            images = []
            for i in range(min(batch_size, len(self.test_images))):
                img_path = os.path.join(self.test_images_dir, self.test_images[i])
                if os.path.exists(img_path):
                    img = utils.load_image(img_path)
                    # Resize to target size
                    img = fn.resize(img, width=width, height=height, backend=self.backend)
                    images.append(img)
            
            if images:
                # If we have real images, duplicate to reach batch size
                while len(images) < batch_size:
                    images.extend(images[:min(len(images), batch_size - len(images))])
                return torch.cat(images[:batch_size], dim=0)
        
        # Fallback to synthetic data
        return utils.create_test_batch(batch_size, height, width, channels=3)
    
    def measure_performance(self, func_name, func, *args, **kwargs):
        """Measure performance of a function"""
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = func(*args, **kwargs)
        
        # Synchronize if using GPU
        if self.backend == types.HIP:
            torch.cuda.synchronize()
        
        # Measure
        start_time = time.perf_counter()
        for _ in range(self.test_iterations):
            _ = func(*args, **kwargs)
        
        if self.backend == types.HIP:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = (total_time / self.test_iterations) * 1000  # ms
        
        return avg_time
    
    def benchmark_brightness(self):
        """Benchmark brightness augmentation"""
        print("\n[PERF] Brightness Augmentation")
        print(f"{'Batch':<8} {'Size':<12} {'Time (ms)':<12} {'Images/sec':<12}")
        print("-" * 50)
        
        results = []
        for batch_size in self.batch_sizes:
            for height, width in self.image_sizes:
                images = self.create_test_batch(batch_size, height, width)
                
                avg_time = self.measure_performance(
                    'brightness',
                    fn.brightness,
                    images,
                    alpha=1.5,
                    beta=10.0,
                    backend=self.backend
                )
                
                images_per_sec = (batch_size * 1000.0) / avg_time
                
                print(f"{batch_size:<8} {f'{width}x{height}':<12} {avg_time:<12.2f} {images_per_sec:<12.1f}")
                
                results.append({
                    'batch_size': batch_size,
                    'image_size': (height, width),
                    'time_ms': avg_time,
                    'throughput': images_per_sec
                })
        
        self.results['brightness'] = results
        return results
    
    def benchmark_gamma_correction(self):
        """Benchmark gamma correction"""
        print("\n[PERF] Gamma Correction")
        print(f"{'Batch':<8} {'Size':<12} {'Time (ms)':<12} {'Images/sec':<12}")
        print("-" * 50)
        
        results = []
        for batch_size in [1, 8, 16]:  # Smaller set for gamma
            for height, width in [(224, 224), (480, 640)]:
                images = self.create_test_batch(batch_size, height, width)
                
                avg_time = self.measure_performance(
                    'gamma_correction',
                    fn.gamma_correction,
                    images,
                    gamma=0.8,
                    backend=self.backend
                )
                
                images_per_sec = (batch_size * 1000.0) / avg_time
                
                print(f"{batch_size:<8} {f'{width}x{height}':<12} {avg_time:<12.2f} {images_per_sec:<12.1f}")
                
                results.append({
                    'batch_size': batch_size,
                    'image_size': (height, width),
                    'time_ms': avg_time,
                    'throughput': images_per_sec
                })
        
        self.results['gamma_correction'] = results
        return results
    
    def benchmark_resize(self):
        """Benchmark resize operation"""
        print("\n[PERF] Resize")
        print(f"{'Batch':<8} {'From':<12} {'To':<12} {'Time (ms)':<12} {'Images/sec':<12}")
        print("-" * 65)
        
        results = []
        resize_targets = [(224, 224), (256, 256), (512, 512)]
        
        for batch_size in [1, 8, 16]:
            for height, width in [(480, 640), (720, 1280)]:
                images = self.create_test_batch(batch_size, height, width)
                
                for target_h, target_w in resize_targets:
                    avg_time = self.measure_performance(
                        'resize',
                        fn.resize,
                        images,
                        width=target_w,
                        height=target_h,
                        backend=self.backend
                    )
                    
                    images_per_sec = (batch_size * 1000.0) / avg_time
                    
                    print(f"{batch_size:<8} {f'{width}x{height}':<12} {f'{target_w}x{target_h}':<12} {avg_time:<12.2f} {images_per_sec:<12.1f}")
                    
                    results.append({
                        'batch_size': batch_size,
                        'from_size': (height, width),
                        'to_size': (target_h, target_w),
                        'time_ms': avg_time,
                        'throughput': images_per_sec
                    })
        
        self.results['resize'] = results
        return results
    
    def benchmark_rotate(self):
        """Benchmark rotate operation"""
        print("\n[PERF] Rotate")
        print(f"{'Batch':<8} {'Size':<12} {'Angle':<8} {'Time (ms)':<12} {'Images/sec':<12}")
        print("-" * 55)
        
        results = []
        angles = [15.0, 30.0, 45.0, 90.0]
        
        for batch_size in [1, 8]:
            for height, width in [(224, 224), (480, 640)]:
                images = self.create_test_batch(batch_size, height, width)
                
                for angle in angles:
                    avg_time = self.measure_performance(
                        'rotate',
                        fn.rotate,
                        images,
                        angle=angle,
                        backend=self.backend
                    )
                    
                    images_per_sec = (batch_size * 1000.0) / avg_time
                    
                    print(f"{batch_size:<8} {f'{width}x{height}':<12} {angle:<8.1f} {avg_time:<12.2f} {images_per_sec:<12.1f}")
                    
                    results.append({
                        'batch_size': batch_size,
                        'image_size': (height, width),
                        'angle': angle,
                        'time_ms': avg_time,
                        'throughput': images_per_sec
                    })
        
        self.results['rotate'] = results
        return results
    
    def benchmark_all_augmentations(self):
        """Benchmark all 10 augmentations with standard parameters"""
        print("\n[PERF] All Augmentations Comparison")
        print(f"{'Augmentation':<20} {'Batch=8':<12} {'224x224':<12} {'Images/sec':<12}")
        print("-" * 60)
        
        # Standard test configuration
        batch_size = 8
        height, width = 224, 224
        # Use real images for more realistic benchmarking
        images = self.create_test_batch(batch_size, height, width, use_real_images=True)
        
        augmentations = [
            ('brightness', lambda img: fn.brightness(img, alpha=1.5, beta=10.0, backend=self.backend)),
            ('gamma_correction', lambda img: fn.gamma_correction(img, gamma=0.8, backend=self.backend)),
            ('contrast', lambda img: fn.contrast(img, contrast_factor=2.0, backend=self.backend)),
            ('hue', lambda img: fn.hue(img, hue_shift=45.0, backend=self.backend)),
            ('flip', lambda img: fn.flip(img, horizontal=True, backend=self.backend)),
            ('resize', lambda img: fn.resize(img, width=256, height=256, backend=self.backend)),
            ('rotate', lambda img: fn.rotate(img, angle=30.0, backend=self.backend)),
            ('crop', lambda img: fn.crop(img, x1=50, y1=50, crop_width=150, crop_height=150, backend=self.backend)),
            ('vignette', lambda img: fn.vignette(img, intensity=0.7, backend=self.backend)),
            ('pixelate', lambda img: fn.pixelate(img, pixelation_percentage=70.0, backend=self.backend))
        ]
        
        comparison_results = []
        
        for name, aug_func in augmentations:
            avg_time = self.measure_performance(name, aug_func, images)
            images_per_sec = (batch_size * 1000.0) / avg_time
            
            print(f"{name:<20} {avg_time:<12.2f} {f'{width}x{height}':<12} {images_per_sec:<12.1f}")
            
            comparison_results.append({
                'augmentation': name,
                'time_ms': avg_time,
                'throughput': images_per_sec
            })
        
        self.results['comparison'] = comparison_results
        return comparison_results
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("\n" + "="*70)
        print(f"PyRPP Performance Benchmarks - Backend: {self.backend}")
        print("="*70)
        
        # Individual benchmarks
        self.benchmark_brightness()
        self.benchmark_gamma_correction()
        self.benchmark_resize()
        self.benchmark_rotate()
        
        # Comparison
        self.benchmark_all_augmentations()
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*70)
        print("Performance Summary")
        print("="*70)
        
        if 'comparison' in self.results:
            print("\nAugmentation Ranking (8 batch, 224x224):")
            sorted_results = sorted(
                self.results['comparison'], 
                key=lambda x: x['throughput'], 
                reverse=True
            )
            
            for i, result in enumerate(sorted_results, 1):
                print(f"{i:2d}. {result['augmentation']:20s}: {result['throughput']:8.1f} images/sec")
        
        # Compare with expected C++ performance
        print("\n" + "-"*50)
        print("Performance vs C++ Reference:")
        print("-"*50)
        
        # These are example reference values - adjust based on actual C++ performance
        cpp_reference = {
            'brightness': 15000,  # images/sec for batch=8, 224x224
            'gamma_correction': 12000,
            'resize': 8000,
            'flip': 20000,
            'rotate': 5000
        }
        
        if 'comparison' in self.results:
            for result in self.results['comparison'][:5]:
                aug_name = result['augmentation']
                if aug_name in cpp_reference:
                    cpp_perf = cpp_reference[aug_name]
                    py_perf = result['throughput']
                    ratio = (py_perf / cpp_perf) * 100
                    print(f"{aug_name:20s}: {ratio:5.1f}% of C++ performance")

def main():
    parser = argparse.ArgumentParser(description='PyRPP Performance Tests')
    parser.add_argument('--backend', choices=['HOST', 'HIP'], default='HOST',
                        help='Backend to use (default: HOST)')
    parser.add_argument('--test', help='Run specific benchmark')
    
    args = parser.parse_args()
    
    backend = types.HIP if args.backend == 'HIP' else types.HOST
    perf_tests = PerformanceTests(backend)
    
    if args.test:
        # Run specific benchmark
        benchmark_method = getattr(perf_tests, f'benchmark_{args.test}', None)
        if benchmark_method:
            benchmark_method()
        else:
            print(f"Benchmark '{args.test}' not found")
            return 1
    else:
        # Run all benchmarks
        perf_tests.run_all_benchmarks()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
