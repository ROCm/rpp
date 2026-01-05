#!/usr/bin/env python3
"""
PyRPP Test Runner
Runs all test suites: unit, performance, and QA tests
"""

import sys
import os
import subprocess
import argparse
import time

def run_test_suite(test_file, backend='HOST', test_name=None):
    """Run a test suite and return results"""
    cmd = [sys.executable, test_file, '--backend', backend]
    if test_name:
        cmd.extend(['--test', test_name])
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print('='*70)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    
    return result.returncode == 0

def verify_augmentation_coverage():
    """Verify all 10 augmentations are tested"""
    augmentations = [
        'brightness',
        'gamma_correction',
        'contrast',
        'hue',
        'flip',
        'resize',
        'rotate',
        'crop',
        'vignette',
        'pixelate'
    ]
    
    print("\n" + "="*70)
    print("PyRPP Augmentation Coverage")
    print("="*70)
    
    # Check test files for coverage
    test_files = {
        'test_unit.py': 'Unit Tests',
        'test_performance.py': 'Performance Tests',
        'test_qa.py': 'QA Tests'
    }
    
    coverage = {}
    
    for test_file, test_name in test_files.items():
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
                
            covered = []
            for aug in augmentations:
                # Check if augmentation is tested
                if f"test_{aug}" in content or f"fn.{aug}" in content:
                    covered.append(aug)
            
            coverage[test_name] = covered
            
            print(f"\n{test_name}:")
            for aug in augmentations:
                status = "✓" if aug in covered else "✗"
                print(f"  {status} {aug}")
    
    # Summary
    all_covered = set()
    for covered in coverage.values():
        all_covered.update(covered)
    
    print(f"\nOverall Coverage: {len(all_covered)}/{len(augmentations)} augmentations")
    
    missing = set(augmentations) - all_covered
    if missing:
        print("\nMissing coverage for:")
        for aug in missing:
            print(f"  - {aug}")
    
    return len(all_covered) == len(augmentations)

def main():
    parser = argparse.ArgumentParser(description='PyRPP Test Runner')
    parser.add_argument('--backend', choices=['HOST', 'HIP'], default='HOST',
                        help='Backend to use (default: HOST)')
    parser.add_argument('--suite', choices=['unit', 'perf', 'qa', 'all'], 
                        default='all', help='Test suite to run')
    parser.add_argument('--test', help='Run specific test within suite')
    parser.add_argument('--coverage-only', action='store_true',
                        help='Only check test coverage')
    
    args = parser.parse_args()
    
    if args.coverage_only:
        verify_augmentation_coverage()
        return 0
    
    # Test configuration
    test_suites = {
        'unit': ('test_unit.py', 'Unit Tests'),
        'perf': ('test_performance.py', 'Performance Tests'),
        'qa': ('test_qa.py', 'Quality Assurance Tests')
    }
    
    results = {}
    
    if args.suite == 'all':
        # Run all test suites
        for suite_key, (test_file, test_name) in test_suites.items():
            if os.path.exists(test_file):
                success = run_test_suite(test_file, args.backend)
                results[test_name] = 'PASS' if success else 'FAIL'
            else:
                print(f"\nWarning: {test_file} not found")
                results[test_name] = 'SKIP'
    else:
        # Run specific suite
        if args.suite in test_suites:
            test_file, test_name = test_suites[args.suite]
            if os.path.exists(test_file):
                success = run_test_suite(test_file, args.backend, args.test)
                results[test_name] = 'PASS' if success else 'FAIL'
            else:
                print(f"\nError: {test_file} not found")
                return 1
    
    # Coverage check
    verify_augmentation_coverage()
    
    # Final summary
    print("\n" + "="*70)
    print("PyRPP Test Summary")
    print("="*70)
    print(f"\nBackend: {args.backend}")
    print("\nTest Results:")
    
    all_passed = True
    for test_name, result in results.items():
        print(f"  {test_name:30s}: {result}")
        if result != 'PASS':
            all_passed = False
    
    print("\nTest Images:")
    print("  All tests use consistent images from:")
    print("  ../utilities/test_suite/TEST_IMAGES/three_images_mixed_src1/")
    
    print("\nAugmentations Tested (10):")
    print("  Color: brightness, gamma_correction, contrast, hue")
    print("  Geometric: flip, resize, rotate, crop")
    print("  Effects: vignette, pixelate")
    
    if all_passed:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
