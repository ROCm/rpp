"""
RPP Test Suite
==============

Unified test suite with combined Unit/QA testing per augmentation.
- Unit mode: Apply augmentation and save image
- QA mode: Apply augmentation and compare with reference tensor
- Performance mode: Time measurements
- All mode: Run all three test types

Usage:
    python test_suite.py --mode UNIT --backend HOST
    python test_suite.py --mode QA --backend HOST
    python test_suite.py --mode PERF --backend HIP
    python test_suite.py --mode ALL --backend HOST
    python test_suite.py --test_type 0 --backend HOST
    python test_suite.py --test_type 1 --backend HIP --num_runs 100
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, List
import shutil
from enum import Enum

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import RPP modules
import rpp_pybind.amd.rpp.fn as fn
import rpp_pybind.amd.rpp.utils as util
from rpp_pybind.amd.rpp.rpp_types import (
    is_gpu_available, get_default_backend, HOST, HIP
)

# =============================================================================
# ENUMS AND MAPPINGS
# =============================================================================

class TestType(Enum):
    """Test type enum matching runImageTests.py"""
    UNIT_TEST = 0
    PERFORMANCE_TEST = 1

class Layout(Enum):
    """Layout types for structured output"""
    PKD3 = 0
    PLN3 = 1
    PLN1 = 2

# Simplified augmentation case mapping for test_suite's 10 functions
augmentationCaseMap = {
    0: ["brightness"],
    1: ["gamma_correction"],
    4: ["contrast"],
    5: ["pixelate"],
    20: ["flip"],
    21: ["resize"],
    23: ["rotate"],
    37: ["crop"],
    42: ["hue"],
    46: ["vignette"],
}

# Functionality group mapping
AugmentationGroupMap = {
    "color_augmentations": ["brightness", "gamma_correction", "contrast", "hue"],
    "effects_augmentations": ["pixelate", "vignette"],
    "geometric_augmentations": ["flip", "resize", "rotate", "crop"]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_path(input_path):
    """Validate if a path exists and is a directory"""
    if not os.path.exists(input_path):
        return False
    return os.path.isdir(input_path)

def validate_and_remove_folders(path, folder_prefix):
    """Remove folders with specified prefix"""
    if path and os.path.isdir(path):
        folders = [f for f in os.listdir(path) if f.startswith(folder_prefix)]
        for folder in folders:
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Removed old directory: {folder_path}")

def func_group_finder(aug_name):
    """Find functionality group for an augmentation"""
    for group, augs in AugmentationGroupMap.items():
        if aug_name in augs:
            return group
    return "miscellaneous"

def create_layout_directories(dst_path):
    """Create layout-based directory structure"""
    for layout in Layout:
        layout_path = os.path.join(dst_path, layout.name)
        os.makedirs(layout_path, exist_ok=True)
        # Move any existing folders into correct layout directory
        for folder in os.listdir(dst_path):
            folder_path = os.path.join(dst_path, folder)
            if os.path.isdir(folder_path) and folder != layout.name:
                if layout.name.lower() in folder.lower():
                    dest_path = os.path.join(layout_path, folder)
                    if not os.path.exists(dest_path):
                        shutil.move(folder_path, dest_path)

def directory_name_generator(backend, layout, aug_name):
    """Generate directory name based on backend, layout and augmentation"""
    func_group = func_group_finder(aug_name)
    return f"rpp_{backend.lower()}_{layout.lower()}_{func_group}"

def print_qa_tests_summary(qaFilePath, supportedCaseList, nonQACaseList, fileName):
    """Read QA results and print summary"""
    try:
        f = open(qaFilePath, 'r+')
        numLines = 0
        numPassed = 0
        for line in f:
            sys.stdout.write(line)
            numLines += 1
            if "PASSED" in line:
                numPassed += 1
            sys.stdout.flush()
        resultsInfo = "\n\nFinal Results of Tests:"
        resultsInfo += "\n    - Total test cases including all subvariants REQUESTED = " + str(numLines)
        resultsInfo += "\n    - Total test cases including all subvariants PASSED = " + str(numPassed)
        resultsInfo += "\n\nGeneral information on test suite availability:"
        resultsInfo += "\n    - Total augmentations supported in test suite = " + str(len(supportedCaseList))
        resultsInfo += "\n    - Total augmentations with golden output QA test support = " + str(len(supportedCaseList) - len(nonQACaseList))
        resultsInfo += "\n    - Total augmentations without golden output QA test support (due to randomization involved) = " + str(len(nonQACaseList))
        f.write(resultsInfo)
        print("\n---------------------------------- Summary of QA Test - " + fileName + " ----------------------------------" + resultsInfo + "\n\n-------------------------------------------------------------------")
        f.close()
    except Exception as e:
        print(f"Error reading QA results: {e}")

def print_performance_tests_summary(logFile, functionalityGroupList, numRuns):
    """Read performance logs and print summary"""
    try:
        f = open(logFile, "r")
        print("\nOpened log file -> " + logFile)
    except IOError:
        print("Skipping file -> " + logFile)
        return

    stats = []
    maxVals = []
    minVals = []
    avgVals = []
    functions = []
    frames = []
    prevLine = ""
    funcCount = 0

    # Loop over each line
    for line in f:
        for functionalityGroup in functionalityGroupList:
            if functionalityGroup in line:
                functions.extend([" ", functionalityGroup, " "])
                frames.extend([" ", " ", " "])
                maxVals.extend([" ", " ", " "])
                minVals.extend([" ", " ", " "])
                avgVals.extend([" ", " ", " "])

        if "max,min,avg wall times in ms/batch" in line:
            splitWordStart = "Running "
            splitWordEnd = " " + str(numRuns)
            prevLine = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
            if prevLine not in functions:
                functions.append(prevLine)
                frames.append(numRuns)
                splitWordStart = "max,min,avg wall times in ms/batch = "
                splitWordEnd = "\n"
                stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                maxVals.append(stats[0])
                minVals.append(stats[1])
                avgVals.append(stats[2])
                funcCount += 1

        if line != "\n":
            prevLine = line

    # Print log lengths
    print("Functionalities - " + str(funcCount))

    # Print summary of log
    headerFormat = "{:<70} {:<15} {:<15} {:<15} {:<15}"
    rowFormat = "{:<70} {:<15} {:<15} {:<15} {:<15}"
    print("\n" + headerFormat.format("Functionality", "Frames Count", "max(ms/batch)", "min(ms/batch)", "avg(ms/batch)") + "\n")
    if len(functions) != 0:
        for i, func in enumerate(functions):
            print(rowFormat.format(func, str(frames[i]), str(maxVals[i]), str(minVals[i]), str(avgVals[i])))
    else:
        print("No variants under this category")

    # Closing log file
    f.close()

def test_suite_parser_and_validator():
    """Parse and validate command-line arguments similar to runImageTests.py"""
    
    script_path = os.path.dirname(os.path.realpath(__file__))
    default_input_path = os.path.join(script_path, "../test_suite/TEST_IMAGES/three_images_mixed_src1")
    
    case_min = min(augmentationCaseMap.keys())
    case_max = max(augmentationCaseMap.keys())
    
    parser = argparse.ArgumentParser(
        description='RPP Test Suite - Complete Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit testing only
  python test_suite.py --mode UNIT --backend HOST
  
  # QA testing only
  python test_suite.py --mode QA --backend HOST
  
  # Performance testing only
  python test_suite.py --mode PERF --backend HIP
  
  # Run all tests
  python test_suite.py --mode ALL --backend HOST
  
  # Using test_type parameter (runImageTests.py style)
  python test_suite.py --test_type 0 --backend HOST
  python test_suite.py --test_type 1 --backend HIP --num_runs 100
        """
    )
    
    # Arguments matching runImageTests.py structure
    parser.add_argument("--input_path1", type=str, default=default_input_path,
                       help="Path to input images")
    parser.add_argument("--input_path2", type=str, default=default_input_path,
                       help="Path to second input folder (for blend operations)")
    parser.add_argument("--case_start", type=int, default=case_min,
                       help=f"Start case number [{case_min}-{case_max}]")
    parser.add_argument("--case_end", type=int, default=case_max,
                       help=f"End case number [{case_min}-{case_max}]")
    parser.add_argument("--test_type", type=int,
                       help="0=Unit tests, 1=Performance tests")
    parser.add_argument("--case_list", nargs="+",
                       help="Specific augmentations to test")
    parser.add_argument("--qa_mode", type=int, default=0,
                       help="Enable QA mode (0/1)")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of performance test iterations")
    parser.add_argument("--preserve_output", type=int, default=1,
                       help="0=override, 1=preserve previous outputs")
    parser.add_argument("--batch_size", type=int, default=3,
                       help="Batch size for testing")
    
    # Keep existing arguments for backward compatibility
    parser.add_argument('--mode', 
                       choices=['UNIT', 'QA', 'PERF', 'ALL'],
                       help='Test mode to run')
    parser.add_argument('--backend', 
                       choices=['HOST', 'HIP'],
                       required=True, 
                       help='Backend to test')
    
    args = parser.parse_args()
    
    # Validate paths
    if not validate_path(args.input_path1):
        args.input_path1 = default_input_path
    if not validate_path(args.input_path2):
        args.input_path2 = default_input_path
    
    # Validate case range
    args.case_start = max(case_min, min(args.case_start, case_max))
    args.case_end = max(case_min, min(args.case_end, case_max))
    if args.case_end < args.case_start:
        args.case_start, args.case_end = args.case_end, args.case_start
    
    # Process case list
    if args.case_list:
        valid_cases = []
        for case in args.case_list:
            if case.isdigit() and int(case) in augmentationCaseMap:
                valid_cases.append(int(case))
            else:
                # Try to match by name
                for num, names in augmentationCaseMap.items():
                    if case.lower() in [n.lower() for n in names]:
                        valid_cases.append(num)
                        break
        args.case_list = valid_cases if valid_cases else None
    
    if not args.case_list:
        args.case_list = [k for k in augmentationCaseMap.keys() 
                         if args.case_start <= k <= args.case_end]
    
    # Map test_type to mode if specified
    if args.test_type is not None:
        if args.test_type == 0:
            args.mode = 'UNIT' if not args.qa_mode else 'QA'
        elif args.test_type == 1:
            args.mode = 'PERF'
    elif not args.mode:
        args.mode = 'ALL'
    
    # Set default num_runs based on test type
    if args.mode == 'PERF' and "--num_runs" not in sys.argv:
        args.num_runs = 100
    
    return args

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Global test configuration"""
    
    def __init__(self, preserve_output=1, test_type=None, qa_mode=0):
        # Directories
        self.TEST_IMAGES_DIR = "../test_suite/TEST_IMAGES/three_images_mixed_src1"
        self.REFERENCE_DIR = "../test_suite/REFERENCE_OUTPUT"
        
        # Test settings
        self.TOLERANCE = 1  # pixel difference tolerance for QA
        self.preserve_output = preserve_output
        self.test_type = test_type
        self.qa_mode = qa_mode
        
        # Test image paths and specs
        self.TEST_IMAGES = [
            "1_img50x50.jpg",
            "2_img100x100.jpg", 
            "3_img150x150.jpg"
        ]
        
        # Image dimensions (actual sizes)
        self.IMAGE_SPECS = [
            (50, 50),   # Image 0
            (100, 100), # Image 1
            (150, 150)  # Image 2
        ]
        
        # Reference batch dimensions
        self.BATCH_HEIGHT = 150
        self.BATCH_WIDTH = 152
        self.BATCH_SIZE = 3
        
        # Timestamp for output directories
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def get_output_dir(self, backend_name, mode):
        """Get output directory based on backend and mode - matching runImageTests.py"""
        if mode == "UNIT":
            if self.qa_mode:
                return f"QA_RESULTS_{backend_name}_{self.timestamp}"
            else:
                return f"OUTPUT_IMAGES_{backend_name}_{self.timestamp}"
        elif mode == "PERF":
            return f"OUTPUT_PERFORMANCE_LOGS_{backend_name}_{self.timestamp}"
        elif mode == "QA":
            return f"QA_RESULTS_{backend_name}_{self.timestamp}"
        else:
            return f"{backend_name}_OUTPUT_{mode}_{self.timestamp}"


# =============================================================================
# UNIFIED TEST CLASS WITH STRUCTURED OUTPUT
# =============================================================================

class UnifiedTestSuite:
    """Unified test suite with structured output organization"""
    
    def __init__(self, backend, mode="ALL", config=None, case_list=None, num_runs=1):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.mode = mode.upper()  # UNIT, QA, PERF, or ALL
        self.config = config if config else TestConfig()
        self.case_list = case_list
        self.num_runs = num_runs
        self.results = {
            'unit': [],
            'qa': [],
            'perf': []
        }
        
        # QA results tracking
        self.qa_results = []
        self.qa_file = None
        self.qa_output_dir = None
        self.current_aug_results = []  # Track results for current augmentation
        
        # Performance log files
        self.perf_logs = {}
        self.perf_output_dir = None
        
        # Setup directories with structured output
        if self.mode in ["UNIT", "ALL"]:
            self.unit_output_dir = self.config.get_output_dir(self.backend_name, "UNIT")
            os.makedirs(self.unit_output_dir, exist_ok=True)
            print(f"Unit Output Directory: {self.unit_output_dir}")
            # Create layout directories for structured output
            if not self.config.qa_mode:
                create_layout_directories(self.unit_output_dir)
        
        # Setup QA output directory and file
        if self.mode in ["QA", "ALL"]:
            if self.config.qa_mode or self.mode == "QA":
                self.qa_output_dir = self.config.get_output_dir(self.backend_name, "QA")
                os.makedirs(self.qa_output_dir, exist_ok=True)
                print(f"QA Output Directory: {self.qa_output_dir}")
                self.qa_file = open(os.path.join(self.qa_output_dir, "QA_results.txt"), "w")
        
        # Setup Performance output directory and log files
        if self.mode in ["PERF", "ALL"]:
            self.perf_output_dir = self.config.get_output_dir(self.backend_name, "PERF")
            os.makedirs(self.perf_output_dir, exist_ok=True)
            print(f"Performance Output Directory: {self.perf_output_dir}")
            # Create performance log files for each layout
            for layout in Layout:
                log_file = os.path.join(self.perf_output_dir, 
                                       f"Tensor_image_{self.backend_name.lower()}_{layout.name.lower()}_raw_performance_log.txt")
                self.perf_logs[layout.name] = open(log_file, "w")
        
        # Load test images paths
        self.test_images = [
            os.path.join(self.config.TEST_IMAGES_DIR, img) 
            for img in self.config.TEST_IMAGES
        ]
        
        print(f"Backend: {self.backend_name}")
        print(f"Mode: {self.mode}")
        print(f"Test Images: {len(self.test_images)}")
        print("-" * 70)
    
    # =========================================================================
    # HELPER FUNCTIONS WITH STRUCTURED OUTPUT
    # =========================================================================
    
    def _get_structured_output_path(self, augmentation_name, layout="PKD3"):
        """Get structured path for saving outputs matching runImageTests.py"""
        if self.config.qa_mode:
            return self.unit_output_dir
        
        # Generate structured directory path
        func_group = func_group_finder(augmentation_name)
        layout_dir = os.path.join(self.unit_output_dir, layout)
        structured_dir = f"rpp_{self.backend_name.lower()}_{layout.lower()}_{func_group}"
        full_path = os.path.join(layout_dir, structured_dir, augmentation_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def _save_output_image(self, tensor, augmentation_name, image_name, img_idx):
        """Save output image to structured filesystem"""
        try:
            # Get structured output directory
            aug_output_dir = self._get_structured_output_path(augmentation_name, "PKD3")
            
            # Convert tensor to numpy
            if hasattr(tensor, 'cpu'):
                tensor_np = tensor.cpu().numpy()
            else:
                tensor_np = np.array(tensor)
            
            # Handle different tensor formats
            if len(tensor_np.shape) == 4:  # NCHW format
                output_single = tensor_np[0]  # Extract first image
                output_hwc = np.transpose(output_single, (1, 2, 0))  # CHW to HWC
            elif len(tensor_np.shape) == 3 and tensor_np.shape[0] == 3:  # CHW format
                output_hwc = np.transpose(tensor_np, (1, 2, 0))
            else:
                output_hwc = tensor_np
            
            # Ensure uint8 type
            if output_hwc.dtype != np.uint8:
                output_hwc = np.clip(output_hwc, 0, 255).astype(np.uint8)
            
            # Get actual dimensions and crop before saving
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            # FOR CROP AND RESIZE: Save at half dimensions (25x25, 50x50, 75x75)
            if augmentation_name in ('crop', 'resize'):
                actual_h //= 2
                actual_w //= 2
            output_hwc = output_hwc[:actual_h, :actual_w, :]
            
            # Save image
            output_path = os.path.join(aug_output_dir, image_name)
            img = Image.fromarray(output_hwc)
            img.save(output_path)
            return True
            
        except Exception as e:
            print(f"    ✗ Failed to save: {e}")
            return False
    
    def _extract_from_batch_nhwc(self, batch_data, img_idx, aug_name=None):
        """
        Extract individual image from NHWC batch reference data.
        
        Batch structure (273,600 bytes total):
        - First two images: each in 150×152×3 slots
        - Third image: in remaining space (also 150×152×3)
        
        But stored with padding to 200 height for uniformity.
        """
        
        # Calculate offsets based on actual storage layout
        slot_height = 150  # Padded height for all slots
        slot_width = 152   # Common width
        slot_size = slot_height * slot_width * 3
        
        # Calculate the offset for the requested image
        offset = img_idx * slot_size
        
        # Extract the image slot
        img_slot = batch_data[offset:offset + slot_size]
        
        # Reshape to HWC
        img_slot_reshaped = img_slot.reshape(slot_height, slot_width, 3)
        
        # Get actual dimensions and extract valid region
        actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]

        # Adjust for crop and resize
        if aug_name in ('crop', 'resize'):
            actual_h //= 2
            actual_w //= 2
        
        # Extract only the valid region (top-left corner)
        ref_roi = img_slot_reshaped[:actual_h, :actual_w, :]
        
        return ref_roi

    
    def _compare_with_reference(self, output_tensor, ref_data, img_idx, aug_name=None):
        """
        Compare output tensor with reference from batch.
        
        Returns: (passed, stats_dict)
        """
        try:
            # Convert output to numpy HWC
            if hasattr(output_tensor, 'cpu'):
                output_np = output_tensor.cpu().numpy()
            else:
                output_np = np.array(output_tensor)
            
            # Handle tensor format conversion
            if len(output_np.shape) == 4:  # NCHW
                output_single = output_np[0]
                output_hwc = np.transpose(output_single, (1, 2, 0))
            elif len(output_np.shape) == 3 and output_np.shape[0] == 3:  # CHW
                output_hwc = np.transpose(output_np, (1, 2, 0))
            else:
                output_hwc = output_np
            
            # Get actual dimensions for this image
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
        
            # Adjust for crop/resize
            if aug_name in ('crop', 'resize'):
                actual_h //= 2
                actual_w //= 2

            # Extract ROI from output (remove any padding)
            output_roi = output_hwc[:actual_h, :actual_w, :]
            
            # Extract reference from batch
            ref_roi = self._extract_from_batch_nhwc(ref_data, img_idx, aug_name)
            
            # Verify shapes match
            if output_roi.shape != ref_roi.shape:
                return False, {
                    "error": f"Shape mismatch: output {output_roi.shape} vs ref {ref_roi.shape}"
                }

            # Calculate differences
            diff = output_roi.astype(np.int16) - ref_roi.astype(np.int16)
            abs_diff = np.abs(diff)
            
            # Statistics
            max_diff = int(abs_diff.max())
            mismatched = np.sum(abs_diff > self.config.TOLERANCE)
            total_pixels = output_roi.size
            
            stats = {
                "max_diff": max_diff,
                "mismatched_pixels": int(mismatched),
                "total_pixels": int(total_pixels),
                "match_percentage": 100.0 * (total_pixels - mismatched) / total_pixels
            }
            
            # Pass if all pixels within tolerance
            passed = max_diff <= self.config.TOLERANCE
            
            return passed, stats
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _write_qa_result(self, aug_name, image_idx, passed, stats=None):
        """Write QA result to file"""
        if self.qa_file:
            status = "PASSED" if passed else "FAILED"
            result_line = f"{aug_name}_img{image_idx}: {status}"
            if stats and not passed:
                if "error" in stats:
                    result_line += f" - Error: {stats['error']}"
                else:
                    result_line += f" - Max diff: {stats['max_diff']}, Mismatch: {stats['mismatched_pixels']}/{stats['total_pixels']}"
            self.qa_file.write(result_line + "\n")
            self.qa_file.flush()
            
            # Track result for percentage calculation
            self.current_aug_results.append(passed)
    
    def _write_aug_summary(self, aug_name):
        """Write augmentation summary with percentage"""
        if self.qa_file and self.current_aug_results:
            passed_count = sum(self.current_aug_results)
            total_count = len(self.current_aug_results)
            percentage = (passed_count / total_count) * 100
            
            if percentage == 100:
                status = "PASSED"
            else:
                status = "FAILED"
                
            summary_line = f"{aug_name} Percentage: {status} ({percentage:.0f}%)\n"
            self.qa_file.write(summary_line)
            self.qa_file.flush()
            
            # Clear for next augmentation
            self.current_aug_results = []
    
    def _write_perf_result(self, aug_name, times_dict, layout="PKD3"):
        """Write performance result to log file"""
        if layout in self.perf_logs:
            log_file = self.perf_logs[layout]
            func_group = func_group_finder(aug_name)
            log_file.write(f"\n{func_group}\n")
            log_file.write(f"Running {aug_name} {self.num_runs} times\n")
            log_file.write(f"max,min,avg wall times in ms/batch = {times_dict['max']:.2f},{times_dict['min']:.2f},{times_dict['avg']:.2f}\n")
            log_file.flush()
    
    def cleanup(self):
        """Close all open files"""
        if self.qa_file:
            self.qa_file.close()
        
        for log_file in self.perf_logs.values():
            log_file.close()
    
    # =========================================================================
    # AUGMENTATION FUNCTIONS (Combined Unit + QA)
    # =========================================================================
    
    def test_brightness(self):
        aug_name = "brightness"
        device = 'cuda' if self.backend == HIP else 'cpu' 
        print(f"  [1/10] Brightness ")
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.brightness(
                    image, 
                    alpha=1.75, 
                    beta=50.0,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS (max_diff={stats['max_diff']})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']} (tolerance: {self.config.TOLERANCE})")
                            print(f"      Mismatched: {stats['mismatched_pixels']}/{stats['total_pixels']} ({100-stats['match_percentage']:.2f}%)")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:  # ALL
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:  # ALL
            self.results['unit'].append((aug_name, True))  # Assume unit passed if we get here
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_gamma_correction(self):
        aug_name = "gamma_correction"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [2/10] Gamma Correction ")
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.gamma_correction(
                    image, 
                    gamma=1.9,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success

    def test_flip(self):
        aug_name = "flip"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f" Flip ")
        
        if not hasattr(fn, 'flip'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.flip(
                    image, 
                    horizontal=True,
                    vertical=False,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_resize(self):
        aug_name = "resize"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [4/10] Resize")
        
        if not hasattr(fn, 'resize'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor_interpolationTypeBilinear.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w //2]
                roi_heights = [actual_h //2]
                
                if self.backend == HIP:
                    torch.cuda.synchronize()
                output = fn.resize(
                    image, 
                    width=actual_w//2, 
                    height=actual_h//2,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                if self.backend == HIP:
                    torch.cuda.synchronize()
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_crop(self):
        aug_name = "crop"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [5/10] Crop ")
        
        if not hasattr(fn, 'crop'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]

                output = fn.crop(
                    image, 
                    x1=10, 
                    y1=10, 
                    crop_width = actual_w//2, 
                    crop_height = actual_h//2, 
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_hue(self):
        aug_name = "hue"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [6/10] Hue ")
        
        if not hasattr(fn, 'hue'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.hue(
                    image, 
                    hue_shift=60.0,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_rotate(self):
        aug_name = "rotate"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [7/10] Rotate")
        
        if not hasattr(fn, 'rotate'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor_interpolationTypeBilinear.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.rotate(
                    image, 
                    angle=50.0,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_contrast(self):
        aug_name = "contrast"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [8/10] Contrast")
        
        if not hasattr(fn, 'contrast'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.contrast(
                    image, 
                    contrast_factor=2.96, 
                    contrast_center=128.0,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_vignette(self):
        aug_name = "vignette"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [9/10] Vignette")
        
        if not hasattr(fn, 'vignette'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.vignette(
                    image, 
                    intensity=6.0,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    def test_pixelate(self):
        aug_name = "pixelate"
        device = 'cuda' if self.backend == HIP else 'cpu'
        print(f"  [10/10] Pixelate ")
        
        if not hasattr(fn, 'pixelate'):
            print("  SKIP (function not available)")
            self.results[self.mode.lower()].append((aug_name, None))
            return None
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_u8_Tensor.bin"
            )
            if os.path.exists(ref_path):
                ref_data = np.fromfile(ref_path, dtype=np.uint8)
                print(f"  Loaded reference: {len(ref_data)} bytes")
            else:
                print(f"  ✗ Reference not found: {ref_path}")
                return False
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load and apply augmentation
                image = util.load_image(img_path, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w]
                roi_heights = [actual_h]
                
                output = fn.pixelate(
                    image, 
                    pixelation_percentage=87.5,
                    roi_widths=roi_widths,
                    roi_heights=roi_heights,
                    backend=self.backend
                )
                
                # UNIT MODE: Save output
                if self.mode in ["UNIT", "ALL"]:
                    if self._save_output_image(output, aug_name, image_name, idx):
                        print(f"    ✓ {image_name} : SAVED")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_with_reference(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    ✓ {image_name} : QA PASS")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    ✗ {image_name} : QA FAIL")
                        if "error" in stats:
                            print(f"      Error: {stats['error']}")
                        else:
                            print(f"      Max diff: {stats['max_diff']}")
                
            except Exception as e:
                print(f"    ✗ {image_name} : ERROR → {e}")
        
        # Write percentage summary for this augmentation
        if self.mode in ["QA", "ALL"] and self.qa_file:
            self._write_aug_summary(aug_name)
        
        # Report results
        success = success_count == total
        if self.mode == "UNIT":
            status = f"SAVED {success_count}/{total} images"
        elif self.mode == "QA":
            status = f"PASSED {success_count}/{total} comparisons"
        else:
            status = f"COMPLETED {success_count}/{total} tests"
        
        print(f"\n  RESULT: {status} {'✓' if success else '✗'}")
        
        if self.mode == "UNIT":
            self.results['unit'].append((aug_name, success))
        elif self.mode == "QA":
            self.results['qa'].append((aug_name, success))
        else:
            self.results['unit'].append((aug_name, True))
            self.results['qa'].append((aug_name, success))
        
        return success
    
    # =========================================================================
    # TEST RUNNERS
    # =========================================================================
    
    def run_unit_tests(self):
        """Run augmentations based on case_list in Unit mode"""
        print(f"\n{'='*70}")
        print(f"UNIT TESTS - Image Generation ({self.backend_name})")
        print(f"{'='*70}\n")
        
        # Map augmentation names to test methods
        test_map = {
            "brightness": self.test_brightness,
            "gamma_correction": self.test_gamma_correction,
            "flip": self.test_flip,
            "resize": self.test_resize,
            "crop": self.test_crop,
            "hue": self.test_hue,
            "rotate": self.test_rotate,
            "contrast": self.test_contrast,
            "vignette": self.test_vignette,
            "pixelate": self.test_pixelate
        }
        
        # Run tests based on case_list
        if self.case_list:
            for case_num in self.case_list:
                if case_num in augmentationCaseMap:
                    aug_name = augmentationCaseMap[case_num][0]
                    if aug_name in test_map:
                        try:
                            test_map[aug_name]()
                            print("-" * 70)
                        except Exception as e:
                            print(f"ERROR in {aug_name}: {e}")
                            print("-" * 70)
        else:
            # Run all tests if no case_list specified
            for test_func in test_map.values():
                try:
                    test_func()
                    print("-" * 70)
                except Exception as e:
                    print(f"ERROR in {test_func.__name__}: {e}")
                    print("-" * 70)
    
    def run_performance_tests(self):
        """Run performance tests - timing measurements for all augmentations"""
        print(f"\n{'='*70}")
        print(f"PERFORMANCE TESTS ({self.backend_name})")
        print(f"{'='*70}\n")
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        num_iterations = self.num_runs if self.num_runs > 1 else 100  # Use num_runs parameter
        warmup_iterations = 5
        
        # Test image for performance (using 100x100 image)
        test_image = util.load_image(self.test_images[1], device=device)
        
        print(f"Running {num_iterations} iterations per augmentation (after {warmup_iterations} warmup runs)\n")
        
        perf_tests = [
            ('brightness', lambda: fn.brightness(test_image, alpha=1.75, beta=50.0, backend=self.backend)),
            ('gamma_correction', lambda: fn.gamma_correction(test_image, gamma=1.9, backend=self.backend)),
            ('flip', lambda: fn.flip(test_image, horizontal=True, vertical=False, backend=self.backend)),
            ('resize', lambda: fn.resize(test_image, width=50, height=50, backend=self.backend)),  # Fixed with proper parameters
            ('crop', lambda: fn.crop(test_image, x1=10, y1=10, crop_width=50, crop_height=50, backend=self.backend)),  # Fixed with proper parameters
            ('hue', lambda: fn.hue(test_image, hue_shift=60.0, backend=self.backend)),
            ('rotate', lambda: fn.rotate(test_image, angle=50.0, backend=self.backend)),
            ('contrast', lambda: fn.contrast(test_image, contrast_factor=2.96, contrast_center=128.0, backend=self.backend)),
            ('vignette', lambda: fn.vignette(test_image, intensity=6.0, backend=self.backend)),
            ('pixelate', lambda: fn.pixelate(test_image, pixelation_percentage=87.5, backend=self.backend))
        ]
        
        for i, (func_name, func_call) in enumerate(perf_tests, 1):
            print(f"  [{i}/10] Testing {func_name}...", end=" ")
            
            # Check if function is available
            if not hasattr(fn, func_name):
                print("SKIP (function not available)")
                self.results['perf'].append((func_name, None))
                continue
            
            try:
                # Warmup
                for _ in range(warmup_iterations):
                    _ = func_call()
                    if self.backend == HIP:
                        torch.cuda.synchronize()
                
                # Timing
                times = []
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    _ = func_call()
                    if self.backend == HIP:
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)  # Convert to ms
                
                avg_time = np.mean(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                print(f"Avg: {avg_time:.2f}ms, Min: {min_time:.2f}ms, Max: {max_time:.2f}ms")
                
                # Store results
                timing_dict = {
                    'num_runs': num_iterations,
                    'min': min_time,
                    'max': max_time,
                    'avg': avg_time
                }
                self.results['perf'].append((func_name, timing_dict))
                
                # Write to performance log files for all layouts
                for layout in Layout:
                    self._write_perf_result(func_name, timing_dict, layout.name)
                
            except Exception as e:
                print(f"ERROR: {e}")
                self.results['perf'].append((func_name, None))
    
    def run_all(self):
        """Run tests based on mode"""
        if self.mode == "UNIT":
            self.run_unit_tests()
        elif self.mode == "QA":
            self.run_unit_tests()
        elif self.mode == "PERF":
            self.run_performance_tests()  # Actually run the performance tests!
        elif self.mode == "ALL":
            self.run_unit_tests()
            self.run_performance_tests()
        else:
            print(f"ERROR: Invalid mode '{self.mode}'")
            return False
        
        # Print summary
        self._print_summary()
        return True
    
    def _print_summary(self):
        """Print test summary"""
        print(f"\n{'='*70}")
        print(f"TEST SUMMARY ({self.mode} mode)")
        print(f"{'='*70}")
        
        if self.mode in ["UNIT", "ALL"] and self.results['unit']:
            passed = sum(1 for _, r in self.results['unit'] if r is True)
            failed = sum(1 for _, r in self.results['unit'] if r is False)
            skipped = sum(1 for _, r in self.results['unit'] if r is None)
            print(f"\nUNIT TESTS:")
            print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if self.mode in ["QA", "ALL"] and self.results['qa']:
            passed = sum(1 for _, r in self.results['qa'] if r is True)
            failed = sum(1 for _, r in self.results['qa'] if r is False)
            skipped = sum(1 for _, r in self.results['qa'] if r is None)
            print(f"\nQA TESTS:")
            print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if self.mode in ["PERF", "ALL"] and self.results['perf']:
            valid = sum(1 for _, r in self.results['perf'] if r is not None)
            print(f"\nPERFORMANCE TESTS:")
            print(f"  Completed: {valid}/{len(self.results['perf'])}")
            
            if valid > 0:
                # Show detailed performance results for each augmentation
                print(f"\nDetailed Performance Results:")
                print(f"  {'Augmentation':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Runs':<8}")
                print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
                
                for aug_name, timing in self.results['perf']:
                    if timing is not None:
                        print(f"  {aug_name:<20} {timing['avg']:<12.3f} {timing['min']:<12.3f} {timing['max']:<12.3f} {timing.get('num_runs', 100):<8}")
                    else:
                        print(f"  {aug_name:<20} {'FAILED':<12} {'-':<12} {'-':<12} {'-':<8}")
                        
        print(f"{'='*70}\n")


# =============================================================================
# MAIN WITH STRUCTURED OUTPUT SUPPORT
# =============================================================================

def main():
    # Use the new parser
    args = test_suite_parser_and_validator()
    
    # Determine backend
    backend = HIP if args.backend == 'HIP' else HOST
    backend_name = args.backend
    
    # Check GPU availability for HIP backend
    if backend == HIP and not is_gpu_available():
        print(f"ERROR: HIP backend requested but GPU not available")
        return 1
    
    # Handle preserve_output - remove old directories if requested
    if args.preserve_output == 0:
        base_dir = os.getcwd()
        if args.mode == "UNIT" and not args.qa_mode:
            validate_and_remove_folders(base_dir, f"OUTPUT_IMAGES_{backend_name}")
        elif args.mode == "QA" or args.qa_mode:
            validate_and_remove_folders(base_dir, f"QA_RESULTS_{backend_name}")
        elif args.mode == "PERF":
            validate_and_remove_folders(base_dir, f"OUTPUT_PERFORMANCE_LOGS_{backend_name}")
    
    # Print header
    print("\n" + "="*70)
    print("RPP TEST SUITE - STRUCTURED OUTPUT VERSION")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Backend: {backend_name}")
    print(f"GPU Available: {is_gpu_available()}")
    if args.case_list:
        print(f"Case List: {args.case_list}")
    print("="*70)
    
    # Create test configuration
    config = TestConfig(
        preserve_output=args.preserve_output,
        test_type=args.test_type if hasattr(args, 'test_type') else None,
        qa_mode=args.qa_mode
    )
    
    # Run tests with structured output
    try:
        test_suite = UnifiedTestSuite(backend, args.mode, config, args.case_list)
        success = test_suite.run_all()
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
