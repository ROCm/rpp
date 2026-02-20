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
==============

Unified test suite with combined Unit/QA testing per augmentation.
- Unit mode: Apply augmentation and save image
- QA mode: Apply augmentation and compare with reference tensor
- Performance mode: Time measurements
- Supports both u8 and f32 bitdepths

Usage:
    python test_suite.py --test_type 0 --backend HOST --bitdepth u8 f16
    python test_suite.py --test_type 0 --backend HIP
    python test_suite.py --test_type 1 --backend HIP --num_runs 100 --bitdepth f32
    python test_suite.py --test_type 0 --backend HOST --bitdepth u8 f32 --qa_mode 1
    python test_suite.py --test_type 1 --backend HIP --num_runs 100 --bitdepth f32
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

class BitDepth(Enum):
    """Supported bit depths"""
    U8 = "u8"
    F32 = "f32"
    F16 = "f16"   # Add half-precision float
    I8 = "i8"

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

def directory_name_generator(backend, layout, aug_name, bitdepth="u8"):
    """Generate directory name based on backend, layout, augmentation and bitdepth"""
    func_group = func_group_finder(aug_name)
    return f"rpp_{backend.lower()}_{layout.lower()}_{bitdepth}_{func_group}"

def load_image_with_bitdepth(img_path, bitdepth='u8', device='cpu'):
    """
    Load image with specified bitdepth.
    
    Args:
        img_path: Path to image file
        bitdepth: 'u8' or 'f32'
        device: 'cpu' or 'cuda'
    
    Returns:
        PyTorch tensor in specified bitdepth
    """
    # Load image (returns float tensor with values 0-255)
    image = util.load_image(img_path, device=device)
    if bitdepth == 'f32':
        # Normalize to [0, 1] range for f32
        image = image / 255.0
    elif bitdepth == 'f16':
        # Convert to half precision
        image = image / 255.0
        image = image.half()  # Convert to float16
    elif bitdepth == 'i8':
        # Convert to signed int8 [-128, 127]
        image = image - 128
        image = image.to(torch.int8)
    else:
        image = image.to(torch.uint8)
    return image

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
    current_category = ""

    # Loop over each line
    for line in f:
        # Process actual performance data
        if "max,min,avg wall times in ms/batch" in line:
            # Extract function name from previous line
            if "Running " in prevLine:
                splitWordStart = "Running "
                splitWordEnd = " " + str(numRuns)
                func_name = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
                if func_name and func_name not in functions:
                    functions.append(func_name)
                    frames.append(numRuns)
                    
                    # Extract timing data from current line
                    splitWordStart = "max,min,avg wall times in ms/batch = "
                    splitWordEnd = "\n"
                    stats = line.partition(splitWordStart)[2].partition(splitWordEnd)[0].split(",")
                    if len(stats) == 3:
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
            if func:  # Skip empty entries
                if func.startswith("---"):
                    # Print category separator
                    print("\n" + func)
                else:
                    # Print actual function data
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
        description='RPP Test Suite - Complete Version with f32 Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit testing only (u8)
  python test_suite.py --test_type 0 --backend HOST --bitdepth u8
  
  # QA testing only (f32)
  python test_suite.py --test_type 0 --backend HOST --bitdepth f32 --qa_mode 1
  
  # Performance testing only
  python test_suite.py --test_type 1 --backend HIP --bitdepth f32
  
  # Using test_type parameter (runImageTests.py style)
  python test_suite.py --test_type 0 --backend HOST --bitdepth f32
  python test_suite.py --test_type 1 --backend HIP --num_runs 100 --bitdepth u8
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
    parser.add_argument("--test_type", type=int, default=0,
                       help="0=Unit/QA tests (based on qa_mode), 1=Performance tests")
    parser.add_argument("--case_list", nargs="+",
                       help="Specific augmentations to test")
    parser.add_argument("--qa_mode", type=int, default=0,
                       help="Enable QA mode when test_type=0 (0/1)")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of performance test iterations")
    parser.add_argument("--preserve_output", type=int, default=1,
                       help="0=override, 1=preserve previous outputs")
    parser.add_argument("--batch_size", type=int, default=3,
                       help="Batch size for testing")
    
    # Bitdepth support - now optional, will run multiple bitdepths if not specified
    parser.add_argument("--bitdepth", 
                       choices=['u8', 'f32', 'f16','i8'],
                       nargs='+',
                       default=None,
                       help='Specific bit depth for testing. If not specified, runs multiple bitdepths based on mode')
    
    parser.add_argument('--backend', 
                       choices=['HOST', 'HIP'],
                       help='Backend to test. If not specified, runs both HOST and HIP')
    
    args = parser.parse_args()
    
    if not validate_path(args.input_path1):
        print(f"Warning: input_path1 '{args.input_path1}' not found, falling back to default.")
        args.input_path1 = default_input_path
    if not validate_path(args.input_path2):
        args.input_path2 = default_input_path

    # Store the default path so QA mode can always reference it
    args.default_input_path = default_input_path
    
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
        args.case_list = [
            k for k in sorted(augmentationCaseMap.keys())
            if args.case_start <= k <= args.case_end
        ]
    
    # Determine mode based on test_type and qa_mode (no more --mode argument)
    if args.test_type == 0:
        args.mode = 'UNIT' if not args.qa_mode else 'QA'
    elif args.test_type == 1:
        args.mode = 'PERF'
    else:
        print(f"Invalid test_type: {args.test_type}. Must be 0 or 1")
        sys.exit(1)
    
    # Set default num_runs for performance tests
    if args.test_type == 1 and "--num_runs" not in sys.argv:
        args.num_runs = 100
    
    return args

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig:
    """Global test configuration"""
    
    def __init__(self, preserve_output=1, test_type=None, qa_mode=0, bitdepth='u8',
                 input_path=None):
        # Directories
        self.DEFAULT_IMAGES_DIR = "../test_suite/TEST_IMAGES/three_images_mixed_src1"
        self.REFERENCE_DIR = "../test_suite/REFERENCE_OUTPUT"

        # QA always uses default images; UNIT/PERF use input_path if valid
        if qa_mode:
            self.TEST_IMAGES_DIR = self.DEFAULT_IMAGES_DIR
        else:
            if input_path and validate_path(input_path):
                self.TEST_IMAGES_DIR = input_path
            else:
                self.TEST_IMAGES_DIR = self.DEFAULT_IMAGES_DIR
        
        # Test settings
        if bitdepth == 'u8':
            self.TOLERANCE = 1
        elif bitdepth == 'f32':
            self.TOLERANCE = 0.01
        elif bitdepth == 'f16':
            self.TOLERANCE = 0.1  # Less precision than f32
        elif bitdepth == 'i8':
            self.TOLERANCE = 1  # Same as u8
        self.preserve_output = preserve_output
        self.test_type = test_type
        self.qa_mode = qa_mode
        self.bitdepth = bitdepth
        
        # Test image paths and specs - dynamically discover for non-QA tests
        if qa_mode:
            # QA mode requires specific test images for comparison
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
        else:
            # Try to discover images dynamically for unit/perf testing
            if not self._discover_images():
                # Fall back to default images if discovery fails
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
    
    def _discover_images(self):
        """Dynamically discover images in the input directory"""
        try:
            # Get list of image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = []
            
            for file in sorted(os.listdir(self.TEST_IMAGES_DIR)):
                if file.lower().endswith(image_extensions):
                    image_files.append(file)
            
            if not image_files:
                print(f"No image files found in {self.TEST_IMAGES_DIR}")
                return False
            
            print(f"Discovered {len(image_files)} image(s) in {self.TEST_IMAGES_DIR}")
            
            # Process all discovered images (no limit)
            # image_files = image_files[:3]  # REMOVED: No longer limiting to 3 images
            
            # Discover dimensions for each image
            self.TEST_IMAGES = []
            self.IMAGE_SPECS = []
            
            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(self.TEST_IMAGES_DIR, img_file)
                try:
                    # Use PIL to get image dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        self.TEST_IMAGES.append(img_file)
                        self.IMAGE_SPECS.append((height, width))  # Store as (height, width)
                        print(f"  - [{idx+1}/{len(image_files)}] {img_file}: {width}x{height}")
                except Exception as e:
                    print(f"  - [{idx+1}/{len(image_files)}] Failed to read {img_file}: {e}")
                    continue
            
            if not self.TEST_IMAGES:
                print(f"Could not read any images from {self.TEST_IMAGES_DIR}")
                return False
            
            # Update BATCH_SIZE to match actual number of images discovered
            self.BATCH_SIZE = len(self.TEST_IMAGES)
            print(f"  - Batch size set to: {self.BATCH_SIZE}")
            
            return True
            
        except Exception as e:
            print(f"Error discovering images: {e}")
            return False
    
    def get_output_dir(self, backend_name, mode):
        """Get output directory based on backend and mode with bitdepth"""
        bitdepth_suffix = f"_{self.bitdepth.upper()}" if self.bitdepth else ""
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
    
    def __init__(self, backend, mode="ALL", config=None, case_list=None, num_runs=1, bitdepth='u8'):
        self.backend = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.mode = mode.upper()  # UNIT, QA, PERF, or ALL
        self.config = config if config else TestConfig(bitdepth=bitdepth)
        self.case_list = case_list
        self.num_runs = num_runs
        self.bitdepth = bitdepth
        self.results = {
            'unit': [],
            'qa': [],
            'perf': []
        }
        
        # QA results tracking
        self.qa_results = []
        self.qa_file = None
        self.qa_output_dir = None
        self.owns_qa_file = True  # Track whether we own the QA file
        self.current_aug_results = []  # Track results for current augmentation
        
        # Performance log files
        self.perf_logs = {}
        self.perf_output_dir = None
        self.perf_log_file = None  # Initialize to None
        self.owns_perf_log = True  # Track whether we own the log file
        
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
                # Don't create QA file here if it will be overridden later
                self.qa_file = None
        
        # Setup Performance output directory and log files
        if self.mode in ["PERF", "ALL"]:
            self.perf_output_dir = self.config.get_output_dir(self.backend_name, "PERF")
            os.makedirs(self.perf_output_dir, exist_ok=True)
            print(f"Performance Output Directory: {self.perf_output_dir}")
            # Don't create log file here if it will be provided externally
            # This will be handled by set_shared_perf_log() method
        
        # Load test images paths
        self.test_images = [
            os.path.join(self.config.TEST_IMAGES_DIR, img) 
            for img in self.config.TEST_IMAGES
        ]
        
        print(f"Backend: {self.backend_name}")
        print(f"Mode: {self.mode}")
        print(f"BitDepth: {self.bitdepth}")
        print(f"Test Images: {len(self.test_images)}")
        print("-" * 70)
    
    # =========================================================================
    # HELPER FUNCTIONS WITH STRUCTURED OUTPUT
    # =========================================================================
    
    def _get_structured_output_path(self, augmentation_name, layout="PKD3"):
        """Get structured path for saving outputs matching runImageTests.py with bitdepth"""
        if self.config.qa_mode:
            return self.unit_output_dir
        
        # Generate structured directory path with bitdepth
        func_group = func_group_finder(augmentation_name)
        layout_dir = os.path.join(self.unit_output_dir, f"{layout}")
        structured_dir = f"rpp_{self.backend_name.lower()}_{layout.lower()}_{self.bitdepth}_{func_group}"
        full_path = os.path.join(layout_dir, structured_dir, augmentation_name)
        os.makedirs(full_path, exist_ok=True)
        return full_path
    
    def _save_output_image(self, tensor, augmentation_name, image_name, img_idx):
        """Save output image to structured filesystem (handles both u8 and f32)"""
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
            
            # Get actual dimensions and crop before saving
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            # FOR CROP AND RESIZE: Save at half dimensions (25x25, 50x50, 75x75)
            if augmentation_name in ('crop', 'resize'):
                actual_h //= 2
                actual_w //= 2
            output_hwc = output_hwc[:actual_h, :actual_w, :]
            
            if self.bitdepth == 'f32':
                jpg_path = os.path.join(aug_output_dir, image_name.replace('.jpg', '.jpg'))
                output_u8 = np.clip(output_hwc * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(output_u8)
                img.save(jpg_path)
            elif self.bitdepth == 'f16':
                jpg_path = os.path.join(aug_output_dir, image_name.replace('.jpg', '.jpg'))
                output_u8 = np.clip(output_hwc * 255, 0, 255).astype(np.uint8)
                img = Image.fromarray(output_u8)
                img.save(jpg_path)
            elif self.bitdepth == 'i8':
                jpg_path = os.path.join(aug_output_dir, image_name.replace('.jpg', '.jpg'))
                out_f = output_hwc.astype(np.float32)
                output_u8 = np.clip(out_f + 128.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(output_u8)
                img.save(jpg_path)
            else:
                # Ensure uint8 type for u8
                if output_hwc.dtype != np.uint8:
                    output_hwc = np.clip(output_hwc, 0, 255).astype(np.uint8)
                
                # Save as image for u8
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
        Handles both u8 and f32 formats.
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

    def _compare_output(self, output_tensor, ref_data, img_idx, aug_name=None):
        """
        Compare output tensor with reference from batch (handles u8 and f32).
        
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
            if self.bitdepth in ['f32']:
                # Handle floating point comparison
                dtype = np.float32 if self.bitdepth == 'f32' else np.float16
                diff = np.abs(output_roi.astype(dtype) - ref_roi.astype(dtype))
                max_diff = float(diff.max())
                mismatched = np.sum(diff > self.config.TOLERANCE)
                total_pixels = output_roi.size
                
                stats = {
                    "max_diff": max_diff,
                    "mismatched_pixels": int(mismatched),
                    "total_pixels": int(total_pixels),
                    "match_percentage": 100.0 * (total_pixels - mismatched) / total_pixels
                }
                
                # Pass if all pixels within tolerance
                passed = max_diff <= self.config.TOLERANCE
            else:
                # For u8, compare as integers
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
            result_line = f"{aug_name}_img{image_idx}_{self.bitdepth}: {status}"
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
                
            summary_line = f"{aug_name} ({self.bitdepth}) Percentage: {status} ({percentage:.0f}%)\n"
            self.qa_file.write(summary_line)
            self.qa_file.flush()
            
            # Clear for next augmentation
            self.current_aug_results = []
    
    def _write_perf_result(self, aug_name, times_dict, layout="PKD3"):
        """Write performance result to log file (only write once, not per layout)"""
        # Only write for PKD3 to avoid duplication (since all layouts use the same file)
        if layout == "PKD3" and hasattr(self, 'perf_log_file'):
            func_group = func_group_finder(aug_name)
            self.perf_log_file.write(f"\n{func_group} ({self.bitdepth})\n")
            self.perf_log_file.write(f"Running {aug_name} {self.num_runs} times\n")
            self.perf_log_file.write(f"max,min,avg wall times in ms/batch = {times_dict['max']:.2f},{times_dict['min']:.2f},{times_dict['avg']:.2f}\n")
            self.perf_log_file.flush()
    
    def cleanup(self):
        """Close all open files"""
        # Close QA file only if we own it
        if self.qa_file and self.owns_qa_file:
            self.qa_file.close()
        
        # Close performance log file only if we own it
        if hasattr(self, 'perf_log_file') and self.perf_log_file and self.owns_perf_log:
            self.perf_log_file.close()
    
    def set_shared_perf_log(self, perf_log_file):
        """Set a shared performance log file from external source"""
        self.perf_log_file = perf_log_file
        self.owns_perf_log = False  # We don't own this file, so won't close it
        # All layouts use the same file handle
        for layout in Layout:
            self.perf_logs[layout.name] = perf_log_file
    
    def set_shared_qa_file(self, qa_file, qa_output_dir):
        """Set a shared QA file from external source"""
        self.qa_file = qa_file
        self.qa_output_dir = qa_output_dir
        self.owns_qa_file = False  # We don't own this file, so won't close it
    
    # =========================================================================
    # AUGMENTATION FUNCTIONS (Combined Unit + QA with f32 support)
    # =========================================================================
    
    def test_brightness(self):
        aug_name = "brightness"
        device = 'cuda' if self.backend == HIP else 'cpu' 
        print(f" Brightness Augmentation ({self.bitdepth})")
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
                print(f"  Loaded reference: {len(ref_data)} bytes ({self.bitdepth})")
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"   {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"   {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth}, max_diff={stats['max_diff']})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f" Gamma Correction Augmentation ({self.bitdepth})")
        
        # Load reference data for QA mode
        ref_data = None
        if self.mode in ["QA", "ALL"]:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR,
                aug_name,
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f" Flip Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f" Resize Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor_interpolationTypeBilinear.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
                # Get actual dimensions for ROI
                actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                roi_widths = [actual_w //2]
                roi_heights = [actual_h //2]

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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f"  Crop Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f"  Hue Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f"  Rotate Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor_interpolationTypeBilinear.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status} ")
        
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
        print(f"  Contrast Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f" Vignette Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status} ")
        
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
        print(f" Pixelate Augmentation ({self.bitdepth})")
        
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
                f"{aug_name}_{self.bitdepth}_Tensor.bin"
            )
            if os.path.exists(ref_path):
                dtype = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)
            
            # Clear results tracking for this augmentation
            self.current_aug_results = []
        
        success_count = 0
        total = len(self.test_images)
        
        # Process each test image
        for idx, img_path in enumerate(self.test_images):
            image_name = os.path.basename(img_path)
            
            try:
                # Load image with specified bitdepth
                image = load_image_with_bitdepth(img_path, bitdepth=self.bitdepth, device=device)
                
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
                        print(f"    {image_name} : SAVED ({self.bitdepth})")
                        if self.mode == "UNIT":
                            success_count += 1
                    else:
                        print(f"    {image_name} : SAVE FAILED")
                
                # QA MODE: Compare with reference
                if self.mode in ["QA", "ALL"] and ref_data is not None:
                    passed, stats = self._compare_output(output, ref_data, idx, aug_name)
                    self._write_qa_result(aug_name, idx, passed, stats)
                    
                    if passed:
                        print(f"    {image_name} : QA PASS ({self.bitdepth})")
                        if self.mode == "QA" or self.mode == "ALL":
                            success_count += 1
                    else:
                        print(f"    {image_name} : QA FAIL ({self.bitdepth})")
                
            except Exception as e:
                print(f"    {image_name} : ERROR → {e}")
        
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
        
        print(f"\n  RESULT: {status}")
        
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
        print(f"UNIT TESTS - Image Generation ({self.backend_name}, {self.bitdepth})")
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
        print(f"PERFORMANCE TESTS ({self.backend_name}, {self.bitdepth})")
        print(f"{'='*70}\n")
        
        device = 'cuda' if self.backend == HIP else 'cpu'
        num_iterations = self.num_runs if self.num_runs > 1 else 100  # Use num_runs parameter
        warmup_iterations = 5
        
        # Test image for performance (using 100x100 image)
        test_image = load_image_with_bitdepth(self.test_images[1], bitdepth=self.bitdepth, device=device)
        
        print(f"Running {num_iterations} iterations per augmentation (after {warmup_iterations} warmup runs)\n")
        print(f"BitDepth: {self.bitdepth}\n")
        
        # Define all available performance tests
        all_perf_tests = [
            (0, 'brightness', lambda: fn.brightness(test_image, alpha=1.75, beta=50.0 if self.bitdepth == 'u8' else 50.0/255.0, backend=self.backend)),
            (1, 'gamma_correction', lambda: fn.gamma_correction(test_image, gamma=1.9, backend=self.backend)),
            (20, 'flip', lambda: fn.flip(test_image, horizontal=True, vertical=False, backend=self.backend)),
            (21, 'resize', lambda: fn.resize(test_image, width=50, height=50, backend=self.backend)),
            (37, 'crop', lambda: fn.crop(test_image, x1=10, y1=10, crop_width=50, crop_height=50, backend=self.backend)),
            (42, 'hue', lambda: fn.hue(test_image, hue_shift=60.0, backend=self.backend)),
            (23, 'rotate', lambda: fn.rotate(test_image, angle=50.0, backend=self.backend)),
            (4, 'contrast', lambda: fn.contrast(test_image, contrast_factor=2.96, contrast_center=128.0 if self.bitdepth == 'u8' else 128.0/255.0, backend=self.backend)),
            (46, 'vignette', lambda: fn.vignette(test_image, intensity=6.0, backend=self.backend)),
            (5, 'pixelate', lambda: fn.pixelate(test_image, pixelation_percentage=87.5, backend=self.backend))
        ]
        
        # Filter tests based on case_list if provided
        if self.case_list:
            perf_tests = [(name, func) for case_id, name, func in all_perf_tests if case_id in self.case_list]
        else:
            perf_tests = [(name, func) for case_id, name, func in all_perf_tests]
        
        # Report which tests will be run
        if self.case_list:
            print(f"Running tests for cases: {self.case_list}")
            print(f"Functions to test: {[name for name, _ in perf_tests]}\n")
        
        total_tests = len(perf_tests)
        for i, (func_name, func_call) in enumerate(perf_tests, 1):
            print(f"  [{i}/{total_tests}] Testing {func_name}...", end=" ")
            
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
        print(f"TEST SUMMARY ({self.mode} mode, {self.bitdepth})")
        print(f"{'='*70}")
        
        if self.mode in ["UNIT", "ALL"] and self.results['unit']:
            passed = sum(1 for _, r in self.results['unit'] if r is True)
            failed = sum(1 for _, r in self.results['unit'] if r is False)
            skipped = sum(1 for _, r in self.results['unit'] if r is None)
            print(f"\nUNIT TESTS:")
            print(f"  BitDepth: {self.bitdepth}")
            print(f"  Saved: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if self.mode in ["QA", "ALL"] and self.results['qa']:
            passed = sum(1 for _, r in self.results['qa'] if r is True)
            failed = sum(1 for _, r in self.results['qa'] if r is False)
            skipped = sum(1 for _, r in self.results['qa'] if r is None)
            print(f"\nQA TESTS:")
            print(f"  BitDepth: {self.bitdepth}")
            print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
        
        if self.mode in ["PERF", "ALL"] and self.results['perf']:
            valid = sum(1 for _, r in self.results['perf'] if r is not None)
            print(f"\nPERFORMANCE TESTS:")
            print(f"  BitDepth: {self.bitdepth}")
            print(f"  Completed: {valid}/{len(self.results['perf'])}")
            
            if valid > 0:
                # Show detailed performance results for each augmentation
                print(f"\nDetailed Performance Results ({self.bitdepth}):")
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
    
    # Determine backends to test
    backends_to_test = []
    if args.backend:
        # Backend was specified
        backend = HIP if args.backend == 'HIP' else HOST
        backend_name = args.backend
        
        # Check GPU availability for HIP backend
        if backend == HIP and not is_gpu_available():
            print(f"ERROR: HIP backend requested but GPU not available")
            return 1
        backends_to_test.append((backend, backend_name))
    else:
        # No backend specified, run both if possible
        backends_to_test.append((HOST, 'HOST'))
        if is_gpu_available():
            backends_to_test.append((HIP, 'HIP'))
        else:
            print("Note: GPU not available, skipping HIP backend")
    
    # Determine bitdepths to test
    bitdepths_to_test = []
    if args.bitdepth:
        # Bitdepth was specified - handle both single value and list
        if isinstance(args.bitdepth, list):
            bitdepths_to_test = args.bitdepth
        else:
            bitdepths_to_test = [args.bitdepth]
    else:
        # No bitdepth specified - always run all bitdepths by default
        bitdepths_to_test = ['u8', 'i8', 'f32', 'f16']
        print("Note: No bitdepth specified. Running tests for all bitdepths (u8, i8, f32, f16)")
    
    # Print header
    print("\n" + "="*70)
    print("RPP TEST SUITE - MULTI-BITDEPTH SUPPORT")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Backends to test: {[b[1] for b in backends_to_test]}")
    print(f"BitDepths to test: {bitdepths_to_test}")
    print(f"GPU Available: {is_gpu_available()}")
    if args.case_list:
        print(f"Case List: {args.case_list}")
    print("="*70)
    
    
    # Run tests for each backend and bitdepth combination
    overall_success = True
    qa_summaries = {}  # Store QA results for overall summary
    
    for backend, backend_name in backends_to_test:
        # Handle preserve_output - remove old directories if requested
        if args.preserve_output == 0:
            base_dir = os.getcwd()
            if args.mode == "UNIT" and not args.qa_mode:
                validate_and_remove_folders(base_dir, f"OUTPUT_IMAGES_{backend_name}")
            elif args.mode == "QA" or args.qa_mode:
                validate_and_remove_folders(base_dir, f"QA_RESULTS_{backend_name}")
            elif args.mode == "PERF":
                validate_and_remove_folders(base_dir, f"OUTPUT_PERFORMANCE_LOGS_{backend_name}")
        
        # Generate single timestamp per backend
        backend_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create single QA output directory and file per backend
        qa_file = None
        qa_output_dir = None
        if args.mode in ["QA", "ALL"] or args.qa_mode:
            qa_output_dir = f"QA_RESULTS_{backend_name}_{backend_timestamp}"
            os.makedirs(qa_output_dir, exist_ok=True)
            qa_file = open(os.path.join(qa_output_dir, "QA_results.txt"), "w")
            print(f"QA Output Directory: {qa_output_dir}")
        
        # Create single performance log directory and file per backend
        perf_log_file = None
        perf_output_dir = None
        if args.mode in ["PERF", "ALL"]:
            perf_output_dir = f"OUTPUT_PERFORMANCE_LOGS_{backend_name}_{backend_timestamp}"
            os.makedirs(perf_output_dir, exist_ok=True)
            perf_log_file = open(os.path.join(perf_output_dir, 
                                              f"Tensor_image_{backend_name.lower()}_raw_performance_log.txt"), "w")
            print(f"Performance Output Directory: {perf_output_dir}")
        
        # Track results for overall summary
        backend_qa_results = []
        
        for bitdepth in bitdepths_to_test:
            print(f"\n{'='*70}")
            print(f"Running tests: Backend={backend_name}, BitDepth={bitdepth}")
            print(f"{'='*70}")
            
            # Check if QA mode is requested for unsupported bitdepths
            if (args.mode == 'QA' or (args.mode == 'UNIT' and args.qa_mode)) and bitdepth in ['i8', 'f16']:
                print(f"Note: QA support is not available for {bitdepth} bitdepth. Skipping QA tests for {bitdepth}.")
                if args.mode == 'QA':
                    # Skip this bitdepth entirely for QA-only mode
                    continue
            
            current_mode_is_qa = args.mode in ['QA'] or args.qa_mode
            input_path_for_config = (
                args.default_input_path if current_mode_is_qa else args.input_path1
            )

            config = TestConfig(
                preserve_output=args.preserve_output,
                test_type=args.test_type if hasattr(args, 'test_type') else None,
                qa_mode=args.qa_mode,
                bitdepth=bitdepth,
                input_path=input_path_for_config,
            )
            config.timestamp = backend_timestamp
            
            # Run tests with structured output
            try:
                # Override QA file if it exists
                if qa_file:
                    # Temporarily override the config to not create new QA directory
                    original_get_output_dir = config.get_output_dir
                    config.get_output_dir = lambda bn, md: qa_output_dir if md == "QA" else original_get_output_dir(bn, md)
                
                test_suite = UnifiedTestSuite(
                    backend, 
                    args.mode, 
                    config, 
                    args.case_list,
                    num_runs=args.num_runs,
                    bitdepth=bitdepth
                )
                
                # Override the QA file to use the shared one
                if qa_file:
                    # Set the shared QA file
                    test_suite.set_shared_qa_file(qa_file, qa_output_dir)
                
                # If performance mode and shared backend log exists, use it
                if args.mode in ["PERF", "ALL"] and perf_log_file:
                    # Set the shared performance log file
                    test_suite.set_shared_perf_log(perf_log_file)
                    
                    # Write section header for this bitdepth
                    perf_log_file.write(f"\n{'='*50}\n")
                    perf_log_file.write(f"BitDepth: {bitdepth}\n")
                    perf_log_file.write(f"{'='*50}\n")
                    perf_log_file.flush()
                
                success = test_suite.run_all()
                
                # Store QA results for summary
                if hasattr(test_suite, 'results') and 'qa' in test_suite.results:
                    for aug_name, result in test_suite.results['qa']:
                        backend_qa_results.append((bitdepth, aug_name, result))
                
                # Cleanup will handle file closure correctly based on ownership
                test_suite.cleanup()
                
            except Exception as e:
                print(f"\nERROR for {backend_name}/{bitdepth}: {e}")
                import traceback
                traceback.print_exc()
                overall_success = False
        
        # Write overall QA summary for this backend
        if qa_file and backend_qa_results:
            qa_file.write("\n" + "="*70 + "\n")
            qa_file.write("OVERALL SUMMARY\n")
            qa_file.write("="*70 + "\n")
            
            # Group by augmentation
            aug_results = {}
            total_tests_requested = 0
            total_tests_passed = 0
            
            for bitdepth, aug_name, result in backend_qa_results:
                if aug_name not in aug_results:
                    aug_results[aug_name] = {}
                aug_results[aug_name][bitdepth] = "PASSED" if result else "FAILED"
            
            # Write summary in requested format
            for aug_name in sorted(aug_results.keys()):
                for bitdepth in bitdepths_to_test:
                    if bitdepth in aug_results[aug_name]:
                        qa_file.write(f"{bitdepth}_{aug_name}: {aug_results[aug_name][bitdepth]}\n")
            
            # Count total test cases
            qa_file.flush()
            qa_file_path = os.path.join(qa_output_dir, "QA_results.txt")
            with open(qa_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if '_img' in line and ': PASSED' in line:
                        total_tests_requested += 1
                        total_tests_passed += 1
                    elif '_img' in line and ': FAILED' in line:
                        total_tests_requested += 1
            
            # Write Final Results summary
            qa_file.write("\n")
            qa_file.write("Final Results of Tests:\n")
            qa_file.write(f"    - Total test cases including all subvariants REQUESTED = {total_tests_requested}\n")
            qa_file.write(f"    - Total test cases including all subvariants PASSED = {total_tests_passed}\n")
            
            # Add general information about test suite
            supported_augmentations = list(augmentationCaseMap.values())
            total_supported = len(supported_augmentations)
            non_qa_functions = [] 
            
            qa_file.write("\nGeneral information on test suite availability:\n")
            qa_file.write(f"    - Total augmentations supported in test suite = {total_supported}\n")
            qa_file.write(f"    - Total augmentations with golden output QA test support = {total_supported}\n")
            qa_file.write(f"    - Total augmentations without golden output QA test support (due to randomization involved) = {len(non_qa_functions)}\n")
            
            qa_file.close()
            print(f"\nQA results saved to: {qa_output_dir}/QA_results.txt")
            
            # Print the summary
            print("\n" + "="*70)
            print("Final Results of Tests:")
            print(f"    - Total test cases including all subvariants REQUESTED = {total_tests_requested}")
            print(f"    - Total test cases including all subvariants PASSED = {total_tests_passed}")
            print("\nGeneral information on test suite availability:")
            print(f"    - Total augmentations supported in test suite = {total_supported}")
            print(f"    - Total augmentations with golden output QA test support = {total_supported}")
            print(f"    - Total augmentations without golden output QA test support (due to randomization involved) = {len(non_qa_functions)}")
            print("="*70)
        
        # Close performance log for this backend
        if perf_log_file:
            perf_log_file.close()
            print(f"\nPerformance results saved to: {perf_output_dir}/Tensor_image_{backend_name.lower()}_raw_performance_log.txt")
            
            # Print performance summary
            print_performance_tests_summary(
                os.path.join(perf_output_dir, f"Tensor_image_{backend_name.lower()}_raw_performance_log.txt"),
                list(AugmentationGroupMap.keys()),
                args.num_runs
            )
                
    print("\n" + "="*70)
    print("OVERALL TEST SUMMARY")
    print("="*70)
    print(f"Backends tested: {[b[1] for b in backends_to_test]}")
    print(f"BitDepths tested: {bitdepths_to_test}")
    print(f"Overall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    print("="*70 + "\n")
    
    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())
