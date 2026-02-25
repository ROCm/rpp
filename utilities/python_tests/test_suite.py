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
- Unit mode   : test_type=0, qa_mode=0  → Apply augmentation and save image
- QA mode     : test_type=0, qa_mode=1  → Apply augmentation and compare with reference tensor
- Performance : test_type=1             → Time measurements
- Supports u8, f32, f16, i8 bitdepths

Usage:
    python test_suite.py --test_type 0 --backend HOST --bitdepth u8 f16
    python test_suite.py --test_type 0 --backend HIP
    python test_suite.py --test_type 1 --backend HIP --num_runs 100 --bitdepth f32
    python test_suite.py --test_type 0 --qa_mode 1 --backend HOST --bitdepth u8 f32 
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
# import matplotlib.pyplot as plt                          # UNUSED: matplotlib not used anywhere in this file
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
# from rpp_pybind.amd.rpp.utils import convert_nchw_to_nhwc, convert_nhwc_to_nchwq   # UNUSED: replaced by util.convert_* calls
from rpp_pybind.amd.rpp.rpp_types import (
    is_gpu_available, get_default_backend, HOST, HIP
)

# =============================================================================
# ENUMS AND MAPPINGS
# =============================================================================

class TestType(Enum):
    """Test type enum
      UNIT_TEST        = test_type 0, qa_mode 0
      QA_TEST          = test_type 0, qa_mode 1
      PERFORMANCE_TEST = test_type 1
    """
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
    F16 = "f16"
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

# =============================================================================
# AUGMENTATION GROUPING
# =============================================================================

AugmentationGroupMap = {
    "color_augmentations":     ["brightness", "gamma_correction", "contrast", "hue"],
    "effects_augmentations":   ["pixelate", "vignette"],
    "geometric_augmentations": ["flip", "resize", "rotate", "crop"]
}

def get_augmentation_group(augmentation_name):
    """Get the group name for a given augmentation"""
    for group_name, augmentations in AugmentationGroupMap.items():
        if augmentation_name in augmentations:
            return group_name
    return "other_augmentations"

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

def directory_name_generator(backend, layout, aug_name, bitdepth="u8"):
    """Generate directory name based on backend, layout, augmentation and bitdepth"""
    func_group = func_group_finder(aug_name)
    return f"rpp_{backend.lower()}_{layout.lower()}_{bitdepth}_{func_group}"

def load_image_with_bitdepth(img_path, bitdepth='u8', device='cpu'):
    """
    Load image with specified bitdepth.

    Args:
        img_path : Path to image file
        bitdepth : 'u8', 'f32', 'f16', or 'i8'
        device   : 'cpu' or 'cuda'

    Returns:
        PyTorch tensor in specified bitdepth
    """
    image = util.load_image(img_path, device=device)
    if bitdepth == 'f32':
        image = image / 255.0
    elif bitdepth == 'f16':
        image = (image / 255.0).half()
    elif bitdepth == 'i8':
        image = (image - 128).to(torch.int8)
    else:
        image = image.to(torch.uint8)
    return image

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
    # current_category = ""                             # UNUSED: assigned but never read

    for line in f:
        if "max,min,avg wall times in ms/batch" in line:
            if "Running " in prevLine:
                splitWordStart = "Running "
                splitWordEnd = " " + str(numRuns)
                func_name = prevLine.partition(splitWordStart)[2].partition(splitWordEnd)[0]
                if func_name and func_name not in functions:
                    functions.append(func_name)
                    frames.append(numRuns)
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

    print("Functionalities - " + str(funcCount))

    headerFormat = "{:<70} {:<15} {:<15} {:<15} {:<15}"
    rowFormat    = "{:<70} {:<15} {:<15} {:<15} {:<15}"
    print("\n" + headerFormat.format("Functionality", "Frames Count", "max(ms/batch)", "min(ms/batch)", "avg(ms/batch)") + "\n")

    if len(functions) != 0:
        for i, func in enumerate(functions):
            if func:
                if func.startswith("---"):
                    print("\n" + func)
                else:
                    print(rowFormat.format(func, str(frames[i]), str(maxVals[i]), str(minVals[i]), str(avgVals[i])))
    else:
        print("No variants under this category")

    f.close()

def get_mode_str(test_type, qa_mode):
    """Return a short mode label for display purposes.

    test_type=0, qa_mode=0  →  'UNIT'
    test_type=0, qa_mode=1  →  'QA'
    test_type=1             →  'PERF'
    """
    if test_type == 1:
        return "PERF"
    return "QA" if qa_mode else "UNIT"

def test_suite_parser_and_validator():
    """Parse and validate command-line arguments.

    Mode is determined entirely by --test_type and --qa_mode:
      test_type=0, qa_mode=0  →  UNIT mode  (save images)
      test_type=0, qa_mode=1  →  QA   mode  (compare with reference)
      test_type=1             →  PERF mode  (timing)
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    default_input_path = os.path.join(script_path, "../test_suite/TEST_IMAGES/three_images_mixed_src1")

    case_min = min(augmentationCaseMap.keys())
    case_max = max(augmentationCaseMap.keys())

    parser = argparse.ArgumentParser(
        description='RPP Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit testing (test_type=0, qa_mode=0)
  python test_suite.py --test_type 0 --backend HOST --bitdepth u8

  # QA testing (test_type=0, qa_mode=1)
  python test_suite.py --test_type 0 --backend HOST --bitdepth f32 --qa_mode 1

  # Performance testing (test_type=1)
  python test_suite.py --test_type 1 --backend HIP --num_runs 100 --bitdepth f32
        """
    )

    parser.add_argument("--input_path1", type=str, default=default_input_path,
                        help="Path to input images")
    parser.add_argument("--input_path2", type=str, default=default_input_path,
                        help="Path to second input folder (for blend operations)")
    parser.add_argument("--case_start", type=int, default=case_min,
                        help=f"Start case number [{case_min}-{case_max}]")
    parser.add_argument("--case_end", type=int, default=case_max,
                        help=f"End case number [{case_min}-{case_max}]")
    parser.add_argument("--test_type", type=int, default=0,
                        help="0 = Unit/QA tests (governed by --qa_mode), 1 = Performance tests")
    parser.add_argument("--case_list", nargs="+",
                        help="Specific augmentations to test")
    parser.add_argument("--qa_mode", type=int, default=0,
                        help="0 = Unit mode (save images), 1 = QA mode (compare with reference). "
                             "Only effective when --test_type 0.")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of performance test iterations")
    parser.add_argument("--preserve_output", type=int, default=1,
                        help="0 = override previous outputs, 1 = preserve them")
    parser.add_argument("--batch_size", type=int, default=3,
                        help="Batch size for testing")
    parser.add_argument("--bitdepth",
                        choices=['u8', 'f32', 'f16', 'i8'],
                        nargs='+',
                        default=None,
                        help="Bit depth(s) to test. If omitted, all bitdepths are tested.")
    parser.add_argument('--backend',
                        choices=['HOST', 'HIP'],
                        help='Backend to use. If omitted, both HOST and HIP are tested.')

    args = parser.parse_args()

    # Validate paths
    if not validate_path(args.input_path1):
        print(f"Warning: input_path1 '{args.input_path1}' not found, falling back to default.")
        args.input_path1 = default_input_path
    if not validate_path(args.input_path2):
        args.input_path2 = default_input_path

    args.default_input_path = default_input_path

    # Validate case range
    args.case_start = max(case_min, min(args.case_start, case_max))
    args.case_end   = max(case_min, min(args.case_end,   case_max))
    if args.case_end < args.case_start:
        args.case_start, args.case_end = args.case_end, args.case_start

    # Process case list
    if args.case_list:
        valid_cases = []
        for case in args.case_list:
            if case.isdigit() and int(case) in augmentationCaseMap:
                valid_cases.append(int(case))
            else:
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

    # Validate test_type
    if args.test_type not in (0, 1):
        print(f"Invalid test_type: {args.test_type}. Must be 0 (Unit/QA) or 1 (Performance).")
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

    def __init__(self, preserve_output=1, test_type=0, qa_mode=0, bitdepth='u8',
                 input_path=None):
        self.DEFAULT_IMAGES_DIR = "../test_suite/TEST_IMAGES/three_images_mixed_src1"
        self.REFERENCE_DIR      = "../test_suite/REFERENCE_OUTPUT"

        # QA always uses default images; UNIT/PERF use input_path if valid
        if qa_mode:
            self.TEST_IMAGES_DIR = self.DEFAULT_IMAGES_DIR
        else:
            if input_path and validate_path(input_path):
                self.TEST_IMAGES_DIR = input_path
            else:
                self.TEST_IMAGES_DIR = self.DEFAULT_IMAGES_DIR

        # Tolerance per bitdepth
        self.TOLERANCE = {'u8': 1, 'f32': 0.01, 'f16': 0.1, 'i8': 1}.get(bitdepth, 1)

        self.preserve_output = preserve_output
        self.test_type  = test_type
        self.qa_mode    = qa_mode
        self.bitdepth   = bitdepth

        if qa_mode:
            # QA mode requires specific test images for comparison
            self.TEST_IMAGES = [
                "1_img50x50.jpg",
                "2_img100x100.jpg",
                "3_img150x150.jpg"
            ]
            self.IMAGE_SPECS = [(50, 50), (100, 100), (150, 150)]
        else:
            # Try to discover images dynamically for unit/perf testing
            if not self._discover_images():
                self.TEST_IMAGES = [
                    "1_img50x50.jpg",
                    "2_img100x100.jpg",
                    "3_img150x150.jpg"
                ]
                self.IMAGE_SPECS = [(50, 50), (100, 100), (150, 150)]

        self.LAYOUT_VARIANTS = [
            ('PKD3', 'PKD3'),
            ('PKD3', 'PLN3'),
            ('PLN3', 'PLN3'),
            ('PLN3', 'PKD3'),
            ('PLN1', 'PLN1'),
        ]

        # Reference batch dimensions
        self.BATCH_HEIGHT = 150
        self.BATCH_WIDTH  = 152
        self.BATCH_SIZE   = 3

        self.AUGMENTATION_PARAMS = {
            'brightness':       {'alpha': 1.75, 'beta': 50.0},
            'gamma_correction': {'gamma': 1.9},
            'contrast':         {'contrast_factor': 2.96, 'contrast_center': 128.0},
            'flip':             {'horizontal': True, 'vertical': False},
            'resize':           {},
            'crop':             {'x1': 10, 'y1': 10},
            'hue':              {'hue_shift': 60.0},
            'rotate':           {'angle': 50.0},
            'vignette':         {'intensity': 6.0},
            'pixelate':         {'pixelation_percentage': 87.5},
        }

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _discover_images(self):
        """Dynamically discover images in the input directory"""
        try:
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = [
                f for f in sorted(os.listdir(self.TEST_IMAGES_DIR))
                if f.lower().endswith(image_extensions)
            ]
            if not image_files:
                print(f"No image files found in {self.TEST_IMAGES_DIR}")
                return False

            self.TEST_IMAGES = []
            self.IMAGE_SPECS = []

            for idx, img_file in enumerate(image_files):
                img_path = os.path.join(self.TEST_IMAGES_DIR, img_file)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        self.TEST_IMAGES.append(img_file)
                        self.IMAGE_SPECS.append((height, width))
                except Exception as e:
                    print(f"  - [{idx+1}/{len(image_files)}] Failed to read {img_file}: {e}")

            if not self.TEST_IMAGES:
                return False

            self.BATCH_SIZE = len(self.TEST_IMAGES)
            return True

        except Exception as e:
            print(f"Error discovering images: {e}")
            return False

    def get_output_dir(self, backend_name, test_type, qa_mode):
        """Get output directory based on backend, test_type, and qa_mode."""
        now = datetime.now()
        ts  = f"{now.year}-{now.month}-{now.day:02d}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}"
        if test_type == 1:
            return f"OUTPUT_PERFORMANCE_LOGS_{backend_name}_{ts}"
        elif qa_mode:
            return f"QA_RESULTS_{backend_name}_{ts}"
        else:
            return f"OUTPUT_IMAGES_{backend_name}_{ts}"


# =============================================================================
# UNIFIED TEST CLASS
# =============================================================================

class UnifiedTestSuite:
    """Unified test suite driven by test_type and qa_mode."""

    def __init__(self, backend, test_type=0, qa_mode=0, config=None,
                 case_list=None, num_runs=1, bitdepth='u8', shared_output_dir=None):
        self.backend      = backend
        self.backend_name = "HIP" if backend == HIP else "HOST"
        self.test_type    = test_type   # 0 = Unit/QA, 1 = Perf
        self.qa_mode      = qa_mode     # 0 = Unit,    1 = QA   (only when test_type==0)
        self.config       = config if config else TestConfig(
                                test_type=test_type, qa_mode=qa_mode, bitdepth=bitdepth)
        self.case_list    = case_list
        self.num_runs     = num_runs
        self.bitdepth     = bitdepth
        self.results      = {'unit': [], 'qa': [], 'perf': []}

        # Case number → (name, method) mapping
        self.case_to_test_map = {
            0:  ('brightness',       self.test_brightness),
            1:  ('gamma_correction', self.test_gamma_correction),
            4:  ('contrast',         self.test_contrast),
            5:  ('pixelate',         self.test_pixelate),
            20: ('flip',             self.test_flip),
            21: ('resize',           self.test_resize),
            23: ('rotate',           self.test_rotate),
            37: ('crop',             self.test_crop),
            42: ('hue',              self.test_hue),
            46: ('vignette',         self.test_vignette),
        }

        # QA tracking
        self.qa_results          = []
        self.qa_file             = None
        self.qa_output_dir       = None
        self.owns_qa_file        = True
        self.current_aug_results = []

        # Performance log
        self.perf_logs       = {}
        self.perf_output_dir = None
        self.perf_log_file   = None
        self.owns_perf_log   = True

        # -----------------------------------------------------------------
        # Setup output directories
        # -----------------------------------------------------------------
        # Setup output directories
        if test_type == 0:                                  # Unit or QA
            if not qa_mode and shared_output_dir:
                # Use the shared directory passed from main()
                self.unit_output_dir = shared_output_dir
                print(f"Using Shared Unit Output Directory: {self.unit_output_dir}")
            else:
                # Create new directory (for QA mode or if no shared dir provided)
                self.unit_output_dir = self.config.get_output_dir(
                    self.backend_name, test_type, qa_mode)
                os.makedirs(self.unit_output_dir, exist_ok=True)
                label = 'QA' if qa_mode else 'Unit'
                print(f"{label} Output Directory: {self.unit_output_dir}")
                if qa_mode:
                    self.qa_output_dir = self.unit_output_dir
                    self.qa_file = None   # set via set_shared_qa_file

        elif test_type == 1:                                # Performance
            self.perf_output_dir = None                   # set via set_shared_perf_log

        # Load test image paths
        self.test_images = [
            os.path.join(self.config.TEST_IMAGES_DIR, img)
            for img in self.config.TEST_IMAGES
        ]

        mode_label = get_mode_str(test_type, qa_mode)

    # =========================================================================
    # IMAGE SAVING  (Unit mode)
    # =========================================================================

    def _save_output_image(self, tensor, augmentation_name, image_name, img_idx,
                           layout_variant=None, input_layout=None):
        """Save output image with correct bitdepth scaling."""
        try:
            group_name = get_augmentation_group(augmentation_name)

            if input_layout is None and layout_variant:
                input_layout, output_layout = layout_variant.split('-')
            elif layout_variant:
                output_layout = layout_variant.split('-')[1]
            else:
                output_layout = input_layout

            rpp_dir_name     = f"rpp_{self.backend_name.lower()}_{input_layout.lower()}_{group_name}"
            detailed_variant = (
                f"{augmentation_name}_{self.bitdepth}_Tensor_"
                f"{self.backend_name}_{input_layout}_to{output_layout}"
            )
            aug_output_dir = os.path.join(
                self.unit_output_dir, input_layout, rpp_dir_name, detailed_variant)
            os.makedirs(aug_output_dir, exist_ok=True)

            tensor_np = tensor.detach().cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)

            # Convert to HWC
            if tensor_np.ndim == 4:
                img = tensor_np[0]
                output_hwc = (img[0] if img.shape[0] == 1
                              else np.transpose(img, (1, 2, 0)) if img.shape[0] == 3
                              else img)
            elif tensor_np.ndim == 3:
                output_hwc = (tensor_np[0] if tensor_np.shape[0] == 1
                              else np.transpose(tensor_np, (1, 2, 0)) if tensor_np.shape[0] == 3
                              else tensor_np)
            else:
                output_hwc = tensor_np

            # Crop to ROI
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            if augmentation_name in ("crop", "resize"):
                actual_h //= 2
                actual_w //= 2
            output_hwc = (output_hwc[:actual_h, :actual_w, :]
                          if output_hwc.ndim == 3
                          else output_hwc[:actual_h, :actual_w])

            # Bitdepth → uint8
            if self.bitdepth == "u8":
                output_u8 = np.clip(output_hwc, 0, 255).astype(np.uint8)
            elif self.bitdepth in ("f32", "f16"):
                output_u8 = np.clip(output_hwc * 255, 0, 255).round().astype(np.uint8)
            elif self.bitdepth == "i8":
                output_u8 = np.clip(output_hwc.astype(np.float32) + 128.0,
                                    0, 255).round().astype(np.uint8)
            else:
                raise ValueError(f"Unsupported bitdepth: {self.bitdepth}")

            Image.fromarray(output_u8).save(os.path.join(aug_output_dir, image_name))
            return True

        except Exception as e:
            print(f"    ✗ Failed to save: {e}")
            return False

    # =========================================================================
    # REFERENCE EXTRACTION HELPERS
    # =========================================================================

    def _extract_from_batch_nhwc(self, batch_data, img_idx, extract_h=None, extract_w=None, aug_name=None):
        slot_height, slot_width = 150, 152
        slot_size = slot_height * slot_width * 3
        offset = img_idx * slot_size
        img_slot_reshaped = batch_data[offset:offset + slot_size].reshape(slot_height, slot_width, 3)

        if extract_h is None or extract_w is None:
            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
        else:
            actual_h, actual_w = extract_h, extract_w

        if aug_name in ('crop', 'resize'):
            actual_h //= 2
            actual_w //= 2
        return img_slot_reshaped[:actual_h, :actual_w, :]

    def _extract_from_batch_pln1(self, batch_data, img_idx, aug_name=None):
        slot_height, slot_width = 150, 152
        rgb_slot_size     = slot_height * slot_width * 3
        pln1_slot_size    = slot_height * slot_width
        pln1_offset_start = rgb_slot_size * 3
        offset = pln1_offset_start + img_idx * pln1_slot_size
        img_slot_reshaped = batch_data[offset:offset + pln1_slot_size].reshape(slot_height, slot_width)

        actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
        if aug_name in ('crop', 'resize'):
            actual_h //= 2
            actual_w //= 2
        return img_slot_reshaped[:actual_h, :actual_w]

    # =========================================================================
    # QA COMPARISON
    # =========================================================================

    def _compare_output(self, output_tensor, ref_data, img_idx, is_grayscale=False, aug_name=None):
        """Compare output tensor with reference. Returns (passed, stats_dict)."""
        try:
            output_np = (output_tensor.cpu().numpy()
                         if hasattr(output_tensor, 'cpu') else np.array(output_tensor))

            if is_grayscale:
                if output_np.ndim == 4:
                    output_hw = output_np[0, 0, :, :]
                elif output_np.ndim == 3:
                    output_hw = output_np[0, :, :]
                else:
                    output_hw = output_np
            else:
                if output_np.ndim == 4:
                    output_single = output_np[0]
                    output_hwc = (np.transpose(output_single, (1, 2, 0))
                                  if output_single.shape[0] == 3 else output_single)
                elif output_np.ndim == 3 and output_np.shape[0] == 3:
                    output_hwc = np.transpose(output_np, (1, 2, 0))
                else:
                    output_hwc = output_np

            actual_h, actual_w = self.config.IMAGE_SPECS[img_idx]
            if aug_name in ('crop', 'resize'):
                actual_h //= 2
                actual_w //= 2

            if is_grayscale:
                output_roi = output_hw[:actual_h, :actual_w]
                ref_roi    = self._extract_from_batch_pln1(ref_data, img_idx, aug_name=aug_name)
            else:
                output_roi = output_hwc[:actual_h, :actual_w, :]
                ref_roi    = self._extract_from_batch_nhwc(ref_data, img_idx, aug_name=aug_name)

            if output_roi.shape != ref_roi.shape:
                return False, {"error": f"Shape mismatch: output {output_roi.shape} vs ref {ref_roi.shape}"}

            if self.bitdepth == 'f32':
                diff       = np.abs(output_roi.astype(np.float32) - ref_roi.astype(np.float32))
                max_diff   = float(diff.max())
                mismatched = np.sum(diff > self.config.TOLERANCE)
            else:
                diff       = output_roi.astype(np.int16) - ref_roi.astype(np.int16)
                abs_diff   = np.abs(diff)
                max_diff   = int(abs_diff.max())
                mismatched = np.sum(abs_diff > self.config.TOLERANCE)

            total_pixels = output_roi.size
            stats = {
                "max_diff":          max_diff,
                "mismatched_pixels": int(mismatched),
                "total_pixels":      int(total_pixels),
                "match_percentage":  100.0 * (total_pixels - mismatched) / total_pixels,
            }
            return max_diff <= self.config.TOLERANCE, stats

        except Exception as e:
            return False, {"error": str(e)}

    def _write_qa_result(self, aug_name, image_idx, passed, stats=None):
        """Accumulate QA pass/fail for the current augmentation."""
        if self.qa_file:
            self.current_aug_results.append(passed)

    # =========================================================================
    # LAYOUT CONVERSION
    # =========================================================================

    def _convert_layout(self, tensor, from_layout, to_layout):
        if from_layout == to_layout:
            return tensor
        if from_layout == 'NCHW' and to_layout == 'NHWC':
            return util.convert_nchw_to_nhwc(tensor)
        if from_layout == 'NHWC' and to_layout == 'NCHW':
            return util.convert_nhwc_to_nchw(tensor)
        return tensor

    # =========================================================================
    # QA SUMMARY WRITER
    # =========================================================================

    def _write_aug_summary(self, aug_name, layout_str, timing_info=None):
        """Write per-augmentation QA summary to file and console."""
        if not self.qa_file:
            return

        backend_str  = "CPU" if self.backend == HOST else "GPU"
        backend_name = "HOST" if self.backend == HOST else "HIP"
        func_name    = f"{aug_name}_{self.bitdepth}_Tensor_{backend_name}_{layout_str}"

        header = (f"Running {func_name} 1 times "
                  f"(each time with a batch size of {self.config.BATCH_SIZE} images) "
                  f"and computing mean statistics...")
        print(f"\n{header}\n")
        self.qa_file.write(f"{header}\n\n")

        if timing_info:
            clock_line = f"{backend_str} Backend Clock Time: {timing_info['clock_time']:.3f} ms/batch"
            wall_line  = f"{backend_str} Backend Wall Time: {timing_info['wall_time']:.4f} ms/batch"
            print(clock_line)
            print(wall_line)
            print()
            self.qa_file.write(clock_line + "\n")
            self.qa_file.write(wall_line  + "\n\n")

        results_line = f"Results for {aug_name}_{self.bitdepth}_Tensor_{layout_str} :"
        print(results_line)
        self.qa_file.write(results_line + "\n")

        overall_passed = all(self.current_aug_results) if self.current_aug_results else True
        status = "PASSED!" if overall_passed else "FAILED!"
        print(status)
        self.qa_file.write(f"{status}\n")
        self.qa_file.write("\n" + "-" * 90 + "\n")
        self.qa_file.flush()
        self.current_aug_results = []

    # =========================================================================
    # PERFORMANCE RESULT WRITER
    # =========================================================================

    def _write_perf_result(self, aug_name, times_dict, layout_variant="PKD3-PKD3"):
        """Write performance result to log file for all layout variants."""
        if hasattr(self, 'perf_log_file') and self.perf_log_file:
            backend_str = "HIP" if self.backend == HIP else "HOST"
            
            # Include the full layout variant in the function name
            func_name = f"{aug_name}_{self.bitdepth}_Tensor_{backend_str}_{layout_variant.replace('-', '_to')}"
            
            # Write to log file
            self.perf_log_file.write(f"Running {func_name} {self.num_runs} times "
                f"(each time with a batch size of {self.config.BATCH_SIZE} images) "
                f"and computing mean statistics...\n")
            self.perf_log_file.write(
                f"max,min,avg wall times in ms/batch = "
                f"{times_dict['max']:.6f},{times_dict['min']:.6f},{times_dict['avg']:.6f}\n")
            self.perf_log_file.flush()
            
            # Also print to console for visibility
            print(f"  → Logged to file: max={times_dict['max']:.2f}ms, min={times_dict['min']:.2f}ms, avg={times_dict['avg']:.2f}ms")

    # =========================================================================
    # FILE LIFECYCLE
    # =========================================================================

    def cleanup(self):
        if self.qa_file and self.owns_qa_file:
            self.qa_file.close()
        if hasattr(self, 'perf_log_file') and self.perf_log_file and self.owns_perf_log:
            self.perf_log_file.close()

    def set_shared_perf_log(self, perf_log_file, perf_output_dir=None):
        self.perf_log_file  = perf_log_file
        self.owns_perf_log  = False
        if perf_output_dir:
            self.perf_output_dir = perf_output_dir
        for layout in Layout:
            self.perf_logs[layout.name] = perf_log_file

    def set_shared_qa_file(self, qa_file, qa_output_dir):
        self.qa_file       = qa_file
        self.qa_output_dir = qa_output_dir
        self.owns_qa_file  = False

    # =========================================================================
    # CORE AUGMENTATION TEST RUNNER
    # =========================================================================

    def _run_augmentation_test(self, aug_name, aug_function, aug_params, ref_file_suffix=""):
        """Run one augmentation across all layout variants."""
        device = 'cuda' if self.backend == HIP else 'cpu'

        # Load reference data once (QA only: test_type=0, qa_mode=1)
        ref_data = None
        if self.test_type == 0 and self.qa_mode:
            ref_path = os.path.join(
                self.config.REFERENCE_DIR, aug_name,
                f"{aug_name}_{self.bitdepth}_Tensor{ref_file_suffix}.bin"
            )
            if os.path.exists(ref_path):
                dtype    = np.float32 if self.bitdepth == 'f32' else np.uint8
                ref_data = np.fromfile(ref_path, dtype=dtype)

        overall_success_count = 0
        overall_total         = 0
        success               = True

        for input_layout, output_layout in self.config.LAYOUT_VARIANTS:
            start_clock  = time.perf_counter()
            start_wall   = time.time()
            variant_name = f"{input_layout}-{output_layout}"
            grayscale    = (input_layout.upper() == 'PLN1')

            if grayscale and aug_name in ('hue', 'saturation', 'color_twist', 'color_jitter'):
                print(f"  Skipping variant: {variant_name} (color augmentation not applicable to grayscale)")
                continue

            variant_success = 0

            for idx, img_path in enumerate(self.test_images):
                image_name = os.path.basename(img_path)
                if self.test_type == 0 and self.qa_mode:
                    overall_total += 1

                try:
                    image = util.load_image(img_path, grayscale=grayscale, device=device)
                    if self.bitdepth == 'f32':
                        image = image.to(torch.float32) / 255.0
                    elif self.bitdepth == 'f16':
                        image = (image.to(torch.float32) / 255.0).half()
                    elif self.bitdepth == 'i8':
                        image = (image - 128).to(torch.int8)
                    else:
                        image = image.to(torch.uint8)

                    if input_layout == 'PKD3':
                        image = self._convert_layout(image, 'NCHW', 'NHWC')
                        input_layout_str = 'NHWC'
                    else:
                        input_layout_str = 'NCHW'

                    output_layout_str = 'NHWC' if output_layout == 'PKD3' else 'NCHW'

                    actual_h, actual_w = self.config.IMAGE_SPECS[idx]
                    params = aug_params.copy()

                    if aug_name == 'crop':
                        params['crop_width']  = actual_w // 2
                        params['crop_height'] = actual_h // 2
                        roi_widths  = [actual_w // 2]
                        roi_heights = [actual_h // 2]
                    elif aug_name == 'resize':
                        params['width']  = actual_w // 2
                        params['height'] = actual_h // 2
                        roi_widths  = [actual_w // 2]
                        roi_heights = [actual_h // 2]
                    else:
                        roi_widths  = [actual_w]
                        roi_heights = [actual_h]

                    output = aug_function(
                        image,
                        roi_widths=roi_widths,
                        roi_heights=roi_heights,
                        input_layout=input_layout_str,
                        output_layout=output_layout_str,
                        backend=self.backend,
                        **params
                    )

                    # ---- Unit mode (test_type=0, qa_mode=0): save images ----
                    if self.test_type == 0 and not self.qa_mode:
                        if len(output.shape) == 4 and output.shape[1] == 3:
                            output_for_save = self._convert_layout(output, 'NCHW', 'NHWC')
                        else:
                            output_for_save = output

                        if self._save_output_image(output_for_save, aug_name, image_name, idx,
                                                   layout_variant=variant_name):
                            print(f"  {image_name} ({variant_name}): SAVED")
                            variant_success += 1
                        else:
                            print(f"  {image_name} ({variant_name}): SAVE FAILED")

                    # ---- QA mode (test_type=0, qa_mode=1): compare with reference ----
                    if self.test_type == 0 and self.qa_mode and ref_data is not None:
                        if grayscale:
                            output_for_qa = output
                        else:
                            if len(output.shape) == 4:
                                if output.shape[1] == 3:
                                    output_for_qa = self._convert_layout(output, 'NCHW', 'NHWC')
                                elif output.shape[3] == 3:
                                    output_for_qa = output
                                else:
                                    output_for_qa = output
                            else:
                                output_for_qa = output

                        passed, stats = self._compare_output(
                            output_for_qa, ref_data, idx,
                            is_grayscale=grayscale, aug_name=aug_name)

                        if passed:
                            variant_success += 1
                            overall_success_count += 1
                        else:
                            if "error" in stats:
                                print(f"  {variant_name}/{image_name}: QA FAIL - {stats['error']}")
                            else:
                                print(f"  {variant_name}/{image_name}: QA FAIL (diff={stats['max_diff']})")
                                print(f"  (tolerance: {self.config.TOLERANCE})")
                                print(f"  Mismatched: {stats['mismatched_pixels']}/{stats['total_pixels']} "
                                      f"({100-stats['match_percentage']:.2f}%)")

                except Exception as e:
                    print(f" {image_name} ({variant_name}): ERROR → {e}")
                    import traceback
                    traceback.print_exc()

            # Timing
            timing_info = {
                'clock_time': (time.perf_counter() - start_clock) * 1000,
                'wall_time':  (time.time()         - start_wall)  * 1000,
            }

            if self.test_type == 0 and self.qa_mode and self.qa_file:
                self._write_aug_summary(aug_name, variant_name, timing_info)

            print("\n" + "-" * 90)

            # Record per-variant results
            if self.test_type == 0 and self.qa_mode:
                success = overall_success_count == overall_total
                self.results['qa'].append((aug_name + "_" + variant_name, success))
            elif self.test_type == 0 and not self.qa_mode:
                self.results['unit'].append((aug_name, True))

        return success

    # =========================================================================
    # INDIVIDUAL AUGMENTATION TEST METHODS
    # =========================================================================

    def test_brightness(self):
        params = self.config.AUGMENTATION_PARAMS['brightness']
        print(f"  [brightness] (alpha={params['alpha']}, beta={params['beta']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('brightness', fn.brightness, params)

    def test_gamma_correction(self):
        params = self.config.AUGMENTATION_PARAMS['gamma_correction']
        print(f"  [gamma_correction] (gamma={params['gamma']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('gamma_correction', fn.gamma_correction, params)

    def test_flip(self):
        params = self.config.AUGMENTATION_PARAMS['flip']
        print(f"  [flip] (horizontal={params['horizontal']}, vertical={params['vertical']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('flip', fn.flip, params)

    def test_resize(self):
        params = self.config.AUGMENTATION_PARAMS['resize']
        print("  [resize] (per-image: width/2, height/2)")
        print("  " + "-" * 50)
        return self._run_augmentation_test('resize', fn.resize, params,
                                           ref_file_suffix='_interpolationTypeBilinear')

    def test_crop(self):
        params = self.config.AUGMENTATION_PARAMS['crop']
        print("  [crop] (x1=10, y1=10, per-image: width/2, height/2)")
        print("  " + "-" * 50)
        return self._run_augmentation_test('crop', fn.crop, params)

    def test_hue(self):
        params = self.config.AUGMENTATION_PARAMS['hue']
        print(f"  [hue] (hue_shift={params['hue_shift']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('hue', fn.hue, params)

    def test_rotate(self):
        params = self.config.AUGMENTATION_PARAMS['rotate']
        print(f"  [rotate] (angle={params['angle']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('rotate', fn.rotate, params,
                                           ref_file_suffix='_interpolationTypeBilinear')

    def test_contrast(self):
        params = self.config.AUGMENTATION_PARAMS['contrast']
        print(f"  [contrast] (factor={params['contrast_factor']}, center={params['contrast_center']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('contrast', fn.contrast, params)

    def test_vignette(self):
        params = self.config.AUGMENTATION_PARAMS['vignette']
        print(f"  [vignette] (intensity={params['intensity']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('vignette', fn.vignette, params)

    def test_pixelate(self):
        params = self.config.AUGMENTATION_PARAMS['pixelate']
        print(f"  [pixelate] (percentage={params['pixelation_percentage']})")
        print("  " + "-" * 50)
        return self._run_augmentation_test('pixelate', fn.pixelate, params)

    # =========================================================================
    # TEST DISPATCH HELPERS
    # =========================================================================

    def _get_filtered_tests(self):
        all_tests = [
            (0,  'brightness',       self.test_brightness),
            (1,  'gamma_correction', self.test_gamma_correction),
            (20, 'flip',             self.test_flip),
            (21, 'resize',           self.test_resize),
            (37, 'crop',             self.test_crop),
            (42, 'hue',              self.test_hue),
            (23, 'rotate',           self.test_rotate),
            (4,  'contrast',         self.test_contrast),
            (46, 'vignette',         self.test_vignette),
            (5,  'pixelate',         self.test_pixelate),
        ]
        if self.case_list is None:
            return all_tests
        return [(n, name, fn_) for n, name, fn_ in all_tests if n in self.case_list]

    def run_unit_tests(self):
        """Run augmentations in Unit mode (test_type=0, qa_mode=0)."""
        for _, name, test_func in self._get_filtered_tests():
            try:
                test_func()
            except Exception as e:
                print(f"ERROR in {test_func.__name__}: {e}")
            print("-" * 70)

    def run_qa_tests(self):
        """Run augmentations in QA mode (test_type=0, qa_mode=1)."""
        for _, name, test_func in self._get_filtered_tests():
            try:
                test_func()
            except Exception as e:
                print(f"ERROR in {test_func.__name__}: {e}")
            print("-" * 70)

    def _filter_test_cases(self, test_cases):
        """
        Apply case_list or case_start/case_end filtering
        to a list of (name, callable) test cases.
        """

        # Filter by explicit case_list
        if self.case_list:
            selected = [
                (name, fn) for name, fn in test_cases
                if name in self.case_list
            ]
            if not selected:
                print(f"WARNING: None of the specified cases found in performance tests: {self.case_list}")
            return selected

        # Filter by range
        if hasattr(self, 'case_start') and hasattr(self, 'case_end'):
            if self.case_start is not None and self.case_end is not None:
                return test_cases[self.case_start:self.case_end + 1]

        return test_cases

    def run_performance_tests(self):
        """Run performance tests (test_type=1)."""
        device            = 'cuda' if self.backend == HIP else 'cpu'
        num_iterations    = self.num_runs if self.num_runs > 1 else 100
        warmup_iterations = 5

        base_image = load_image_with_bitdepth(self.test_images[1], bitdepth=self.bitdepth, device=device)
        print(f"Running {num_iterations} iterations per augmentation "
              f"(after {warmup_iterations} warmup runs)\n")

        for input_layout, output_layout in self.config.LAYOUT_VARIANTS:
            variant_name = f"{input_layout}-{output_layout}"
            grayscale    = (input_layout.upper() == 'PLN1')
            print(f"\n--- Layout Variant: {variant_name} ---")

            skip_color_ops = grayscale

            if input_layout == 'PKD3':
                test_image       = self._convert_layout(base_image, 'NCHW', 'NHWC')
                input_layout_str = 'NHWC'
            else:
                test_image       = base_image
                input_layout_str = 'NCHW'

            output_layout_str = 'NHWC' if output_layout == 'PKD3' else 'NCHW'
            beta_val        = 50.0  if self.bitdepth in ['u8', 'i8'] else 50.0  / 255.0
            contrast_center = 128.0 if self.bitdepth in ['u8', 'i8'] else 128.0 / 255.0

            # Create a dictionary mapping case numbers to test functions
            all_perf_tests = {
                0:  ('brightness',       lambda: fn.brightness(test_image, alpha=1.75, beta=beta_val,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                1:  ('gamma_correction', lambda: fn.gamma_correction(test_image, gamma=1.9,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                20: ('flip',            lambda: fn.flip(test_image, horizontal=True, vertical=False,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                21: ('resize',          lambda: fn.resize(test_image, width=50, height=50,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                37: ('crop',            lambda: fn.crop(test_image, x1=10, y1=10, crop_width=50, crop_height=50,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                23: ('rotate',          lambda: fn.rotate(test_image, angle=50.0,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                4:  ('contrast',        lambda: fn.contrast(test_image, contrast_factor=2.96, contrast_center=contrast_center,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                46: ('vignette',        lambda: fn.vignette(test_image, intensity=6.0,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                5:  ('pixelate',        lambda: fn.pixelate(test_image, pixelation_percentage=87.5,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
                42: ('hue',             lambda: fn.hue(test_image, hue_shift=60.0,
                                        input_layout=input_layout_str, output_layout=output_layout_str, backend=self.backend)),
            }
            
            # FILTER based on case_list
            if self.case_list:
                perf_tests = [(all_perf_tests[case][0], all_perf_tests[case][1]) 
                              for case in self.case_list if case in all_perf_tests]
            else:
                # If no case_list, run all tests
                perf_tests = [(name, func) for _, (name, func) in sorted(all_perf_tests.items())]

            for func_name, func_call in perf_tests:
                if skip_color_ops and func_name in ['hue']:
                    continue
                print(f"Testing {func_name}...", end=" ")
                try:
                    for _ in range(warmup_iterations):
                        _ = func_call()
                        if self.backend == HIP:
                            torch.cuda.synchronize()

                    times = []
                    for _ in range(num_iterations):
                        start = time.perf_counter()
                        _ = func_call()
                        if self.backend == HIP:
                            torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)

                    timing_dict = {
                        'num_runs': num_iterations,
                        'min': np.min(times),
                        'max': np.max(times),
                        'avg': np.mean(times),
                    }
                    print(f"Avg: {timing_dict['avg']:.2f}ms")
                    self.results['perf'].append((f"{func_name}_{variant_name}", timing_dict))
                    self._write_perf_result(func_name, timing_dict, variant_name)

                except Exception as e:
                    print(f"ERROR: {e}")
                    self.results['perf'].append((f"{func_name}_{variant_name}", None))

    # =========================================================================
    # TOP-LEVEL ENTRY POINT
    # =========================================================================

    def run_all(self):
        """Dispatch to the correct runner based on test_type and qa_mode."""
        if self.test_type == 1:
            self.run_performance_tests()
        elif self.test_type == 0 and self.qa_mode:
            self.run_qa_tests()
        elif self.test_type == 0 and not self.qa_mode:
            self.run_unit_tests()
        else:
            print(f"ERROR: Invalid combination test_type={self.test_type}, qa_mode={self.qa_mode}")
            return False

        self._print_summary()
        return True

    def _print_summary(self):
        mode_label = get_mode_str(self.test_type, self.qa_mode)
        # print(f"\n{'-'*30}")
        # print(f"TEST SUMMARY ({mode_label} mode, {self.bitdepth})")
        # print(f"{'-'*30}")

        if self.results['unit']:
            passed  = sum(1 for _, r in self.results['unit'] if r is True)
            failed  = sum(1 for _, r in self.results['unit'] if r is False)
            skipped = sum(1 for _, r in self.results['unit'] if r is None)
            # print(f"\nUNIT TESTS:  BitDepth={self.bitdepth}  Saved={passed}  Failed={failed}  Skipped={skipped}")

        if self.results['qa']:
            passed  = sum(1 for _, r in self.results['qa'] if r is True)
            failed  = sum(1 for _, r in self.results['qa'] if r is False)
            skipped = sum(1 for _, r in self.results['qa'] if r is None)
            # print(f"\nQA TESTS:    BitDepth={self.bitdepth}  Passed={passed}  Failed={failed}  Skipped={skipped}")

        if self.results['perf']:
            valid = sum(1 for _, r in self.results['perf'] if r is not None)
            print(f"\nPERFORMANCE TESTS:  BitDepth={self.bitdepth}  Completed={valid}/{len(self.results['perf'])}")
            if valid > 0:
                print(f"\nDetailed Performance Results ({self.bitdepth}):")
                print(f"  {'Augmentation':<25} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Runs':<8}")
                print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
                for aug_name, timing in self.results['perf']:
                    if timing:
                        print(f"  {aug_name:<25} {timing['avg']:<12.3f} {timing['min']:<12.3f} "
                              f"{timing['max']:<12.3f} {timing.get('num_runs',100):<8}")
                    else:
                        print(f"  {aug_name:<25} {'FAILED':<12} {'-':<12} {'-':<12} {'-':<8}")

        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = test_suite_parser_and_validator()

    # ------------------------------------------------------------------
    # Determine backends
    # ------------------------------------------------------------------
    backends_to_test = []
    if args.backend:
        backend      = HIP if args.backend == 'HIP' else HOST
        backend_name = args.backend
        if backend == HIP and not is_gpu_available():
            print("ERROR: HIP backend requested but GPU not available")
            return 1
        backends_to_test.append((backend, backend_name))
    else:
        backends_to_test.append((HOST, 'HOST'))
        if is_gpu_available():
            backends_to_test.append((HIP, 'HIP'))
        else:
            print("Note: GPU not available, skipping HIP backend")

    # ------------------------------------------------------------------
    # Determine bitdepths
    # ------------------------------------------------------------------
    if args.bitdepth:
        bitdepths_to_test = args.bitdepth if isinstance(args.bitdepth, list) else [args.bitdepth]
    else:
        bitdepths_to_test = ['u8', 'i8', 'f32', 'f16']
        print("Note: No bitdepth specified. Running tests for all bitdepths (u8, i8, f32, f16)")

    # ------------------------------------------------------------------
    # Print header
    # ------------------------------------------------------------------
    mode_label = get_mode_str(args.test_type, args.qa_mode)
    print("\n" + "="*70)
    print("RPP TEST SUITE")
    print("="*70)
    print(f"test_type:         {args.test_type}  ({mode_label} mode)")
    print(f"qa_mode:           {args.qa_mode}")
    print(f"Backends to test:  {[b[1] for b in backends_to_test]}")
    print(f"BitDepths to test: {bitdepths_to_test}")
    print(f"GPU Available:     {is_gpu_available()}")
    if args.case_list:
        print(f"Case List:         {args.case_list}")
    print("="*70)

    overall_success = True
    # qa_summaries = {}   # UNUSED: reserved for future multi-backend summary aggregation

    for backend, backend_name in backends_to_test:
        if args.preserve_output == 0:
            base_dir = os.getcwd()
            validate_and_remove_folders(base_dir, f"OUTPUT_IMAGES_{backend_name}")
            validate_and_remove_folders(base_dir, f"QA_RESULTS_{backend_name}")
            validate_and_remove_folders(base_dir, f"OUTPUT_PERFORMANCE_LOGS_{backend_name}")

        backend_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        unit_output_dir = None
        if args.test_type == 0 and not args.qa_mode:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            unit_output_dir = f"OUTPUT_IMAGES_{backend_name}_{timestamp}"
            os.makedirs(unit_output_dir, exist_ok=True)
            print(f"Shared Unit Output Directory: {unit_output_dir}")

        # Create shared QA file  (test_type=0, qa_mode=1)
        qa_file       = None
        qa_output_dir = None
        if args.test_type == 0 and args.qa_mode:
            qa_output_dir = f"QA_RESULTS_{backend_name}_{backend_timestamp}"
            os.makedirs(qa_output_dir, exist_ok=True)
            qa_file = open(os.path.join(qa_output_dir, "QA_results.txt"), "w")
            print(f"QA Output Directory: {qa_output_dir}")

        # Create shared performance log  (test_type=1)
        perf_log_file  = None
        perf_output_dir = None
        if args.test_type == 1:
            perf_output_dir = f"OUTPUT_PERFORMANCE_LOGS_{backend_name}_{backend_timestamp}"
            os.makedirs(perf_output_dir, exist_ok=True)
            perf_log_file = open(
                os.path.join(perf_output_dir,
                             f"Tensor_image_{backend_name.lower()}_raw_performance_log.txt"), "w")
            print(f"Performance Output Directory: {perf_output_dir}")

        backend_qa_results = []

        for bitdepth in bitdepths_to_test:

            # QA not supported for i8/f16
            if args.test_type == 0 and args.qa_mode and bitdepth in ['i8', 'f16']:
                print(f"Note: QA not available for {bitdepth}. Skipping.")
                continue

            input_path_for_config = (
                args.default_input_path if (args.test_type == 0 and args.qa_mode)
                else args.input_path1
            )

            config = TestConfig(
                preserve_output=args.preserve_output,
                test_type=args.test_type,
                qa_mode=args.qa_mode,
                bitdepth=bitdepth,
                input_path=input_path_for_config,
            )
            config.timestamp = backend_timestamp

            try:
                # Override get_output_dir so QA mode reuses the shared directory
                if qa_file:
                    _orig = config.get_output_dir
                    config.get_output_dir = lambda bn, tt, qm: (
                        qa_output_dir if (tt == 0 and qm) else _orig(bn, tt, qm))

                test_suite = UnifiedTestSuite(
                    backend,
                    test_type=args.test_type,
                    qa_mode=args.qa_mode,
                    config=config,
                    case_list=args.case_list,
                    num_runs=args.num_runs,
                    bitdepth=bitdepth,
                    shared_output_dir=unit_output_dir
                )

                if qa_file:
                    test_suite.set_shared_qa_file(qa_file, qa_output_dir)

                if args.test_type == 1 and perf_log_file:
                    test_suite.set_shared_perf_log(perf_log_file, perf_output_dir)
                    perf_log_file.write(f"\n{'='*50}\nBitDepth: {bitdepth}\n{'='*50}\n")
                    perf_log_file.flush()

                success = test_suite.run_all()

                # Collect QA results for summary
                for aug_name, result in test_suite.results.get('qa', []):
                    backend_qa_results.append((bitdepth, aug_name, result))

                test_suite.cleanup()

            except Exception as e:
                print(f"\nERROR for {backend_name}/{bitdepth}: {e}")
                import traceback
                traceback.print_exc()
                overall_success = False

        # ------------------------------------------------------------------
        # Write overall QA summary
        # ------------------------------------------------------------------
        if qa_file and backend_qa_results:
            qa_file.write("\n" + "-"*40 + " Summary of QA Test \n" + "-"*40 + "\n")
            print("\n" + "-"*40 + " Summary of QA Test " + "-"*40 + "\n")

            aug_results           = {}
            total_tests_requested = 0
            total_tests_passed    = 0

            for bitdepth, aug_name, result in backend_qa_results:
                aug_results.setdefault(aug_name, {})[bitdepth] = "PASSED" if result else "FAILED"

            for aug_name in sorted(aug_results):
                for bitdepth in bitdepths_to_test:
                    if bitdepth in aug_results[aug_name]:
                        line = f"{bitdepth}_{aug_name}: {aug_results[aug_name][bitdepth]}"
                        qa_file.write(line + "\n")
                        print(line)

            for bitdepth, aug_name, result in backend_qa_results:
                total_tests_requested += 1
                if result:
                    total_tests_passed += 1

            total_supported  = len(augmentationCaseMap)
            non_qa_functions = []

            summary_lines = [
                "",
                "Final Results of Tests:",
                f"    - Total test cases including all subvariants REQUESTED = {total_tests_requested}",
                f"    - Total test cases including all subvariants PASSED = {total_tests_passed}",
                "",
                "General information on Tensor test suite availability:",
                f"    - Total augmentations supported in Tensor test suite = {total_supported}",
                f"    - Total augmentations with golden output QA test support = {total_supported}",
                f"    - Total augmentations without golden output QA test support "
                f"(due to randomization involved) = {len(non_qa_functions)}",
            ]
            for line in summary_lines:
                qa_file.write(line + "\n")
                print(line)

            qa_file.close()
            print(f"\nQA results saved to: {qa_output_dir}/QA_results.txt")

        # ------------------------------------------------------------------
        # Close performance log
        # ------------------------------------------------------------------
        if perf_log_file:
            perf_log_file.close()
            log_path = os.path.join(perf_output_dir,
                                    f"Tensor_image_{backend_name.lower()}_raw_performance_log.txt")
            print(f"\nPerformance results saved to: {log_path}")
            print_performance_tests_summary(log_path, list(AugmentationGroupMap.keys()), args.num_runs)

    print("\n" + "-"*30 + " OVERALL TEST SUMMARY " + "-"*40)
    print(f"Backends tested:  {[b[1] for b in backends_to_test]}")
    print(f"BitDepths tested: {bitdepths_to_test}")
    print(f"Overall Result:   {'SUCCESS' if overall_success else 'FAILURE'}")
    print("-"*80 + "\n")

    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())
