# PyRPP - Python Bindings for AMD ROCm Performance Primitives

> [!NOTE]
> PyRPP provides Python bindings for AMD's ROCm Performance Primitives (RPP) library, enabling GPU-accelerated image augmentations with a simple, intuitive API designed to be easy to use. The documentation source files reside in the `rpp_pybind` folder of this repository.

AMD PyRPP is a comprehensive, high-performance Python interface for computer vision augmentations on AMD processors with `HIP` (GPU) and `HOST` (CPU) backends.

<p align="center"><img width="35%" src="https://github.com/ROCm/rpp/raw/master/docs/data/rpp_structure_4.png" /></p>

## Supported functionalities and variants

PyRPP provides 10 core augmentations across different categories:

### Color Augmentations (4)
| Function | Description | Parameters |
|----------|-------------|------------|
| **brightness** | Adjust image brightness | `alpha` (0-20), `beta` (0-255) |
| **gamma_correction** | Non-linear brightness adjustment | `gamma` (>0) |
| **contrast** | Adjust image contrast | `contrast_factor` (>0), `contrast_center` (0-255) |
| **hue** | Shift hue values for RGB images | `hue_shift` (0-359) |

### Geometric Augmentations (4)
| Function | Description | Parameters |
|----------|-------------|------------|
| **flip** | Flip images horizontally/vertically | `horizontal` (bool), `vertical` (bool) |
| **resize** | Resize images with bilinear interpolation | `width`, `height` |
| **rotate** | Rotate images by angle | `angle` (degrees) |
| **crop** | Extract rectangular region | `x1`, `y1`, `crop_width`, `crop_height` |

### Effects Augmentations (2)
| Function | Description | Parameters |
|----------|-------------|------------|
| **vignette** | Add darkening effect around edges | `intensity` (0.0-1.0) |
| **pixelate** | Create pixelated/mosaic effect | `pixelation_percentage` (0-100) |

## Prerequisites

### Operating Systems
* Linux
  * Ubuntu - `22.04` / `24.04`
  * RedHat - `8` / `9`
  * SLES - `15 SP7`

### Hardware
* **CPU**: [AMD64](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
* **GPU**: [AMD Radeon™ Graphics](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) / [AMD Instinct™ Accelerators](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)

> [!IMPORTANT] 
> * [ROCm-supported hardware required for HIP backend](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
> * `gfx908` or higher GPU required for GPU backend
> * Install ROCm `6.0.0` or later with [amdgpu-install](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html): **Required** usecase:`rocm`

### Python Requirements
* Python Version `3.8` or later
  ```shell
  python3 --version
  ```

### Dependencies

#### System Libraries
* libturbojpeg for JPEG operations
  ```shell
  sudo apt install libturbojpeg0-dev
  ```

#### Python Packages
* NumPy - Array operations
* PyTorch - Tensor operations
* PyTurboJPEG - Fast JPEG encoding/decoding
  ```shell
  pip install numpy torch PyTurboJPEG
  ```

> [!NOTE]
> All package installs are shown with the `apt` package manager. Use the appropriate package manager for your operating system.

## Installation instructions

The installation process uses the following steps:

* [Prerequisites installation](#prerequisites)
* [RPP library installation](#rpp-library-installation)
* [Python bindings installation](#python-bindings-installation)

### RPP Library Installation

> [!IMPORTANT]
> PyRPP requires the base RPP library to be installed first.

#### Package Install (Recommended)

##### Ubuntu
```shell
sudo apt install rpp rpp-dev
```

##### RHEL
```shell
sudo yum install rpp rpp-devel
```

##### SLES
```shell
sudo zypper install rpp rpp-devel
```

### Python Bindings Installation

#### Option 1: Build with RPP (Recommended)

```shell
# Clone RPP repository
git clone https://github.com/ROCm/rpp.git
cd rpp

# Build RPP with Python support
mkdir build && cd build
cmake -DRPP_PYPACKAGE=ON ..
make -j$(nproc)
sudo make install
```

#### Option 2: Install from PyPI

```shell
pip install rpp-pybind
```

#### Option 3: Development Install

```shell
# Clone repository
git clone https://github.com/ROCm/rpp.git
cd rpp/rpp_pybind

# Development install
pip install -e .
```

## Verify installation

### Quick Verification

```python
import rpp_pybind as rpp
import rpp_pybind.amd.rpp.rpp_types as rpp_type

# Check version
print(f"PyRPP Version: {rpp.__version__}")

# Check backend availability
print(f"GPU Available: {rpp_type.is_gpu_available()}")
print(f"Default Backend: {rpp_type.get_default_backend()}")
```

### Run Test Suite

The comprehensive test suite is located at `utilities/python_tests/test_suite.py` and provides Unit, QA, and Performance testing capabilities.

```bash
# Navigate to RPP root directory
cd /path/to/rpp

# Unit testing (saves output images)
python utilities/python_tests/test_suite.py --test_type 0 --backend HOST --bitdepth u8

# QA testing (compares with reference outputs)
python utilities/python_tests/test_suite.py --test_type 0 --backend HOST --bitdepth f32 --qa_mode 1

# Performance testing (timing measurements)
python utilities/python_tests/test_suite.py --test_type 1 --backend HIP --num_runs 100 --bitdepth f32

# Test specific augmentations
python utilities/python_tests/test_suite.py --test_type 0 --backend HOST --case_list 0 1 4  # brightness, gamma, contrast

# Test multiple bitdepths
python utilities/python_tests/test_suite.py --test_type 0 --backend HOST --bitdepth u8 f32 --qa_mode 1
```

#### Test Suite Options
- `--test_type`: 0 for Unit/QA tests, 1 for Performance tests
- `--backend`: HOST (CPU) or HIP (GPU)
- `--bitdepth`: u8, i8, f32, f16 (can specify multiple)
- `--qa_mode`: 1 to enable QA comparison mode
- `--num_runs`: Number of iterations for performance testing
- `--case_list`: Specific augmentation cases to test

## Usage Examples

### Basic Example

```python
import rpp_pybind as rpp
from rpp_pybind.amd.rpp import fn, utils, rpp_types

# Load images
images = utils.load_images(['image1.jpg', 'image2.jpg'])

# Apply augmentations
output = fn.brightness(images, alpha=1.5, beta=10.0)
output = fn.contrast(output, contrast_factor=1.5)
output = fn.resize(output, width=256, height=256)
output = fn.vignette(output, intensity=0.6)

# Save result
utils.save_image(output[0], 'augmented_result.jpg')
```

### Backend Selection

```python
# Auto-detect backend (GPU if available, else CPU)
output = fn.brightness(images, alpha=1.5)

# Explicitly use CPU backend
output = fn.brightness(images, alpha=1.5, backend=rpp_types.HOST)

# Explicitly use GPU backend
output = fn.brightness(images, alpha=1.5, backend=rpp_types.HIP)
```

### Batch Processing

```python
# Process multiple images with different parameters
horizontal_flips = [True, False, True, False]
vertical_flips = [False, True, False, True]

output = fn.flip(images, 
                 horizontal=horizontal_flips,
                 vertical=vertical_flips,
                 backend=types.HIP)
```

## API Reference

### Module Structure

```
rpp_pybind/
├── __init__.py       # Main module with C++ exports
├── amd/rpp/fn.py           # High-level augmentation functions
├── amd/rpp/rpp_types.py     # Type definitions and helpers
├── amd/rpp/utils.py        # Image loading/saving utilities
└── rpp_pybind.cpp   # C++ pybind11 bindings
```

### fn Module - Augmentation Functions

All functions accept:
- `images`: Input tensor(s)
- `backend`: Optional backend selection (`types.HOST` or `types.HIP`)
- Function-specific parameters

Returns:
- Augmented tensor(s) in the same format as input

### utils Module - Image Utilities

| Function | Description | Parameters |
|----------|-------------|------------|
| `load_image()` | Load single JPEG image | `path`, `device`, `apply_padding` |
| `load_images()` | Load batch of JPEG images | `paths`, `device`, `apply_padding` |
| `save_image()` | Save tensor as JPEG | `tensor`, `path` |
| `create_test_batch()` | Create random test images | `batch_size`, `height`, `width`, `channels`, `device` |

### types Module - Constants and Helpers

#### Enums
- **Backend**: `HOST` (CPU), `HIP` (GPU)
- **Layout**: `NCHW`, `NHWC`
- **DataType**: `U8`, `F32`, `F16`

#### Helper Functions
- `is_gpu_available()`: Check GPU availability
- `get_default_backend()`: Auto-select optimal backend

## Performance Optimization

PyRPP leverages RPP's optimized kernels:
- **CPU Backend**: OpenMP-accelerated SIMD implementations
- **GPU Backend**: HIP/ROCm kernels for AMD GPUs

### Performance Tips

1. **Use GPU backend** when available for maximum performance
2. **Process in batches** rather than individual images
3. **Reuse tensors** to minimize memory allocation overhead
4. **Chain operations** efficiently to minimize data transfers
