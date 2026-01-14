# PyRPP - Python Bindings for AMD ROCm Performance Primitives

Simple Python interface for RPP, designed to be easy to use like rocAL.

## Overview

PyRPP provides Python bindings for AMD's ROCm Performance Primitives (RPP) library, enabling GPU-accelerated image augmentations with a simple, intuitive API.

## Structure

```
rpp_pybind/
├── __init__.py       # Main module with C++ exports
├── fn.py            # High-level augmentation functions
├── types.py         # Type definitions and helpers
├── utils.py         # Image loading/saving utilities
├── rpp_pybind.cpp   # C++ pybind11 bindings
├── test_simple.py   # Simple test runner
└── example_simple.py # Usage examples
```

## Installation

### Prerequisites

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-dev libturbojpeg0-dev

# Install Python packages
pip install numpy torch PyTurboJPEG
```

### Build and Install

```bash
# Build RPP with Python support
cd rpp
mkdir build && cd build
cmake -DBUILD_PYPACKAGE=ON ..
make -j$(nproc)
sudo make install

# The Python module will be installed to your Python site-packages
```

## Usage

### Basic Example

```python
import rpp_pybind as rpp
from rpp_pybind import fn, utils, types

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
output = fn.brightness(images, alpha=1.5, backend=types.HOST)

# Explicitly use GPU backend
output = fn.brightness(images, alpha=1.5, backend=types.HIP)
```

## API Reference

### fn Module - Augmentation Functions

#### Color Augmentations (4)

- **brightness**(images, alpha=1.0, beta=0.0, backend=None)
  - Adjust image brightness
  - `alpha`: Brightness multiplier (0 to 20)
  - `beta`: Brightness offset (0 to 255)

- **gamma_correction**(images, gamma=1.0, backend=None)
  - Apply gamma correction
  - `gamma`: Gamma value (>0)

- **contrast**(images, contrast_factor=1.0, contrast_center=128.0, backend=None)
  - Adjust image contrast
  - `contrast_factor`: Contrast multiplier (>0)
  - `contrast_center`: Center value for contrast

- **hue**(images, hue_shift=0.0, backend=None)
  - Adjust hue for RGB images
  - `hue_shift`: Hue shift in degrees (0 to 359)
  - Note: RGB images only (3 channels)

#### Geometric Augmentations (4)

- **flip**(images, horizontal=False, vertical=False, backend=None)
  - Flip images horizontally and/or vertically
  - `horizontal`: Boolean or list
  - `vertical`: Boolean or list

- **resize**(images, width, height, backend=None)
  - Resize images to specified dimensions
  - `width`: Target width
  - `height`: Target height

- **rotate**(images, angle=0.0, backend=None)
  - Rotate images by given angle
  - `angle`: Rotation in degrees (positive = counter-clockwise)

- **crop**(images, x1, y1, crop_width, crop_height, backend=None)
  - Crop images to specified region
  - `x1`, `y1`: Top-left coordinates
  - `crop_width`, `crop_height`: Crop dimensions

#### Effects Augmentations (2)

- **vignette**(images, intensity=0.5, backend=None)
  - Apply vignette effect
  - `intensity`: Effect strength (0.0 to 1.0)

- **pixelate**(images, pixelation_percentage=50.0, backend=None)
  - Apply pixelate effect
  - `pixelation_percentage`: Pixelation level (0 to 100)

### utils Module - Image Utilities

- **load_image**(path, device='cpu', apply_padding=True)
  - Load single JPEG image as tensor

- **load_images**(paths, device='cpu', apply_padding=True)
  - Load batch of JPEG images

- **save_image**(tensor, path)
  - Save tensor as JPEG image

- **create_test_batch**(batch_size, height, width, channels=3, device='cpu')
  - Create random test images

### types Module - Enums and Constants

- **Backends**: `HOST` (CPU), `HIP` (GPU)
- **Layouts**: `NCHW`, `NHWC`
- **Data Types**: `U8`, `F32`, `F16`
- **Helper Functions**:
  - `is_gpu_available()`: Check if GPU is available
  - `get_default_backend()`: Auto-select backend

## Testing

```bash
# Run all tests
python test_simple.py

# Run specific test
python test_simple.py --test brightness

# Use GPU backend
python test_simple.py --backend HIP
```

## Examples

```bash
# Run examples
python example_simple.py
```

## Supported Augmentations

PyRPP provides 10 augmentations from different categories:

### Color Augmentations (4)
1. **Brightness** - Adjust image brightness with alpha/beta parameters
2. **Gamma Correction** - Non-linear brightness adjustment
3. **Contrast** - Adjust image contrast around a center value
4. **Hue** - Shift hue values for RGB images

### Geometric Augmentations (4)
5. **Flip** - Horizontal/vertical image flipping
6. **Resize** - Resize images using bilinear interpolation
7. **Rotate** - Rotate images by specified angle
8. **Crop** - Extract rectangular region from images

### Effects Augmentations (2)
9. **Vignette** - Add darkening effect around edges
10. **Pixelate** - Create pixelated/mosaic effect

## Direct C++ API Access

For advanced users, you can access the C++ functions directly:

```python
import rpp_pybind as rpp

# Create handle
handle = rpp.rppCreate(batch_size=4, backend=rpp.types.HOST)

# Call C++ function directly
rpp.brightness(input_tensor, output_tensor, alpha_list, beta_list, handle, backend)

# Destroy handle
rpp.rppDestroy(handle, backend)
```

## Performance

PyRPP leverages RPP's optimized kernels for both CPU and GPU backends:
- **CPU**: OpenMP-accelerated SIMD implementations
- **GPU**: HIP/ROCm kernels for AMD GPUs

## License

MIT License - see LICENSE file for details.
