# py_rpp — Python Bindings for RPP

Python extension module (`py_rpp.so`) for [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives), built with pybind11. Runs on the **HIP (GPU) backend**.

---

## Requirements

- ROCm / TheRock install (via `install_therock.sh`)
- Python 3.8+
- pybind11 (`sudo apt install python3-pybind11`)
- numpy

---

## Build

```bash
# 1. Set up ROCm environment
source ~/amd-workspace/AMD-important-scripts/set_rocm_env/set_rocm_env.sh

# 2. Configure and build
cd ~/amd-workspace/AMD-stack/rpp/py_rpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

`cmake` auto-discovers `ROCM_PATH` from the TheRock state file — no manual flag needed.  
The compiled module lands at `build/py_rpp/py_rpp.cpython-*.so`.

---

## Usage

```python
import sys
sys.path.insert(0, "build")   # or install the package

import numpy as np
import py_rpp

# Batch of 2 RGB images, 64x64, NHWC layout
images = np.random.randint(0, 200, (2, 64, 64, 3), dtype=np.uint8)

alpha = np.array([1.5, 0.8], dtype=np.float32)   # per-image multiplier [0, 20]
beta  = np.array([10.0, 0.0], dtype=np.float32)  # per-image offset     [0, 255]

result = py_rpp.brightness(images, alpha, beta)
# result: np.ndarray, uint8, shape (2, 64, 64, 3)
# formula: output = clamp(alpha * input + beta, 0, 255)
```

---

## API

### `py_rpp.brightness(src, alpha, beta)`

Apply brightness augmentation to a batch of images on the GPU.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `src` | `np.uint8` | `(N, H, W, C)` | Input images, NHWC layout, C-contiguous |
| `alpha` | `np.float32` | `(N,)` | Per-image brightness multiplier `[0, 20]` |
| `beta` | `np.float32` | `(N,)` | Per-image brightness offset `[0, 255]` |
| **returns** | `np.uint8` | `(N, H, W, C)` | Output images, same shape as `src` |

---

## Test

```bash
# From the py_rpp/ root (after building)
source ~/amd-workspace/AMD-important-scripts/set_rocm_env/set_rocm_env.sh
pytest tests/test_brightness.py -v
```

`set_rocm_env.sh` must be sourced before running tests so that `librpp.so` is on `LD_LIBRARY_PATH`.

---

## Structure

```
py_rpp/
  CMakeLists.txt          — build system (pybind11 + RPP + HIP)
  conftest.py             — pytest sys.path setup
  src/
    py_rpp.cpp            — pybind11 binding
  py_rpp/
    __init__.py           — Python package init
  tests/
    test_brightness.py    — pytest tests for brightness
```

---

## Extending

To add a new op:

1. Add a function in `src/py_rpp.cpp` following the `brightness()` pattern
2. Register it in `PYBIND11_MODULE`
3. Re-export it in `py_rpp/__init__.py`
4. Add a test file under `tests/`
