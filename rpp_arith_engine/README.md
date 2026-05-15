# rpp_arith_engine

A high-level GPU tensor engine built on top of [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives).

Manages its own HIP device buffer and RPP handle. Exposes C++ operator overloading for tensor arithmetic and a chainable API for color augmentations — all executing on the GPU.

---

## Tensor format

All tensors are **float32, NHWC layout** (Batch × Height × Width × Channels).  
Pixel values are expected in **[0, 1]**.

---

## Build

```bash
# 1. Set up ROCm environment
source ~/amd-workspace/AMD-important-scripts/set_rocm_env/set_rocm_env.sh

# 2. Configure and build
cd ~/amd-workspace/AMD-stack/rpp/rpp_arith_engine
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This produces:
- `build/librpp_arith_engine.so` — the shared library
- `build/test_arith_engine`      — the smoke test binary

---

## Run the tests

```bash
# From the build/ directory (after building)
source ~/amd-workspace/AMD-important-scripts/set_rocm_env/set_rocm_env.sh
cd ~/amd-workspace/AMD-stack/rpp/rpp_arith_engine/build

LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./test_arith_engine
```

Expected output:

```
=================================================
  rppArithmeticEngine — smoke tests
=================================================

[test_scalar_add]       PASS  scalar add: 0.5 + 0.3 = 0.8
[test_scalar_subtract]  PASS  scalar sub: 1.0 - 0.4 = 0.6
[test_scalar_multiply]  PASS  scalar mul: 0.5 * 2.0 = 1.0
[test_scalar_divide]    PASS  scalar div: 0.8 / 4.0 = 0.2
[test_tensor_add]       PASS  tensor add: 0.3 + 0.5 = 0.8  (batch preserved)
[test_tensor_subtract]  PASS  tensor sub: 0.9 - 0.4 = 0.5
[test_chained_color_augmentations]  PASS  (identity chain + LUT resolution)
[test_brightness_augmentation]      PASS  0.4 * alpha=2 + beta=0 = 0.8
[test_move_semantics]               PASS  data and shape preserved
[test_shape_mismatch_throws]        PASS  std::invalid_argument raised

=================================================
  Results: 13 passed, 0 failed
=================================================
```

---

## API

### Construction

```cpp
// Create an engine: allocates GPU buffer + RPP handle
rppArithmeticEngine eng(uint32_t N, uint32_t H, uint32_t W, uint32_t C);
```

### Host I/O

```cpp
eng.upload(const float* hostPtr);   // copy host → GPU
eng.download(float* hostPtr);       // copy GPU → host
```

### Scalar arithmetic — returns a new engine

| Expression   | GPU operation              |
|--------------|---------------------------|
| `eng + s`    | `clamp(1*x + s)`           |
| `eng - s`    | `clamp(1*x - s)`           |
| `eng * s`    | `clamp(s*x + 0)`           |
| `eng / s`    | `clamp((1/s)*x + 0)`       |

Backed by `rppt_brightness`. Output is clamped to [0, 1].

> **Note:** `rppt_brightness` for F32 internally divides beta by 255, so the engine
> scales your scalar offset by 255 before calling RPP. You always work in [0, 1] space.

### Tensor-tensor arithmetic — same shape required, returns a new engine

```cpp
auto c = a + b;   // rppt_tensor_add_tensor
auto c = a - b;   // rppt_tensor_subtract_tensor
auto c = a * b;   // rppt_tensor_multiply_tensor
auto c = a / b;   // rppt_tensor_divide_tensor
```

### Color augmentations — in-place, chainable

```cpp
eng.brightness(alpha, beta);                         // clamp(alpha*x + beta)
eng.gammaCorrection(gamma);                          // x ^ gamma  (256-entry LUT)
eng.colorTwist(brightness, contrast, hue, sat);      // combined adjustment
eng.histogramEqualization();                         // per-channel equalization
eng.exposure(factor);                                // exposure shift
eng.contrast(factor, center);                        // contrast adjustment
eng.hue(shift);                                      // hue rotation (degrees)
eng.saturation(factor);                              // saturation scaling
```

All methods take `std::vector<float>` with one value per image in the batch.  
All return `*this` for chaining.

---

## Usage example

```cpp
#include "rpp_arith_engine.hpp"
#include <vector>

int main()
{
    const uint32_t N=2, H=64, W=64, C=3;

    // Fill two images with 0.5
    std::vector<float> src(N*H*W*C, 0.5f);

    // Create engine and upload
    rppArithmeticEngine eng(N, H, W, C);
    eng.upload(src.data());

    // Scalar arithmetic
    auto brighter = eng * 1.5f;    // [0.75, 0.75, ...]
    auto shifted  = eng + 0.2f;    // [0.70, 0.70, ...]
    auto combined = brighter + shifted;  // tensor add

    // Chain augmentations (in-place)
    eng.gammaCorrection({1.2f, 0.8f})          // per-image gamma
       .brightness({1.0f, 1.0f}, {0.0f, 0.0f}) // identity pass
       .colorTwist({1.1f, 0.9f},               // brightness
                   {1.2f, 0.8f},               // contrast
                   {10.0f, -5.0f},             // hue shift (degrees)
                   {1.0f,  1.0f});             // saturation

    // Download result
    std::vector<float> out(N*H*W*C);
    eng.download(out.data());

    return 0;
}
```

---

## Known behaviour

| Behaviour | Reason |
|---|---|
| Scalar `+`/`-` operate in [0,1] space; you pass [0,1] offsets | `rppt_brightness` F32 scales beta by `1/255` internally; the engine compensates |
| `gammaCorrection` has ~1/255 quantization error for non-boundary values | Uses a 256-entry LUT; boundary values (0.0, 1.0) are exact |
| Scalar ops clamp output to [0, 1] | `rppt_brightness` pixel-checks for F32 |

---

## Integration with the main RPP build

```bash
cmake /path/to/rpp -DBUILD_ARITH_ENGINE=ON
make -j$(nproc)
```
