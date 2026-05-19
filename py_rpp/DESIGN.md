# py_rpp — Design & Architecture

## 1. Purpose

RPP (ROCm Performance Primitives) is a C library with no official Python API.
`py_rpp` is a pybind11 Python extension that exposes RPP GPU image-augmentation
ops as ordinary Python functions accepting and returning NumPy arrays.

Goals:
- Zero boilerplate for the caller — NumPy in, NumPy out.
- Scalar-to-batch broadcasting: pass one number, it applies to every image.
- Layout transparency: accepts NHWC or NCHW; converts internally.
- Dual-backend: same call signature for HIP (GPU) and CPU.
- All host↔device memory management is hidden from the caller.

---

## 2. Repository Layout

```
py_rpp/
├── CMakeLists.txt              — build system (pybind11 + RPP + HIP)
├── conftest.py                 — project-root sys.path fix for pytest
├── pytest.ini                  — marker declarations, test root config
├── src/
│   ├── common.hpp              — RAII primitives, dispatch templates, helpers
│   ├── py_rpp.cpp              — thin pybind11 entry point (registers all ops)
│   └── ops/                   — one .hpp/.cpp pair per operation category
│       ├── color.hpp/.cpp      — brightness, gamma, hue, saturation, …
│       ├── geometric.hpp/.cpp  — flip, crop, rotate, resize
│       ├── filter.hpp/.cpp     — box_filter, gaussian_filter, median_filter
│       ├── morphological.hpp/.cpp — erode, dilate
│       ├── bitwise.hpp/.cpp    — bitwise_and/or/xor/not
│       ├── statistical.hpp/.cpp — tensor_mean/min/max, threshold
│       └── effects.hpp/.cpp    — gaussian_noise, salt_and_pepper, vignette, pixelate
├── py_rpp/
│   ├── __init__.py             — public API surface; re-exports fn.py layer
│   ├── fn.py                   — ergonomic wrappers (broadcasting, layout, backend)
│   ├── utils.py                — GPU detection, layout conversion
│   └── types.py                — backend/layout string constants
└── tests/
    ├── conftest.py             — shared fixtures (backend, make_batch, …)
    ├── test_brightness.py
    ├── test_color.py
    ├── test_bitwise.py
    ├── test_effects.py
    ├── test_filter.py
    ├── test_geometric.py
    ├── test_morphological.py
    └── test_statistical.py
```

After `cmake + make`, the build tree adds:

```
build/
└── py_rpp/
    ├── _py_rpp.cpython-312-x86_64-linux-gnu.so   — compiled C++ extension
    ├── __init__.py                                 — staged copy
    ├── fn.py                                       — staged copy
    ├── types.py                                    — staged copy
    └── utils.py                                    — staged copy
```

---

## 3. Two-Layer Architecture

```
╔════════════════════════════════════════════════════════════╗
║  Python caller                                             ║
║                                                            ║
║  import py_rpp                                             ║
║  out = py_rpp.brightness(images, alpha=1.5, beta=10)       ║
║         (scalar alpha/beta, NCHW input — all fine)         ║
╚═══════════════════════╦════════════════════════════════════╝
                        │
              ┌─────────▼──────────┐
              │   py_rpp/fn.py     │  ← HIGH-LEVEL Python layer
              │                    │
              │ • _ensure_nhwc()   │  NCHW → NHWC transpose
              │ • _param()         │  scalar → float32 (N,) array
              │ • _param_nc()      │  scalar → float32 (N*C,) array
              │ • _flag()          │  bool → uint32 (N,) array
              │ • _bk()            │  None → detect GPU → 'hip'/'cpu'
              └─────────┬──────────┘
                        │  NHWC uint8 arrays, fully-typed params
              ┌─────────▼──────────────────────────────────────┐
              │   _py_rpp  (C++ pybind11 extension)            │  ← LOW-LEVEL C++ layer
              │                                                 │
              │  src/py_rpp.cpp  — PYBIND11_MODULE(_py_rpp)    │
              │  src/ops/*.cpp   — one file per category        │
              │  src/common.hpp  — RAII + dispatch templates    │
              │                                                 │
              │  Per-call for image→image ops:                  │
              │    validate → ParamBuf → run_img_op             │
              │      → GpuBuf (hipMalloc) → rppt_xxx           │
              │      → hipDeviceSynchronize → D2H → numpy out  │
              └─────────┬──────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │   HIP Runtime      │  hipMalloc / hipFree
              │   librpp.so        │  rppt_brightness / rppt_flip / …
              └─────────┬──────────┘
                        │
                     GPU / CPU
```

### Why two layers?

The C++ binding (`_py_rpp`) is intentionally low-level: it only accepts
C-contiguous NHWC uint8 arrays and fully-shaped parameter arrays. This keeps
the C++ simple and auditable — no Python-level logic in C++.

`fn.py` is the ergonomic layer. It handles all the "nice to have" features:
scalar broadcasting, layout auto-detection, auto-backend selection. Doing
this in Python means faster iteration, easier testing, and no recompile needed
for API changes.

Callers import `py_rpp` (which re-exports `fn.py`) and never need to know
about `_py_rpp` directly.

---

## 4. C++ Layer (`src/`)

### 4.1 `common.hpp` — shared infrastructure

Everything that every op needs lives here. Ops include only their own `.hpp`.

#### RAII primitives

**`GpuBuf`** — wraps `hipMalloc`/`hipFree`:
```cpp
struct GpuBuf {
    void* ptr;
    explicit GpuBuf(size_t bytes);     // hipMalloc; throws on failure
    ~GpuBuf();                         // hipFree; always fires (stack unwind safe)
    void upload(const void*, size_t);  // hipMemcpy H2D
    void download(void*, size_t);      // hipMemcpy D2H
};
```

**`RppHandle`** — wraps `rppCreate`/`rppDestroy`:
```cpp
struct RppHandle {
    rppHandle_t h;
    RppHandle(size_t batch_size, Backend bk);
    ~RppHandle();
};
```

A new handle is created per op call. The reason: `rppCreate` allocates
per-batch-size metadata and stream state. Sharing a handle across calls with
different batch sizes (common in dynamic inference workloads) would require
a handle pool or lock. Per-call creation is simpler, correct, and the cost is
negligible compared to the actual memory transfers.

**`ParamBuf`** — unified CPU/HIP parameter buffer:
```cpp
struct ParamBuf {
    ParamBuf(const void* host, size_t bytes, Backend bk);
    void* ptr() const;  // GPU pointer on HIP, host pointer on CPU
};
```

Eliminates the HIP/CPU branch in every op's parameter handling. On HIP it
allocates a GPU buffer and uploads immediately. On CPU it just holds the host
pointer. Every op calls `.ptr()` and passes it to the RPP kernel — the same
line of code works for both backends.

#### Descriptor builders

**`make_nhwc_desc(n, h, w, c)`** — fills `RpptDesc` with C-contiguous NHWC strides:
```
nStride = C × W × H   (bytes to next image)
hStride = C × W       (bytes to next row)
wStride = C           (bytes to next pixel)
cStride = 1           (bytes to next channel)
```
This matches a NumPy C-contiguous `uint8` array in NHWC layout exactly.

**`make_full_rois(N, W, H)`** — returns `std::vector<RpptROI>` covering each full image,
using `XYWH` format (`x=0, y=0, w=W, h=H`).

#### Validation

- `validate_4d_u8(bi, name)` — enforces `ndim==4`; dtype enforcement is handled by
  pybind11's `forcecast` on the array argument type.
- `validate_1d_f32(bi, n, name)` — length check for per-image float parameters.
- `validate_1d_u32(bi, n, name)` — length check for per-image uint32 parameters.
- `validate_two_images(s1, s2)` — calls both 4D checks and asserts identical shapes.

#### Dispatch templates

**`run_img_op`** — core HIP/CPU dispatch for image-to-image ops:
```cpp
template<typename Fn>
py::array_t<uint8_t> run_img_op(
    const void* src_host, uint32_t N, uint32_t H, uint32_t W, uint32_t C,
    Backend bk, Fn&& fn);
```
Allocates GPU buffers (HIP path), uploads src, calls `fn(src, dst, desc, rois, handle)`,
synchronizes, downloads dst. On CPU, calls `fn` directly with host pointers.
`ParamBuf` objects for per-image params are created by the caller before `run_img_op`
and captured by the lambda — the template requires no knowledge of them.

**`run_two_img_op`** — same pattern for two-source ops (bitwise, blend):
```cpp
template<typename Fn>
py::array_t<uint8_t> run_two_img_op(
    const py::buffer_info& s1i, const py::buffer_info& s2i,
    uint32_t N, uint32_t H, uint32_t W, uint32_t C,
    Backend bk, Fn&& fn);
```
`fn` signature: `(void* src1, void* src2, RpptDesc*, void* dst, RpptROIPtr, rppHandle_t)`.

**`reduction_op`** in `statistical.cpp` (file-static template):
```cpp
template<typename OutT, typename KernelFn>
static py::array_t<OutT> reduction_op(
    const py::buffer_info& si, Backend bk, KernelFn&& kernel_fn, const char* name);
```
Reduction ops (`tensor_mean`, `tensor_min`, `tensor_max`) differ from image→image
ops: their output shape is `(N, C+1)` (per-channel stat + one overall stat per image)
and their dtype may differ from the input. They can't use `run_img_op`. This template
captures the shared dispatch boilerplate for all three, eliminating ~90 lines of
repetition.

Using function templates (not `std::function`) for all dispatch ensures the lambda
is inlined at the call site with zero vtable or allocation overhead.

---

### 4.2 `src/ops/` — operation categories

Each category is a `.hpp`/`.cpp` pair. The `.hpp` declares the function; the `.cpp`
implements it using helpers from `common.hpp`. `py_rpp.cpp` includes only the `.hpp`
files. This keeps compile times manageable as ops accumulate.

| File | Operations |
|---|---|
| `color.cpp` | `brightness`, `gamma_correction`, `hue`, `saturation`, `contrast`, `exposure`, `blend`, `color_twist`, `histogram_equalize` |
| `geometric.cpp` | `flip`, `crop`, `rotate`, `resize` |
| `filter.cpp` | `box_filter`, `gaussian_filter`, `median_filter` |
| `morphological.cpp` | `erode`, `dilate` |
| `bitwise.cpp` | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not` |
| `statistical.cpp` | `tensor_mean`, `tensor_min`, `tensor_max`, `threshold` |
| `effects.cpp` | `gaussian_noise`, `salt_and_pepper_noise`, `vignette`, `pixelate` |

**`crop` and `resize`** cannot use `run_img_op` because their output shape differs
from their input (`out_h × out_w` vs `H × W`). They inline their own HIP/CPU dispatch.

**`pixelate`** requires an additional float32 scratch buffer of the same element count
as the image (RPP uses it as intermediate working storage). The HIP path allocates a
`GpuBuf` for it; the CPU path uses a `std::vector<float>` on the stack.

---

### 4.3 `src/py_rpp.cpp` — pybind11 entry point

A thin file whose only job is to include the op `.hpp` files and write the
`PYBIND11_MODULE(_py_rpp, m)` block. No business logic lives here.

The module is named `_py_rpp` (underscore prefix) following the AMD convention:
the C extension is private; `py_rpp/__init__.py` is the public API. This allows
the Python layer to intercept and transform calls without exposing the C++ ABI.

The module also exposes `rpp_create`/`rpp_destroy` as a power-user API for callers
that want to manage handle lifecycle themselves. These return the handle as
`uintptr_t` (an opaque integer that Python can hold). Normal callers should
use `fn.py` instead, where the handle is created and destroyed per-call inside
`RppHandle`'s RAII scope.

---

## 5. Python Layer (`py_rpp/`)

### 5.1 `types.py` — string constants

Single source of truth for the string literals `"hip"`, `"cpu"`, `"nhwc"`, `"nchw"`.
Callers can use `py_rpp.types.HIP` instead of bare strings to get IDE completion
and catch typos at import time. The C++ side also accepts these strings
(via `parse_backend()`), so no enum conversion is needed.

### 5.2 `utils.py` — GPU detection and layout helpers

**`is_gpu_available()`** — runs `rocm-smi --showbus` via subprocess and returns
`True` if it exits with code 0. Wrapped in a try/except so it always returns a bool.

**`get_default_backend()`** — calls `is_gpu_available()` once and caches the result
in a module-level variable `_default_backend`. Subsequent calls return the cached
value with no subprocess fork. This matters because every `fn.py` wrapper that
receives `backend=None` calls `_bk(backend)`, which calls `get_default_backend()`.
Without the cache, every API call would fork `rocm-smi`.

**`detect_layout(images)`** — heuristic for 4-D arrays:
- `dim[-1] ∈ {1,2,3,4}` and `dim[1]` not → NHWC
- `dim[1] ∈ {1,2,3,4}` and `dim[-1]` not → NCHW
- Both or neither → NHWC (packed uint8 is the common case)

**`to_nhwc(images, layout=None)`** — transposes NCHW → NHWC if needed, then
returns a C-contiguous uint8 array. `layout=None` triggers `detect_layout`.

### 5.3 `fn.py` — ergonomic wrappers

Every public op in `fn.py` follows the same pattern:
```python
def brightness(images, alpha=1.0, beta=0.0, *, input_layout=None, backend=None):
    imgs = _ensure_nhwc(images, input_layout)   # layout + dtype normalization
    N = imgs.shape[0]
    return _C.brightness(imgs,
                         _param(alpha, N),       # scalar → (N,) float32
                         _param(beta,  N),
                         _bk(backend))           # None → 'hip' or 'cpu'
```

#### Broadcasting helpers

**`_param(value, n, dtype=float32)`** — scalar or sequence → 1-D array of length `n`.
Raises `ValueError` if a sequence has the wrong length.

**`_flag(value, n)`** — bool or sequence → uint32 (N,) of 0/1. Used for flip
`horizontal`/`vertical` and crop `x`/`y` offsets.

**`_param_nc(value, n, c, dtype=float32)`** — for `threshold`, whose RPP kernel
reads `param[imageIndex * C + channelIndex]` (i.e., N×C floats total). Accepts:
- scalar → broadcast to all images and all channels
- shape `(n,)` → repeat C times per image
- shape `(n, c)` → flatten row-major
- shape `(n*c,)` → pass through

This design means callers can use the most natural representation:
`min_val=0.0` (same threshold for everything) or `min_val=np.array([[10,20,30],[15,25,35]])` (per-channel per-image).

#### Salt-and-pepper scale convention

`salt_value` and `pepper_value` are accepted in `[0, 255]` (the natural uint8 scale).
They are divided by `255.0` before forwarding to RPP, which internally requires
normalised floats in `[0, 1]`. The division happens at the Python boundary in `fn.py`
so the C++ layer never needs to know about this convention mismatch.

### 5.4 `__init__.py` — public API surface

Imports `fn.py` names directly so callers can write `py_rpp.brightness(...)` instead
of `py_rpp.fn.brightness(...)`. Also exposes `_py_rpp`, `types`, `utils`, and `fn`
as submodules for advanced use.

```
py_rpp.__version__ = "0.2.0"
```

---

## 6. Build System (`CMakeLists.txt`)

### Helper functions

Three CMake functions encapsulate reusable logic:

**`py_rpp_find_rocm(out_var)`** — resolves the ROCm installation path without
requiring the caller to export environment variables:
```
Priority 1: $ROCM_PATH env var  (set by set_rocm_env.sh)
Priority 2: INSTALL_DIR= line in .therock_last_install state file
            (searched in ~/amd-workspace/, ~/workspace/, /scratch/…, ~/)
Priority 3: /opt/rocm  (system install fallback)
```

**`py_rpp_stage_files(dest_root, file1, …)`** — copies Python package files
into the build tree using `configure_file(... COPYONLY)`. This is what makes
`PYTHONPATH=build` work without an install step: the `.so` lands in
`build/py_rpp/` and the pure-Python files are staged alongside it.

**`py_rpp_configure_hip_target(target)`** — adds `hip::host` link and the
`RPP_BACKEND_HIP`/`GPU_SUPPORT` compile definitions. Factored out so a future
second target (e.g., a CPU-only `.so` for CI without ROCm) can be added
without repeating the logic.

### Key ordering constraint

`CMAKE_PREFIX_PATH` must include `${ROCM_PATH}/lib/cmake` **before** any
`find_package(rpp)` call. `rpp-config.cmake` internally calls
`find_dependency(HIP)`, which only succeeds if the HIP CMake config is already
discoverable on the prefix path. This is prepended unconditionally at the top of
the file:
```cmake
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake")
```

### Output layout

The `.so` is placed under `build/py_rpp/` via:
```cmake
set_target_properties(_py_rpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py_rpp"
)
```

This means `PYTHONPATH=build python3 -c "import py_rpp"` works immediately after
`make`, with no install step and no `PYTHONPATH` gymnastics.

---

## 7. Test Infrastructure

### `conftest.py` (project root)

Inserts `build/` at `sys.path[0]` so that `import py_rpp` finds the compiled
package in the build tree rather than the bare source directory (which has no
`.so`). Without this, pytest's automatic sys.path manipulation would resolve
`py_rpp/__init__.py` from the source tree first, then fail to import `_py_rpp`.

### `tests/conftest.py` — shared fixtures

**`backend` fixture** — parametrized over `["cpu", "hip"]`. The `"hip"` variant
carries `@pytest.mark.skipif(not GPU_AVAILABLE)`, which is evaluated once at
collection time by running `rocm-smi --showbus`. Every test that accepts a
`backend` argument automatically runs twice; the HIP variant is silently skipped
on machines without a GPU.

**`make_batch(n, h, w, c, fill=None)`** — returns an `(N, H, W, C)` uint8 batch.
With `fill=None` it returns random pixel values; with `fill=k` every pixel is `k`.

**`make_nchw(n, h, w, c, fill=None)`** — same but transposed to `(N, C, H, W)`,
used by layout-detection tests.

**`pix(arr, n=0)`** — returns `int(arr[n, 0, 0, 0])`. A one-liner for spot-checks
in single-pixel tests.

### Test structure per module

Each test module covers:
1. **Identity / passthrough** — op with neutral parameters leaves pixels unchanged.
2. **Known-value** — single pixel with known math, verify exact output.
3. **Boundary / clipping** — values that should saturate at 0 or 255.
4. **Batch independence** — different params per image; verify they don't bleed.
5. **Layout** — NCHW input is accepted and produces the same result as NHWC.
6. **Input validation** — wrong ndim, wrong param length, wrong dtype raise `ValueError`.

### Running

```bash
cd /path/to/rpp/py_rpp

# All tests (HIP skipped if no GPU)
python3 -m pytest tests/ --tb=short -q

# CPU only
python3 -m pytest tests/ -m "not hip" -q

# Single module, verbose
python3 -m pytest tests/test_color.py -v

# With C++ call-trace logging (rppCreate calls, backend selection)
PY_RPP_DEBUG=1 python3 -m pytest tests/test_brightness.py -v -s
```

Note: RPP itself does not expose a runtime debug log level via environment
variable in this version. `PY_RPP_DEBUG=1` is the only logging layer available.

---

## 8. Known RPP 3.1.2 Quirks

These are bugs in the RPP CPU library, not in the Python bindings. The affected
tests work around them by using batch size N=1.

### Zero-parameter batch-skip bug

When a per-image parameter is exactly zero (or computes to zero internally),
the RPP 3.1.2 CPU library silently skips writing output for images past index 0
in any N>1 batch. Affected ops and the zero condition:

| Op | Zero condition |
|---|---|
| `saturation` | `factor=1.0` → internal `delta = factor - 1 = 0` |
| `hue` | `hue_shift=0.0` |
| `rotate` | `angle=0.0` |
| `pixelate` | `pixelation_percentage=0.0` |

Tests that would naturally use `factor=1.0` or `angle=0.0` as the identity
condition use N=1 to avoid the bug.

### SIMD/OMP boundary artifact

After calling a tensor reduction op (`tensor_min`, `tensor_max`) on the CPU
backend in the same process, subsequent image-to-image ops that call flip
produce a 4-byte corruption at the start of image[1] in N=2 batches. The
corruption is deterministic and confined to bytes `[N*H*W*C, N*H*W*C + 3]`
(i.e., the first pixel of image index 1). Suspected cause: a SIMD or
OpenMP thread-boundary write that lands just outside the destination buffer.

Affected test: `test_threshold_all_in_range` (uses `tensor_min`/`tensor_max`
before `threshold`). Workaround: N=1.

---

## 9. Adding a New Op

Adding one op requires touching at most four files. No architecture changes needed.

**Step 1 — implement in `src/ops/<category>.cpp`:**
```cpp
py::array_t<uint8_t> my_op(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> param,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    auto pi = param.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(pi, N, "param");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf p_buf(pi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_my_op(s, desc, d, desc,
                static_cast<Rpp32f*>(p_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_my_op");
        });
}
```

**Step 2 — declare in `src/ops/<category>.hpp`:**
```cpp
py::array_t<uint8_t> my_op(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> param,
    const std::string& backend);
```

**Step 3 — register in `src/py_rpp.cpp`:**
```cpp
m.def("my_op", &ops::my_op,
      py::arg("src"), py::arg("param"), py::arg("backend") = "hip",
      "One-line description.");
```

**Step 4 — wrap in `py_rpp/fn.py`:**
```python
def my_op(images, param=1.0, *, input_layout=None, backend=None):
    """Docstring."""
    imgs = _ensure_nhwc(images, input_layout)
    N = imgs.shape[0]
    return _C.my_op(imgs, _param(param, N), _bk(backend))
```

**Step 5 — re-export in `py_rpp/__init__.py`** and add `tests/test_<category>.py`.

If the output shape differs from the input (like `crop`, `resize`, `tensor_mean`),
inline the HIP/CPU dispatch directly in the `.cpp` instead of using `run_img_op`.

---

## 10. Known Constraints

| Constraint | Reason |
|---|---|
| Input dtype must be uint8 | All bound RPP kernels use the U8 code path; F32/F16 paths are not wrapped |
| Input layout must be 4-D | All RPP tensor ops require N/H/W/C; lower dims unsupported |
| `hipDeviceSynchronize()` after every op | No async queue exposed; guarantees output is ready when the function returns |
| Per-call `rppCreate`/`rppDestroy` | Avoids shared-state and thread-safety concerns; overhead is negligible |
| Kernel size must be odd (filters/morphological) | RPP requirement for symmetric kernels |
| No multi-GPU | `rppCreate` receives `nullptr` stream; always uses device 0 |
| `threshold` params are N×C | RPP threshold kernel indexes `param[imageIndex * C + channel]`, not per-image |
