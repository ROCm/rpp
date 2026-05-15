#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rppdefs.h>
#include <rpp/rppt_tensor_color_augmentations.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ─── Debug logging ────────────────────────────────────────────────────────────
//
// Compile-time: cmake -DLOG_LEVEL_DEBUG=ON ..   (always-on)
// Runtime:      PY_RPP_DEBUG=1 pytest ...       (toggle without rebuild)

#ifdef LOG_LEVEL_DEBUG
static constexpr bool g_debug_compiled = true;
#else
static constexpr bool g_debug_compiled = false;
#endif

static bool debug_enabled() {
    static bool val = g_debug_compiled || (std::getenv("PY_RPP_DEBUG") != nullptr);
    return val;
}

#define LOG_ENTRY(fn) \
    do { if (debug_enabled()) \
        std::fprintf(stderr, "[py_rpp] ENTER  %-30s\n", fn); } while(0)

#define LOG_EXIT(fn) \
    do { if (debug_enabled()) \
        std::fprintf(stderr, "[py_rpp] EXIT   %-30s\n", fn); } while(0)

#define LOG_DEBUG(...) \
    do { if (debug_enabled()) { \
        std::fprintf(stderr, "[py_rpp]   " __VA_ARGS__); \
        std::fprintf(stderr, "\n"); } } while(0)

// ─── Error helpers ────────────────────────────────────────────────────────────

static void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}

static void check_rpp(RppStatus s, const char* what) {
    if (s != RPP_SUCCESS)
        throw std::runtime_error(std::string(what) + " failed (RppStatus=" + std::to_string(s) + ")");
}

static void check_rpp_create(rppStatus_t s, const char* what) {
    if (s != rppStatusSuccess)
        throw std::runtime_error(std::string(what) + " failed (rppStatus_t=" + std::to_string(s) + ")");
}

// ─── RAII wrappers ────────────────────────────────────────────────────────────

struct GpuBuf {
    void* ptr = nullptr;
    size_t bytes;
    explicit GpuBuf(size_t n) : bytes(n) {
        LOG_ENTRY("GpuBuf::GpuBuf");
        LOG_DEBUG("hipMalloc  %zu bytes", n);
        check_hip(hipMalloc(&ptr, n), "hipMalloc");
        LOG_DEBUG("ptr        %p", ptr);
        LOG_EXIT("GpuBuf::GpuBuf");
    }
    ~GpuBuf() {
        if (ptr) {
            LOG_DEBUG("hipFree    %p  (%zu bytes)", ptr, bytes);
            (void)hipFree(ptr);
        }
    }
    void upload(const void* host, size_t n) {
        LOG_DEBUG("hipMemcpy  H2D  host=%p → dev=%p  %zu bytes", host, ptr, n);
        check_hip(hipMemcpy(ptr, host, n, hipMemcpyHostToDevice), "hipMemcpy H2D");
    }
    void download(void* host, size_t n) const {
        LOG_DEBUG("hipMemcpy  D2H  dev=%p → host=%p  %zu bytes", ptr, host, n);
        check_hip(hipMemcpy(host, ptr, n, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    }
};

struct RppHandle {
    rppHandle_t h;
    RppHandle(size_t batch_size) {
        LOG_ENTRY("RppHandle::RppHandle");
        LOG_DEBUG("rppCreate  batch_size=%zu  backend=HIP", batch_size);
        check_rpp_create(
            rppCreate(&h, batch_size, 0, nullptr, RPP_HIP_BACKEND),
            "rppCreate");
        LOG_DEBUG("handle     %p", (void*)h);
        LOG_EXIT("RppHandle::RppHandle");
    }
    ~RppHandle() {
        LOG_DEBUG("rppDestroy handle=%p", (void*)h);
        rppDestroy(h, RPP_HIP_BACKEND);
    }
};

// ─── Descriptor builder ───────────────────────────────────────────────────────

static RpptDesc make_nhwc_desc(Rpp32u n, Rpp32u h, Rpp32u w, Rpp32u c) {
    RpptDesc d{};
    d.numDims       = 4;
    d.offsetInBytes = 0;
    d.dataType      = RpptDataType::U8;
    d.layout        = RpptLayout::NHWC;
    d.n = n; d.h = h; d.w = w; d.c = c;
    d.strides.nStride = c * w * h;
    d.strides.hStride = c * w;
    d.strides.wStride = c;
    d.strides.cStride = 1;
    LOG_DEBUG("RpptDesc   N=%u H=%u W=%u C=%u  strides(n=%u h=%u w=%u c=%u)",
              n, h, w, c,
              d.strides.nStride, d.strides.hStride,
              d.strides.wStride, d.strides.cStride);
    return d;
}

// ─── brightness ──────────────────────────────────────────────────────────────

py::array_t<uint8_t> brightness(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> alpha,
    py::array_t<float,   py::array::c_style | py::array::forcecast> beta)
{
    LOG_ENTRY("brightness");

    // ── validate ──────────────────────────────────────────────────────────────
    auto si = src.request();
    if (si.ndim != 4)
        throw std::invalid_argument("src must be 4-D (N, H, W, C)");

    auto ai = alpha.request();
    auto bi = beta.request();
    const Rpp32u N = si.shape[0];
    const Rpp32u H = si.shape[1];
    const Rpp32u W = si.shape[2];
    const Rpp32u C = si.shape[3];

    LOG_DEBUG("input      shape=(%u,%u,%u,%u)  dtype=uint8", N, H, W, C);
    LOG_DEBUG("alpha      length=%zu", ai.size);
    LOG_DEBUG("beta       length=%zu", bi.size);

    if (ai.size != N || bi.size != N)
        throw std::invalid_argument("alpha and beta must each have length == batch size N");

    // ── GPU buffers ───────────────────────────────────────────────────────────
    const size_t img_bytes   = (size_t)N * H * W * C * sizeof(uint8_t);
    const size_t param_bytes = (size_t)N * sizeof(float);
    const size_t roi_bytes   = (size_t)N * sizeof(RpptROI);

    LOG_DEBUG("allocating src_d  (%zu bytes)", img_bytes);
    GpuBuf src_d(img_bytes);
    LOG_DEBUG("allocating dst_d  (%zu bytes)", img_bytes);
    GpuBuf dst_d(img_bytes);
    LOG_DEBUG("allocating alpha_d (%zu bytes)", param_bytes);
    GpuBuf alpha_d(param_bytes);
    LOG_DEBUG("allocating beta_d  (%zu bytes)", param_bytes);
    GpuBuf beta_d(param_bytes);
    LOG_DEBUG("allocating roi_d   (%zu bytes)", roi_bytes);
    GpuBuf roi_d(roi_bytes);

    src_d.upload(si.ptr, img_bytes);
    alpha_d.upload(ai.ptr, param_bytes);
    beta_d.upload(bi.ptr, param_bytes);

    // ── full-image XYWH ROIs for every image in the batch ─────────────────────
    std::vector<RpptROI> rois(N);
    for (Rpp32u i = 0; i < N; ++i) {
        rois[i].xywhROI.xy.x      = 0;
        rois[i].xywhROI.xy.y      = 0;
        rois[i].xywhROI.roiWidth  = static_cast<int>(W);
        rois[i].xywhROI.roiHeight = static_cast<int>(H);
        LOG_DEBUG("roi[%u]    XYWH(0,0,%d,%d)", i,
                  rois[i].xywhROI.roiWidth, rois[i].xywhROI.roiHeight);
    }
    roi_d.upload(rois.data(), roi_bytes);

    // ── descriptors + handle ──────────────────────────────────────────────────
    LOG_DEBUG("building src descriptor");
    RpptDesc src_desc = make_nhwc_desc(N, H, W, C);
    LOG_DEBUG("building dst descriptor");
    RpptDesc dst_desc = make_nhwc_desc(N, H, W, C);
    RppHandle handle(N);

    // ── kernel call ───────────────────────────────────────────────────────────
    LOG_DEBUG("calling    rppt_brightness  roiType=XYWH  backend=HIP");
    check_rpp(
        rppt_brightness(
            src_d.ptr, &src_desc,
            dst_d.ptr, &dst_desc,
            static_cast<Rpp32f*>(alpha_d.ptr),
            static_cast<Rpp32f*>(beta_d.ptr),
            static_cast<RpptROIPtr>(roi_d.ptr),
            RpptRoiType::XYWH,
            handle.h,
            RPP_HIP_BACKEND),
        "rppt_brightness");

    LOG_DEBUG("hipDeviceSynchronize");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    // ── copy result back ──────────────────────────────────────────────────────
    auto out = py::array_t<uint8_t>({N, H, W, C});
    dst_d.download(out.mutable_data(), img_bytes);

    LOG_EXIT("brightness");
    return out;
}

// ─── Module ───────────────────────────────────────────────────────────────────

PYBIND11_MODULE(py_rpp, m) {
    m.doc() = "Python bindings for RPP (ROCm Performance Primitives) — HIP backend";

    m.def("brightness", &brightness,
        py::arg("src"), py::arg("alpha"), py::arg("beta"),
        R"doc(
Apply brightness augmentation to a batch of images on the GPU.

Args:
    src   (np.ndarray): uint8, shape (N, H, W, C), NHWC layout, C-contiguous.
    alpha (np.ndarray): float32, shape (N,), per-image multiplier in [0, 20].
    beta  (np.ndarray): float32, shape (N,), per-image offset   in [0, 255].

Returns:
    np.ndarray: uint8, same shape as src.  output = clamp(alpha*src + beta, 0, 255)

Debug:
    Set PY_RPP_DEBUG=1 in the environment to enable call-trace logging to stderr.
    Or build with -DLOG_LEVEL_DEBUG=ON for always-on logging.
)doc");
}
