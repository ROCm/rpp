#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <rpp/rppdefs.h>
#include <rpp/rppt_tensor_color_augmentations.h>
#include <rpp/rppt_tensor_arithmetic_operations.h>
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <rpp/rppt_tensor_filter_augmentations.h>
#include <rpp/rppt_tensor_morphological_operations.h>
#include <rpp/rppt_tensor_bitwise_operations.h>
#include <rpp/rppt_tensor_statistical_operations.h>
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ─── Debug logging ────────────────────────────────────────────────────────────
#ifdef LOG_LEVEL_DEBUG
static constexpr bool g_debug_compiled = true;
#else
static constexpr bool g_debug_compiled = false;
#endif

inline bool debug_enabled() {
    static bool val = g_debug_compiled || (std::getenv("PY_RPP_DEBUG") != nullptr);
    return val;
}

#define LOG_DEBUG(...) \
    do { if (debug_enabled()) { \
        std::fprintf(stderr, "[py_rpp]   " __VA_ARGS__); \
        std::fprintf(stderr, "\n"); } } while(0)

// ─── Backend ──────────────────────────────────────────────────────────────────
enum class Backend { HIP, CPU };

inline Backend parse_backend(const std::string& s) {
    if (s == "hip" || s == "HIP" || s == "gpu" || s == "GPU") return Backend::HIP;
    if (s == "cpu" || s == "CPU") return Backend::CPU;
    throw std::invalid_argument("backend must be 'hip' or 'cpu', got: " + s);
}

inline RppBackend to_rpp_backend(Backend b) {
    return b == Backend::HIP ? RPP_HIP_BACKEND : RPP_HOST_BACKEND;
}

// ─── Error helpers ────────────────────────────────────────────────────────────
inline void check_hip(hipError_t err, const char* what) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
}

inline void check_rpp(RppStatus s, const char* what) {
    if (s != RPP_SUCCESS)
        throw std::runtime_error(std::string(what) +
            " failed (RppStatus=" + std::to_string(static_cast<int>(s)) + ")");
}

inline void check_rpp_create(rppStatus_t s, const char* what) {
    if (s != rppStatusSuccess)
        throw std::runtime_error(std::string(what) +
            " failed (rppStatus_t=" + std::to_string(static_cast<int>(s)) + ")");
}

// ─── RAII GPU buffer ──────────────────────────────────────────────────────────
struct GpuBuf {
    void* ptr = nullptr;
    size_t bytes;
    explicit GpuBuf(size_t n) : bytes(n) {
        LOG_DEBUG("hipMalloc %zu bytes", n);
        check_hip(hipMalloc(&ptr, n), "hipMalloc");
    }
    ~GpuBuf() { if (ptr) (void)hipFree(ptr); }
    GpuBuf(const GpuBuf&) = delete;
    GpuBuf& operator=(const GpuBuf&) = delete;

    void upload(const void* host, size_t n) {
        check_hip(hipMemcpy(ptr, host, n, hipMemcpyHostToDevice), "hipMemcpy H2D");
    }
    void download(void* host, size_t n) const {
        check_hip(hipMemcpy(host, ptr, n, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    }
};

// ─── RAII RPP handle ──────────────────────────────────────────────────────────
struct RppHandle {
    rppHandle_t h;
    RppBackend bk;
    RppHandle(size_t n, Backend backend = Backend::HIP) {
        bk = to_rpp_backend(backend);
        uint32_t threads = (backend == Backend::CPU) ? 4 : 0;
        LOG_DEBUG("rppCreate N=%zu %s", n, backend == Backend::HIP ? "HIP" : "CPU");
        check_rpp_create(rppCreate(&h, n, threads, nullptr, bk), "rppCreate");
    }
    ~RppHandle() { rppDestroy(h, bk); }
    RppHandle(const RppHandle&) = delete;
    RppHandle& operator=(const RppHandle&) = delete;
};

// ─── ParamBuf: GPU alloc+upload on HIP; host-pointer passthrough on CPU ───────
//
// Eliminates the HIP/CPU branch in every op: just create one ParamBuf per
// param array and call .ptr() to get the right pointer for rppt_xxx.
struct ParamBuf {
    Backend bk_;
    std::unique_ptr<GpuBuf> gpu_;
    const void* host_ptr_;

    ParamBuf(const void* host, size_t bytes, Backend bk)
        : bk_(bk), host_ptr_(host)
    {
        if (bk == Backend::HIP) {
            gpu_ = std::make_unique<GpuBuf>(bytes);
            gpu_->upload(host, bytes);
        }
    }
    void* ptr() const {
        return bk_ == Backend::HIP ? gpu_->ptr : const_cast<void*>(host_ptr_);
    }
};

// ─── Descriptors ──────────────────────────────────────────────────────────────
// Layout-aware descriptor builder. Strides are set to match a C-contiguous
// numpy array in either NHWC or NCHW memory order.
inline RpptDesc make_desc(Rpp32u n, Rpp32u h, Rpp32u w, Rpp32u c,
                           RpptLayout layout = RpptLayout::NHWC) {
    RpptDesc d{};
    d.numDims = 4; d.offsetInBytes = 0;
    d.dataType = RpptDataType::U8; d.layout = layout;
    d.n = n; d.h = h; d.w = w; d.c = c;
    if (layout == RpptLayout::NHWC) {
        d.strides.nStride = c * w * h;
        d.strides.hStride = c * w;
        d.strides.wStride = c;
        d.strides.cStride = 1;
    } else { // NCHW
        d.strides.nStride = c * h * w;
        d.strides.cStride = h * w;
        d.strides.hStride = w;
        d.strides.wStride = 1;
    }
    return d;
}

inline RpptDesc make_nhwc_desc(Rpp32u n, Rpp32u h, Rpp32u w, Rpp32u c) {
    return make_desc(n, h, w, c, RpptLayout::NHWC);
}

// ─── ROI helpers ──────────────────────────────────────────────────────────────
inline std::vector<RpptROI> make_full_rois(uint32_t N, uint32_t H, uint32_t W) {
    std::vector<RpptROI> rois(N);
    for (uint32_t i = 0; i < N; ++i) {
        rois[i].xywhROI.xy.x      = 0;
        rois[i].xywhROI.xy.y      = 0;
        rois[i].xywhROI.roiWidth  = static_cast<int>(W);
        rois[i].xywhROI.roiHeight = static_cast<int>(H);
    }
    return rois;
}

// ─── Validation ───────────────────────────────────────────────────────────────
inline void validate_4d_u8(const py::buffer_info& bi, const char* name) {
    if (bi.ndim != 4)
        throw std::invalid_argument(
            std::string(name) + " must be 4-D (N,H,W,C), got " +
            std::to_string(bi.ndim) + "-D");
}

inline void validate_1d_f32(const py::buffer_info& bi,
                             py::ssize_t expected_n, const char* name) {
    if (bi.ndim != 1 || bi.size != expected_n)
        throw std::invalid_argument(
            std::string(name) + " must be 1-D float32 of length " +
            std::to_string(expected_n) + ", got [" +
            std::to_string(bi.shape[0]) + "]");
}

inline void validate_1d_u32(const py::buffer_info& bi,
                             py::ssize_t expected_n, const char* name) {
    if (bi.ndim != 1 || bi.size != expected_n)
        throw std::invalid_argument(
            std::string(name) + " must be 1-D uint32 of length " +
            std::to_string(expected_n) + ", got [" +
            std::to_string(bi.shape[0]) + "]");
}

// Validate that two image arrays have identical 4-D uint8 shape.
inline void validate_two_images(const py::buffer_info& s1, const py::buffer_info& s2) {
    validate_4d_u8(s1, "src1"); validate_4d_u8(s2, "src2");
    if (s1.shape[0] != s2.shape[0] || s1.shape[1] != s2.shape[1] ||
        s1.shape[2] != s2.shape[2] || s1.shape[3] != s2.shape[3])
        throw std::invalid_argument("src1 and src2 must have the same shape");
}

// ─── run_img_op: core HIP/CPU dispatch for image→image ops ───────────────────
//
// Allocates GPU image buffers, uploads src, calls fn, downloads dst.
// fn signature: (void* src, void* dst, RpptDesc*, RpptROIPtr, rppHandle_t)
// ParamBuf objects for per-image params are created by the caller before
// calling run_img_op; fn captures them by reference.
template<typename Fn>
py::array_t<uint8_t> run_img_op(
    const void* src_host, size_t img_bytes,
    uint32_t N, uint32_t H, uint32_t W, uint32_t C,
    const RpptDesc& desc,
    const std::vector<RpptROI>& rois_host,
    RppHandle& handle,
    Backend bk,
    Fn&& fn)
{
    auto out = py::array_t<uint8_t>(
        {(py::ssize_t)N, (py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)C});
    {
        py::gil_scoped_release release;
        if (bk == Backend::HIP) {
            // Some kernels (filter, morphological) require a prefix region before
            // the image data in GPU memory; desc.offsetInBytes encodes that size.
            uint32_t off = desc.offsetInBytes;
            GpuBuf src_d(img_bytes + off), dst_d(img_bytes + off);
            GpuBuf roi_d(N * sizeof(RpptROI));
            check_hip(hipMemcpy(static_cast<char*>(src_d.ptr) + off,
                                src_host, img_bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
            roi_d.upload(rois_host.data(), N * sizeof(RpptROI));
            fn(src_d.ptr, dst_d.ptr,
               const_cast<RpptDesc*>(&desc),
               static_cast<RpptROIPtr>(roi_d.ptr),
               handle.h);
            check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
            check_hip(hipMemcpy(out.mutable_data(),
                                static_cast<char*>(dst_d.ptr) + off,
                                img_bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
        } else {
            fn(const_cast<void*>(src_host), out.mutable_data(),
               const_cast<RpptDesc*>(&desc),
               const_cast<RpptROIPtr>(rois_host.data()),
               handle.h);
        }
    }
    return out;
}

// Convenience overload: builds desc, rois, and handle internally.
template<typename Fn>
py::array_t<uint8_t> run_img_op(
    const void* src_host,
    uint32_t N, uint32_t H, uint32_t W, uint32_t C,
    Backend bk, Fn&& fn)
{
    size_t img_bytes = static_cast<size_t>(N) * H * W * C;
    RpptDesc desc = make_nhwc_desc(N, H, W, C);
    auto rois = make_full_rois(N, H, W);
    RppHandle handle(N, bk);
    return run_img_op(src_host, img_bytes, N, H, W, C, desc, rois, handle, bk,
                      std::forward<Fn>(fn));
}

// Convenience overload for filter/morphological kernels:
// RPP HIP filter kernels require offsetInBytes = 12 * (kernel_size / 2) in the
// descriptor so they have a valid prefix region in the GPU buffer before the image.
template<typename Fn>
py::array_t<uint8_t> run_img_op(
    const void* src_host,
    uint32_t N, uint32_t H, uint32_t W, uint32_t C,
    uint32_t kernel_size, Backend bk, Fn&& fn)
{
    size_t img_bytes = static_cast<size_t>(N) * H * W * C;
    RpptDesc desc = make_nhwc_desc(N, H, W, C);
    desc.offsetInBytes = (bk == Backend::HIP) ? 12 * (kernel_size / 2) : 0;
    auto rois = make_full_rois(N, H, W);
    RppHandle handle(N, bk);
    return run_img_op(src_host, img_bytes, N, H, W, C, desc, rois, handle, bk,
                      std::forward<Fn>(fn));
}

// ─── run_two_img_op: dispatch for two-source same-shape ops ──────────────────
//
// fn signature: (void* src1, void* src2, RpptDesc*, void* dst, RpptROIPtr, rppHandle_t)
template<typename Fn>
py::array_t<uint8_t> run_two_img_op(
    const py::buffer_info& s1i, const py::buffer_info& s2i,
    uint32_t N, uint32_t H, uint32_t W, uint32_t C,
    Backend bk, Fn&& fn)
{
    size_t img_bytes = static_cast<size_t>(N) * H * W * C;
    RpptDesc desc = make_nhwc_desc(N, H, W, C);
    auto rois = make_full_rois(N, H, W);
    auto out = py::array_t<uint8_t>(
        {(py::ssize_t)N, (py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)C});
    RppHandle handle(N, bk);
    {
        py::gil_scoped_release release;
        if (bk == Backend::HIP) {
            GpuBuf s1_d(img_bytes), s2_d(img_bytes), dst_d(img_bytes),
                   roi_d(N * sizeof(RpptROI));
            s1_d.upload(s1i.ptr, img_bytes);
            s2_d.upload(s2i.ptr, img_bytes);
            roi_d.upload(rois.data(), N * sizeof(RpptROI));
            fn(s1_d.ptr, s2_d.ptr, &desc, dst_d.ptr,
               static_cast<RpptROIPtr>(roi_d.ptr), handle.h);
            check_hip(hipDeviceSynchronize(), "sync");
            dst_d.download(out.mutable_data(), img_bytes);
        } else {
            fn(const_cast<void*>(s1i.ptr), const_cast<void*>(s2i.ptr),
               &desc, out.mutable_data(),
               const_cast<RpptROIPtr>(rois.data()),
               handle.h);
        }
    }
    return out;
}
