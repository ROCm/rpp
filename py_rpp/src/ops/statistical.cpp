#include "statistical.hpp"

namespace ops {

// Reduction ops write per-image stats to an output that differs from the input
// in both shape (N × (C+1)) and dtype — so run_img_op can't handle them.
// This template captures the shared dispatch boilerplate for all three ops.
template<typename OutT, typename KernelFn>
static py::array_t<OutT> reduction_op(
    const py::buffer_info& si, Backend bk, KernelFn&& kernel_fn, const char* name)
{
    const uint32_t N=si.shape[0], H=si.shape[1], W=si.shape[2], C=si.shape[3];
    const uint32_t out_len = N * (C + 1);
    const size_t img_bytes = static_cast<size_t>(N) * H * W * C;
    const size_t out_bytes = out_len * sizeof(OutT);
    RppBackend rb = to_rpp_backend(bk);
    RpptDesc desc = make_nhwc_desc(N, H, W, C);
    auto rois = make_full_rois(N, H, W);
    RppHandle handle(N, bk);
    auto out = py::array_t<OutT>({(py::ssize_t)N, (py::ssize_t)(C + 1)});
    {
        py::gil_scoped_release release;
        if (bk == Backend::HIP) {
            GpuBuf src_d(img_bytes), res_d(out_bytes), roi_d(N * sizeof(RpptROI));
            src_d.upload(si.ptr, img_bytes);
            roi_d.upload(rois.data(), N * sizeof(RpptROI));
            check_rpp(kernel_fn(src_d.ptr, &desc, res_d.ptr, out_len,
                static_cast<RpptROIPtr>(roi_d.ptr), RpptRoiType::XYWH, handle.h, rb), name);
            check_hip(hipDeviceSynchronize(), "sync");
            res_d.download(out.mutable_data(), out_bytes);
        } else {
            check_rpp(kernel_fn(si.ptr, &desc, out.mutable_data(), out_len,
                rois.data(), RpptRoiType::XYWH, handle.h, rb), name);
        }
    }
    return out;
}

py::array_t<float> tensor_mean(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    return reduction_op<float>(si, parse_backend(backend), rppt_tensor_mean, "rppt_tensor_mean");
}

py::array_t<uint8_t> tensor_min(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    return reduction_op<uint8_t>(si, parse_backend(backend), rppt_tensor_min, "rppt_tensor_min");
}

py::array_t<uint8_t> tensor_max(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    return reduction_op<uint8_t>(si, parse_backend(backend), rppt_tensor_max, "rppt_tensor_max");
}

py::array_t<uint8_t> threshold(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> min_val,
    py::array_t<float,   py::array::c_style | py::array::forcecast> max_val,
    const std::string& backend)
{
    auto si  = src.request();     validate_4d_u8(si, "src");
    auto mni = min_val.request(); auto mxi = max_val.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    // RPP threshold kernel reads minTensor[batchIndex*C + c] — needs N*C floats.
    const uint32_t param_len = N * C;
    validate_1d_f32(mni, param_len, "min_val"); validate_1d_f32(mxi, param_len, "max_val");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf mn_buf(mni.ptr, param_len * sizeof(float), bk);
    ParamBuf mx_buf(mxi.ptr, param_len * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_threshold(s, desc, d, desc,
                static_cast<Rpp32f*>(mn_buf.ptr()),
                static_cast<Rpp32f*>(mx_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_threshold");
        });
}

} // namespace ops
