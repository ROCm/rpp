#include "geometric.hpp"

namespace ops {

py::array_t<uint8_t> flip(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> src,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> horizontal,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> vertical,
    const std::string& backend)
{
    auto si  = src.request();        validate_4d_u8(si, "src");
    auto hi  = horizontal.request(); auto vi = vertical.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_u32(hi, N, "horizontal"); validate_1d_u32(vi, N, "vertical");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf h_buf(hi.ptr, N * sizeof(uint32_t), bk);
    ParamBuf v_buf(vi.ptr, N * sizeof(uint32_t), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t rh) {
            check_rpp(rppt_flip(s, desc, d, desc,
                static_cast<Rpp32u*>(h_buf.ptr()),
                static_cast<Rpp32u*>(v_buf.ptr()),
                rois, RpptRoiType::XYWH, rh, rb), "rppt_flip");
        });
}

py::array_t<uint8_t> crop(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> src,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> x,
    py::array_t<uint32_t, py::array::c_style | py::array::forcecast> y,
    uint32_t out_h, uint32_t out_w,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    auto xi = x.request();   auto yi = y.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_u32(xi, N, "x"); validate_1d_u32(yi, N, "y");
    if (out_h == 0 || out_w == 0)
        throw std::invalid_argument("out_h and out_w must be > 0");

    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    size_t src_bytes = static_cast<size_t>(N) * H * W * C;
    size_t dst_bytes = static_cast<size_t>(N) * out_h * out_w * C;

    // Source descriptor uses full source dimensions (strides calculated on src)
    RpptDesc src_desc = make_nhwc_desc(N, H, W, C);
    // Destination descriptor uses the output dimensions
    RpptDesc dst_desc = make_nhwc_desc(N, out_h, out_w, C);

    // Build per-image XYWH ROIs using the per-image (x, y) and uniform (out_w, out_h)
    const uint32_t* x_ptr = static_cast<const uint32_t*>(xi.ptr);
    const uint32_t* y_ptr = static_cast<const uint32_t*>(yi.ptr);
    std::vector<RpptROI> rois(N);
    for (uint32_t i = 0; i < N; ++i) {
        rois[i].xywhROI.xy.x      = static_cast<int>(x_ptr[i]);
        rois[i].xywhROI.xy.y      = static_cast<int>(y_ptr[i]);
        rois[i].xywhROI.roiWidth  = static_cast<int>(out_w);
        rois[i].xywhROI.roiHeight = static_cast<int>(out_h);
    }

    auto out = py::array_t<uint8_t>(
        {(py::ssize_t)N, (py::ssize_t)out_h, (py::ssize_t)out_w, (py::ssize_t)C});
    RppHandle handle(N, bk);

    if (bk == Backend::HIP) {
        GpuBuf src_d(src_bytes), dst_d(dst_bytes), roi_d(N * sizeof(RpptROI));
        src_d.upload(si.ptr, src_bytes);
        roi_d.upload(rois.data(), N * sizeof(RpptROI));
        check_rpp(rppt_crop(src_d.ptr, &src_desc, dst_d.ptr, &dst_desc,
            static_cast<RpptROIPtr>(roi_d.ptr), RpptRoiType::XYWH,
            handle.h, rb), "rppt_crop");
        check_hip(hipDeviceSynchronize(), "sync");
        dst_d.download(out.mutable_data(), dst_bytes);
    } else {
        check_rpp(rppt_crop(si.ptr, &src_desc, out.mutable_data(), &dst_desc,
            rois.data(), RpptRoiType::XYWH, handle.h, rb), "rppt_crop");
    }
    return out;
}

py::array_t<uint8_t> rotate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> angle,
    const std::string& backend)
{
    auto si = src.request();   validate_4d_u8(si, "src");
    auto ai = angle.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(ai, N, "angle");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf a_buf(ai.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_rotate(s, desc, d, desc,
                static_cast<Rpp32f*>(a_buf.ptr()),
                RpptInterpolationType::BILINEAR,
                rois, RpptRoiType::XYWH, h, rb), "rppt_rotate");
        });
}

py::array_t<uint8_t> resize(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t out_h, uint32_t out_w,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    if (out_h == 0 || out_w == 0)
        throw std::invalid_argument("out_h and out_w must be > 0");

    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    size_t src_bytes = static_cast<size_t>(N) * H * W * C;
    size_t dst_bytes = static_cast<size_t>(N) * out_h * out_w * C;

    RpptDesc src_desc = make_nhwc_desc(N, H, W, C);
    RpptDesc dst_desc = make_nhwc_desc(N, out_h, out_w, C);
    auto rois = make_full_rois(N, W, H);

    // dstImgSizes: one RpptImagePatch per image (uniform here)
    std::vector<RpptImagePatch> dst_sizes(N);
    for (uint32_t i = 0; i < N; ++i) {
        dst_sizes[i].width  = out_w;
        dst_sizes[i].height = out_h;
    }

    auto out = py::array_t<uint8_t>(
        {(py::ssize_t)N, (py::ssize_t)out_h, (py::ssize_t)out_w, (py::ssize_t)C});
    RppHandle handle(N, bk);

    if (bk == Backend::HIP) {
        GpuBuf src_d(src_bytes), dst_d(dst_bytes), roi_d(N * sizeof(RpptROI));
        GpuBuf sizes_d(N * sizeof(RpptImagePatch));
        src_d.upload(si.ptr, src_bytes);
        roi_d.upload(rois.data(), N * sizeof(RpptROI));
        sizes_d.upload(dst_sizes.data(), N * sizeof(RpptImagePatch));
        check_rpp(rppt_resize(src_d.ptr, &src_desc, dst_d.ptr, &dst_desc,
            static_cast<RpptImagePatchPtr>(sizes_d.ptr),
            RpptInterpolationType::BILINEAR,
            static_cast<RpptROIPtr>(roi_d.ptr), RpptRoiType::XYWH,
            handle.h, rb), "rppt_resize");
        check_hip(hipDeviceSynchronize(), "sync");
        dst_d.download(out.mutable_data(), dst_bytes);
    } else {
        check_rpp(rppt_resize(si.ptr, &src_desc, out.mutable_data(), &dst_desc,
            dst_sizes.data(), RpptInterpolationType::BILINEAR,
            rois.data(), RpptRoiType::XYWH, handle.h, rb), "rppt_resize");
    }
    return out;
}

} // namespace ops
