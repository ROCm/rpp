#include "effects.hpp"

namespace ops {

py::array_t<uint8_t> gaussian_noise(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> mean,
    py::array_t<float,   py::array::c_style | py::array::forcecast> std_dev,
    uint32_t seed,
    const std::string& backend)
{
    auto si  = src.request();    validate_4d_u8(si, "src");
    auto mni = mean.request();   auto sdi = std_dev.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(mni, N, "mean"); validate_1d_f32(sdi, N, "std_dev");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf mn_buf(mni.ptr, N * sizeof(float), bk);
    ParamBuf sd_buf(sdi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_gaussian_noise(s, desc, d, desc,
                static_cast<Rpp32f*>(mn_buf.ptr()),
                static_cast<Rpp32f*>(sd_buf.ptr()),
                seed, rois, RpptRoiType::XYWH, h, rb), "rppt_gaussian_noise");
        });
}

py::array_t<uint8_t> salt_and_pepper_noise(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> noise_prob,
    py::array_t<float,   py::array::c_style | py::array::forcecast> salt_prob,
    py::array_t<float,   py::array::c_style | py::array::forcecast> salt_value,
    py::array_t<float,   py::array::c_style | py::array::forcecast> pepper_value,
    uint32_t seed,
    const std::string& backend)
{
    auto si  = src.request();          validate_4d_u8(si, "src");
    auto npi = noise_prob.request();   auto spi = salt_prob.request();
    auto svi = salt_value.request();   auto pvi = pepper_value.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(npi, N, "noise_prob");  validate_1d_f32(spi, N, "salt_prob");
    validate_1d_f32(svi, N, "salt_value");  validate_1d_f32(pvi, N, "pepper_value");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf np_buf(npi.ptr, N * sizeof(float), bk);
    ParamBuf sp_buf(spi.ptr, N * sizeof(float), bk);
    ParamBuf sv_buf(svi.ptr, N * sizeof(float), bk);
    ParamBuf pv_buf(pvi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_salt_and_pepper_noise(s, desc, d, desc,
                static_cast<Rpp32f*>(np_buf.ptr()),
                static_cast<Rpp32f*>(sp_buf.ptr()),
                static_cast<Rpp32f*>(sv_buf.ptr()),
                static_cast<Rpp32f*>(pv_buf.ptr()),
                seed, rois, RpptRoiType::XYWH, h, rb), "rppt_salt_and_pepper_noise");
        });
}

py::array_t<uint8_t> vignette(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> intensity,
    const std::string& backend)
{
    auto si = src.request();       validate_4d_u8(si, "src");
    auto ii = intensity.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(ii, N, "intensity");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf i_buf(ii.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_vignette(s, desc, d, desc,
                static_cast<Rpp32f*>(i_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_vignette");
        });
}

py::array_t<uint8_t> pixelate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    float pixelation_percentage,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    if (pixelation_percentage < 0.0f || pixelation_percentage > 100.0f)
        throw std::invalid_argument("pixelation_percentage must be in [0, 100]");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    size_t img_bytes = static_cast<size_t>(N) * H * W * C;
    // rppt_pixelate requires a float32 scratch buffer of the same element count
    size_t scratch_bytes = static_cast<size_t>(N) * H * W * C * sizeof(float);
    RpptDesc desc = make_nhwc_desc(N, H, W, C);
    auto rois = make_full_rois(N, W, H);
    RppHandle handle(N, bk);
    auto out = py::array_t<uint8_t>(
        {(py::ssize_t)N, (py::ssize_t)H, (py::ssize_t)W, (py::ssize_t)C});

    if (bk == Backend::HIP) {
        GpuBuf src_d(img_bytes), dst_d(img_bytes), scratch_d(scratch_bytes),
               roi_d(N * sizeof(RpptROI));
        src_d.upload(si.ptr, img_bytes);
        roi_d.upload(rois.data(), N * sizeof(RpptROI));
        check_rpp(rppt_pixelate(src_d.ptr, &desc, dst_d.ptr, &desc,
            scratch_d.ptr, pixelation_percentage,
            static_cast<RpptROIPtr>(roi_d.ptr), RpptRoiType::XYWH,
            handle.h, rb), "rppt_pixelate");
        check_hip(hipDeviceSynchronize(), "sync");
        dst_d.download(out.mutable_data(), img_bytes);
    } else {
        std::vector<float> scratch_host(static_cast<size_t>(N) * H * W * C, 0.0f);
        check_rpp(rppt_pixelate(si.ptr, &desc, out.mutable_data(), &desc,
            scratch_host.data(), pixelation_percentage,
            rois.data(), RpptRoiType::XYWH,
            handle.h, rb), "rppt_pixelate");
    }
    return out;
}

} // namespace ops
