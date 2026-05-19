#include "color.hpp"

namespace ops {

py::array_t<uint8_t> brightness(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> alpha,
    py::array_t<float,   py::array::c_style | py::array::forcecast> beta,
    const std::string& backend)
{
    auto si = src.request();   validate_4d_u8(si, "src");
    auto ai = alpha.request(); auto bi = beta.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(ai, N, "alpha"); validate_1d_f32(bi, N, "beta");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf a_buf(ai.ptr, N * sizeof(float), bk);
    ParamBuf b_buf(bi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_brightness(s, desc, d, desc,
                static_cast<Rpp32f*>(a_buf.ptr()),
                static_cast<Rpp32f*>(b_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_brightness");
        });
}

py::array_t<uint8_t> gamma_correction(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> gamma,
    const std::string& backend)
{
    auto si = src.request();   validate_4d_u8(si, "src");
    auto gi = gamma.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(gi, N, "gamma");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf g_buf(gi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_gamma_correction(s, desc, d, desc,
                static_cast<Rpp32f*>(g_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_gamma_correction");
        });
}

py::array_t<uint8_t> hue(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> hue_shift,
    const std::string& backend)
{
    auto si = src.request();   validate_4d_u8(si, "src");
    auto hi = hue_shift.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(hi, N, "hue_shift");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf h_buf(hi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h_) {
            check_rpp(rppt_hue(s, desc, d, desc,
                static_cast<Rpp32f*>(h_buf.ptr()),
                rois, RpptRoiType::XYWH, h_, rb), "rppt_hue");
        });
}

py::array_t<uint8_t> saturation(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> factor,
    const std::string& backend)
{
    auto si = src.request();   validate_4d_u8(si, "src");
    auto fi = factor.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(fi, N, "factor");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf f_buf(fi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_saturation(s, desc, d, desc,
                static_cast<Rpp32f*>(f_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_saturation");
        });
}

py::array_t<uint8_t> contrast(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> factor,
    py::array_t<float,   py::array::c_style | py::array::forcecast> center,
    const std::string& backend)
{
    auto si = src.request();    validate_4d_u8(si, "src");
    auto fi = factor.request(); auto ci = center.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(fi, N, "factor"); validate_1d_f32(ci, N, "center");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf f_buf(fi.ptr, N * sizeof(float), bk);
    ParamBuf c_buf(ci.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_contrast(s, desc, d, desc,
                static_cast<Rpp32f*>(f_buf.ptr()),
                static_cast<Rpp32f*>(c_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_contrast");
        });
}

py::array_t<uint8_t> exposure(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> factor,
    const std::string& backend)
{
    auto si = src.request();   validate_4d_u8(si, "src");
    auto fi = factor.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(fi, N, "factor");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf f_buf(fi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_exposure(s, desc, d, desc,
                static_cast<Rpp32f*>(f_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_exposure");
        });
}

py::array_t<uint8_t> blend(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    py::array_t<float,   py::array::c_style | py::array::forcecast> alpha,
    const std::string& backend)
{
    auto s1i = src1.request(); auto s2i = src2.request();
    validate_two_images(s1i, s2i);
    auto ai = alpha.request();
    const uint32_t N = s1i.shape[0], H = s1i.shape[1], W = s1i.shape[2], C = s1i.shape[3];
    validate_1d_f32(ai, N, "alpha");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf a_buf(ai.ptr, N * sizeof(float), bk);
    return run_two_img_op(s1i, s2i, N, H, W, C, bk,
        [&](void* s1, void* s2, RpptDesc* desc, void* dst, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_blend(s1, s2, desc, dst, desc,
                static_cast<Rpp32f*>(a_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_blend");
        });
}

py::array_t<uint8_t> color_twist(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> brightness,
    py::array_t<float,   py::array::c_style | py::array::forcecast> contrast,
    py::array_t<float,   py::array::c_style | py::array::forcecast> hue_shift,
    py::array_t<float,   py::array::c_style | py::array::forcecast> saturation,
    const std::string& backend)
{
    auto si  = src.request();        validate_4d_u8(si, "src");
    auto bri = brightness.request(); auto coi = contrast.request();
    auto hui = hue_shift.request();  auto sai = saturation.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(bri, N, "brightness"); validate_1d_f32(coi, N, "contrast");
    validate_1d_f32(hui, N, "hue_shift");  validate_1d_f32(sai, N, "saturation");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf br_buf(bri.ptr, N * sizeof(float), bk);
    ParamBuf co_buf(coi.ptr, N * sizeof(float), bk);
    ParamBuf hu_buf(hui.ptr, N * sizeof(float), bk);
    ParamBuf sa_buf(sai.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_color_twist(s, desc, d, desc,
                static_cast<Rpp32f*>(br_buf.ptr()),
                static_cast<Rpp32f*>(co_buf.ptr()),
                static_cast<Rpp32f*>(hu_buf.ptr()),
                static_cast<Rpp32f*>(sa_buf.ptr()),
                rois, RpptRoiType::XYWH, h, rb), "rppt_color_twist");
        });
}

py::array_t<uint8_t> histogram_equalize(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_histogram_equalize(s, desc, d, desc,
                rois, RpptRoiType::XYWH, h, rb), "rppt_histogram_equalize");
        });
}

} // namespace ops
