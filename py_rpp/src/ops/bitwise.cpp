#include "bitwise.hpp"

namespace ops {

py::array_t<uint8_t> bitwise_and(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    const std::string& backend)
{
    auto s1i = src1.request(); auto s2i = src2.request();
    validate_two_images(s1i, s2i);
    const uint32_t N = s1i.shape[0], H = s1i.shape[1], W = s1i.shape[2], C = s1i.shape[3];
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_two_img_op(s1i, s2i, N, H, W, C, bk,
        [&](void* s1, void* s2, RpptDesc* desc, void* dst, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_bitwise_and(s1, s2, desc, dst, desc,
                rois, RpptRoiType::XYWH, h, rb), "rppt_bitwise_and");
        });
}

py::array_t<uint8_t> bitwise_or(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    const std::string& backend)
{
    auto s1i = src1.request(); auto s2i = src2.request();
    validate_two_images(s1i, s2i);
    const uint32_t N = s1i.shape[0], H = s1i.shape[1], W = s1i.shape[2], C = s1i.shape[3];
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_two_img_op(s1i, s2i, N, H, W, C, bk,
        [&](void* s1, void* s2, RpptDesc* desc, void* dst, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_bitwise_or(s1, s2, desc, dst, desc,
                rois, RpptRoiType::XYWH, h, rb), "rppt_bitwise_or");
        });
}

py::array_t<uint8_t> bitwise_xor(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src1,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src2,
    const std::string& backend)
{
    auto s1i = src1.request(); auto s2i = src2.request();
    validate_two_images(s1i, s2i);
    const uint32_t N = s1i.shape[0], H = s1i.shape[1], W = s1i.shape[2], C = s1i.shape[3];
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_two_img_op(s1i, s2i, N, H, W, C, bk,
        [&](void* s1, void* s2, RpptDesc* desc, void* dst, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_bitwise_xor(s1, s2, desc, dst, desc,
                rois, RpptRoiType::XYWH, h, rb), "rppt_bitwise_xor");
        });
}

py::array_t<uint8_t> bitwise_not(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_bitwise_not(s, desc, d, desc,
                rois, RpptRoiType::XYWH, h, rb), "rppt_bitwise_not");
        });
}

} // namespace ops
