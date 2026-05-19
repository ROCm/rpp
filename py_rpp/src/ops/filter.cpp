#include "filter.hpp"

namespace ops {

py::array_t<uint8_t> box_filter(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    if (kernel_size % 2 == 0)
        throw std::invalid_argument("kernel_size must be odd");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_box_filter(s, desc, d, desc,
                kernel_size, RpptImageBorderType::REPLICATE,
                rois, RpptRoiType::XYWH, h, rb), "rppt_box_filter");
        });
}

py::array_t<uint8_t> gaussian_filter(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    py::array_t<float,   py::array::c_style | py::array::forcecast> std_dev,
    uint32_t kernel_size,
    const std::string& backend)
{
    auto si = src.request();  validate_4d_u8(si, "src");
    auto sdi = std_dev.request();
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    validate_1d_f32(sdi, N, "std_dev");
    if (kernel_size % 2 == 0)
        throw std::invalid_argument("kernel_size must be odd");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    ParamBuf sd_buf(sdi.ptr, N * sizeof(float), bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_gaussian_filter(s, desc, d, desc,
                static_cast<Rpp32f*>(sd_buf.ptr()),
                kernel_size, RpptImageBorderType::REPLICATE,
                rois, RpptRoiType::XYWH, h, rb), "rppt_gaussian_filter");
        });
}

py::array_t<uint8_t> median_filter(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size,
    const std::string& backend)
{
    auto si = src.request(); validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    if (kernel_size % 2 == 0)
        throw std::invalid_argument("kernel_size must be odd");
    Backend bk = parse_backend(backend);
    RppBackend rb = to_rpp_backend(bk);
    return run_img_op(si.ptr, N, H, W, C, bk,
        [&](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            check_rpp(rppt_median_filter(s, desc, d, desc,
                kernel_size, RpptImageBorderType::REPLICATE,
                rois, RpptRoiType::XYWH, h, rb), "rppt_median_filter");
        });
}

} // namespace ops
