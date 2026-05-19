#include "morphological.hpp"

namespace ops {

// Both erode and dilate have split APIs in RPP:
//   rppt_erode_host / rppt_dilate_host  — CPU (always available)
//   rppt_erode      / rppt_dilate       — HIP (guarded by GPU_SUPPORT)
// The CPU path is taken via run_img_op's else branch; the HIP path needs GPU_SUPPORT.

py::array_t<uint8_t> erode(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size,
    const std::string& backend)
{
    auto si = src.request();
    validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    if (kernel_size % 2 == 0)
        throw std::invalid_argument("kernel_size must be odd");
    Backend bk = parse_backend(backend);
    return run_img_op(si.ptr, N, H, W, C, kernel_size, bk,
        [bk, kernel_size](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            if (bk == Backend::CPU) {
                check_rpp(rppt_erode_host(s, desc, d, desc,
                    kernel_size, rois, RpptRoiType::XYWH, h), "rppt_erode_host");
            } else {
#ifdef GPU_SUPPORT
                check_rpp(rppt_erode(s, desc, d, desc,
                    kernel_size, rois, RpptRoiType::XYWH, h, RPP_HIP_BACKEND), "rppt_erode");
#else
                throw std::runtime_error("erode: built without GPU_SUPPORT");
#endif
            }
        });
}

py::array_t<uint8_t> dilate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> src,
    uint32_t kernel_size,
    const std::string& backend)
{
    auto si = src.request();
    validate_4d_u8(si, "src");
    const uint32_t N = si.shape[0], H = si.shape[1], W = si.shape[2], C = si.shape[3];
    if (kernel_size % 2 == 0)
        throw std::invalid_argument("kernel_size must be odd");
    Backend bk = parse_backend(backend);
    return run_img_op(si.ptr, N, H, W, C, kernel_size, bk,
        [bk, kernel_size](void* s, void* d, RpptDesc* desc, RpptROIPtr rois, rppHandle_t h) {
            if (bk == Backend::CPU) {
                check_rpp(rppt_dilate_host(s, desc, d, desc,
                    kernel_size, rois, RpptRoiType::XYWH, h), "rppt_dilate_host");
            } else {
#ifdef GPU_SUPPORT
                check_rpp(rppt_dilate(s, desc, d, desc,
                    kernel_size, rois, RpptRoiType::XYWH, h, RPP_HIP_BACKEND), "rppt_dilate");
#else
                throw std::runtime_error("dilate: built without GPU_SUPPORT");
#endif
            }
        });
}

} // namespace ops
