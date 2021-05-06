#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include <hip/rpp_hip_common.hpp>
#include "kernel/rpp_hip_host_decls.hpp"

/******************** color_twist ********************/

RppStatus
color_twist_hip( Rpp8u* srcPtr,RppiSize srcSize, Rpp8u* dstPtr, float alpha, float beta, float hue_shift, float sat, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "color_twist.cpp", "color_twist_batch", vld, vgd, "")(srcPtr,
                                                                                   dstPtr,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                                   plnpkdind,
                                                                                   plnpkdind);

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::string kernel_file = "color_twist.cpp";
    std::string kernel_name = "color_twist_batch";
    get_kernel_name(kernel_name, tensor_info);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);

    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "",  kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                      handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                      handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                      handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                      handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                      handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                      handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                      handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                      handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                      handle.GetInitHandle()->mem.mgpu.inc,
                                                                      handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                      in_plnpkdind,
                                                                      out_plnpkdind);

#elif defined(STATIC)

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_color_twist_batch(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_color_twist_batch_fp16(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_color_twist_batch_fp32(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
color_twist_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_color_twist_batch_int8(srcPtr, dstPtr, handle, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** resize_crop_mirror ********************/

RppStatus
resize_crop_mirror_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::string kernel_file = "resize.cpp";
    std::string kernel_name = "resize_crop_mirror_batch";
    get_kernel_name(kernel_name, tensor_info);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);

    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", kernel_file , kernel_name, vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                      handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                      handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                      handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                      handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                      handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                      handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                      handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                      tensor_info._in_channels,
                                                                      handle.GetInitHandle()->mem.mgpu.inc,
                                                                      handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                      in_plnpkdind,
                                                                      out_plnpkdind);

#elif defined(STATIC)

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_resize_crop_mirror_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_resize_crop_mirror_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_resize_crop_mirror_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
resize_crop_mirror_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_resize_crop_mirror_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** crop ********************/

RppStatus
crop_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::string kernel_file  = "crop.cpp";
    std::string kernel_name = "crop_batch";
    get_kernel_name(kernel_name, tensor_info);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);

    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width , max_height, handle.GetBatchSize()};

    handle.AddKernel("", "",kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                    dstPtr,
                                                                    handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                    handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                    handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                    handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                    tensor_info._in_channels,
                                                                    handle.GetInitHandle()->mem.mgpu.inc,
                                                                    handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                    in_plnpkdind,
                                                                    out_plnpkdind);

#elif defined(STATIC)

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_batch_u8_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_batch_u8_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_batch_u8_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

/******************** crop_mirror_normalize ********************/

RppStatus
crop_mirror_normalize_hip(
    Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiSize dstSize, Rpp32u crop_pox_x, Rpp32u crop_pos_y,
    Rpp32f mean, Rpp32f std_dev, Rpp32u mirrorFlag, Rpp32u outputFormatToggle, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

    std::string kernel_file = "crop_mirror_normalize.cpp";
    std::string kernel_name = "crop_mirror_normalize_batch";
    get_kernel_name(kernel_name, tensor_info);

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);

    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

#elif defined(STATIC)

    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    hip_exec_crop_mirror_normalize_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_mirror_normalize_batch_u8_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_mirror_normalize_batch_u8_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_mirror_normalize_batch_u8_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_mirror_normalize_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_mirror_normalize_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_mirror_normalize_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format);
    int out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.cdstSize.height, handle.GetInitHandle()->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_mirror_normalize_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}