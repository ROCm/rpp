#ifndef HIP_DECLARATIONS_INLINE_H
#define HIP_DECLARATIONS_INLINE_H

#include "rpp.h"
#include "hip/rpp/handle.hpp"
#include "hip/rpp_hip_common.hpp"
#include "kernel/rpp_hip_host_decls.hpp"

/******************** color_convert ********************/

template <typename T, typename U>
RppStatus
color_convert_hip_batch(T* srcPtr, U* dstPtr, RppiColorConvertMode convert_mode, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    std::string kernel_name ;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    if (convert_mode == RGB_HSV)
        kernel_name = "convert_batch_rgb_hsv";
    if (convert_mode == HSV_RGB)
        kernel_name = "convert_batch_hsv_rgb";
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "hue.cpp", kernel_name, vld, vgd, "")(srcPtr,
                                                                   dstPtr,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                   handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                   handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                   handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                   handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                   handle.GetInitHandle()->mem.mgpu.inc,
                                                                   plnpkdind);

    return RPP_SUCCESS;
}

/******************** non_linear_blend ********************/

template <typename T, typename U>
RppStatus
non_linear_blend_hip_batch_tensor(T* srcPtr1, T* srcPtr2, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "non_linear_blend.cpp";
    std::string kernel_name = "non_linear_blend_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1,
                                                                     srcPtr2,
                                                                     dstPtr,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** water ********************/

template <typename T, typename U>
RppStatus
water_hip_batch_tensor(T* srcPtr, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "water.cpp";
    std::string kernel_name = "water_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[4].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[5].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** erase ********************/

template <typename T, typename U>
RppStatus
erase_hip_batch_tensor(T* srcPtr, U* dstPtr, Rpp32u* anchor_box_info, T* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "erase.cpp";
    std::string kernel_name = "erase_batch";
    std::string kernel_pln1_name = "erase_pln1_batch";
    get_kernel_name(kernel_name, tensor_info);
    get_kernel_name(kernel_pln1_name, tensor_info);

    if (tensor_info._in_channels == 3)
        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                         dstPtr,
                                                                         anchor_box_info,
                                                                         colors,
                                                                         box_offset,
                                                                         handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                         handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                         handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                         handle.GetInitHandle()->mem.mgpu.inc,
                                                                         handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                         in_plnpkdind,
                                                                         out_plnpkdind);
    else
        handle.AddKernel("", "", kernel_file, kernel_pln1_name, vld, vgd, "")(srcPtr,
                                                                              dstPtr,
                                                                              anchor_box_info,
                                                                              colors,
                                                                              box_offset,
                                                                              handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                              handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                              handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                              handle.GetInitHandle()->mem.mgpu.inc,
                                                                              handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                              in_plnpkdind,
                                                                              out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

template <typename T, typename U>
RppStatus
color_cast_hip_batch_tensor(T* srcPtr, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "color_cast.cpp";
    std::string kernel_name = "color_cast_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     handle.GetInitHandle()->mem.mgpu.ucharArr[0].ucharmem,
                                                                     handle.GetInitHandle()->mem.mgpu.ucharArr[1].ucharmem,
                                                                     handle.GetInitHandle()->mem.mgpu.ucharArr[2].ucharmem,
                                                                     handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** lut ********************/

template <typename T, typename U>
RppStatus
lut_hip_batch_tensor(T* srcPtr, U* dstPtr, T* lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "look_up_table.cpp";
    std::string kernel_name = "look_up_table_batch_tensor";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     lut,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** crop_and_patch ********************/

template <typename T, typename U>
RppStatus
crop_and_patch_hip_batch_tensor(T* srcPtr1, T* srcPtr2, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "crop_and_patch.cpp";
    std::string kernel_name = "crop_and_patch_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1,
                                                                     srcPtr2,
                                                                     dstPtr,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[6].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[7].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** glitch ********************/

template <typename T, typename U>
RppStatus
glitch_hip_batch_tensor(T* srcPtr, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "glitch.cpp";
    std::string kernel_name = "glitch_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                     dstPtr,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                     handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                     handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                     handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle.GetInitHandle()->mem.mgpu.inc,
                                                                     handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                     in_plnpkdind,
                                                                     out_plnpkdind);

    return RPP_SUCCESS;
}

/******************** tensor_transpose ********************/

template <typename T, typename U>
RppStatus
tensor_transpose_hip(T* srcPtr, U* dstPtr,  Rpp32u* in_dims, Rpp32u *perm, RPPTensorDataType data_type, rpp::Handle& handle)
{
    unsigned int out_dims[4];
    out_dims[0] = in_dims[perm[0]];
    out_dims[1] = in_dims[perm[1]];
    out_dims[2] = in_dims[perm[2]];
    out_dims[3] = in_dims[perm[3]];

    unsigned int in_strides[4], out_strides[4];
    in_strides[0] = in_dims[1] * in_dims[2] * in_dims[3];
    in_strides[1] = in_dims[2] * in_dims[3];
    in_strides[2] = in_dims[3];
    in_strides[3] = 1;

    out_strides[0] = out_dims[1] * out_dims[2] * out_dims[3];
    out_strides[1] = out_dims[2] * out_dims[3];
    out_strides[2] = out_dims[3];
    out_strides[3] = 1;

    Rpp32u *d_perm, *d_in_strides, *d_out_strides, *d_out_dims;
    hipMalloc(&d_perm, 4 * sizeof(Rpp32u));
    hipMalloc(&d_in_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_strides, 4 * sizeof(Rpp32u));
    hipMalloc(&d_out_dims, 4 * sizeof(Rpp32u));
    hipMemcpy(d_perm, perm, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_in_strides, in_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_strides, out_strides, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_out_dims, out_dims, 4 * sizeof(Rpp32u), hipMemcpyHostToDevice);

    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{out_dims[0], out_dims[1], out_dims[2] * out_dims[3]};
    std::string kernel_name = "tensor_transpose";
    if(data_type == RPPTensorDataType::U8)
        kernel_name = "tensor_transpose";
    if(data_type == RPPTensorDataType::FP32)
        kernel_name = "tensor_transpose_fp32";
    if(data_type == RPPTensorDataType::FP16)
        kernel_name = "tensor_transpose_fp16";
    if(data_type == RPPTensorDataType::I8)
        kernel_name = "tensor_transpose_int8";

    handle.AddKernel("", "", "tensor.cpp", kernel_name, vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      d_out_dims,
                                                                      d_perm,
                                                                      d_out_strides,
                                                                      d_in_strides);

    return RPP_SUCCESS;
}

#endif // HIP_DECLARATIONS_INLINE_H