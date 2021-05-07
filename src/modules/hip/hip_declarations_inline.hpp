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