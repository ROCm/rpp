#ifndef HIP_DECLARATIONS_INLINE_H
#define HIP_DECLARATIONS_INLINE_H

#include "rpp.h"
#include "hip/rpp/handle.hpp"
#include "hip/rpp_hip_common.hpp"

template <typename T, typename U>
RppStatus
resize_hip_batch_tensor(T* srcPtr, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    unsigned int padding = 0;
    unsigned int type = 0;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "resize.cpp";
    std::string kernel_name = "resize_crop_batch";
    get_kernel_name(kernel_name, tensor_info);
    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight, 
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        padding,
                                                                        type,
                                                                        in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus
resize_crop_hip_batch_tensor(T* srcPtr, U* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    unsigned int padding = 10;
    unsigned int type = 1;
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
//     int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "resize.cpp";
    std::string kernel_name = "resize_crop_batch";
    get_kernel_name(kernel_name, tensor_info);
    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
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
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        padding,
                                                                        type,
                                                                        in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus
rotate_hip_batch_tensor(T *srcPtr, U *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    // int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "rotate.cpp";
    std::string kernel_name = "rotate_batch";
    get_kernel_name(kernel_name, tensor_info);
    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.height,
                                                                        handle.GetInitHandle()->mem.mgpu.dstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                        handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,   
                                                                        handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.maxDstSize.width,
                                                                        handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                        handle.GetInitHandle()->mem.mgpu.dstBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle.GetInitHandle()->mem.mgpu.inc,
                                                                        handle.GetInitHandle()->mem.mgpu.dstInc,
                                                                        in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus
warp_affine_hip_batch_tensor(T *srcPtr, U *dstPtr, rpp::Handle &handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    // int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.cdstSize.height, handle_obj->mem.mgpu.cdstSize.width, handle.GetBatchSize(), &max_height, &max_width);
    Rpp32f *hipAffine;
    hipMalloc(&hipAffine, handle.GetBatchSize() * 6 * sizeof(Rpp32f));
    hipMemcpy(hipAffine, affine, handle.GetBatchSize() * 6 * sizeof(Rpp32f), hipMemcpyHostToDevice);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width ,max_height , handle.GetBatchSize()};
    std::string kernel_file  = "warp_affine.cpp";
    std::string kernel_name = "warp_affine_batch";
    get_kernel_name(kernel_name, tensor_info);
    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                        hipAffine,
                                                                        handle_obj->mem.mgpu.srcSize.height,
                                                                        handle_obj->mem.mgpu.srcSize.width,
                                                                        handle_obj->mem.mgpu.dstSize.height,
                                                                        handle_obj->mem.mgpu.dstSize.width,
                                                                        handle_obj->mem.mgpu.roiPoints.x,
                                                                        handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                        handle_obj->mem.mgpu.roiPoints.y,
                                                                        handle_obj->mem.mgpu.roiPoints.roiHeight,   
                                                                        handle_obj->mem.mgpu.maxSrcSize.width,
                                                                        handle_obj->mem.mgpu.maxDstSize.width,
                                                                        handle_obj->mem.mgpu.srcBatchIndex,
                                                                        handle_obj->mem.mgpu.dstBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle_obj->mem.mgpu.inc,
                                                                        handle_obj->mem.mgpu.dstInc,
                                                                        in_plnpkdind, out_plnpkdind);
    hipFree(hipAffine);
    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus
color_convert_hip_batch ( T* srcPtr,
                 U* dstPtr,  RppiColorConvertMode convert_mode,
                 RppiChnFormat chnFormat, unsigned int channel,
                 rpp::Handle& handle){
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
    handle.AddKernel("", "", "hue.cpp", kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                    handle.GetInitHandle()->mem.mgpu.inc,
                                                                    plnpkdind
                                                                    );
    return RPP_SUCCESS;
}

#endif // HIP_DECLARATIONS_INLINE_H