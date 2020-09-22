#include <cl/rpp_cl_common.hpp>
#include "cl_declarations.hpp"

RppStatus
non_linear_blend_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, 
                                                            RPPTensorFunctionMetaData &tensor_info)
{

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "non_linear_blend.cl";
    std::string kernel_name = "non_linear_blend_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1, srcPtr2, dstPtr,
                                                                     handle_obj->mem.mgpu.floatArr[0].floatmem,
                                                                     handle_obj->mem.mgpu.roiPoints.x,
                                                                     handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                     handle_obj->mem.mgpu.roiPoints.y,
                                                                     handle_obj->mem.mgpu.roiPoints.roiHeight,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

RppStatus
water_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle &handle, 
                                                            RPPTensorFunctionMetaData &tensor_info)
{

    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "water.cl";
    std::string kernel_name = "water_batch";
    get_kernel_name(kernel_name, tensor_info);

    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr,
                                                                     handle_obj->mem.mgpu.floatArr[0].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[1].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[2].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[3].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[4].floatmem,
                                                                     handle_obj->mem.mgpu.floatArr[5].floatmem,
                                                                     handle_obj->mem.mgpu.roiPoints.x,
                                                                     handle_obj->mem.mgpu.roiPoints.roiWidth,
                                                                     handle_obj->mem.mgpu.roiPoints.y,
                                                                     handle_obj->mem.mgpu.roiPoints.roiHeight,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);
    return RPP_SUCCESS;
}

RppStatus
erase_cl_batch(cl_mem srcPtr, cl_mem dstPtr, cl_mem anchor_box_info, cl_mem colors, cl_mem box_offset,
                             rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "erase.cl";
    std::string kernel_name = "erase_batch";
    get_kernel_name(kernel_name, tensor_info);
    std::cout << kernel_file << "\t" << kernel_name << std::endl;
    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr, anchor_box_info, colors, box_offset,
                                                                     handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}

RppStatus
lut_cl_batch(cl_mem srcPtr, cl_mem dstPtr, cl_mem lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    int batch_size = handle.GetBatchSize();
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{16, 16, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
    std::string kernel_file = "look_up_table.cl";
    std::string kernel_name = "look_up_table_tensor";
    get_kernel_name(kernel_name, tensor_info);
    std::cout << kernel_file << "\t" << kernel_name << std::endl;
    handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr, dstPtr, lut,
                                                                     handle_obj->mem.mgpu.srcSize.height,
                                                                     handle_obj->mem.mgpu.srcSize.width,
                                                                     handle_obj->mem.mgpu.maxSrcSize.width,
                                                                     handle_obj->mem.mgpu.srcBatchIndex,
                                                                     tensor_info._in_channels,
                                                                     handle_obj->mem.mgpu.inc,
                                                                     handle_obj->mem.mgpu.dstInc,
                                                                     in_plnpkdind, out_plnpkdind);

    return RPP_SUCCESS;
}
