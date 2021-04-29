#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include "kernel/rpp_hip_host_decls.hpp"

/******************** non_linear_blend ********************/

RppStatus
non_linear_blend_hip_batch_tensor(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
        std::string kernel_file = "non_linear_blend.cpp";
        std::string kernel_name = "non_linear_blend_batch";
        get_kernel_name(kernel_name, tensor_info);

        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1,
                                                                        srcPtr2,
                                                                        dstPtr,
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
                                                                        in_plnpkdind,
                                                                        out_plnpkdind);

    #elif defined(STATIC)

        hip_exec_non_linear_blend_batch(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
non_linear_blend_hip_batch_tensor_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_non_linear_blend_batch_fp16(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
non_linear_blend_hip_batch_tensor_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_non_linear_blend_batch_fp32(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
non_linear_blend_hip_batch_tensor_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_non_linear_blend_batch_int8(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

/******************** water ********************/

RppStatus
water_hip_batch_tensor(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
        std::string kernel_file = "water.cpp";
        std::string kernel_name = "water_batch";
        get_kernel_name(kernel_name, tensor_info);

        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                        dstPtr,
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
                                                                        handle_obj->mem.mgpu.maxSrcSize.height,
                                                                        handle_obj->mem.mgpu.maxSrcSize.width,
                                                                        handle_obj->mem.mgpu.srcBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle_obj->mem.mgpu.inc,
                                                                        handle_obj->mem.mgpu.dstInc,
                                                                        in_plnpkdind,
                                                                        out_plnpkdind);

    #elif defined(STATIC)

        hip_exec_water_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
water_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_water_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
water_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_water_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
water_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_water_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}
/******************** erase ********************/

RppStatus
erase_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp32u* anchor_box_info, Rpp8u* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

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
                                                                            handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                            handle_obj->mem.mgpu.srcSize.height,
                                                                            handle_obj->mem.mgpu.srcSize.width,
                                                                            handle_obj->mem.mgpu.maxSrcSize.width,
                                                                            handle_obj->mem.mgpu.srcBatchIndex,
                                                                            handle_obj->mem.mgpu.inc,
                                                                            handle_obj->mem.mgpu.dstInc,
                                                                            in_plnpkdind,
                                                                            out_plnpkdind);
        else
            handle.AddKernel("", "", kernel_file, kernel_pln1_name, vld, vgd, "")(srcPtr,
                                                                                dstPtr,
                                                                                anchor_box_info,
                                                                                colors,
                                                                                box_offset,
                                                                                handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                                handle_obj->mem.mgpu.srcSize.height,
                                                                                handle_obj->mem.mgpu.srcSize.width,
                                                                                handle_obj->mem.mgpu.maxSrcSize.width,
                                                                                handle_obj->mem.mgpu.srcBatchIndex,
                                                                                handle_obj->mem.mgpu.inc,
                                                                                handle_obj->mem.mgpu.dstInc,
                                                                                in_plnpkdind,
                                                                                out_plnpkdind);

    #elif defined(STATIC)

        hip_exec_erase_batch(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
erase_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, Rpp32u* anchor_box_info, Rpp16f* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_erase_batch_fp16(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
erase_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, Rpp32u* anchor_box_info, Rpp32f* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_erase_batch_fp32(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
erase_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp32u* anchor_box_info, Rpp8s* colors, Rpp32u* box_offset, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_erase_batch_int8(srcPtr, dstPtr, anchor_box_info, colors, handle, box_offset, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

RppStatus
color_cast_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
        std::string kernel_file = "color_cast.cpp";
        std::string kernel_name = "color_cast_batch";
        get_kernel_name(kernel_name, tensor_info);

        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        handle_obj->mem.mgpu.ucharArr[0].ucharmem,
                                                                        handle_obj->mem.mgpu.ucharArr[1].ucharmem,
                                                                        handle_obj->mem.mgpu.ucharArr[2].ucharmem,
                                                                        handle_obj->mem.mgpu.floatArr[3].floatmem,
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
                                                                        in_plnpkdind,
                                                                        out_plnpkdind);

    #elif defined(STATIC)

            hip_exec_color_cast_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
color_cast_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_color_cast_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
color_cast_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_color_cast_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
color_cast_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_color_cast_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}
/******************** lut ********************/

RppStatus
lut_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, Rpp8u* lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
        std::string kernel_file = "look_up_table.cpp";
        std::string kernel_name = "look_up_table_batch_tensor";
        get_kernel_name(kernel_name, tensor_info);

        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        lut,
                                                                        handle_obj->mem.mgpu.srcSize.height,
                                                                        handle_obj->mem.mgpu.srcSize.width,
                                                                        handle_obj->mem.mgpu.maxSrcSize.width,
                                                                        handle_obj->mem.mgpu.srcBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle_obj->mem.mgpu.inc,
                                                                        handle_obj->mem.mgpu.dstInc,
                                                                        in_plnpkdind,
                                                                        out_plnpkdind);

    #elif defined(STATIC)

            hip_exec_lut_batch(srcPtr, dstPtr, lut, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
lut_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, Rpp8s* lut, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_lut_batch_int8(srcPtr, dstPtr, lut, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

/******************** crop_and_patch ********************/

RppStatus
crop_and_patch_hip_batch_tensor(Rpp8u* srcPtr1, Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
        std::string kernel_file = "crop_and_patch.cpp";
        std::string kernel_name = "crop_and_patch_batch";
        get_kernel_name(kernel_name, tensor_info);

        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr1,
                                                                        srcPtr2,
                                                                        dstPtr,
                                                                        handle_obj->mem.mgpu.srcSize.height,
                                                                        handle_obj->mem.mgpu.srcSize.width,
                                                                        handle_obj->mem.mgpu.dstSize.height,
                                                                        handle_obj->mem.mgpu.dstSize.width,
                                                                        handle_obj->mem.mgpu.uintArr[4].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[5].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[6].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[7].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[2].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[3].uintmem,
                                                                        handle_obj->mem.mgpu.maxSrcSize.width,
                                                                        handle_obj->mem.mgpu.maxDstSize.width,
                                                                        handle_obj->mem.mgpu.srcBatchIndex,
                                                                        handle_obj->mem.mgpu.dstBatchIndex,
                                                                        tensor_info._in_channels,
                                                                        handle_obj->mem.mgpu.inc,
                                                                        handle_obj->mem.mgpu.dstInc,
                                                                        in_plnpkdind,
                                                                        out_plnpkdind);

    #elif defined(STATIC)

        hip_exec_crop_and_patch_batch(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
crop_and_patch_hip_batch_tensor_fp16(Rpp16f *srcPtr1, Rpp16f *srcPtr2, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_crop_and_patch_batch_fp16(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
crop_and_patch_hip_batch_tensor_fp32(Rpp32f *srcPtr1, Rpp32f *srcPtr2, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_and_patch_batch_fp32(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}

RppStatus
crop_and_patch_hip_batch_tensor_int8(Rpp8s *srcPtr1, Rpp8s *srcPtr2, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

#if defined (HIPRTC)

#elif defined(STATIC)

    hip_exec_crop_and_patch_batch_int8(srcPtr1, srcPtr2, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

#endif

    return RPP_SUCCESS;
}


/******************** glitch ********************/

RppStatus
glitch_hip_batch_tensor(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};
        std::string kernel_file = "glitch.cpp";
        std::string kernel_name = "glitch_batch";
        get_kernel_name(kernel_name, tensor_info);

        handle.AddKernel("", "", kernel_file, kernel_name, vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        handle_obj->mem.mgpu.uintArr[0].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[1].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[2].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[3].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[4].uintmem,
                                                                        handle_obj->mem.mgpu.uintArr[5].uintmem,
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
                                                                        in_plnpkdind,
                                                                        out_plnpkdind);

    #elif defined(STATIC)

            hip_exec_glitch_batch(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
glitch_hip_batch_tensor_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_glitch_batch_fp16(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

        return RPP_SUCCESS;
}

RppStatus
glitch_hip_batch_tensor_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_glitch_batch_fp32(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}

RppStatus
glitch_hip_batch_tensor_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle &handle, RPPTensorFunctionMetaData &tensor_info)
{
    int in_plnpkdind = getplnpkdind(tensor_info._in_format), out_plnpkdind = getplnpkdind(tensor_info._out_format);
    InitHandle *handle_obj = handle.GetInitHandle();
    Rpp32u max_height, max_width;
    max_size(handle_obj->mem.mgpu.csrcSize.height, handle_obj->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    #if defined (HIPRTC)

    #elif defined(STATIC)

        hip_exec_glitch_batch_int8(srcPtr, dstPtr, handle, tensor_info, in_plnpkdind, out_plnpkdind, max_height, max_width);

    #endif

    return RPP_SUCCESS;
}