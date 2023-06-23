/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_statistical_operations.h"
#include "cpu/host_tensor_statistical_operations.hpp"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_statistical_operations.hpp"
#endif // HIP_COMPILE

/******************** image_sum ********************/

RppStatus rppt_image_min_max_host(RppPtr_t srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t imageMinMaxArr,
                                  Rpp32u imageMinMaxArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (imageMinMaxArrLength < srcDescPtr->n * 2)      // min, max pair for each image
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMinMaxArrLength < srcDescPtr->n * 8)  // min, max pair of each channel, and min, max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        image_min_max_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(imageMinMaxArr),
                                        imageMinMaxArrLength,
                                        roiTensorPtrSrc,
                                        roiType,
                                        layoutParams);
    }
    /*else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        image_min_max_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           static_cast<Rpp32f*>(imageMinMaxArr),
                                           imageMinMaxArrLength,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        image_min_max_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                           srcDescPtr,
                                           static_cast<Rpp32f*>(imageMinMaxArr),
                                           imageMinMaxArrLength,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        image_min_max_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp32f*>(imageMinMaxArr),
                                         imageMinMaxArrLength,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams);
    }*/

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** image_min ********************/

RppStatus rppt_image_min_gpu(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             Rpp32f *imageMinArr,
                             Rpp32u imageMinArrLength,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE

    if (srcDescPtr->c == 1)
    {
        if (imageMinArrLength < srcDescPtr->n)   // min and max of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMinArrLength < srcDescPtr->n * 3)   // min and max of each channel, and overall min and max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_image_min_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  imageMinArr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }


    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

RppStatus rppt_image_max_gpu(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             Rpp32f *imageMaxArr,
                             Rpp32u imageMaxArrLength,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c == 1)
    {
        if (imageMaxArrLength < srcDescPtr->n)   // max of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (imageMaxArrLength < srcDescPtr->n * 3)   // max of each channel, and overall max of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_image_max_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  imageMaxArr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
    }


    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT