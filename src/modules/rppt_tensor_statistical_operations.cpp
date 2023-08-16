/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

/******************** tensor_mean ********************/

RppStatus rppt_tensor_mean_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t tensorMeanArr,
                                Rpp32u tensorMeanArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (tensorMeanArrLength < srcDescPtr->n)      // mean of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorMeanArrLength < srcDescPtr->n * 4)  // mean of each channel, and total mean of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        tensor_mean_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp32f*>(tensorMeanArr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        tensor_mean_f16_f16_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 static_cast<Rpp32f*>(tensorMeanArr),
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        tensor_mean_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 static_cast<Rpp32f*>(tensorMeanArr),
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        tensor_mean_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp32f*>(tensorMeanArr),
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** tensor_stddev ********************/

RppStatus rppt_tensor_stddev_host(RppPtr_t srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  RppPtr_t tensorStddevArr,
                                  Rpp32u tensorStddevArrLength,
                                  Rpp32f *meanTensor,
                                  int flag,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rppHandle_t rppHandle)
{
    if (srcDescPtr->c == 1)
    {
        if (tensorStddevArrLength < srcDescPtr->n)      // mean of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorStddevArrLength < srcDescPtr->n * 4)  // mean of each channel, and total mean of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        if(!flag || flag == 1)
            custom_stddev_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(tensorStddevArr),
                                     meanTensor,
                                     flag,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
        if(flag == 2)
            tensor_stddev_u8_u8_host(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(tensorStddevArr),
                                     meanTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        if(!flag || flag == 1)
            custom_stddev_f16_f16_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       static_cast<Rpp32f*>(tensorStddevArr),
                                       meanTensor,
                                       flag,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
        if(flag == 2)
            tensor_stddev_f16_f16_host(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       static_cast<Rpp32f*>(tensorStddevArr),
                                       meanTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        if(!flag || flag == 1)
            custom_stddev_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       static_cast<Rpp32f*>(tensorStddevArr),
                                       meanTensor,
                                       flag,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
        if(flag == 2)
            tensor_stddev_f32_f32_host(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       static_cast<Rpp32f*>(tensorStddevArr),
                                       meanTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        if(!flag || flag == 1)
            custom_stddev_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(tensorStddevArr),
                                     meanTensor,
                                     flag,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
        if(flag == 2)
            tensor_stddev_i8_i8_host(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp32f*>(tensorStddevArr),
                                     meanTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** tensor_mean ********************/

RppStatus rppt_tensor_mean_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t tensorMeanArr,
                               Rpp32u tensorMeanArrLength,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (srcDescPtr->c == 1)
    {
        if (tensorMeanArrLength < srcDescPtr->n)      // Mean of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorMeanArrLength < srcDescPtr->n * 4)  // Mean of each channel, and total Mean of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_tensor_mean(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp32f*>(tensorMeanArr),
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_tensor_mean(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             static_cast<Rpp32f*>(tensorMeanArr),
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_tensor_mean(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             static_cast<Rpp32f*>(tensorMeanArr),
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_tensor_mean(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp32f*>(tensorMeanArr),
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** tensor_stddev ********************/

RppStatus rppt_tensor_stddev_gpu(RppPtr_t srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 RppPtr_t tensorStddevArr,
                                 Rpp32u tensorStddevArrLength,
                                 Rpp32f *meanTensor,
                                 int flag,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    if (srcDescPtr->c == 1)
    {
        if (tensorStddevArrLength < srcDescPtr->n)      // Mean of single channel
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
        copy_param_float(meanTensor, rpp::deref(rppHandle), paramIndex++);
    }
    else if (srcDescPtr->c == 3)
    {
        if (tensorStddevArrLength < srcDescPtr->n * 4)  // Mean of each channel, and total Mean of all 3 channels
            return RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH;
        copy_reduction_param_float(meanTensor, rpp::deref(rppHandle), paramIndex++);
    }
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > REDUCTION_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > REDUCTION_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > REDUCTION_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > REDUCTION_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
        }
    }

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        hip_exec_tensor_stddev(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp32f*>(tensorStddevArr),
                               flag,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        hip_exec_tensor_stddev(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                               srcDescPtr,
                               static_cast<Rpp32f*>(tensorStddevArr),
                               flag,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        hip_exec_tensor_stddev(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                               srcDescPtr,
                               static_cast<Rpp32f*>(tensorStddevArr),
                               flag,
                               roiTensorPtrSrc,
                               roiType,
                               rpp::deref(rppHandle));
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        hip_exec_tensor_stddev(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp32f*>(tensorStddevArr),
                               flag,
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
