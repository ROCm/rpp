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
#include "rppt_tensor_geometric_augmentations.h"
#include "cpu/host_tensor_geometric_augmentations.hpp"

#ifdef HIP_COMPILE
#include <hip/hip_fp16.h>
#include "hip/hip_tensor_geometric_augmentations.hpp"
#endif // HIP_COMPILE

/******************** crop ********************/

RppStatus rppt_crop_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** crop mirror normalize ********************/

RppStatus rppt_crop_mirror_normalize_host(RppPtr_t srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          RppPtr_t dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *meanTensor,
                                          Rpp32f *stdDevTensor,
                                          Rpp32u *mirrorTensor,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        crop_mirror_normalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                meanTensor,
                                                stdDevTensor,
                                                mirrorTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        crop_mirror_normalize_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        crop_mirror_normalize_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        crop_mirror_normalize_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                meanTensor,
                                                stdDevTensor,
                                                mirrorTensor,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams);
    }
    return RPP_SUCCESS;
}

/******************** flip ********************/

RppStatus rppt_flip_host(RppPtr_t srcPtr,
                         RpptDescPtr srcDescPtr,
                         RppPtr_t dstPtr,
                         RpptDescPtr dstDescPtr,
                         Rpp32u *horizontalTensor,
                         Rpp32u *verticalTensor,
                         RpptROIPtr roiTensorPtrSrc,
                         RpptRoiType roiType,
                         rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        flip_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               horizontalTensor,
                               verticalTensor,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        flip_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 horizontalTensor,
                                 verticalTensor,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        flip_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 horizontalTensor,
                                 verticalTensor,
                                 roiTensorPtrSrc,
                                 roiType,
                                 layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        flip_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                               srcDescPtr,
                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                               dstDescPtr,
                               horizontalTensor,
                               verticalTensor,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** resize ********************/

RppStatus rppt_resize_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           RpptImagePatchPtr dstImgSizes,
                           RpptInterpolationType interpolationType,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if(interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_nn_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        dstImgSizes,
                                        roiTensorPtrSrc,
                                        roiType,
                                        srcLayoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_nn_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          dstImgSizes,
                                          roiTensorPtrSrc,
                                          roiType,
                                          srcLayoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_nn_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                          srcDescPtr,
                                          (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                          dstDescPtr,
                                          dstImgSizes,
                                          roiTensorPtrSrc,
                                          roiType,
                                          srcLayoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_nn_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        dstImgSizes,
                                        roiTensorPtrSrc,
                                        roiType,
                                        srcLayoutParams);
        }
    }
    else if(interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_bilinear_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              dstImgSizes,
                                              roiTensorPtrSrc,
                                              roiType,
                                              srcLayoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_bilinear_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                roiTensorPtrSrc,
                                                roiType,
                                                srcLayoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_bilinear_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                roiTensorPtrSrc,
                                                roiType,
                                                srcLayoutParams);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_bilinear_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                               srcDescPtr,
                                               static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                               dstDescPtr,
                                               dstImgSizes,
                                               roiTensorPtrSrc,
                                               roiType,
                                               srcLayoutParams);
        }
    }
    else
    {
        RpptDesc tempDesc;
        tempDesc = *srcDescPtr;
        RpptDescPtr tempDescPtr = &tempDesc;
        tempDescPtr->h = dstDescPtr->h;
        tempDescPtr->strides.nStride = srcDescPtr->w * dstDescPtr->h * srcDescPtr->c;

        // The channel stride changes with the change in the height for PLN images
        if(srcDescPtr->layout == RpptLayout::NCHW)
            tempDescPtr->strides.cStride = srcDescPtr->w * dstDescPtr->h;

        if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
        {
            resize_separable_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.tempFloatmem,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
        {
            resize_separable_host_tensor(static_cast<Rpp32f*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.tempFloatmem,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType);
        }
        else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
        {
            resize_separable_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.tempFloatmem,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType);
        }
        else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
        {
            resize_separable_host_tensor(static_cast<Rpp16f*>(srcPtr) + srcDescPtr->offsetInBytes,
                                         srcDescPtr,
                                         static_cast<Rpp16f*>(dstPtr) + dstDescPtr->offsetInBytes,
                                         dstDescPtr,
                                         rpp::deref(rppHandle).GetInitHandle()->mem.mcpu.tempFloatmem,
                                         tempDescPtr,
                                         dstImgSizes,
                                         roiTensorPtrSrc,
                                         roiType,
                                         srcLayoutParams,
                                         interpolationType);
        }
    }

    return RPP_SUCCESS;
}

/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize_host(RppPtr_t srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            RppPtr_t dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSizes,
                                            RpptInterpolationType interpolationType,
                                            Rpp32f *meanTensor,
                                            Rpp32f *stdDevTensor,
                                            Rpp32u *mirrorTensor,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            rppHandle_t rppHandle)
{
    RppLayoutParams srcLayoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        resize_mirror_normalize_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  dstImgSizes,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  srcLayoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_mirror_normalize_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    dstImgSizes,
                                                    meanTensor,
                                                    stdDevTensor,
                                                    mirrorTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    srcLayoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_mirror_normalize_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                    srcDescPtr,
                                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                    dstDescPtr,
                                                    dstImgSizes,
                                                    meanTensor,
                                                    stdDevTensor,
                                                    mirrorTensor,
                                                    roiTensorPtrSrc,
                                                    roiType,
                                                    srcLayoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        resize_mirror_normalize_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                  srcDescPtr,
                                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                  dstDescPtr,
                                                  dstImgSizes,
                                                  meanTensor,
                                                  stdDevTensor,
                                                  mirrorTensor,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  srcLayoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resize_mirror_normalize_u8_f32_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   dstImgSizes,
                                                   meanTensor,
                                                   stdDevTensor,
                                                   mirrorTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   srcLayoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        resize_mirror_normalize_u8_f16_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                   srcDescPtr,
                                                   (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                   dstDescPtr,
                                                   dstImgSizes,
                                                   meanTensor,
                                                   stdDevTensor,
                                                   mirrorTensor,
                                                   roiTensorPtrSrc,
                                                   roiType,
                                                   srcLayoutParams);
    }

    return RPP_SUCCESS;
}

/********************************************************************************************************************/
/*********************************************** RPP_GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** crop ********************/

RppStatus rppt_crop_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** crop mirror normalize ********************/

RppStatus rppt_crop_mirror_normalize_gpu(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *meanTensor,
                                         Rpp32f *stdDevTensor,
                                         Rpp32u *mirrorTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_float(meanTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_crop_mirror_normalize_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_crop_mirror_normalize_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_crop_mirror_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** warp_affine ********************/

RppStatus rppt_warp_affine_gpu(RppPtr_t srcPtr,
                               RpptDescPtr srcDescPtr,
                               RppPtr_t dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *affineTensor,
                               RpptInterpolationType interpolationType,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
        return RPP_ERROR_NOT_IMPLEMENTED;

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_warp_affine_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_warp_affine_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_warp_affine_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                    srcDescPtr,
                                    static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                    dstDescPtr,
                                    affineTensor,
                                    interpolationType,
                                    roiTensorPtrSrc,
                                    roiType,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** flip ********************/

RppStatus rppt_flip_gpu(RppPtr_t srcPtr,
                        RpptDescPtr srcDescPtr,
                        RppPtr_t dstPtr,
                        RpptDescPtr dstDescPtr,
                        Rpp32u *horizontalTensor,
                        Rpp32u *verticalTensor,
                        RpptROIPtr roiTensorPtrSrc,
                        RpptRoiType roiType,
                        rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    Rpp32u paramIndex = 0;
    copy_param_uint(horizontalTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(verticalTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_flip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_flip_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_flip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_flip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** resize_mirror_normalize ********************/

RppStatus rppt_resize_mirror_normalize_gpu(RppPtr_t srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           RppPtr_t dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptImagePatchPtr dstImgSizes,
                                           RpptInterpolationType interpolationType,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           Rpp32u *mirrorTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if (interpolationType != RpptInterpolationType::BILINEAR)
        return RPP_ERROR_NOT_IMPLEMENTED;

    Rpp32u paramIndex = 0;
    if(srcDescPtr->c == 3)
    {
        copy_param_float3(meanTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float3(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    }
    else if(srcDescPtr->c == 1)
    {
        copy_param_float(meanTensor, rpp::deref(rppHandle), paramIndex++);
        copy_param_float(stdDevTensor, rpp::deref(rppHandle), paramIndex++);
    }
    copy_param_uint(mirrorTensor, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_mirror_normalize_tensor((half*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (half*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }

    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_mirror_normalize_tensor((Rpp32f*)(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                srcDescPtr,
                                                (Rpp32f*)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                (Rpp32f *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
                                                roiTensorPtrSrc,
                                                roiType,
                                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_resize_mirror_normalize_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                (half *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                dstDescPtr,
                                                dstImgSizes,
                                                interpolationType,
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