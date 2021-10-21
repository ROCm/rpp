/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <rppt_tensor_augmentations.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
    #include "hip/hip_tensor_augmentations.hpp"
#elif defined(OCL_COMPILE)
    #include <cl/rpp_cl_common.hpp>
    #include "cl/cl_declarations.hpp"
#endif //backend

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;

#include "cpu/host_tensor_augmentations.hpp"

/******************** brightness ********************/

RppStatus
rppt_brightness_gpu(RppPtr_t srcPtr,
                    RpptDescPtr srcDescPtr,
                    RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr,
                    Rpp32f *alphaTensor,
                    Rpp32f *betaTensor,
                    RpptROIPtr roiTensorPtrSrc,
                    RpptRoiType roiType,
                    rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

#elif defined (HIP_COMPILE)

    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(betaTensor, rpp::deref(rppHandle), paramIndex++);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        if (dstDescPtr->dataType == RpptDataType::U8)
        {
            brightness_hip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        if (dstDescPtr->dataType == RpptDataType::F16)
        {
            brightness_hip_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        if (dstDescPtr->dataType == RpptDataType::F32)
        {
            brightness_hip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        if (dstDescPtr->dataType == RpptDataType::I8)
        {
            brightness_hip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }

#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppt_brightness_host(RppPtr_t srcPtr,
                     RpptDescPtr srcDescPtr,
                     RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr,
                     Rpp32f *alphaTensor,
                     Rpp32f *betaTensor,
                     RpptROIPtr roiTensorPtrSrc,
                     RpptRoiType roiType,
                     rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        brightness_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     alphaTensor,
                                     betaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        brightness_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       alphaTensor,
                                       betaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        brightness_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       alphaTensor,
                                       betaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        brightness_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     alphaTensor,
                                     betaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

RppStatus
rppt_gamma_correction_gpu(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          Rpp32f *gammaTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

#elif defined (HIP_COMPILE)

    Rpp32u paramIndex = 0;
    copy_param_float(gammaTensor, rpp::deref(rppHandle), paramIndex++);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        if (dstDescPtr->dataType == RpptDataType::U8)
        {
            gamma_correction_hip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        if (dstDescPtr->dataType == RpptDataType::F16)
        {
            gamma_correction_hip_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        if (dstDescPtr->dataType == RpptDataType::F32)
        {
            gamma_correction_hip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                        srcDescPtr,
                                        (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                        dstDescPtr,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        if (dstDescPtr->dataType == RpptDataType::I8)
        {
            gamma_correction_hip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                        srcDescPtr,
                                        static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                        dstDescPtr,
                                        roiTensorPtrSrc,
                                        roiType,
                                        rpp::deref(rppHandle));
        }
    }

#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppt_gamma_correction_host(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32f *gammaTensor,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        gamma_correction_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           gammaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        gamma_correction_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        gamma_correction_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                             srcDescPtr,
                                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                             dstDescPtr,
                                             gammaTensor,
                                             roiTensorPtrSrc,
                                             roiType,
                                             layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        gamma_correction_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                           srcDescPtr,
                                           static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                           dstDescPtr,
                                           gammaTensor,
                                           roiTensorPtrSrc,
                                           roiType,
                                           layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** blend ********************/

RppStatus
rppt_blend_gpu(RppPtr_t srcPtr1,
               RppPtr_t srcPtr2,
               RpptDescPtr srcDescPtr,
               RppPtr_t dstPtr,
               RpptDescPtr dstDescPtr,
               Rpp32f *alphaTensor,
               RpptROIPtr roiTensorPtrSrc,
               RpptRoiType roiType,
               rppHandle_t rppHandle)
{
#ifdef OCL_COMPILE

#elif defined (HIP_COMPILE)

    Rpp32u paramIndex = 0;
    copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        if (dstDescPtr->dataType == RpptDataType::U8)
        {
            blend_hip_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                             static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        if (dstDescPtr->dataType == RpptDataType::F16)
        {
            blend_hip_tensor((half*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                             (half*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        if (dstDescPtr->dataType == RpptDataType::F32)
        {
            blend_hip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                             (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                             srcDescPtr,
                             (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        if (dstDescPtr->dataType == RpptDataType::I8)
        {
            blend_hip_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                             static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                             srcDescPtr,
                             static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                             dstDescPtr,
                             roiTensorPtrSrc,
                             roiType,
                             rpp::deref(rppHandle));
        }
    }

#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppt_blend_host(RppPtr_t srcPtr1,
                RppPtr_t srcPtr2,
                RpptDescPtr srcDescPtr,
                RppPtr_t dstPtr,
                RpptDescPtr dstDescPtr,
                Rpp32f *alphaTensor,
                RpptROIPtr roiTensorPtrSrc,
                RpptRoiType roiType,
                rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        blend_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                alphaTensor,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        blend_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  (Rpp16f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  alphaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        blend_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr1) + srcDescPtr->offsetInBytes),
                                  (Rpp32f*) (static_cast<Rpp8u*>(srcPtr2) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  alphaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        blend_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr1) + srcDescPtr->offsetInBytes,
                                static_cast<Rpp8s*>(srcPtr2) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                alphaTensor,
                                roiTensorPtrSrc,
                                roiType,
                                layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** color_jitter ********************/

// RppStatus
// rppt_color_jitter_gpu(RppPtr_t srcPtr,
//                       RpptDescPtr srcDescPtr,
//                       RppPtr_t dstPtr,
//                       RpptDescPtr dstDescPtr,
//                       Rpp32f *alphaTensor,
//                       Rpp32f *betaTensor,
//                       RpptROIPtr roiTensorPtrSrc,
//                       RpptRoiType roiType,
//                       rppHandle_t rppHandle)
// {
// #ifdef OCL_COMPILE

// #elif defined (HIP_COMPILE)

//     Rpp32u paramIndex = 0;
//     copy_param_float(alphaTensor, rpp::deref(rppHandle), paramIndex++);
//     copy_param_float(betaTensor, rpp::deref(rppHandle), paramIndex++);

//     if (srcDescPtr->dataType == RpptDataType::U8)
//     {
//         if (dstDescPtr->dataType == RpptDataType::U8)
//         {
//             color_jitter_hip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offset,
//                                     srcDescPtr,
//                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offset,
//                                     dstDescPtr,
//                                     roiTensorPtrSrc,
//                                     roiType,
//                                     rpp::deref(rppHandle));
//         }
//     }
//     else if (srcDescPtr->dataType == RpptDataType::F16)
//     {
//         if (dstDescPtr->dataType == RpptDataType::F16)
//         {
//             color_jitter_hip_tensor(static_cast<half*>(srcPtr) + srcDescPtr->offset,
//                                     srcDescPtr,
//                                     static_cast<half*>(dstPtr) + dstDescPtr->offset,
//                                     dstDescPtr,
//                                     roiTensorPtrSrc,
//                                     roiType,
//                                     rpp::deref(rppHandle));
//         }
//     }
//     else if (srcDescPtr->dataType == RpptDataType::F32)
//     {
//         if (dstDescPtr->dataType == RpptDataType::F32)
//         {
//             color_jitter_hip_tensor(static_cast<Rpp32f*>(srcPtr) + srcDescPtr->offset,
//                                     srcDescPtr,
//                                     static_cast<Rpp32f*>(dstPtr) + dstDescPtr->offset,
//                                     dstDescPtr,
//                                     roiTensorPtrSrc,
//                                     roiType,
//                                     rpp::deref(rppHandle));
//         }
//     }
//     else if (srcDescPtr->dataType == RpptDataType::I8)
//     {
//         if (dstDescPtr->dataType == RpptDataType::I8)
//         {
//             color_jitter_hip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offset,
//                                     srcDescPtr,
//                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offset,
//                                     dstDescPtr,
//                                     roiTensorPtrSrc,
//                                     roiType,
//                                     rpp::deref(rppHandle));
//         }
//     }

// #endif //BACKEND

//     return RPP_SUCCESS;
// }

RppStatus
rppt_color_jitter_host(RppPtr_t srcPtr,
                       RpptDescPtr srcDescPtr,
                       RppPtr_t dstPtr,
                       RpptDescPtr dstDescPtr,
                       Rpp32f *brightnessTensor,
                       Rpp32f *contrastTensor,
                       Rpp32f *hueTensor,
                       Rpp32f *saturationTensor,
                       RpptROIPtr roiTensorPtrSrc,
                       RpptRoiType roiType,
                       rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_jitter_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       brightnessTensor,
                                       contrastTensor,
                                       hueTensor,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_jitter_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         brightnessTensor,
                                         contrastTensor,
                                         hueTensor,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_jitter_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                         srcDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                         dstDescPtr,
                                         brightnessTensor,
                                         contrastTensor,
                                         hueTensor,
                                         saturationTensor,
                                         roiTensorPtrSrc,
                                         roiType,
                                         layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_jitter_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                       srcDescPtr,
                                       static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                       dstDescPtr,
                                       brightnessTensor,
                                       contrastTensor,
                                       hueTensor,
                                       saturationTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

RppStatus
rppt_color_cast_gpu(RppPtr_t srcPtr,
                    RpptDescPtr srcDescPtr,
                    RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr,
                    RpptRGBA *rgbaTensor,
                    RpptROIPtr roiTensorPtrSrc,
                    RpptRoiType roiType,
                    rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
    {
        return RPP_INVALID_ARGUMENTS;
    }

#ifdef OCL_COMPILE

#elif defined (HIP_COMPILE)

    copy_param_RpptRGBA(rgbaTensor, rpp::deref(rppHandle));

    if (srcDescPtr->dataType == RpptDataType::U8)
    {
        if (dstDescPtr->dataType == RpptDataType::U8)
        {
            color_cast_hip_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F16)
    {
        if (dstDescPtr->dataType == RpptDataType::F16)
        {
            color_cast_hip_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::F32)
    {
        if (dstDescPtr->dataType == RpptDataType::F32)
        {
            color_cast_hip_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                  srcDescPtr,
                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }
    else if (srcDescPtr->dataType == RpptDataType::I8)
    {
        if (dstDescPtr->dataType == RpptDataType::I8)
        {
            color_cast_hip_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  roiTensorPtrSrc,
                                  roiType,
                                  rpp::deref(rppHandle));
        }
    }

#endif //BACKEND

    return RPP_SUCCESS;
}

RppStatus
rppt_color_cast_host(RppPtr_t srcPtr,
                     RpptDescPtr srcDescPtr,
                     RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr,
                     RpptRGBA *rgbaTensor,
                     RpptROIPtr roiTensorPtrSrc,
                     RpptRoiType roiType,
                     rppHandle_t rppHandle)
{
    if (srcDescPtr->c != 3)
    {
        return RPP_INVALID_ARGUMENTS;
    }

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        color_cast_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     rgbaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        color_cast_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        color_cast_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                       srcDescPtr,
                                       (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                       dstDescPtr,
                                       rgbaTensor,
                                       roiTensorPtrSrc,
                                       roiType,
                                       layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        color_cast_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                     srcDescPtr,
                                     static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                     dstDescPtr,
                                     rgbaTensor,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }

    return RPP_SUCCESS;
}
