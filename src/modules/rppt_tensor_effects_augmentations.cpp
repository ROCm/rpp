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

#include <random>
#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_effects_augmentations.h"
#include "cpu/host_tensor_effects_augmentations.hpp"

#ifdef HIP_COMPILE
#include <hip/hip_fp16.h>
#include "hip/hip_tensor_effects_augmentations.hpp"
#endif // HIP_COMPILE

/******************** gridmask ********************/

RppStatus rppt_gridmask_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32u tileWidth,
                             Rpp32f gridRatio,
                             Rpp32f gridAngle,
                             RpptUintVector2D translateVector,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        gridmask_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   tileWidth,
                                   gridRatio,
                                   gridAngle,
                                   translateVector,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        gridmask_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     tileWidth,
                                     gridRatio,
                                     gridAngle,
                                     translateVector,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        gridmask_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                     srcDescPtr,
                                     (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                     dstDescPtr,
                                     tileWidth,
                                     gridRatio,
                                     gridAngle,
                                     translateVector,
                                     roiTensorPtrSrc,
                                     roiType,
                                     layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        gridmask_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                   srcDescPtr,
                                   static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                   dstDescPtr,
                                   tileWidth,
                                   gridRatio,
                                   gridAngle,
                                   translateVector,
                                   roiTensorPtrSrc,
                                   roiType,
                                   layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** spatter ********************/

RppStatus rppt_spatter_host(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            RpptRGB spatterColor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].xywhROI.roiWidth > SPATTER_MAX_WIDTH) || (roiTensorPtrSrc[i].xywhROI.roiHeight > SPATTER_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
            if ((roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x > SPATTER_MAX_XDIM) || (roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y > SPATTER_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        spatter_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  spatterColor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        spatter_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    spatterColor,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        spatter_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                    srcDescPtr,
                                    (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                    dstDescPtr,
                                    spatterColor,
                                    roiTensorPtrSrc,
                                    roiType,
                                    layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        spatter_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                  srcDescPtr,
                                  static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                  dstDescPtr,
                                  spatterColor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);
    }

    return RPP_SUCCESS;
}

/******************** salt_and_pepper_noise ********************/

RppStatus rppt_salt_and_pepper_noise_host(RppPtr_t srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          RppPtr_t dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *noiseProbabilityTensor,
                                          Rpp32f *saltProbabilityTensor,
                                          Rpp32f *saltValueTensor,
                                          Rpp32f *pepperValueTensor,
                                          Rpp32u seed,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rppHandle_t rppHandle)
{
    for(int i = 0; i < srcDescPtr->n; i++)
        if (!RPPINRANGE(noiseProbabilityTensor[i], 0, 1) || !RPPINRANGE(saltProbabilityTensor[i], 0, 1) || !RPPINRANGE(saltValueTensor[i], 0, 1) || !RPPINRANGE(pepperValueTensor[i], 0, 1))
            return RPP_ERROR_INVALID_ARGUMENTS;

    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    RpptXorwowState xorwowInitialState[SIMD_FLOAT_VECTOR_LENGTH];
    rpp_host_rng_xorwow_f32_initialize_multiseed_stream<SIMD_FLOAT_VECTOR_LENGTH>(xorwowInitialState, seed);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        salt_and_pepper_noise_u8_u8_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                noiseProbabilityTensor,
                                                saltProbabilityTensor,
                                                saltValueTensor,
                                                pepperValueTensor,
                                                xorwowInitialState,
                                                roiTensorPtrSrc,
                                                roiType,
                                                layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        salt_and_pepper_noise_f16_f16_host_tensor((Rpp16f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp16f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  noiseProbabilityTensor,
                                                  saltProbabilityTensor,
                                                  saltValueTensor,
                                                  pepperValueTensor,
                                                  xorwowInitialState,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        salt_and_pepper_noise_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                                  srcDescPtr,
                                                  (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                                  dstDescPtr,
                                                  noiseProbabilityTensor,
                                                  saltProbabilityTensor,
                                                  saltValueTensor,
                                                  pepperValueTensor,
                                                  xorwowInitialState,
                                                  roiTensorPtrSrc,
                                                  roiType,
                                                  layoutParams);
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        salt_and_pepper_noise_i8_i8_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                                srcDescPtr,
                                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                                dstDescPtr,
                                                noiseProbabilityTensor,
                                                saltProbabilityTensor,
                                                saltValueTensor,
                                                pepperValueTensor,
                                                xorwowInitialState,
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

/******************** gridmask ********************/

RppStatus rppt_gridmask_gpu(RppPtr_t srcPtr,
                            RpptDescPtr srcDescPtr,
                            RppPtr_t dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32u tileWidth,
                            Rpp32f gridRatio,
                            Rpp32f gridAngle,
                            RpptUintVector2D translateVector,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hipMemset((void *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), 0, dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(Rpp8u));
        hip_exec_gridmask_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 tileWidth,
                                 gridRatio,
                                 gridAngle,
                                 translateVector,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hipMemset((void *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), 0, dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(half));
        hip_exec_gridmask_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 tileWidth,
                                 gridRatio,
                                 gridAngle,
                                 translateVector,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hipMemset((void *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), 0, dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(Rpp32f));
        hip_exec_gridmask_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                 srcDescPtr,
                                 (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                 dstDescPtr,
                                 tileWidth,
                                 gridRatio,
                                 gridAngle,
                                 translateVector,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hipMemset((void *)(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes), -128, dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(Rpp8s));
        hip_exec_gridmask_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                 srcDescPtr,
                                 static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                 dstDescPtr,
                                 tileWidth,
                                 gridRatio,
                                 gridAngle,
                                 translateVector,
                                 roiTensorPtrSrc,
                                 roiType,
                                 rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** spatter ********************/

RppStatus rppt_spatter_gpu(RppPtr_t srcPtr,
                           RpptDescPtr srcDescPtr,
                           RppPtr_t dstPtr,
                           RpptDescPtr dstDescPtr,
                           RpptRGB spatterColor,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    RpptROI roiTensorPtrSrcHost[dstDescPtr->n];
    hipMemcpy(roiTensorPtrSrcHost, roiTensorPtrSrc, dstDescPtr->n * sizeof(RpptROI), hipMemcpyDeviceToHost);
    if (roiType == RpptRoiType::XYWH)
    {
        for(int i = 0; i < dstDescPtr->n; i++)
            if ((roiTensorPtrSrcHost[i].xywhROI.roiWidth > SPATTER_MAX_WIDTH) || (roiTensorPtrSrcHost[i].xywhROI.roiHeight > SPATTER_MAX_HEIGHT))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }
    else if (roiType == RpptRoiType::LTRB)
    {
        for(int i = 0; i < dstDescPtr->n; i++)
            if ((roiTensorPtrSrcHost[i].ltrbROI.rb.x - roiTensorPtrSrcHost[i].ltrbROI.lt.x > SPATTER_MAX_XDIM) || (roiTensorPtrSrcHost[i].ltrbROI.rb.y - roiTensorPtrSrcHost[i].ltrbROI.lt.y > SPATTER_MAX_YDIM))
                return RPP_ERROR_HIGH_SRC_DIMENSION;
    }

    std::random_device rd;  // Random number engine seed
    std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
    Rpp32u maskLocArrHostX[dstDescPtr->n], maskLocArrHostY[dstDescPtr->n];
    for(int i = 0; i < dstDescPtr->n; i++)
    {
        std::uniform_int_distribution<> distribX(0, SPATTER_MAX_WIDTH - roiTensorPtrSrcHost[i].xywhROI.roiWidth);
        std::uniform_int_distribution<> distribY(0, SPATTER_MAX_HEIGHT - roiTensorPtrSrcHost[i].xywhROI.roiHeight);
        maskLocArrHostX[i] = distribX(gen);
        maskLocArrHostY[i] = distribY(gen);
    }

    Rpp32u paramIndex = 0;
    copy_param_uint(maskLocArrHostX, rpp::deref(rppHandle), paramIndex++);
    copy_param_uint(maskLocArrHostY, rpp::deref(rppHandle), paramIndex++);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_spatter_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                spatterColor,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_spatter_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                spatterColor,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_spatter_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                srcDescPtr,
                                (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                dstDescPtr,
                                spatterColor,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_spatter_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                srcDescPtr,
                                static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                dstDescPtr,
                                spatterColor,
                                roiTensorPtrSrc,
                                roiType,
                                rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

/******************** salt_and_pepper_noise ********************/

RppStatus rppt_salt_and_pepper_noise_gpu(RppPtr_t srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         RppPtr_t dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *noiseProbabilityTensor,
                                         Rpp32f *saltProbabilityTensor,
                                         Rpp32f *saltValueTensor,
                                         Rpp32f *pepperValueTensor,
                                         Rpp32u seed,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    for(int i = 0; i < srcDescPtr->n; i++)
        if (!RPPINRANGE(noiseProbabilityTensor[i], 0, 1) || !RPPINRANGE(saltProbabilityTensor[i], 0, 1) || !RPPINRANGE(saltValueTensor[i], 0, 1) || !RPPINRANGE(pepperValueTensor[i], 0, 1))
            return RPP_ERROR_INVALID_ARGUMENTS;

    Rpp32u paramIndex = 0;
    copy_param_float(noiseProbabilityTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(saltProbabilityTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(saltValueTensor, rpp::deref(rppHandle), paramIndex++);
    copy_param_float(pepperValueTensor, rpp::deref(rppHandle), paramIndex++);

    RpptXorwowState xorwowInitialState;
    xorwowInitialState.x[0] = 0x75BCD15 + seed;
    xorwowInitialState.x[1] = 0x159A55E5 + seed;
    xorwowInitialState.x[2] = 0x1F123BB5 + seed;
    xorwowInitialState.x[3] = 0x5491333 + seed;
    xorwowInitialState.x[4] = 0x583F19 + seed;
    xorwowInitialState.counter = 0x64F0C9 + seed;

    RpptXorwowState *d_xorwowInitialStatePtr;
    d_xorwowInitialStatePtr = (RpptXorwowState *) rpp::deref(rppHandle).GetInitHandle()->mem.mgpu.maskArr.floatmem;
    hipMemcpy(d_xorwowInitialStatePtr, &xorwowInitialState, sizeof(RpptXorwowState), hipMemcpyHostToDevice);

    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        hip_exec_salt_and_pepper_noise_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              d_xorwowInitialStatePtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_salt_and_pepper_noise_tensor((half*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (half*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              d_xorwowInitialStatePtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_salt_and_pepper_noise_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                                              srcDescPtr,
                                              (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                                              dstDescPtr,
                                              d_xorwowInitialStatePtr,
                                              roiTensorPtrSrc,
                                              roiType,
                                              rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        hip_exec_salt_and_pepper_noise_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                                              srcDescPtr,
                                              static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                                              dstDescPtr,
                                              d_xorwowInitialStatePtr,
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
