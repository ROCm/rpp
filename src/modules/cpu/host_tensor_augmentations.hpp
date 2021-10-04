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

#ifndef HOST_TENSOR_AUGMENTATIONS_HPP
#define HOST_TENSOR_AUGMENTATIONS_HPP

#include "cpu/rpp_cpu_simd.hpp"
#include <cpu/rpp_cpu_common.hpp>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

/************ brightness ************/

RppStatus brightness_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment

                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment

                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);
                    dstPtrTempR++;
                    srcPtrTemp++;

                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);
                    dstPtrTempG++;
                    srcPtrTemp++;

                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);
                    dstPtrTempB++;
                    srcPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment

                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment

                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempR)) * alpha) + beta);
                    dstPtrTemp++;
                    srcPtrTempR++;

                    *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempG)) * alpha) + beta);
                    dstPtrTemp++;
                    srcPtrTempG++;

                    *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempB)) * alpha) + beta);
                    dstPtrTemp++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);

                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                        p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                        p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment

                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);

                        dstPtrTemp++;
                        srcPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount] * 0.0039216; // 1/255

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);
                    dstPtrTempR++;
                    srcPtrTemp++;

                    *dstPtrTempG = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);
                    dstPtrTempG++;
                    srcPtrTemp++;

                    *dstPtrTempB = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);
                    dstPtrTempB++;
                    srcPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTempR * alpha + beta);
                    dstPtrTemp++;
                    srcPtrTempR++;

                    *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTempG * alpha + beta);
                    dstPtrTemp++;
                    srcPtrTempG++;

                    *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTempB * alpha + beta);
                    dstPtrTemp++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, p);

                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment

                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);

                        dstPtrTemp++;
                        srcPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount] * 0.0039216; // 1/255

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp * alpha + beta);
                    dstPtrTempR++;
                    srcPtrTemp++;

                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp * alpha + beta);
                    dstPtrTempG++;
                    srcPtrTemp++;

                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp * alpha + beta);
                    dstPtrTempB++;
                    srcPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~11;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR * alpha + beta);
                    dstPtrTemp++;
                    srcPtrTempR++;

                    *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG * alpha + beta);
                    dstPtrTemp++;
                    srcPtrTempG++;

                    *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB * alpha + beta);
                    dstPtrTemp++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        Rpp32f srcPtrTemp_ps[4], dstPtrTemp_ps[4];

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp16f) *(srcPtrTemp + cnt);
                        }

                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, p);

                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment

                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p);

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        }

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp * alpha + beta);

                        dstPtrTemp++;
                        srcPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus brightness_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment

                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment

                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);
                    dstPtrTempR++;
                    srcPtrTemp++;

                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);
                    dstPtrTempG++;
                    srcPtrTemp++;

                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);
                    dstPtrTempB++;
                    srcPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);

                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment

                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment

                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempR) + 128) * alpha) + beta - 128);
                    dstPtrTemp++;
                    srcPtrTempR++;

                    *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempG) + 128) * alpha) + beta - 128);
                    dstPtrTemp++;
                    srcPtrTempG++;

                    *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempB) + 128) * alpha) + beta - 128);
                    dstPtrTemp++;
                    srcPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                    {
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);

                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                        p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                        p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment

                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);

                        dstPtrTemp++;
                        srcPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ color_jitter ************/

RppStatus color_jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8u *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *brightnessTensor,
                                         Rpp32f *contrastTensor,
                                         Rpp32f *hueTensor,
                                         Rpp32f *saturationTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }

        Rpp32f brightnessParam = brightnessTensor[batchCount];
        Rpp32f contrastParam = contrastTensor[batchCount];
        Rpp32f hueParam = hueTensor[batchCount];
        Rpp32f saturationParam = saturationTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *ctm = static_cast<Rpp32f*>(aligned_alloc(64, 64));
        FILL_COLOR_JITTER_CTM(ctm);
        compute_color_jitter_ctm_host(brightnessParam, contrastParam, hueParam, saturationParam, ctm);

        __m128 pCtm[12];
        for(int i = 0; i < 12; i++)
        {
            pCtm[i] = _mm_set1_ps(ctm[i]);
        }

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Color Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12], pResult[3];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);
                    compute_color_jitter_host(p, pResult, pCtm);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp+=3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12], pResult[3];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_color_jitter_host(p, pResult, pCtm);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NHWC -> NHWC)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128 p[12], pResult[3];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);
                    compute_color_jitter_host(p, pResult, pCtm);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);

                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK(ctm[0] * srcPtrTemp[0] + ctm[1] * srcPtrTemp[1] + ctm[2] * srcPtrTemp[2] + ctm[3]);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK(ctm[4] * srcPtrTemp[0] + ctm[5] * srcPtrTemp[1] + ctm[6] * srcPtrTemp[2] + ctm[7]);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK(ctm[8] * srcPtrTemp[0] + ctm[9] * srcPtrTemp[1] + ctm[10] * srcPtrTemp[2] + ctm[11]);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Color Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~47;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128 p[12], pResult[3];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_color_jitter_host(p, pResult, pCtm);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK(ctm[0] * *srcPtrTempR + ctm[1] * *srcPtrTempG + ctm[2] * *srcPtrTempB + ctm[3]);
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK(ctm[4] * *srcPtrTempR + ctm[5] * *srcPtrTempG + ctm[6] * *srcPtrTempB + ctm[7]);
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK(ctm[8] * *srcPtrTempR + ctm[9] * *srcPtrTempG + ctm[10] * *srcPtrTempB + ctm[11]);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

#endif // HOST_TENSOR_AUGMENTATIONS_HPP
