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

RppStatus brightness_u8_u8_host_tensor(Rpp8u* srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u* dstPtr,
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

        __m128i const zero = _mm_setzero_si128();
        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);
        __m128 p0, p1, p2, p3;
        __m128i px0, px1, px2, px3;

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

                __m128i px4, px5, px6, px7;
                __m128 p0R, p1R, p2R, p3R;
                __m128 p0G, p1G, p2G, p3G;
                __m128 p0B, p1B, p2B, p3B;

                __m128i mask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    RPP_LOAD48_U8PKD3_TO_F32PLN3

                    p0R = _mm_fmadd_ps(p0R, pMul, pAdd);    // brightness adjustment
                    p1R = _mm_fmadd_ps(p1R, pMul, pAdd);    // brightness adjustment
                    p2R = _mm_fmadd_ps(p2R, pMul, pAdd);    // brightness adjustment
                    p3R = _mm_fmadd_ps(p3R, pMul, pAdd);    // brightness adjustment

                    p0G = _mm_fmadd_ps(p0G, pMul, pAdd);    // brightness adjustment
                    p1G = _mm_fmadd_ps(p1G, pMul, pAdd);    // brightness adjustment
                    p2G = _mm_fmadd_ps(p2G, pMul, pAdd);    // brightness adjustment
                    p3G = _mm_fmadd_ps(p3G, pMul, pAdd);    // brightness adjustment

                    p0B = _mm_fmadd_ps(p0B, pMul, pAdd);    // brightness adjustment
                    p1B = _mm_fmadd_ps(p1B, pMul, pAdd);    // brightness adjustment
                    p2B = _mm_fmadd_ps(p2B, pMul, pAdd);    // brightness adjustment
                    p3B = _mm_fmadd_ps(p3B, pMul, pAdd);    // brightness adjustment

                    RPP_STORE48_F32PLN3_TO_U8PLN3

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

                __m128i px4, px5, px6, px7;
                __m128 p0R, p1R, p2R, p3R;
                __m128 p0G, p1G, p2G, p3G;
                __m128 p0B, p1B, p2B, p3B;

                __m128 const pZero = _mm_setzero_ps();
                __m128i mask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    RPP_LOAD48_U8PLN3_TO_F32PLN3

                    p0R = _mm_fmadd_ps(p0R, pMul, pAdd);    // brightness adjustment
                    p1R = _mm_fmadd_ps(p1R, pMul, pAdd);    // brightness adjustment
                    p2R = _mm_fmadd_ps(p2R, pMul, pAdd);    // brightness adjustment
                    p3R = _mm_fmadd_ps(p3R, pMul, pAdd);    // brightness adjustment

                    p0G = _mm_fmadd_ps(p0G, pMul, pAdd);    // brightness adjustment
                    p1G = _mm_fmadd_ps(p1G, pMul, pAdd);    // brightness adjustment
                    p2G = _mm_fmadd_ps(p2G, pMul, pAdd);    // brightness adjustment
                    p3G = _mm_fmadd_ps(p3G, pMul, pAdd);    // brightness adjustment

                    p0B = _mm_fmadd_ps(p0B, pMul, pAdd);    // brightness adjustment
                    p1B = _mm_fmadd_ps(p1B, pMul, pAdd);    // brightness adjustment
                    p2B = _mm_fmadd_ps(p2B, pMul, pAdd);    // brightness adjustment
                    p3B = _mm_fmadd_ps(p3B, pMul, pAdd);    // brightness adjustment

                    RPP_STORE48_F32PLN3_TO_U8PKD3

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
                        RPP_LOAD16_U8_TO_F32

                        p0 = _mm_fmadd_ps(p0, pMul, pAdd);    // brightness adjustment
                        p1 = _mm_fmadd_ps(p1, pMul, pAdd);    // brightness adjustment
                        p2 = _mm_fmadd_ps(p2, pMul, pAdd);    // brightness adjustment
                        p3 = _mm_fmadd_ps(p3, pMul, pAdd);    // brightness adjustment

                        RPP_STORE16_F32_TO_U8

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

RppStatus brightness_f32_f32_host_tensor(Rpp32f* srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f* dstPtr,
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

                __m128 p0R, p0G, p0B, p0A;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=12)
                {
                    RPP_LOAD12_F32PKD3_TO_F32PLN3

                    p0R = _mm_fmadd_ps(p0R, pMul, pAdd);    // brightness adjustment
                    p0G = _mm_fmadd_ps(p0G, pMul, pAdd);    // brightness adjustment
                    p0B = _mm_fmadd_ps(p0B, pMul, pAdd);    // brightness adjustment

                    RPP_STORE12_F32PLN3_TO_F32PLN3

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

                __m128 p0R, p0G, p0B, p0A;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                {
                    RPP_LOAD12_F32PLN3_TO_F32PLN3

                    p0R = _mm_fmadd_ps(p0R, pMul, pAdd);    // brightness adjustment
                    p0G = _mm_fmadd_ps(p0G, pMul, pAdd);    // brightness adjustment
                    p0B = _mm_fmadd_ps(p0B, pMul, pAdd);    // brightness adjustment

                    RPP_STORE12_F32PLN3_TO_F32PKD3

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

                    __m128 p0;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount+=4)
                    {
                        RPP_LOAD4_F32_TO_F32

                        p0 = _mm_fmadd_ps(p0, pMul, pAdd);    // brightness adjustment

                        RPP_STORE4_F32_TO_F32

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

#endif // HOST_TENSOR_AUGMENTATIONS_HPP
