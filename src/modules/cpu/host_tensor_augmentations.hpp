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

template <typename T>
RppStatus brightness_host_tensor(T* srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T* dstPtr,
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

        RpptROI roi;
        RpptROIPtr roiPtr;
        roiPtr = &roi;
        roiPtr->xywhROI.xy.x = RPPMAX2(roiPtrDefault->xywhROI.xy.x, roiPtrImage->xywhROI.xy.x);
        roiPtr->xywhROI.xy.y = RPPMAX2(roiPtrDefault->xywhROI.xy.y, roiPtrImage->xywhROI.xy.y);
        roiPtr->xywhROI.roiWidth = RPPMIN2(roiPtrDefault->xywhROI.roiWidth - roiPtrImage->xywhROI.xy.x, roiPtrImage->xywhROI.roiWidth);
        roiPtr->xywhROI.roiHeight = RPPMIN2(roiPtrDefault->xywhROI.roiHeight - roiPtrImage->xywhROI.xy.y, roiPtrImage->xywhROI.roiHeight);

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = bufferLength & ~15;

        __m128i const zero = _mm_setzero_si128();
        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);
        __m128 p0, p1, p2, p3;
        __m128i px0, px1, px2, px3;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        for(int c = 0; c < layoutParams.channelParam; c++)
        {
            T *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
                    px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
                    px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
                    p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
                    p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
                    p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
                    p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
                    p0 = _mm_mul_ps(p0, pMul);
                    p1 = _mm_mul_ps(p1, pMul);
                    p2 = _mm_mul_ps(p2, pMul);
                    p3 = _mm_mul_ps(p3, pMul);
                    px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
                    px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
                    px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
                    px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));

                    px0 = _mm_packus_epi32(px0, px1);
                    px1 = _mm_packus_epi32(px2, px3);
                    px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

                    srcPtrTemp +=16;
                    dstPtrTemp +=16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);

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

    return RPP_SUCCESS;
}

#endif // HOST_TENSOR_AUGMENTATIONS_HPP
