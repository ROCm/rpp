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
RppStatus brightness_host_tensor(T* srcPtr, RppTensorDescPtr srcDesc, T* dstPtr, RppTensorDescPtr dstDesc,
                                Rpp32f *alpha_tensor, Rpp32f *beta_tensor, RpptRoi *roiTensorSrc)
{
    if(srcDesc->layout == RpptTensorLayout::NCHW)
    {
        int n = srcDesc->n;
        int channel = srcDesc->c;
        int h = srcDesc->h;
        int w = srcDesc->w;
        RpptRoi roi_default = {0, w, 0, h};
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(n)
        for(int batchCount = 0; batchCount < n; batchCount++)
        {
            Rpp32u imageDimMax = srcDesc->stride[2];
            pRpptRoi proi = roiTensorSrc? &roiTensorSrc[batchCount] : &roi_default;
            Rpp32f alpha = alpha_tensor[batchCount];
            Rpp32f beta = beta_tensor[batchCount];
            Rpp32u x1 = std::max(0, proi->x1);
            Rpp32u y1 = std::max(0, proi->y1);;
            Rpp32u x2 = std::min(proi->x2, w);
            Rpp32u y2 = std::min(proi->y2, h);
            Rpp32u remainingElementsAfterROI = (srcDesc->stride[1] - x2);
            T *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDesc->stride[3];
            dstPtrImage = dstPtr + batchCount * dstDesc->stride[3];

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * srcDesc->stride[2]);
                dstPtrChannel = dstPtrImage + (c * dstDesc->stride[2]);

                for(int i = y1; i < y2; i++)
                {
                  T *srcPtrTemp, *dstPtrTemp;
                  srcPtrTemp = srcPtrChannel + (i * srcDesc->stride[1]) + x1;
                  dstPtrTemp = dstPtrChannel + (i * dstDesc->stride[1]) + x1;
                  Rpp32u bufferLength = x2 - x1;
                  Rpp32u alignedLength = bufferLength & ~15;

                  __m128i const zero = _mm_setzero_si128();
                  __m128 pMul = _mm_set1_ps(alpha);
                  __m128 pAdd = _mm_set1_ps(beta);
                  __m128 p0, p1, p2, p3;
                  __m128i px0, px1, px2, px3;
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
                        *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
                    }
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }
    else if (srcDesc->layout == RpptTensorLayout::NHWC)
    {
        int n = srcDesc->n;
        int c = srcDesc->c;
        int h = srcDesc->h;
        int w = srcDesc->w;
        RpptRoi roi_default = {0, w, 0, h};
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(n)
        for(int batchCount = 0; batchCount < n; batchCount ++)
        {
            Rpp32u imageDimMax = srcDesc->stride[2];
            pRpptRoi proi = roiTensorSrc ? &roiTensorSrc[batchCount] : &roi_default;
            Rpp32u x1 = std::max(0, proi->x1);
            Rpp32u y1 = std::max(0, proi->y1);;
            Rpp32u x2 = std::min(proi->x2, w);
            Rpp32u y2 = std::min(proi->y2, h);
            Rpp32u roi_width = x2-x1;
            Rpp32f alpha = alpha_tensor[batchCount];
            Rpp32f beta = beta_tensor[batchCount];
            Rpp32u elementsBeforeROI = c * x1;
            Rpp32u remainingElementsAfterROI = srcDesc->stride[2] - x2 * srcDesc->stride[1];
            T *srcPtrImage, *dstPtrImage;
            srcPtrImage = srcPtr + batchCount * srcDesc->stride[3];
            dstPtrImage = dstPtr + batchCount * dstDesc->stride[3];
            Rpp32u elementsInRow = c * w;
            Rpp32u elementsInRowMax = srcDesc->stride[1];

            for(int i = y1; i < y2; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * srcDesc->stride[2]);
                dstPtrTemp = dstPtrImage + (i * dstDesc->stride[2]);

                memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                srcPtrTemp += elementsBeforeROI;
                dstPtrTemp += elementsBeforeROI;
                Rpp32u bufferLength = srcDesc->stride[1] * roi_width;
                Rpp32u alignedLength = bufferLength & ~15;

                __m128i const zero = _mm_setzero_si128();
                __m128 pMul = _mm_set1_ps(alpha);
                __m128 pAdd = _mm_set1_ps(beta);
                __m128 p0, p1, p2, p3;
                __m128i px0, px1, px2, px3;

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
                    *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
                }

                srcPtrTemp += remainingElementsAfterROI;
                dstPtrTemp += remainingElementsAfterROI;
            }
        }
    }
    return RPP_SUCCESS;
}
#endif