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
                                 RpptDescPtr srcDesc,
                                 T* dstPtr,
                                 RpptDescPtr dstDesc,
                                 Rpp32f *alphaTensor,
                                 Rpp32f *betaTensor,
                                 RpptROI *roiTensorSrc)
{
    if(srcDesc->layout == RpptLayout::NCHW)
    {
        // int n = srcDesc->n;
        // int c = srcDesc->c;
        // int h = srcDesc->h;
        // int w = srcDesc->w;
        // RpptROI roi_default = {0, w, 0, h};
//         omp_set_dynamic(0);
// #pragma omp parallel for num_threads(srcDesc->n)
//         for(int batchCount = 0; batchCount < srcDesc->n; batchCount++)
//         {
//             // Rpp32u imageDimMax = srcDesc->strides.cStride;
//             // RpptROIPtr proi = roiTensorSrc? &roiTensorSrc[batchCount] : &roi_default;
//             Rpp32f alpha = alphaTensor[batchCount];
//             Rpp32f beta = betaTensor[batchCount];
//             // Rpp32u x1 = std::max(0, proi->x1);
//             // Rpp32u y1 = std::max(0, proi->y1);;
//             // Rpp32u x2 = std::min(proi->x2, w);
//             // Rpp32u y2 = std::min(proi->y2, h);
//             Rpp32u srcRemainingElementsAfterROI = srcDesc->strides.hStride - srcDesc->w;
//             Rpp32u dstRemainingElementsAfterROI = dstDesc->strides.hStride - dstDesc->w;

//             T *srcPtrImage, *dstPtrImage;
//             srcPtrImage = srcPtr + batchCount * srcDesc->strides.nStride;
//             dstPtrImage = dstPtr + batchCount * dstDesc->strides.nStride;

//             T *srcPtrChannel, *dstPtrChannel;
//             srcPtrChannel = srcPtrImage;
//             dstPtrChannel = dstPtrImage;

//             for(int chnl = 0; chnl < srcDesc->c; chnl++)
//             {
//                 // T *srcPtrChannel, *dstPtrChannel;
//                 // srcPtrChannel = srcPtrImage + (chnl * srcDesc->stride[2]);
//                 // dstPtrChannel = dstPtrImage + (chnl * dstDesc->stride[2]);

//                 T *srcPtrTemp, *dstPtrTemp;
//                 srcPtrTemp = srcPtrChannel;
//                 dstPtrTemp = dstPtrChannel;

//                 for(int i = 0; i < srcDesc->h; i++)
//                 {
//                     // T *srcPtrTemp, *dstPtrTemp;
//                     // srcPtrTemp = srcPtrChannel + (i * srcDesc->stride[1]) + x1;
//                     // dstPtrTemp = dstPtrChannel + (i * dstDesc->stride[1]) + x1;
//                     Rpp32u bufferLength = srcDesc->w;
//                     Rpp32u alignedLength = bufferLength & ~15;

//                     __m128i const zero = _mm_setzero_si128();
//                     __m128 pMul = _mm_set1_ps(alpha);
//                     __m128 pAdd = _mm_set1_ps(beta);
//                     __m128 p0, p1, p2, p3;
//                     __m128i px0, px1, px2, px3;
//                     int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
//                     {
//                         px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);

//                         px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
//                         px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
//                         p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
//                         p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
//                         p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
//                         p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15

//                         p0 = _mm_mul_ps(p0, pMul);
//                         p1 = _mm_mul_ps(p1, pMul);
//                         p2 = _mm_mul_ps(p2, pMul);
//                         p3 = _mm_mul_ps(p3, pMul);
//                         px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
//                         px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
//                         px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
//                         px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));

//                         px0 = _mm_packus_epi32(px0, px1);
//                         px1 = _mm_packus_epi32(px2, px3);
//                         px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

//                         _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

//                         srcPtrTemp +=16;
//                         dstPtrTemp +=16;
//                     }
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
//                     }
//                     srcPtrTemp += srcRemainingElementsAfterROI;
//                     dstPtrTemp += dstRemainingElementsAfterROI;
//                 }
//                 srcPtrChannel += srcDesc->strides.cStride;
//                 dstPtrChannel += dstDesc->strides.cStride;
//             }
//         }
    }
    else if (srcDesc->layout == RpptLayout::NHWC)
    {
        printf("\nHere  in packed!");
        printf("\n n = %d", srcDesc->n);
        printf("\n c = %d", srcDesc->c);
        printf("\n h = %d", srcDesc->h);
        printf("\n w = %d", srcDesc->w);
        // int n = srcDesc->n;
        // int h = srcDesc->h;
        // int w = srcDesc->w;
        // int c = srcDesc->c;
        // RpptROI roi_default = {0, w, 0, h};
//         omp_set_dynamic(0);
// #pragma omp parallel for num_threads(srcDesc->n)
//         for(int batchCount = 0; batchCount < srcDesc->n; batchCount ++)
//         {
//             // Rpp32u imageDimMax = srcDesc->stride[2];
//             // RpptROIPtr proi = roiTensorSrc ? &roiTensorSrc[batchCount] : &roi_default;
//             // Rpp32u x1 = std::max(0, proi->x1);
//             // Rpp32u y1 = std::max(0, proi->y1);;
//             // Rpp32u x2 = std::min(proi->x2, w);
//             // Rpp32u y2 = std::min(proi->y2, h);
//             // Rpp32u roi_width = x2-x1;
//             Rpp32f alpha = alphaTensor[batchCount];
//             Rpp32f beta = betaTensor[batchCount];
//             // Rpp32u elementsBeforeROI = c * x1;
//             Rpp32u srcRemainingElementsAfterROI = srcDesc->strides.hStride - srcDesc->w;
//             T *srcPtrImage, *dstPtrImage;
//             srcPtrImage = srcPtr + batchCount * srcDesc->stride[3];
//             dstPtrImage = dstPtr + batchCount * dstDesc->stride[3];
//             Rpp32u elementsInRow = c * w;
//             Rpp32u elementsInRowMax = srcDesc->stride[1];

//             for(int i = y1; i < y2; i++)
//             {
//                 T *srcPtrTemp, *dstPtrTemp;
//                 srcPtrTemp = srcPtrImage + (i * srcDesc->stride[2]);
//                 dstPtrTemp = dstPtrImage + (i * dstDesc->stride[2]);

//                 memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
//                 srcPtrTemp += elementsBeforeROI;
//                 dstPtrTemp += elementsBeforeROI;
//                 Rpp32u bufferLength = srcDesc->stride[1] * roi_width;
//                 Rpp32u alignedLength = bufferLength & ~15;

//                 __m128i const zero = _mm_setzero_si128();
//                 __m128 pMul = _mm_set1_ps(alpha);
//                 __m128 pAdd = _mm_set1_ps(beta);
//                 __m128 p0, p1, p2, p3;
//                 __m128i px0, px1, px2, px3;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
//                 {
//                     px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);
//                     px1 = _mm_unpackhi_epi8(px0, zero);    // pixels 8-15
//                     px0 = _mm_unpacklo_epi8(px0, zero);    // pixels 0-7
//                     p0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3
//                     p1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7
//                     p2 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 8-11
//                     p3 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 12-15
//                     p0 = _mm_mul_ps(p0, pMul);
//                     p1 = _mm_mul_ps(p1, pMul);
//                     p2 = _mm_mul_ps(p2, pMul);
//                     p3 = _mm_mul_ps(p3, pMul);
//                     px0 = _mm_cvtps_epi32(_mm_add_ps(p0, pAdd));
//                     px1 = _mm_cvtps_epi32(_mm_add_ps(p1, pAdd));
//                     px2 = _mm_cvtps_epi32(_mm_add_ps(p2, pAdd));
//                     px3 = _mm_cvtps_epi32(_mm_add_ps(p3, pAdd));

//                     px0 = _mm_packus_epi32(px0, px1);
//                     px1 = _mm_packus_epi32(px2, px3);
//                     px0 = _mm_packus_epi16(px0, px1);    // pixels 0-15

//                     _mm_storeu_si128((__m128i *)dstPtrTemp, px0);

//                     srcPtrTemp +=16;
//                     dstPtrTemp +=16;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     *dstPtrTemp++ = (T) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp++)) * alpha) + beta);
//                 }

//                 srcPtrTemp += remainingElementsAfterROI;
//                 dstPtrTemp += remainingElementsAfterROI;
//             }
//         }
    }
    return RPP_SUCCESS;
}
#endif