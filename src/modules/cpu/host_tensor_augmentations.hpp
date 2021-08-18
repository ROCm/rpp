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

                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTemp);           // load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04
                    px1 =  _mm_loadu_si128((__m128i *)(srcPtrTemp + 12));    // load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08
                    px2 =  _mm_loadu_si128((__m128i *)(srcPtrTemp + 24));    // load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12
                    px3 =  _mm_loadu_si128((__m128i *)(srcPtrTemp + 36));    // load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16

                    px0 = _mm_shuffle_epi8(px0, mask);    // shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04
                    px1 = _mm_shuffle_epi8(px1, mask);    // shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08
                    px2 = _mm_shuffle_epi8(px2, mask);    // shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12
                    px3 = _mm_shuffle_epi8(px3, mask);    // shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16

                    px4 = _mm_unpackhi_epi8(px0, zero);    // unpack 8 hi-pixels of px0
                    px5 = _mm_unpackhi_epi8(px1, zero);    // unpack 8 hi-pixels of px1
                    px6 = _mm_unpackhi_epi8(px2, zero);    // unpack 8 hi-pixels of px2
                    px7 = _mm_unpackhi_epi8(px3, zero);    // unpack 8 hi-pixels of px3

                    px0 = _mm_unpacklo_epi8(px0, zero);    // unpack 8 lo-pixels of px0
                    px1 = _mm_unpacklo_epi8(px1, zero);    // unpack 8 lo-pixels of px1
                    px2 = _mm_unpacklo_epi8(px2, zero);    // unpack 8 lo-pixels of px2
                    px3 = _mm_unpacklo_epi8(px3, zero);    // unpack 8 lo-pixels of px3

                    p0R = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // unpack 4 lo-pixels of px0 - Contains R01-04
                    p1R = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // unpack 4 lo-pixels of px1 - Contains R05-08
                    p2R = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // unpack 4 lo-pixels of px2 - Contains R09-12
                    p3R = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px3, zero));    // unpack 4 lo-pixels of px3 - Contains R13-16

                    p0G = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // unpack 4 hi-pixels of px0 - Contains G01-04
                    p1G = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // unpack 4 hi-pixels of px1 - Contains G05-08
                    p2G = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px2, zero));    // unpack 4 hi-pixels of px2 - Contains G09-12
                    p3G = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px3, zero));    // unpack 4 hi-pixels of px3 - Contains G13-16

                    p0B = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px4, zero));    // unpack 4 lo-pixels of px4 - Contains B01-04
                    p1B = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px5, zero));    // unpack 4 lo-pixels of px5 - Contains B05-08
                    p2B = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px6, zero));    // unpack 4 lo-pixels of px6 - Contains B09-12
                    p3B = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px7, zero));    // unpack 4 lo-pixels of px7 - Contains B13-16

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

                    px4 = _mm_cvtps_epi32(p0R);    // convert to int32 for R
                    px5 = _mm_cvtps_epi32(p1R);    // convert to int32 for R
                    px6 = _mm_cvtps_epi32(p2R);    // convert to int32 for R
                    px7 = _mm_cvtps_epi32(p3R);    // convert to int32 for R
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 for R
                    px5 = _mm_packus_epi32(px6, px7);    // pack pixels 8-15 for R
                    px0 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 for R

                    px4 = _mm_cvtps_epi32(p0G);    // convert to int32 for G
                    px5 = _mm_cvtps_epi32(p1G);    // convert to int32 for G
                    px6 = _mm_cvtps_epi32(p2G);    // convert to int32 for G
                    px7 = _mm_cvtps_epi32(p3G);    // convert to int32 for G
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 for G
                    px5 = _mm_packus_epi32(px6, px7);    // pack pixels 8-15 for G
                    px1 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 for G

                    px4 = _mm_cvtps_epi32(p0B);    // convert to int32 for B
                    px5 = _mm_cvtps_epi32(p1B);    // convert to int32 for B
                    px6 = _mm_cvtps_epi32(p2B);    // convert to int32 for B
                    px7 = _mm_cvtps_epi32(p3B);    // convert to int32 for B
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 for B
                    px5 = _mm_packus_epi32(px6, px7);    // pack pixels 8-15 for B
                    px2 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 for B

                    _mm_storeu_si128((__m128i *)dstPtrTempR, px0);    // store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16]
                    _mm_storeu_si128((__m128i *)dstPtrTempG, px1);    // store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16]
                    _mm_storeu_si128((__m128i *)dstPtrTempB, px2);    // store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16]

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

                    px0 =  _mm_loadu_si128((__m128i *)srcPtrTempR);    // load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16]
                    px1 =  _mm_loadu_si128((__m128i *)srcPtrTempG);    // load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16]
                    px2 =  _mm_loadu_si128((__m128i *)srcPtrTempB);    // load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16]

                    px3 = _mm_unpackhi_epi8(px0, zero);    // unpack 8 hi-pixels of px0
                    px4 = _mm_unpackhi_epi8(px1, zero);    // unpack 8 hi-pixels of px1
                    px5 = _mm_unpackhi_epi8(px2, zero);    // unpack 8 hi-pixels of px2

                    px0 = _mm_unpacklo_epi8(px0, zero);    // unpack 8 lo-pixels of px0
                    px1 = _mm_unpacklo_epi8(px1, zero);    // unpack 8 lo-pixels of px1
                    px2 = _mm_unpacklo_epi8(px2, zero);    // unpack 8 lo-pixels of px2

                    p0R = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px0, zero));    // pixels 0-3 of original px0 containing 16 R values
                    p1R = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px0, zero));    // pixels 4-7 of original px0 containing 16 R values
                    p2R = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px3, zero));    // pixels 8-11 of original px0 containing 16 R values
                    p3R = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px3, zero));    // pixels 12-15 of original px0 containing 16 R values

                    p0G = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px1, zero));    // pixels 0-3 of original px1 containing 16 G values
                    p1G = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px1, zero));    // pixels 4-7 of original px1 containing 16 G values
                    p2G = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px4, zero));    // pixels 8-11 of original px1 containing 16 G values
                    p3G = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px4, zero));    // pixels 12-15 of original px1 containing 16 G values

                    p0B = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px2, zero));    // pixels 0-3 of original px1 containing 16 G values
                    p1B = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px2, zero));    // pixels 4-7 of original px1 containing 16 G values
                    p2B = _mm_cvtepi32_ps(_mm_unpacklo_epi16(px5, zero));    // pixels 8-11 of original px1 containing 16 G values
                    p3B = _mm_cvtepi32_ps(_mm_unpackhi_epi16(px5, zero));    // pixels 12-15 of original px1 containing 16 G values

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

                    px4 = _mm_cvtps_epi32(p0R);    // convert to int32 for R01-04
                    px5 = _mm_cvtps_epi32(p0G);    // convert to int32 for G01-04
                    px6 = _mm_cvtps_epi32(p0B);    // convert to int32 for B01-04
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 as R01-04|G01-04
                    px5 = _mm_packus_epi32(px6, pZero);    // pack pixels 8-15 as B01-04|X01-04
                    px0 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00]

                    px4 = _mm_cvtps_epi32(p1R);    // convert to int32 for R05-08
                    px5 = _mm_cvtps_epi32(p1G);    // convert to int32 for G05-08
                    px6 = _mm_cvtps_epi32(p1B);    // convert to int32 for B05-08
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 as R05-08|G05-08
                    px5 = _mm_packus_epi32(px6, pZero);    // pack pixels 8-15 as B05-08|X01-04
                    px1 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00]

                    px4 = _mm_cvtps_epi32(p2R);    // convert to int32 for R09-12
                    px5 = _mm_cvtps_epi32(p2G);    // convert to int32 for G09-12
                    px6 = _mm_cvtps_epi32(p2B);    // convert to int32 for B09-12
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 as R09-12|G09-12
                    px5 = _mm_packus_epi32(px6, pZero);    // pack pixels 8-15 as B09-12|X01-04
                    px2 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00]

                    px4 = _mm_cvtps_epi32(p3R);    // convert to int32 for R13-16
                    px5 = _mm_cvtps_epi32(p3G);    // convert to int32 for G13-16
                    px6 = _mm_cvtps_epi32(p3B);    // convert to int32 for B13-16
                    px4 = _mm_packus_epi32(px4, px5);    // pack pixels 0-7 as R13-16|G13-16
                    px5 = _mm_packus_epi32(px6, pZero);    // pack pixels 8-15 as B13-16|X01-04
                    px3 = _mm_packus_epi16(px4, px5);    // pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00]

                    px0 = _mm_shuffle_epi8(px0, mask);    // shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00]
                    px1 = _mm_shuffle_epi8(px1, mask);    // shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00]
                    px2 = _mm_shuffle_epi8(px2, mask);    // shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00]
                    px3 = _mm_shuffle_epi8(px3, mask);    // shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00]

                    _mm_storeu_si128((__m128i *)dstPtrTemp, px0);           // store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16]
                    _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), px1);    // store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16]
                    _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), px2);    // store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16]
                    _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), px3);    // store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16]

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

#endif // HOST_TENSOR_AUGMENTATIONS_HPP
