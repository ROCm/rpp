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
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_src_loc(int row , int col, Rpp32s *locArray, RpptDescPtr srcDescPtr, RpptChannelOffsets *rgbOffsets, RpptROI roi, int batchCount, int channelValue)
{
    int xR, yR, xG, yG, xB, yB;

    xR = col + rgbOffsets[batchCount].r.x;
    yR = row + rgbOffsets[batchCount].r.y;

    xG = col + rgbOffsets[batchCount].g.x;
    yG = row + rgbOffsets[batchCount].g.y;

    xB = col + rgbOffsets[batchCount].b.x;
    yB = row + rgbOffsets[batchCount].b.y;

    if (xR >= roi.xywhROI.roiWidth || xR < roi.xywhROI.xy.x || yR >= roi.xywhROI.roiHeight || yR < roi.xywhROI.xy.y)
    {
        xR = col;
        yR = row;
    }

    if (xG >= roi.xywhROI.roiWidth || xG < roi.xywhROI.xy.x || yG >= roi.xywhROI.roiHeight || yG < roi.xywhROI.xy.y)
    {
        xG = col;
        yG = row;
    }

    if (xB >= roi.xywhROI.roiWidth || xB < roi.xywhROI.xy.x || yB >= roi.xywhROI.roiHeight || yB < roi.xywhROI.xy.y)
    {
        xB = col;
        yB = row;
    }

    locArray[0] = yR * srcDescPtr->strides.hStride + xR * channelValue;
    locArray[1] = yG * srcDescPtr->strides.hStride + xG * channelValue;
    locArray[2] = yB * srcDescPtr->strides.hStride + xB * channelValue;
}

RppStatus glitch_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        __m128i maskR = _mm_setr_epi8(0, 0x80, 0x80, 3, 0x80, 0x80, 6, 0x80, 0x80, 9, 0x80, 0x80, 12, 0x80, 0x80, 15);
        __m128i maskGB = _mm_setr_epi8(0x80, 1, 2, 0x80, 4, 5, 0x80, 7, 8, 0x80, 10, 11, 0x80, 13, 14, 0x80);
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            __m256i rMask = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            __m256i gMask = _mm256_setr_epi8(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            __m256i bMask = _mm256_setr_epi8(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
            Rpp32u alignedLength = ((int)((roi.xywhROI.roiWidth * 0.75)) / 8) * 8;   // Align dst width to process 16 dst pixels per iteration
            Rpp32s vectorIncrement = 10;
            Rpp32s vectorIncrementPkd = 30;
            Rpp32s remappedSrcLoc;
            Rpp32s remapSrcLocArray[3] = {0};     // Since 4 dst pixels are processed per iteration
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u* dstRowPtrTempR = dstPtrRow;
                Rpp8u* dstRowPtrTempG = dstPtrRow + dstDescPtr->strides.cStride;
                Rpp8u* dstRowPtrTempB = dstPtrRow + 2 * dstDescPtr->strides.cStride;

                for (int vectorLoopCount = 0; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    compute_src_loc(dstLocRow, vectorLoopCount, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 3);
                    rpp_simd_load(rpp_glitch_load24_u8pkd3_to_f32pln3_avx, srcPtrChannel, p, remapSrcLocArray);
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstRowPtrTempR, dstRowPtrTempG, dstRowPtrTempB, p);    // simd stores

                    dstRowPtrTempR += 8;
                    dstRowPtrTempG += 8;
                    dstRowPtrTempB += 8;
                }

                for (int i = alignedLength; i < roi.xywhROI.roiWidth; i++)
                {
                    compute_src_loc(dstLocRow, i, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 3);
                    *dstRowPtrTempR++ = *(srcPtrChannel + remapSrcLocArray[0] + 0);
                    *dstRowPtrTempG++ = *(srcPtrChannel + remapSrcLocArray[1] + 1);
                    *dstRowPtrTempB++ = *(srcPtrChannel + remapSrcLocArray[2] + 2);
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32u vectorIncrement = 16;
            Rpp32u alignedLength = ((int)(roi.xywhROI.roiWidth * 0.75)) & ~15;
            Rpp32s remapSrcLocArray[3] = {0};

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u* dstPtrTemp = dstPtrRow;

                for (int vectorLoopCount = 0; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p[6];
                    compute_src_loc(dstLocRow, vectorLoopCount, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 1);
                    rpp_simd_load(rpp_glitch_load48_u8pln3_to_f32pln3_avx, srcPtrChannel, srcPtrChannel + srcDescPtr->strides.cStride, srcPtrChannel + 2 * srcDescPtr->strides.cStride, p, remapSrcLocArray);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
                    dstPtrTemp += 48;
                }

                for (int i = alignedLength; i < roi.xywhROI.roiWidth; i++)
                {
                    compute_src_loc(dstLocRow, i, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                    {
                        *(dstPtrTemp + c) = *(srcPtrChannel + remapSrcLocArray[c] + c *srcDescPtr->strides.cStride);
                    }
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32u vectorIncrement = 32;
            Rpp32u alignedLength = ((int)(roi.xywhROI.roiWidth * 0.75)) & ~31;
            Rpp32s remapSrcLocArray[3] = {0};

            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u* dstPtrTemp = dstPtrRow;

                for (int vectorLoopCount = 0; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    compute_src_loc(dstLocRow, vectorLoopCount, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                    {
                        __m256i p;
                        p = _mm256_loadu_epi8(srcPtrChannel + (remapSrcLocArray[c] + c * srcDescPtr->strides.cStride)); 

                        _mm256_storeu_epi8((dstPtrTemp + c * srcDescPtr->strides.cStride), p);
                    }
                    dstPtrTemp += 32;
                }

                for (int i = alignedLength; i < roi.xywhROI.roiWidth; i++)
                {
                    compute_src_loc(dstLocRow, i, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 1);
                    for (int c = 0; c < 3; c++)
                    {
                        *(dstPtrTemp + c * dstDescPtr->strides.cStride) = *(srcPtrChannel + remapSrcLocArray[c] + c *srcDescPtr->strides.cStride);
                    }
                    dstPtrTemp += 1;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }

        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            __m256i rMask = _mm256_setr_epi8(0, 0x80, 0x80, 3, 0x80, 0x80, 6, 0x80, 0x80, 9, 0x80, 0x80, 12, 0x80, 0x80, 15, 0x80, 0x80, 18, 0x80, 0x80, 21, 0x80, 0x80, 24, 0x80, 0x80, 27, 0x80, 0x80, 0x80, 0x80);
            __m256i gMask = _mm256_setr_epi8(0x80, 1, 0x80, 0x80, 4, 0x80, 0x80, 7, 0x80, 0x80, 10, 0x80, 0x80, 13, 0x80, 0x80, 16, 0x80, 0x80, 19, 0x80, 0x80, 22, 0x80, 0x80, 25, 0x80, 0x80, 28, 0x80, 0x80, 0x80);
            __m256i bMask = _mm256_setr_epi8(0x80, 0x80, 2, 0x80, 0x80, 5, 0x80, 0x80, 8, 0x80, 0x80, 11, 0x80, 0x80, 14, 0x80, 0x80, 17, 0x80, 0x80, 20, 0x80, 0x80, 23, 0x80, 0x80, 26, 0x80, 0x80, 29, 0x80, 0x80);
            Rpp32u alignedLength = ((int)((roi.xywhROI.roiWidth * 0.75)) / 10) * 10;   // Align dst width to process 16 dst pixels per iteration
            Rpp32s vectorIncrement = 10;
            Rpp32s vectorIncrementPkd = 30;
            Rpp32s remappedSrcLoc;
            Rpp32s remapSrcLocArray[3] = {0};     // Since 4 dst pixels are processed per iteration
            for (int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u* dstPtrTemp = dstPtrRow;

                for (int vectorLoopCount = 0; vectorLoopCount < alignedLength; vectorLoopCount += 10)
                {
                    __m256i r, g, b;
                    compute_src_loc(dstLocRow, vectorLoopCount, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 3);
                    r = _mm256_loadu_epi8(srcPtrChannel + remapSrcLocArray[0]); 
                    g = _mm256_loadu_epi8(srcPtrChannel + remapSrcLocArray[1]);
                    b = _mm256_loadu_epi8(srcPtrChannel + remapSrcLocArray[2]);
                    r = _mm256_shuffle_epi8(r, rMask);
                    g = _mm256_shuffle_epi8(g, gMask);
                    b = _mm256_shuffle_epi8(b, bMask);
                    r = _mm256_or_si256(r,g);
                    r = _mm256_or_si256(r,b);

                    _mm256_storeu_epi8(dstPtrTemp, r);
                    dstPtrTemp += 30;
                }

                for (int i = alignedLength; i < roi.xywhROI.roiWidth; i++)
                {
                    compute_src_loc(dstLocRow, i, remapSrcLocArray, srcDescPtr, rgbOffsets, roi, batchCount, 3);
                    for (int c = 0; c < 3; c++)
                    {
                        *dstPtrTemp++ = *(srcPtrChannel + remapSrcLocArray[c] + c);
                    }
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }

        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = rgbOffsets[batchCount].r.x;
        Rpp32u yOffsetRchn = rgbOffsets[batchCount].r.y;
        Rpp32u xOffsetGchn = rgbOffsets[batchCount].g.x;
        Rpp32u yOffsetGchn = rgbOffsets[batchCount].g.y;
        Rpp32u xOffsetBchn = rgbOffsets[batchCount].b.x;
        Rpp32u yOffsetBchn = rgbOffsets[batchCount].b.y;

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp32f *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride);
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride);
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp32f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + dstDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + 2 * dstDescPtr->strides.cStride;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 4)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 4) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 4) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 4))
                        {
                            __m128 p1[4], p2[4], p3[4];
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempR, p1);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempG, p2);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempB, p3);    // simd loads
                            p1[1] = p2[0];
                            p1[2] = p3[0];
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstRowPtrTempR, dstRowPtrTempG, dstRowPtrTempB, p1);    // simd stores
                            xR += 4;
                            xG += 4;
                            xB += 4;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    srcRowPtrTempR += 3;
                                    dstRowPtrTempR++;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG += 3;
                                    dstRowPtrTempG++;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    srcRowPtrTempB += 3;
                                    dstRowPtrTempB++;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 12;
                        srcRowPtrTempG += 12;
                        srcRowPtrTempB += 12;
                        dstRowPtrTempR += 4;
                        dstRowPtrTempG += 4;
                        dstRowPtrTempB += 4;
                    }
                }
                else
                {
                    Rpp32f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + dstDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + 2 * dstDescPtr->strides.cStride;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR += 3;
                            dstRowPtrTempR++;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG += 3;
                            dstRowPtrTempG++;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            srcRowPtrTempB += 3;
                            dstRowPtrTempB++;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp32f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR * 3;
                    dstRowPtrTempR = dstRowPtr + xR;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR++;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp32f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + xG * 3 + 1;
                    dstRowPtrTempG = dstRowPtr + xG + dstDescPtr->strides.cStride;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG++;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp32f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB * 3 + 2;
                    dstRowPtrTempB = dstRowPtr + xB+ 2 * dstDescPtr->strides.cStride;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB++;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp32f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR++;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp32f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + 1;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + dstDescPtr->strides.cStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG++;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp32f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2 * dstDescPtr->strides.cStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB++;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp32f *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride) + srcDescPtr->strides.cStride * 1;
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride) + srcDescPtr->strides.cStride * 2;
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp32f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR;
                    srcRowPtrTempG = srcRowPtrG + xG;
                    srcRowPtrTempB = srcRowPtrB + xB;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 4)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 4) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 4) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 4))
                        {
                            __m128 p[4];
                            rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcRowPtrTempR, srcRowPtrTempG, srcRowPtrTempB, p);    // simd loads
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstRowPtrTempR, p);    // simd stores
                            xR += 4;
                            xG += 4;
                            xB += 4;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    srcRowPtrTempR++;
                                    dstRowPtrTempR += 3;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG++;
                                    dstRowPtrTempG += 3;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    srcRowPtrTempB++;
                                    dstRowPtrTempB += 3;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 4;
                        srcRowPtrTempG += 4;
                        srcRowPtrTempB += 4;
                        dstRowPtrTempR += 12;
                        dstRowPtrTempG += 12;
                        dstRowPtrTempB += 12;
                    }
                }
                else
                {
                    Rpp32f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR;
                    srcRowPtrTempG = srcRowPtrG + xG;
                    srcRowPtrTempB = srcRowPtrB + xB;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR++;
                            dstRowPtrTempR += 3;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG++;
                            dstRowPtrTempG += 3;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            srcRowPtrTempB++;
                            dstRowPtrTempB += 3;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp32f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR;
                    dstRowPtrTempR = dstRowPtr + xR * 3;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR++;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp32f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + srcDescPtr->strides.cStride + xG ;
                    dstRowPtrTempG = dstRowPtr + xG * 3 + 1;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG++;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp32f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB + 2 * srcDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + xB * 3 + 2;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB++;
                        dstRowPtrTempB += 3;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp32f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR++;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp32f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + srcDescPtr->strides.cStride;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + 1;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG++;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp32f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2 * srcDescPtr->strides.cStride;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB++;
                        dstRowPtrTempB += 3;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp32u vectorIncrement = 8;
            for(int c = 0; c < srcDescPtr->c; c++)
            {
                Rpp32f *srcPtrChannel, *dstPtrChannel;
                Rpp32f *srcPtrChannelRow, *dstPtrChannelRow, *srcPtrChannelRowOffset;
                srcPtrChannel = srcPtrImage + (c * srcDescPtr->strides.cStride);
                dstPtrChannel = dstPtrImage + (c * dstDescPtr->strides.cStride);
                srcPtrChannelRow = srcPtrChannel;
                srcPtrChannelRowOffset = srcPtrChannel + (yOffsets[c] * srcDescPtr->strides.hStride);
                dstPtrChannelRow = dstPtrChannel;
                int currentRow = yOffsets[c];
                for(; currentRow < roi.xywhROI.roiHeight; currentRow++)
                {
                    Rpp32f *srcRowTempOffset, *dstRowTemp, *srcRowTemp;
                    srcRowTempOffset = srcPtrChannelRowOffset + xOffsets[c];
                    srcRowTemp = srcPtrChannelRow + (roi.xywhROI.roiWidth - xOffsets[c]);
                    dstRowTemp = dstPtrChannelRow;
                    int currentCol = xOffsets[c];
                    Rpp32u alignedLength = (roi.xywhROI.roiWidth - currentCol) & ~7;
                    if (((currentRow >= 0) && (currentRow < roi.xywhROI.roiHeight)) && ((currentCol >= 0) && (currentCol < roi.xywhROI.roiWidth)))
                    {
                        for( ; currentCol < alignedLength; currentCol += vectorIncrement)
                        {
                            __m256 p;
                            p = _mm256_loadu_ps(srcRowTempOffset);
                            _mm256_storeu_ps(dstRowTemp, p);
                            srcRowTempOffset += vectorIncrement;
                            dstRowTemp += vectorIncrement;
                        }
                        for(; currentCol < roi.xywhROI.roiWidth; currentCol++)
                            *dstRowTemp++ = *srcRowTempOffset++;
                    }
                    for(int i = 0; i < xOffsets[c]; i++)
                        *dstRowTemp++ = *srcRowTemp++;
                    srcPtrChannelRowOffset += srcDescPtr->strides.hStride;
                    dstPtrChannelRow += dstDescPtr->strides.hStride;
                    srcPtrChannelRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannelRow = srcPtrChannel + ((roi.xywhROI.roiHeight - yOffsets[c]) * srcDescPtr->strides.hStride);
                for(int j = 0; j < yOffsets[c]; j++)
                {
                    Rpp32f *dstRowTemp, *srcRowTemp;
                    srcRowTemp = srcPtrChannelRow;
                    dstRowTemp = dstPtrChannelRow;
                    Rpp32f alignedLength = roi.xywhROI.roiWidth & ~7;
                    int currentCol = 0;
                    for( ; currentCol < alignedLength; currentCol += vectorIncrement)
                    {
                        __m256 p;
                        p = _mm256_loadu_ps(srcRowTemp);
                        _mm256_storeu_ps(dstRowTemp, p);
                        srcRowTemp += vectorIncrement;
                        dstRowTemp += vectorIncrement;
                    }
                    for(; currentCol < roi.xywhROI.roiWidth; currentCol++)
                        *dstRowTemp++ = *srcRowTemp++;
                    srcPtrChannelRow += srcDescPtr->strides.hStride;
                    dstPtrChannelRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp32f *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride);
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride);
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp32f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 4)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 4) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 4) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 4))
                        {
                            __m128 p1[4], p2[4], p3[4];
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempR, p1);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempG, p2);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempB, p3);    // simd loads
                            p1[1] = p2[0];
                            p1[2] = p3[0];
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstRowPtrTempR, p1);    // simd stores
                            xR += 4;
                            xG += 4;
                            xB += 4;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    dstRowPtrTempR += 3;
                                    srcRowPtrTempR += 3;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG += 3;
                                    dstRowPtrTempG += 3;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    dstRowPtrTempB += 3;
                                    srcRowPtrTempB += 3;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 12;
                        srcRowPtrTempG += 12;
                        srcRowPtrTempB += 12;
                        dstRowPtrTempR += 12;
                        dstRowPtrTempG += 12;
                        dstRowPtrTempB += 12;
                    }
                }
                else
                {
                    Rpp32f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR += 3;
                            dstRowPtrTempR += 3;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG += 3;
                            dstRowPtrTempG += 3;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            dstRowPtrTempB += 3;
                            srcRowPtrTempB += 3;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp32f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR * 3;
                    dstRowPtrTempR = dstRowPtr + xR * 3;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp32f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + xG * 3 + 1;
                    dstRowPtrTempG = dstRowPtr + xG * 3 + 1;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp32f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB * 3 + 2;
                    dstRowPtrTempB = dstRowPtr + xB * 3 + 2;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB += 3;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp32f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp32f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + 1;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + 1;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp32f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB += 3;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = rgbOffsets[batchCount].r.x;
        Rpp32u yOffsetRchn = rgbOffsets[batchCount].r.y;
        Rpp32u xOffsetGchn = rgbOffsets[batchCount].g.x;
        Rpp32u yOffsetGchn = rgbOffsets[batchCount].g.y;
        Rpp32u xOffsetBchn = rgbOffsets[batchCount].b.x;
        Rpp32u yOffsetBchn = rgbOffsets[batchCount].b.y;

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp16f *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride);
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride);
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp16f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + dstDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + 2 * dstDescPtr->strides.cStride;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 4)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 4) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 4) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 4))
                        {
                            Rpp32f srcRowPtrTempR_ps[12], srcRowPtrTempG_ps[12], srcRowPtrTempB_ps[12], dstPtrTemp_ps[12];
                            for(int cnt = 0; cnt < 12; cnt++)
                            {
                                *(srcRowPtrTempR_ps + cnt) = (Rpp32f) *(srcRowPtrTempR + cnt);
                                *(srcRowPtrTempG_ps + cnt) = (Rpp32f) *(srcRowPtrTempG + cnt);
                                *(srcRowPtrTempB_ps + cnt) = (Rpp32f) *(srcRowPtrTempB + cnt);
                            }
                            __m128 p1[4], p2[4], p3[4];
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempR_ps, p1);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempG_ps, p2);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempB_ps, p3);    // simd loads
                            p1[1] = p2[0];
                            p1[2] = p3[0];
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p1);    // simd stores
                            for(int cnt = 0; cnt < 4; cnt++)
                            {
                                *(dstRowPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                                *(dstRowPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                                *(dstRowPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                            }
                            xR += 4;
                            xG += 4;
                            xB += 4;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    srcRowPtrTempR += 3;
                                    dstRowPtrTempR++;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG += 3;
                                    dstRowPtrTempG++;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    srcRowPtrTempB += 3;
                                    dstRowPtrTempB++;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 12;
                        srcRowPtrTempG += 12;
                        srcRowPtrTempB += 12;
                        dstRowPtrTempR += 4;
                        dstRowPtrTempG += 4;
                        dstRowPtrTempB += 4;
                    }
                }
                else
                {
                    Rpp16f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + dstDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + 2 * dstDescPtr->strides.cStride;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR += 3;
                            dstRowPtrTempR++;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG += 3;
                            dstRowPtrTempG++;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            srcRowPtrTempB += 3;
                            dstRowPtrTempB++;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp16f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR * 3;
                    dstRowPtrTempR = dstRowPtr + xR;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR++;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp16f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + xG * 3 + 1;
                    dstRowPtrTempG = dstRowPtr + xG + dstDescPtr->strides.cStride;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG++;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp16f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB * 3 + 2;
                    dstRowPtrTempB = dstRowPtr + xB+ 2 * dstDescPtr->strides.cStride;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB++;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp16f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR++;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp16f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + 1;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + dstDescPtr->strides.cStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG++;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp16f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2 * dstDescPtr->strides.cStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB++;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp16f *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride) + srcDescPtr->strides.cStride * 1;
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride) + srcDescPtr->strides.cStride * 2;
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp16f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR;
                    srcRowPtrTempG = srcRowPtrG + xG;
                    srcRowPtrTempB = srcRowPtrB + xB;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 4)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 4) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 4) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 4))
                        {
                            Rpp32f srcRowPtrTemp_ps[12], dstPtrTemp_ps[12];
                            for(int cnt = 0; cnt < 4; cnt++)
                            {
                                *(srcRowPtrTemp_ps + cnt) = (Rpp32f) *(srcRowPtrTempR + cnt);
                                *(srcRowPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcRowPtrTempG + cnt);
                                *(srcRowPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcRowPtrTempB + cnt);
                            }
                            __m128 p[4];
                            rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcRowPtrTemp_ps, srcRowPtrTemp_ps + 4, srcRowPtrTemp_ps + 8, p);    // simd loads
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
                            for(int cnt = 0; cnt < 12; cnt++)
                            {
                                *(dstRowPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            }
                            xR += 4;
                            xG += 4;
                            xB += 4;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    srcRowPtrTempR++;
                                    dstRowPtrTempR += 3;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG++;
                                    dstRowPtrTempG += 3;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    srcRowPtrTempB++;
                                    dstRowPtrTempB += 3;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 4;
                        srcRowPtrTempG += 4;
                        srcRowPtrTempB += 4;
                        dstRowPtrTempR += 12;
                        dstRowPtrTempG += 12;
                        dstRowPtrTempB += 12;
                    }
                }
                else
                {
                    Rpp16f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR;
                    srcRowPtrTempG = srcRowPtrG + xG;
                    srcRowPtrTempB = srcRowPtrB + xB;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR++;
                            dstRowPtrTempR += 3;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG++;
                            dstRowPtrTempG += 3;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            srcRowPtrTempB++;
                            dstRowPtrTempB += 3;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp16f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR;
                    dstRowPtrTempR = dstRowPtr + xR * 3;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR++;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp16f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + srcDescPtr->strides.cStride + xG ;
                    dstRowPtrTempG = dstRowPtr + xG * 3 + 1;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG++;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp16f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB + 2 * srcDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + xB * 3 + 2;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB++;
                        dstRowPtrTempB += 3;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp16f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR++;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp16f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + srcDescPtr->strides.cStride;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + 1;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG++;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp16f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2 * srcDescPtr->strides.cStride;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB++;
                        dstRowPtrTempB += 3;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp32u vectorIncrement = 8;
            for(int c = 0; c < srcDescPtr->c; c++)
            {
                Rpp16f *srcPtrChannel, *dstPtrChannel;
                Rpp16f *srcPtrChannelRow, *dstPtrChannelRow, *srcPtrChannelRowOffset;
                srcPtrChannel = srcPtrImage + (c * srcDescPtr->strides.cStride);
                dstPtrChannel = dstPtrImage + (c * dstDescPtr->strides.cStride);
                srcPtrChannelRow = srcPtrChannel;
                srcPtrChannelRowOffset = srcPtrChannel + (yOffsets[c] * srcDescPtr->strides.hStride);
                dstPtrChannelRow = dstPtrChannel;
                int currentRow = yOffsets[c];
                for(; currentRow < roi.xywhROI.roiHeight; currentRow++)
                {
                    Rpp16f *srcRowTempOffset, *dstRowTemp, *srcRowTemp;
                    srcRowTempOffset = srcPtrChannelRowOffset + xOffsets[c];
                    srcRowTemp = srcPtrChannelRow + (roi.xywhROI.roiWidth - xOffsets[c]);
                    dstRowTemp = dstPtrChannelRow;
                    int currentCol = xOffsets[c];
                    Rpp32u alignedLength = (roi.xywhROI.roiWidth - currentCol) & ~7;
                    if (((currentRow >= 0) && (currentRow < roi.xywhROI.roiHeight)) && ((currentCol >= 0) && (currentCol < roi.xywhROI.roiWidth)))
                    {
                        for( ; currentCol < alignedLength; currentCol += vectorIncrement)
                        {
                            Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                            for(int cnt = 0; cnt < 8; cnt++)
                            {
                                *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcRowTempOffset + cnt);
                            }
                            __m256 p;
                            p = _mm256_loadu_ps(srcPtrTemp_ps);
                            _mm256_storeu_ps(dstPtrTemp_ps, p);
                            for(int cnt = 0; cnt < 8; cnt++)
                            {
                                *(dstRowTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            }
                            srcRowTempOffset += vectorIncrement;
                            dstRowTemp += vectorIncrement;
                        }
                        for(; currentCol < roi.xywhROI.roiWidth; currentCol++)
                            *dstRowTemp++ = *srcRowTempOffset++;
                    }
                    for(int i = 0; i < xOffsets[c]; i++)
                        *dstRowTemp++ = *srcRowTemp++;
                    srcPtrChannelRowOffset += srcDescPtr->strides.hStride;
                    dstPtrChannelRow += dstDescPtr->strides.hStride;
                    srcPtrChannelRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannelRow = srcPtrChannel + ((roi.xywhROI.roiHeight - yOffsets[c]) * srcDescPtr->strides.hStride);
                for(int j = 0; j < yOffsets[c]; j++)
                {
                    Rpp16f *dstRowTemp, *srcRowTemp;
                    srcRowTemp = srcPtrChannelRow;
                    dstRowTemp = dstPtrChannelRow;
                    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~7;
                    int currentCol = 0;
                    for( ; currentCol < alignedLength; currentCol += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                        for(int cnt = 0; cnt < 8; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcRowTemp + cnt);
                        }
                        __m256 p;
                        p = _mm256_loadu_ps(srcPtrTemp_ps);
                        _mm256_storeu_ps(dstPtrTemp_ps, p);
                        for(int cnt = 0; cnt < 8; cnt++)
                        {
                            *(dstRowTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        }
                        srcRowTemp += vectorIncrement;
                        dstRowTemp += vectorIncrement;
                    }
                    for(; currentCol < roi.xywhROI.roiWidth; currentCol++)
                        *dstRowTemp++ = *srcRowTemp++;
                    srcPtrChannelRow += srcDescPtr->strides.hStride;
                    dstPtrChannelRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp16f *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride);
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride);
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp16f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 4)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 4) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 4) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 4))
                        {
                            Rpp32f srcRowPtrTempR_ps[12], srcRowPtrTempG_ps[12], srcRowPtrTempB_ps[12], dstPtrTemp_ps[12];
                            for(int cnt = 0; cnt < 12; cnt++)
                            {
                                *(srcRowPtrTempR_ps + cnt) = (Rpp32f) *(srcRowPtrTempR + cnt);
                                *(srcRowPtrTempG_ps + cnt) = (Rpp32f) *(srcRowPtrTempG + cnt);
                                *(srcRowPtrTempB_ps + cnt) = (Rpp32f) *(srcRowPtrTempB + cnt);
                            }
                            __m128 p1[4], p2[4], p3[4];
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempR_ps, p1);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempG_ps, p2);    // simd loads
                            rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcRowPtrTempB_ps, p3);    // simd loads
                            p1[1] = p2[0];
                            p1[2] = p3[0];
                            rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p1);    // simd stores
                            for(int cnt = 0; cnt < 12; cnt++)
                            {
                                *(dstRowPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                            }
                            xR += 4;
                            xG += 4;
                            xB += 4;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    dstRowPtrTempR += 3;
                                    srcRowPtrTempR += 3;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG += 3;
                                    dstRowPtrTempG += 3;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    dstRowPtrTempB += 3;
                                    srcRowPtrTempB += 3;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 12;
                        srcRowPtrTempG += 12;
                        srcRowPtrTempB += 12;
                        dstRowPtrTempR += 12;
                        dstRowPtrTempG += 12;
                        dstRowPtrTempB += 12;
                    }
                }
                else
                {
                    Rpp16f *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR += 3;
                            dstRowPtrTempR += 3;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG += 3;
                            dstRowPtrTempG += 3;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            dstRowPtrTempB += 3;
                            srcRowPtrTempB += 3;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp16f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR * 3;
                    dstRowPtrTempR = dstRowPtr + xR * 3;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp16f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + xG * 3 + 1;
                    dstRowPtrTempG = dstRowPtr + xG * 3 + 1;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp16f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB * 3 + 2;
                    dstRowPtrTempB = dstRowPtr + xB * 3 + 2;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB += 3;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp16f *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp16f *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + 1;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + 1;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp16f *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB += 3;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        __m128i maskR = _mm_setr_epi8(0, 0x80, 0x80, 3, 0x80, 0x80, 6, 0x80, 0x80, 9, 0x80, 0x80, 12, 0x80, 0x80, 15);
        __m128i maskGB = _mm_setr_epi8(0x80, 1, 2, 0x80, 4, 5, 0x80, 7, 8, 0x80, 10, 11, 0x80, 13, 14, 0x80);
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = rgbOffsets[batchCount].r.x;
        Rpp32u yOffsetRchn = rgbOffsets[batchCount].r.y;
        Rpp32u xOffsetGchn = rgbOffsets[batchCount].g.x;
        Rpp32u yOffsetGchn = rgbOffsets[batchCount].g.y;
        Rpp32u xOffsetBchn = rgbOffsets[batchCount].b.x;
        Rpp32u yOffsetBchn = rgbOffsets[batchCount].b.y;

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp8s *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride);
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride);
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp8s *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + dstDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + 2 * dstDescPtr->strides.cStride;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 16)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 16) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 16) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 16))
                        {
                            __m128 p1[12], p2[12], p3[12];
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcRowPtrTempR, p1);    // simd loads
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcRowPtrTempG, p2);    // simd loads
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcRowPtrTempB, p3);    // simd loads
                            for(int k = 0; k < 4; k++)
                            {
                                p1[k + 4] = p2[k];
                                p1[k + 8] = p3[k];
                            }
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstRowPtrTempR, dstRowPtrTempG, dstRowPtrTempB, p1);    // simd stores
                            xR += 16;
                            xG += 16;
                            xB += 16;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    srcRowPtrTempR += 3;
                                    dstRowPtrTempR++;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG += 3;
                                    dstRowPtrTempG++;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    srcRowPtrTempB += 3;
                                    dstRowPtrTempB++;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 48;
                        srcRowPtrTempG += 48;
                        srcRowPtrTempB += 48;
                        dstRowPtrTempR += 16;
                        dstRowPtrTempG += 16;
                        dstRowPtrTempB += 16;
                    }
                }
                else
                {
                    Rpp8s *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + dstDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + 2 * dstDescPtr->strides.cStride;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR += 3;
                            dstRowPtrTempR++;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG += 3;
                            dstRowPtrTempG++;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            srcRowPtrTempB += 3;
                            dstRowPtrTempB++;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp8s *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR * 3;
                    dstRowPtrTempR = dstRowPtr + xR;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR++;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp8s *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + xG * 3 + 1;
                    dstRowPtrTempG = dstRowPtr + xG + dstDescPtr->strides.cStride;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG++;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp8s *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB * 3 + 2;
                    dstRowPtrTempB = dstRowPtr + xB+ 2 * dstDescPtr->strides.cStride;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB++;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp8s *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR++;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp8s *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + 1;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + dstDescPtr->strides.cStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG++;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp8s *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2 * dstDescPtr->strides.cStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB++;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp8s *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride) + srcDescPtr->strides.cStride * 1;
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride) + srcDescPtr->strides.cStride * 2;
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp8s *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR;
                    srcRowPtrTempG = srcRowPtrG + xG;
                    srcRowPtrTempB = srcRowPtrB + xB;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 16)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 16) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 16) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 16))
                        {
                            __m128 p[12];
                            rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcRowPtrTempR, srcRowPtrTempG, srcRowPtrTempB, p);    // simd loads
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstRowPtrTempR, p);    // simd stores
                            xR += 16;
                            xG += 16;
                            xB += 16;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    srcRowPtrTempR++;
                                    dstRowPtrTempR += 3;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG++;
                                    dstRowPtrTempG += 3;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    srcRowPtrTempB++;
                                    dstRowPtrTempB += 3;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 16;
                        srcRowPtrTempG += 16;
                        srcRowPtrTempB += 16;
                        dstRowPtrTempR += 48;
                        dstRowPtrTempG += 48;
                        dstRowPtrTempB += 48;
                    }
                }
                else
                {
                    Rpp8s *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR;
                    srcRowPtrTempG = srcRowPtrG + xG;
                    srcRowPtrTempB = srcRowPtrB + xB;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR++;
                            dstRowPtrTempR += 3;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG++;
                            dstRowPtrTempG += 3;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            srcRowPtrTempB++;
                            dstRowPtrTempB += 3;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp8s *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR;
                    dstRowPtrTempR = dstRowPtr + xR * 3;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR++;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp8s *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + srcDescPtr->strides.cStride + xG ;
                    dstRowPtrTempG = dstRowPtr + xG * 3 + 1;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG++;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp8s *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB + 2 * srcDescPtr->strides.cStride;
                    dstRowPtrTempB = dstRowPtr + xB * 3 + 2;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB++;
                        dstRowPtrTempB += 3;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp8s *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR++;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp8s *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + srcDescPtr->strides.cStride;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + 1;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG++;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp8s *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2 * srcDescPtr->strides.cStride;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB++;
                        dstRowPtrTempB += 3;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp32u vectorIncrement = 32;
            for(int c = 0; c < srcDescPtr->c; c++)
            {
                Rpp8s *srcPtrChannel, *dstPtrChannel;
                Rpp8s *srcPtrChannelRow, *dstPtrChannelRow, *srcPtrChannelRowOffset;
                srcPtrChannel = srcPtrImage + (c * srcDescPtr->strides.cStride);
                dstPtrChannel = dstPtrImage + (c * dstDescPtr->strides.cStride);
                srcPtrChannelRow = srcPtrChannel;
                srcPtrChannelRowOffset = srcPtrChannel + (yOffsets[c] * srcDescPtr->strides.hStride);
                dstPtrChannelRow = dstPtrChannel;
                int currentRow = yOffsets[c];
                for(; currentRow < roi.xywhROI.roiHeight; currentRow++)
                {
                    Rpp8s *srcRowTempOffset, *dstRowTemp, *srcRowTemp;
                    srcRowTempOffset = srcPtrChannelRowOffset + xOffsets[c];
                    srcRowTemp = srcPtrChannelRow + (roi.xywhROI.roiWidth - xOffsets[c]);
                    dstRowTemp = dstPtrChannelRow;
                    int currentCol = xOffsets[c];
                    Rpp32u alignedLength = (roi.xywhROI.roiWidth - currentCol) & ~31;
                    if (((currentRow >= 0) && (currentRow < roi.xywhROI.roiHeight)) && ((currentCol >= 0) && (currentCol < roi.xywhROI.roiWidth)))
                    {
                        for( ; currentCol < alignedLength; currentCol += vectorIncrement)
                        {
                            __m256i p;
                            p = _mm256_loadu_epi8(srcRowTempOffset);
                            _mm256_storeu_epi8(dstRowTemp, p);
                            srcRowTempOffset += vectorIncrement;
                            dstRowTemp += vectorIncrement;
                        }
                        for(; currentCol < roi.xywhROI.roiWidth; currentCol++)
                            *dstRowTemp++ = *srcRowTempOffset++;
                    }
                    for(int i = 0; i < xOffsets[c]; i++)
                        *dstRowTemp++ = *srcRowTemp++;
                    srcPtrChannelRowOffset += srcDescPtr->strides.hStride;
                    dstPtrChannelRow += dstDescPtr->strides.hStride;
                    srcPtrChannelRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannelRow = srcPtrChannel + ((roi.xywhROI.roiHeight - yOffsets[c]) * srcDescPtr->strides.hStride);
                for(int j = 0; j < yOffsets[c]; j++)
                {
                    Rpp8s *dstRowTemp, *srcRowTemp;
                    srcRowTemp = srcPtrChannelRow;
                    dstRowTemp = dstPtrChannelRow;
                    Rpp32u alignedLength = roi.xywhROI.roiWidth & ~31;
                    int currentCol = 0;
                    for( ; currentCol < alignedLength; currentCol += vectorIncrement)
                    {
                        __m256i p;
                        p = _mm256_loadu_epi8(srcRowTemp);
                        _mm256_storeu_epi8(dstRowTemp, p);
                        srcRowTemp += vectorIncrement;
                        dstRowTemp += vectorIncrement;
                    }
                    for(; currentCol < roi.xywhROI.roiWidth; currentCol++)
                        *dstRowTemp++ = *srcRowTemp++;
                    srcPtrChannelRow += srcDescPtr->strides.hStride;
                    dstPtrChannelRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            int yR = yOffsetRchn;
            int yG = yOffsetGchn;
            int yB = yOffsetBchn;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int xR = xOffsetRchn;
                int xG = xOffsetGchn;
                int xB = xOffsetBchn;
                Rpp8s *srcRowPtrR, *srcRowPtrG, *srcRowPtrB, *dstRowPtr;
                srcRowPtrR = srcPtrImage + (yR * srcDescPtr->strides.hStride);
                srcRowPtrG = srcPtrImage + (yG * srcDescPtr->strides.hStride);
                srcRowPtrB = srcPtrImage + (yB * srcDescPtr->strides.hStride);
                dstRowPtr = dstPtrImage + i * dstDescPtr->strides.hStride;
                if((yR >= 0) && (yR < roi.xywhROI.roiHeight) && (yG >= 0) && (yG < roi.xywhROI.roiHeight) && (yB >= 0) && (yB < roi.xywhROI.roiHeight))
                {
                    Rpp8s *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j += 16)
                    {
                        if((xR >= 0) && (xR <= roi.xywhROI.roiWidth - 16) && (xG >= 0) && (xG <= roi.xywhROI.roiWidth - 16) && (xB >= 0) && (xB < roi.xywhROI.roiWidth - 16))
                        {
                            __m128 p1[12], p2[12], p3[12];
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcRowPtrTempR, p1);    // simd loads
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcRowPtrTempG, p2);    // simd loads
                            rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcRowPtrTempB, p3);    // simd loads
                            for(int k = 0; k < 4; k++)
                            {
                                p1[k + 4] = p2[k];
                                p1[k + 8] = p3[k];
                            }
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstRowPtrTempR, p1);    // simd stores
                            xR += 16;
                            xG += 16;
                            xB += 16;
                        }
                        else
                        {
                            if(xR < roi.xywhROI.roiWidth)
                            {
                                for( ; xR < roi.xywhROI.roiWidth; xR++)
                                {
                                    *dstRowPtrTempR = *srcRowPtrTempR;
                                    dstRowPtrTempR += 3;
                                    srcRowPtrTempR += 3;
                                }
                            }
                            if(xG < roi.xywhROI.roiWidth)
                            {
                                for( ; xG < roi.xywhROI.roiWidth; xG++)
                                {
                                   *dstRowPtrTempG = *srcRowPtrTempG;
                                    srcRowPtrTempG += 3;
                                    dstRowPtrTempG += 3;
                                }
                            }
                            if(xB < roi.xywhROI.roiWidth)
                            {
                                for( ; xB < roi.xywhROI.roiWidth; xB++)
                                {
                                    *dstRowPtrTempB = *srcRowPtrTempB;
                                    dstRowPtrTempB += 3;
                                    srcRowPtrTempB += 3;
                                }
                            }
                            break;
                        }
                        srcRowPtrTempR += 48;
                        srcRowPtrTempG += 48;
                        srcRowPtrTempB += 48;
                        dstRowPtrTempR += 48;
                        dstRowPtrTempG += 48;
                        dstRowPtrTempB += 48;
                    }
                }
                else
                {
                    Rpp8s *srcRowPtrTempR, *srcRowPtrTempG, *srcRowPtrTempB, *dstRowPtrTempR, *dstRowPtrTempG, *dstRowPtrTempB;
                    srcRowPtrTempR = srcRowPtrR + xR * 3;
                    srcRowPtrTempG = srcRowPtrG + xG * 3 + 1;
                    srcRowPtrTempB = srcRowPtrB + xB * 3 + 2;
                    dstRowPtrTempR = dstRowPtr;
                    dstRowPtrTempG = dstRowPtr + 1;
                    dstRowPtrTempB = dstRowPtr + 2;
                    if(yR < roi.xywhROI.roiHeight && xR < roi.xywhROI.roiWidth)
                    {
                        for(; xR < roi.xywhROI.roiWidth; xR++)
                        {
                            *dstRowPtrTempR = *srcRowPtrTempR;
                            srcRowPtrTempR += 3;
                            dstRowPtrTempR += 3;
                        }
                    }
                    if(yG < roi.xywhROI.roiHeight && xG < roi.xywhROI.roiWidth)
                    {
                        for(; xG < roi.xywhROI.roiWidth; xG++)
                        {
                            *dstRowPtrTempG = *srcRowPtrTempG;
                            srcRowPtrTempG += 3;
                            dstRowPtrTempG += 3;
                        }
                    }
                    if(yB < roi.xywhROI.roiHeight && xB < roi.xywhROI.roiWidth)
                    {
                        for(; xB < roi.xywhROI.roiWidth; xB++)
                        {
                            *dstRowPtrTempB = *srcRowPtrTempB;
                            dstRowPtrTempB += 3;
                            srcRowPtrTempB += 3;
                        }
                    }
                }
                if(yR < roi.xywhROI.roiHeight && xR >= roi.xywhROI.roiWidth)
                {
                    xR = xR - xOffsetRchn;
                    Rpp8s *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + i * srcDescPtr->strides.hStride + xR * 3;
                    dstRowPtrTempR = dstRowPtr + xR * 3;
                    for(; xR < roi.xywhROI.roiWidth; xR++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG < roi.xywhROI.roiHeight && xG >= roi.xywhROI.roiWidth)
                {
                    xG = xG - xOffsetGchn;
                    Rpp8s *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + i * srcDescPtr->strides.hStride + xG * 3 + 1;
                    dstRowPtrTempG = dstRowPtr + xG * 3 + 1;
                    for(; xG < roi.xywhROI.roiWidth; xG++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB < roi.xywhROI.roiHeight && xB >= roi.xywhROI.roiWidth)
                {
                    xB = xB - xOffsetBchn;
                    Rpp8s *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + i * srcDescPtr->strides.hStride + xB * 3 + 2;
                    dstRowPtrTempB = dstRowPtr + xB * 3 + 2;
                    for(; xB < roi.xywhROI.roiWidth; xB++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB += 3;
                    }
                }
                if(yR >= roi.xywhROI.roiHeight)
                {
                    int idx = yR - yOffsetRchn;
                    Rpp8s *srcRowPtrTempR, *dstRowPtrTempR;
                    srcRowPtrTempR = srcPtrImage + idx * srcDescPtr->strides.hStride;
                    dstRowPtrTempR = dstPtrImage + idx * dstDescPtr->strides.hStride;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempR = *srcRowPtrTempR;
                        srcRowPtrTempR += 3;
                        dstRowPtrTempR += 3;
                    }
                }
                if(yG >= roi.xywhROI.roiHeight)
                {
                    int idx = yG - yOffsetGchn;
                    Rpp8s *srcRowPtrTempG, *dstRowPtrTempG;
                    srcRowPtrTempG = srcPtrImage + idx * srcDescPtr->strides.hStride + 1;
                    dstRowPtrTempG = dstPtrImage + idx * dstDescPtr->strides.hStride + 1;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempG = *srcRowPtrTempG;
                        srcRowPtrTempG += 3;
                        dstRowPtrTempG += 3;
                    }
                }
                if(yB >= roi.xywhROI.roiHeight)
                {
                    int idx = yB - yOffsetBchn;
                    Rpp8s *srcRowPtrTempB, *dstRowPtrTempB;
                    srcRowPtrTempB = srcPtrImage + idx * srcDescPtr->strides.hStride + 2;
                    dstRowPtrTempB = dstPtrImage + idx * dstDescPtr->strides.hStride + 2;
                    for(int x = 0; x < roi.xywhROI.roiWidth; x++)
                    {
                        *dstRowPtrTempB = *srcRowPtrTempB;
                        srcRowPtrTempB += 3;
                        dstRowPtrTempB += 3;
                    }
                }
                yR++;
                yG++;
                yB++;
            }
        }
    }
    return RPP_SUCCESS;
}