/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void increment_row_ptrs(Rpp8u **srcPtrTemp, Rpp32u kernelSize, Rpp32u increment)
{
    for (int i = 0; i < kernelSize; i++)
        srcPtrTemp[i] += increment;
}

inline void get_kernel_loop_limit(Rpp32s index, Rpp32s &loopLimit, Rpp32u kernelSize, Rpp32u padLength, Rpp32u length)
{
    if ((index >= padLength) && (index < length - padLength))
        loopLimit = kernelSize;
    else
    {
        Rpp32u rowFactor = (index < padLength) ? index : (length - 1 - index);
        loopLimit = kernelSize - padLength + rowFactor;
    }
}

inline void box_filter_generic_u8_u8_host_tensor(Rpp8u **srcPtrTemp, Rpp8u *dstPtrTemp, Rpp32u rowIndex, Rpp32u columnIndex,
                                                 Rpp32u kernelSize, Rpp32u padLength, Rpp32u height, Rpp32u width)
{
    Rpp32f accum = 0.0f;
    Rpp32s rowKernelLoopLimit, columnKernelLoopLimit;

    // find the rowKernelLoopLimit, colKernelLoopLimit based on rowIndex, columnIndex
    get_kernel_loop_limit(rowIndex, rowKernelLoopLimit, kernelSize, padLength, height);
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, kernelSize, padLength, width);

    for (int i = 0; i < rowKernelLoopLimit; i++)
    {
        for (int j = 0; j < columnKernelLoopLimit; j++)
            accum += static_cast<Rpp32f>(srcPtrTemp[i][j]);
    }
    static Rpp32f divFactor = 1.0f / (kernelSize * kernelSize);
    accum *= divFactor;
    *dstPtrTemp = static_cast<Rpp8u>(RPPPIXELCHECK(accum));
}

RppStatus box_filter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f kernelSizeInverseSquare = 1.0 / (kernelSize * kernelSize);
        Rpp16s convolutionFactor = (Rpp16s) std::ceil(65536 * kernelSizeInverseSquare);
        __m128i pxConvolutionFactor = _mm_set1_epi16(convolutionFactor);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // box filter without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (kernelSize == 3)
            {
                Rpp8u *srcPtrRow[3], *dstPtrRow;
                for (int i = 0; i < 3; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 14) * 14;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool firstRow = (i == 0);
                    bool lastRow = (i == (roi.xywhROI.roiHeight - 1));
                    Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, i, k, kernelSize, padLength, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth);
                        dstPtrTemp++;
                    }

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                    {
                        __m128i pxRow[3];
                        if (!firstRow && !lastRow)
                        {
                            pxRow[0] = _mm_loadu_si128((__m128i *)srcPtrTemp[0]);    // load ROW[0][0] .. ROW[0][15]
                            pxRow[1] = _mm_loadu_si128((__m128i *)srcPtrTemp[1]);    // load ROW[1][0] .. ROW[1][15]
                            pxRow[2] = _mm_loadu_si128((__m128i *)srcPtrTemp[2]);    // load ROW[2][0] .. ROW[2][15]
                        }
                        else
                        {
                            pxRow[0] = xmm_px0;
                            pxRow[1] = _mm_loadu_si128((__m128i *)srcPtrTemp[0]);
                            pxRow[2] = _mm_loadu_si128((__m128i *)srcPtrTemp[1]);
                        }

                        __m128i pxLower, pxUpper;
                        pxLower = _mm_unpacklo_epi8(pxRow[0], xmm_px0);                            // ROW0 0-7
                        pxLower = _mm_add_epi16(pxLower, _mm_unpacklo_epi8(pxRow[1], xmm_px0));    // ROW0 0-7 + ROW1 0-7
                        pxLower = _mm_add_epi16(pxLower, _mm_unpacklo_epi8(pxRow[2], xmm_px0));    // upper accum - ROW0 0-7 + ROW1 0-7 + ROW2 0-7

                        pxUpper = _mm_unpackhi_epi8(pxRow[0], xmm_px0);                            // ROW0 8-15
                        pxUpper = _mm_add_epi16(pxUpper, _mm_unpackhi_epi8(pxRow[1], xmm_px0));    // ROW0 8-15 + ROW1 8-15
                        pxUpper = _mm_add_epi16(pxUpper, _mm_unpackhi_epi8(pxRow[2], xmm_px0));    // lower accum - ROW0 8-15 + ROW1 8-15 + ROW2 8-15

                        // shift row wise and add
                        for (int k = 0; k < 3; k++)
                        {
                            __m128i pxTemp[2];
                            pxTemp[0] = _mm_shuffle_epi8(pxRow[k], xmm_pxMask01To15);
                            pxTemp[1] = _mm_shuffle_epi8(pxRow[k], xmm_pxMask02To15);
                            pxLower = _mm_add_epi16(pxLower, _mm_unpacklo_epi8(pxTemp[0], xmm_px0));
                            pxLower = _mm_add_epi16(pxLower, _mm_unpacklo_epi8(pxTemp[1], xmm_px0));
                            pxUpper = _mm_add_epi16(pxUpper, _mm_unpackhi_epi8(pxTemp[0], xmm_px0));
                            pxUpper = _mm_add_epi16(pxUpper, _mm_unpackhi_epi8(pxTemp[1], xmm_px0));
                        }

                        pxLower = _mm_mulhi_epi16(pxLower, pxConvolutionFactor);
                        pxUpper = _mm_mulhi_epi16(pxUpper, pxConvolutionFactor);
                        pxLower = _mm_packus_epi16(pxLower, pxUpper);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxLower);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                        dstPtrTemp += 14;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, i, vectorLoopCount + padLength, kernelSize, padLength, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!firstRow) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if (kernelSize == 9)
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 12) * 12;
                Rpp8u *srcPtrRow[9], *dstPtrRow;
                for (int i = 0; i < 9; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[9] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2],
                                            srcPtrRow[3], srcPtrRow[4], srcPtrRow[5],
                                            srcPtrRow[6], srcPtrRow[7], srcPtrRow[8]};
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, i, k, kernelSize, padLength, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth);
                        dstPtrTemp++;
                        vectorLoopCount++;
                    }
                    // process remaining columns in row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, i, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}