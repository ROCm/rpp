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
    accum *= 0.1111111f;
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
            Rpp8u *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * padLength * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 12) * 12;

            const __m128i xmm_pxMask02To15 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0x80, 0x80);
            const __m128i xmm_pxMask04To15 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0x80, 0x80, 0x80, 0x80);
            const __m128i xmm_pxMask00To01 = _mm_setr_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1);
            const __m128i xmm_pxMask00To03 = _mm_setr_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3);

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

                // process remaining columns
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
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

                    __m128i pxUpper, pxLower;
                    pxUpper = _mm_unpacklo_epi8(pxRow[0], xmm_px0);                            // ROW0 0-7
                    pxUpper = _mm_add_epi16(pxUpper, _mm_unpacklo_epi8(pxRow[1], xmm_px0));    // ROW0 0-7 + ROW1 0-7
                    pxUpper = _mm_add_epi16(pxUpper, _mm_unpacklo_epi8(pxRow[2], xmm_px0));    // upper accum - ROW0 0-7 + ROW1 0-7 + ROW2 0-7

                    pxLower = _mm_unpackhi_epi8(pxRow[0], xmm_px0);                            // ROW0 8-15
                    pxLower = _mm_add_epi16(pxLower, _mm_unpackhi_epi8(pxRow[1], xmm_px0));    // ROW0 8-15 + ROW1 8-15
                    pxLower = _mm_add_epi16(pxLower, _mm_unpackhi_epi8(pxRow[2], xmm_px0));    // lower accum - ROW0 8-15 + ROW1 8-15 + ROW2 8-15

                    // shuffle to get the correct order
                    __m128i pxShift[4];
                    pxShift[0] = _mm_shuffle_epi8(pxUpper, xmm_pxMask02To15); // upper accum 1-7
                    pxShift[0] = _mm_add_epi16(pxShift[0], _mm_shuffle_epi8(pxLower, xmm_pxMask00To01)); // upper accum 1-7 + lower accum 0
                    pxShift[1] = _mm_shuffle_epi8(pxUpper, xmm_pxMask04To15); // upper accum 2-7
                    pxShift[1] = _mm_add_epi16(pxShift[1], _mm_shuffle_epi8(pxLower, xmm_pxMask00To03)); // upper accum 2-7 + lower accum 0-2

                    pxShift[2] = _mm_shuffle_epi8(pxLower, xmm_pxMask02To15); // lower accum 1-7
                    pxShift[3] = _mm_shuffle_epi8(pxLower, xmm_pxMask04To15); // lower accum 2-7

                    __m128i pxResult[2];
                    pxResult[0] = _mm_add_epi16(_mm_add_epi16(pxUpper, pxShift[0]), pxShift[1]); // upper accum 0-7 + upper accum 1-7 + upper accum 2-7 + lower accum 0-2
                    pxResult[0] = _mm_mulhi_epi16(pxResult[0], pxConvolutionFactor);
                    pxResult[1] = _mm_add_epi16(_mm_add_epi16(pxLower, pxShift[2]), pxShift[3]); // lower accum 0-7 + lower accum 1-7 + lower accum 2-7
                    pxResult[1] = _mm_mulhi_epi16(pxResult[1], pxConvolutionFactor);
                    pxResult[0] = _mm_packus_epi16(pxResult[0], pxResult[1]);
                    _mm_storeu_si128((__m128i *)dstPtrTemp, pxResult[0]);

                    srcPtrTemp[0] += 12;
                    srcPtrTemp[1] += 12;
                    srcPtrTemp[2] += 12;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, i, vectorLoopCount + padLength, kernelSize, padLength, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth);
                    srcPtrTemp[0]++;
                    srcPtrTemp[1]++;
                    srcPtrTemp[2]++;
                    dstPtrTemp++;
                }
                srcPtrRow[0] += (!firstRow) ? srcDescPtr->strides.hStride : 0;
                srcPtrRow[1] += (!firstRow) ? srcDescPtr->strides.hStride : 0;
                srcPtrRow[2] += (!firstRow) ? srcDescPtr->strides.hStride : 0;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}