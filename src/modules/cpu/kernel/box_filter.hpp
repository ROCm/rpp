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

const __m128i xmm_pxMask01To15 = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0x80);
const __m128i xmm_pxMask02To15 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0x80, 0x80);
const __m128i xmm_pxMaskReverse1 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
const __m128i xmm_pxMaskReverse2 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
const __m128i xmm_pxMaskReverse3 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5);
const __m128i xmm_pxMaskReverse4 = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
const __m128i xmm_pxMaskReverse5 = _mm_setr_epi8(10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
const __m128i xmm_pxMaskReverse6 = _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
const __m128i xmm_pxMaskReverse7 = _mm_setr_epi8(14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);

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

inline void box_filter_generic_u8_u8_host_tensor(Rpp8u **srcPtrTemp, Rpp8u *dstPtrTemp, Rpp32u columnIndex,
                                                 Rpp32u kernelSize, Rpp32u padLength, Rpp32u width, Rpp32s rowKernelLoopLimit,
                                                 Rpp32f kernelSizeInverseSquare)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit;

    // find the colKernelLoopLimit based on rowIndex, columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, kernelSize, padLength, width);

    for (int i = 0; i < rowKernelLoopLimit; i++)
    {
        for (int j = 0; j < columnKernelLoopLimit; j++)
            accum += static_cast<Rpp32f>(srcPtrTemp[i][j]);
    }
    accum *= kernelSizeInverseSquare;
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
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 24) * 24;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp++;
                    }

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3];
                        pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
                        pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
                        if (rowKernelLoopLimit == 3)
                            pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
                        else
                            pxRow[2] = avx_px0;

                        __m256i pxLower, pxUpper;
                        pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));

                        pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));

                        __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
                        pxLower1 =  _mm256_castsi256_si128(pxLower);
                        pxLower2 =  _mm256_castsi256_si128(pxUpper);
                        pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);
                        pxUpper2 =  _mm256_extracti128_si256(pxUpper, 1);

                        __m128i pxTemp[2];
                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskReverse1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskReverse2);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);

                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskReverse1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskReverse2);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);

                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 1), xmm_pxMaskReverse1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 3), xmm_pxMaskReverse2);
                        pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
                        pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);

                        pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
                        pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
                        pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);
                        pxLower1 = _mm_packus_epi16(pxLower1, pxLower2);
                        pxUpper1 = _mm_packus_epi16(pxUpper1, xmm_px0);
                        __m256i pxResult = _mm256_setr_m128i(pxLower1, pxUpper1);

                        _mm256_storeu_si256((__m256i *)dstPtrTemp, pxResult);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if (kernelSize == 9)
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 16) * 16;
                Rpp8u *srcPtrRow[9], *dstPtrRow;
                for (int i = 0; i < 9; i++)
                    srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[9];
                    for (int i = 0; i < 9; i++)
                        srcPtrTemp[i] = srcPtrRow[i];
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp++;
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m256i pxRow[9];
                        pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
                        pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
                        pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
                        pxRow[3] = _mm256_loadu_si256((__m256i *)srcPtrTemp[3]);
                        pxRow[4] = _mm256_loadu_si256((__m256i *)srcPtrTemp[4]);
                        for (int k = 5; k < rowKernelLoopLimit; k++)
                            pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
                        for (int k = rowKernelLoopLimit; k < 9; k++)
                            pxRow[k] = avx_px0;

                        __m256i pxLower, pxUpper;
                        pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[3], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[4], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[5], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[6], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[7], avx_px0));
                        pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[8], avx_px0));

                        pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[3], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[4], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[5], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[6], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[7], avx_px0));
                        pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[8], avx_px0));

                        __m128i pxLower1, pxLower2, pxUpper1;
                        pxLower1 =  _mm256_castsi256_si128(pxLower);
                        pxLower2 =  _mm256_castsi256_si128(pxUpper);
                        pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);

                        // get the final accumalated result for first 8 elements
                        __m128i pxTemp[7];
                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskReverse1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskReverse2);
                        pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskReverse3);
                        pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 15), xmm_pxMaskReverse4);
                        pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 31), xmm_pxMaskReverse5);
                        pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskReverse6);
                        pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 127), xmm_pxMaskReverse7);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[2]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[3]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[4]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[5]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[6]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxLower2);

                        // get the final accumalated result for next 8 elements
                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskReverse1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskReverse2);
                        pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskReverse3);
                        pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 15), xmm_pxMaskReverse4);
                        pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 31), xmm_pxMaskReverse5);
                        pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskReverse6);
                        pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 127), xmm_pxMaskReverse7);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[2]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[3]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[4]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[5]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxTemp[6]);
                        pxLower2 = _mm_add_epi16(pxLower2, pxUpper1);

                        pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
                        pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
                        pxLower1 = _mm_packus_epi16(pxLower1, pxLower2);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxLower1);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                        dstPtrTemp += 16;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
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