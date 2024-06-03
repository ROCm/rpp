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

const __m128i xmm_pxMaskRotate0To1 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
const __m128i xmm_pxMaskRotate0To3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
const __m128i xmm_pxMaskRotate0To5 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5);
const __m128i xmm_pxMaskRotate0To7 = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
const __m128i xmm_pxMaskRotate0To9 = _mm_setr_epi8(10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
const __m128i xmm_pxMaskRotate0To11 = _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
const __m128i xmm_pxMaskRotate0To13 = _mm_setr_epi8(14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);

template<typename T>
inline void increment_row_ptrs(T **srcPtrTemp, Rpp32u kernelSize, Rpp32u increment)
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
                                                 Rpp32f kernelSizeInverseSquare, Rpp32u channels = 1)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit;

    // find the colKernelLoopLimit based on rowIndex, columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, kernelSize, padLength, width);

    for (int i = 0; i < rowKernelLoopLimit; i++)
    {
        for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            accum += static_cast<Rpp32f>(srcPtrTemp[i][k]);
    }
    accum *= kernelSizeInverseSquare;
    *dstPtrTemp = static_cast<Rpp8u>(RPPPIXELCHECK(accum));
}

inline void box_filter_generic_f32_f32_host_tensor(Rpp32f **srcPtrTemp, Rpp32f *dstPtrTemp, Rpp32u columnIndex,
                                                   Rpp32u kernelSize, Rpp32u padLength, Rpp32u width, Rpp32s rowKernelLoopLimit,
                                                   Rpp32f kernelSizeInverseSquare, Rpp32u channels = 1)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit;

    // find the colKernelLoopLimit based on rowIndex, columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, kernelSize, padLength, width);

    for (int i = 0; i < rowKernelLoopLimit; i++)
    {
        for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            accum += static_cast<Rpp32f>(srcPtrTemp[i][k]);
    }
    accum *= kernelSizeInverseSquare;
    *dstPtrTemp = RPPPIXELCHECKF32(accum);
}

inline void compute_box_filter_3x3_24_host_pln(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // pack lower half of each of 3 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));

    // pack higher half of each of 3 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 =  _mm256_castsi256_si128(pxLower);
    pxLower2 =  _mm256_castsi256_si128(pxUpper);
    pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 =  _mm256_extracti128_si256(pxUpper, 1);

    // perform blend and shuffle operations for the first 8 output values to get required order and add them
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskRotate0To3);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 3), xmm_pxMaskRotate0To3);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // saturate 16 bit values to 8 bit values and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

inline void compute_box_filter_3x3_24_host_pkd(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // pack lower half of each of 3 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));

    // pack higher half of each of 3 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 =  _mm256_castsi256_si128(pxLower);
    pxLower2 =  _mm256_castsi256_si128(pxUpper);
    pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 =  _mm256_extracti128_si256(pxUpper, 1);

    // perform blend and shuffle operations for the first 8 output values to get required order and add them
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskRotate0To11);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 63), xmm_pxMaskRotate0To11);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // saturate 16 bit values to 8 bit values and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

inline void rpp_load_box_filter_3x3_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    else
        pxRow[2] = avx_px0;
}

inline void rpp_load_box_filter_3x3_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[0] + 8);
    pRow[4] = _mm256_loadu_ps(srcPtrTemp[1] + 8);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
    {
        pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
        pRow[5] = _mm256_loadu_ps(srcPtrTemp[2] + 8);
    }
    else
    {
        pRow[2] = avx_px0;
        pRow[5] = avx_px0;
    }
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
        const __m128i pxConvolutionFactor = _mm_set1_epi16(convolutionFactor);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            Rpp8u *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        Rpp8u *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                        // process padLength number of columns in each row
                        // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[3];
                            rpp_load_box_filter_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_3x3_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));

                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength * 3) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength * 3; k++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, k / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3];
                        rpp_load_box_filter_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m128i pxDst[2];
                        compute_box_filter_3x3_24_host_pkd(pxRow, pxDst, pxConvolutionFactor);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));

                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength * 3) / 24) * 24;
                Rpp8u *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp8u *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength * 3; k++)
                    {
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTempChannels[k], k / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    }
                    increment_row_ptrs(dstPtrTempChannels, kernelSize, 1);

                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3];
                        rpp_load_box_filter_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m128i pxDst[2];
                        compute_box_filter_3x3_24_host_pkd(pxRow, pxDst, pxConvolutionFactor);

                        // convert from PKD3 to PLN3 and store channelwise
                        __m128i pxR, pxG, pxB;
                        rpp_convert24_u8pkd3_to_u8pln3(pxDst[0], pxDst[1], pxR, pxG, pxB);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxR);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxG);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxB);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 8);
                    }
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        box_filter_generic_u8_u8_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3][3] = {
                                                {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                                {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
                                                {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
                                              };

                    Rpp8u *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_u8_u8_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[3];
                            rpp_load_box_filter_3x3_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_3x3_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_u8pln3_to_u8pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_u8_u8_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if (kernelSize == 9)
        {
            Rpp8u *srcPtrRow[9], *dstPtrRow;
            for (int i = 0; i < 9; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 16) * 16;
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
                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
                        pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
                        pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 15), xmm_pxMaskRotate0To7);
                        pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 31), xmm_pxMaskRotate0To9);
                        pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
                        pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 127), xmm_pxMaskRotate0To13);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[2]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[3]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[4]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[5]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxTemp[6]);
                        pxLower1 = _mm_add_epi16(pxLower1, pxLower2);

                        // get the final accumalated result for next 8 elements
                        pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
                        pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskRotate0To3);
                        pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
                        pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 15), xmm_pxMaskRotate0To7);
                        pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 31), xmm_pxMaskRotate0To9);
                        pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskRotate0To11);
                        pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 127), xmm_pxMaskRotate0To13);
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

                    // process remaining columns in each row
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

RppStatus box_filter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
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

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f kernelSizeInverseSquare = 1.0 / (kernelSize * kernelSize);
        const __m128 pConvolutionFactor = _mm_set1_ps(kernelSizeInverseSquare);

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            Rpp32f *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength) / 14) * 14;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp32f *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        Rpp32f *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                        // process padLength number of columns in each row
                        // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_f32_f32_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[6];
                            // irrespective of row location, we need to load 2 rows for 3x3 kernel
                            pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
                            pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
                            pRow[3] = _mm256_loadu_ps(srcPtrTemp[0] + 8);
                            pRow[4] = _mm256_loadu_ps(srcPtrTemp[1] + 8);

                            // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
                            if (rowKernelLoopLimit == 3)
                            {
                                pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
                                pRow[5] = _mm256_loadu_ps(srcPtrTemp[2] + 8);
                            }
                            else
                            {
                                pRow[2] = avx_p0;
                                pRow[5] = avx_p0;
                            }

                            // add loaded values from 3 rows
                            __m256 pLower, pUpper;
                            pLower = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
                            pUpper = _mm256_add_ps(_mm256_add_ps(pRow[3], pRow[4]), pRow[5]);

                            // get 4 SSE registers from above 2 AVX registers to arrange as per required order
                            __m128i pLower1, pLower2, pUpper1, pUpper2;
                            pLower1 =  _mm256_castps256_ps128(pLower);
                            pUpper1 =  _mm256_extractf128_ps(pLower, 1);
                            pLower2 =  _mm256_castps256_ps128(pUpper);
                            pUpper2 =  _mm256_extractf128_ps(pUpper, 1);

                            // perform blend and shuffle operations for the first 4 output values to get required order and add them
                            __m128 pTemp[2];
                            pTemp[0] = _mm_blend_ps(pLower1, pUpper1, 1);
                            pTemp[0] = _mm_shuffle_ps(pTemp[0], pTemp[0], 57);
                            pTemp[1] = _mm_blend_ps(pLower1, pUpper1, 3);
                            pTemp[1] = _mm_shuffle_ps(pTemp[1], pTemp[1], 78);
                            pLower1 = _mm_add_ps(pLower1, pTemp[0]);
                            pLower1 = _mm_add_ps(pLower1, pTemp[1]);

                            // perform blend and shuffle operations for the next 4 output values to get required order and add them
                            pTemp[0] = _mm_blend_ps(pUpper1, pLower2, 1);
                            pTemp[0] = _mm_shuffle_ps(pTemp[0], pTemp[0], 57);
                            pTemp[1] = _mm_blend_ps(pUpper1, pLower2, 3);
                            pTemp[1] = _mm_shuffle_ps(pTemp[1], pTemp[1], 78);
                            pUpper1 = _mm_add_ps(pUpper1, pTemp[0]);
                            pUpper1 = _mm_add_ps(pUpper1, pTemp[1]);

                            // perform blend and shuffle operations for the next 4 output values to get required order and add them
                            pTemp[0] = _mm_blend_ps(pLower2, pUpper2, 1);
                            pTemp[0] = _mm_shuffle_ps(pTemp[0], pTemp[0], 57);
                            pTemp[1] = _mm_blend_ps(pLower2, pUpper2, 3);
                            pTemp[1] = _mm_shuffle_ps(pTemp[1], pTemp[1], 78);
                            pLower2 = _mm_add_ps(pLower2, pTemp[0]);
                            pLower2 = _mm_add_ps(pLower2, pTemp[1]);

                            // perform blend and shuffle operations for the next 4 output values to get required order and add them
                            pTemp[0] = _mm_shuffle_ps(pUpper2, pUpper2, 57);
                            pTemp[1] = _mm_shuffle_ps(pUpper2, pUpper2, 78);
                            pUpper2 = _mm_add_ps(pUpper2, pTemp[0]);
                            pUpper2 = _mm_add_ps(pUpper2, pTemp[1]);

                            // multiply with convolution factor
                            pLower1 = _mm_mul_ps(pLower1, pConvolutionFactor);
                            pLower2 = _mm_mul_ps(pLower2, pConvolutionFactor);
                            pUpper1 = _mm_mul_ps(pUpper1, pConvolutionFactor);
                            pUpper2 = _mm_mul_ps(pUpper2, pConvolutionFactor);

                            _mm256_storeu_ps(dstPtrTemp,  _mm256_setr_m128i(pLower1, pUpper1));
                            _mm256_storeu_ps(dstPtrTemp + 8,  _mm256_setr_m128i(pLower2, pUpper2));
                            increment_row_ptrs(srcPtrTemp, kernelSize, 14);
                            dstPtrTemp += 14;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_f32_f32_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - 2 * padLength * 3) / 9) * 9;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength * 3; k++)
                    {
                        box_filter_generic_f32_f32_host_tensor(srcPtrTemp, dstPtrTemp, k / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 9)
                    {
                        __m256 pRow[6];
                        // irrespective of row location, we need to load 2 rows for 3x3 kernel
                        pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
                        pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
                        pRow[3] = _mm256_loadu_ps(srcPtrTemp[0] + 8);
                        pRow[4] = _mm256_loadu_ps(srcPtrTemp[1] + 8);

                        // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
                        if (rowKernelLoopLimit == 3)
                        {
                            pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
                            pRow[5] = _mm256_loadu_ps(srcPtrTemp[2] + 8);
                        }
                        else
                        {
                            pRow[2] = avx_p0;
                            pRow[5] = avx_p0;
                        }

                        // add loaded values from 3 rows
                        __m256 pLower, pUpper;
                        pLower = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
                        pUpper = _mm256_add_ps(_mm256_add_ps(pRow[3], pRow[4]), pRow[5]);

                        // get 4 SSE registers from above 2 AVX registers to arrange as per required order
                        __m128i pLower1, pLower2, pUpper1, pUpper2;
                        pLower1 =  _mm256_castps256_ps128(pLower);
                        pUpper1 =  _mm256_extractf128_ps(pLower, 1);
                        pLower2 =  _mm256_castps256_ps128(pUpper);
                        pUpper2 =  _mm256_extractf128_ps(pUpper, 1);

                        // perform blend and shuffle operations for the first 4 output values to get required order and add them
                        __m128 pTemp[2];
                        pTemp[0] = _mm_blend_ps(pLower1, pUpper1, 7);
                        pTemp[0] = _mm_shuffle_ps(pTemp[0], pTemp[0], 147);
                        pTemp[1] = _mm_blend_ps(pUpper1, pLower2, 3);
                        pTemp[1] = _mm_shuffle_ps(pTemp[1], pTemp[1], 78);
                        pLower1 = _mm_add_ps(pLower1, pTemp[0]);
                        pLower1 = _mm_add_ps(pLower1, pTemp[1]);

                        // perform blend and shuffle operations for the next 4 output values to get required order and add them
                        pTemp[0] = _mm_blend_ps(pUpper1, pLower2, 7);
                        pTemp[0] = _mm_shuffle_ps(pTemp[0], pTemp[0], 147);
                        pTemp[1] = _mm_blend_ps(pLower2, pUpper2, 3);
                        pTemp[1] = _mm_shuffle_ps(pTemp[1], pTemp[1], 78);
                        pUpper1 = _mm_add_ps(pUpper1, pTemp[0]);
                        pUpper1 = _mm_add_ps(pUpper1, pTemp[1]);

                        // perform blend and shuffle operations for the next 4 output values to get required order and add them
                        pTemp[0] = _mm_blend_ps(pLower2, pUpper2, 7);
                        pTemp[0] = _mm_shuffle_ps(pTemp[0], pTemp[0], 147);
                        pTemp[1] = _mm_blend_ps(pUpper2, xmm_p0, 3);
                        pTemp[1] = _mm_shuffle_ps(pTemp[1], pTemp[1], 78);
                        pLower2 = _mm_add_ps(pLower2, pTemp[0]);
                        pLower2 = _mm_add_ps(pLower2, pTemp[1]);

                        // multiply with convolution factor
                        pLower1 = _mm_mul_ps(pLower1, pConvolutionFactor);
                        pLower2 = _mm_mul_ps(pLower2, pConvolutionFactor);
                        pUpper1 = _mm_mul_ps(pUpper1, pConvolutionFactor);

                        _mm256_storeu_ps(dstPtrTemp,  _mm256_setr_m128i(pLower1, pUpper1));
                        _mm256_storeu_ps(dstPtrTemp + 8,  _mm256_setr_m128i(pLower2, xmm_p0));
                        increment_row_ptrs(srcPtrTemp, kernelSize, 9);
                        dstPtrTemp += 9;
                    }
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_f32_f32_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
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