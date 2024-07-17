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
#include "rpp_cpu_filter.hpp"

inline void rpp_store16_float(Rpp32f *dstPtrTemp, __m256 *pDst)
{
    _mm256_storeu_ps(dstPtrTemp, pDst[0]);
    _mm256_storeu_ps(dstPtrTemp + 8, pDst[1]);
}

inline void rpp_store16_float(Rpp16f *dstPtrTemp, __m256 *pDst)
{
    __m128i pxDst[2];
    pxDst[0] = _mm256_cvtps_ph(pDst[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    pxDst[1] = _mm256_cvtps_ph(pDst[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtrTemp, pxDst[0]);
    _mm_storeu_si128((__m128i *)(dstPtrTemp + 8), pxDst[1]);
}

inline void rpp_store_gaussian_filter_3x3_pln_host(Rpp8u *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
}

inline void rpp_store_gaussian_filter_3x3_pln_host(Rpp8s *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
}

inline void rpp_store_gaussian_filter_3x3_pln_host(Rpp32f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_float(dstPtrTemp, pDst);
}

inline void rpp_store_gaussian_filter_3x3_pln_host(Rpp16f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_float(dstPtrTemp, pDst);
}

inline Rpp32f gaussian(int iSquare, int j, Rpp32f mulFactor)
{
    Rpp32f expFactor = - (iSquare + (j * j)) * mulFactor;
    expFactor = std::exp(expFactor);
    return expFactor;
}

inline void create_gaussian_kernel_3x3_host(Rpp32f* filterTensor,
                                       Rpp32f* stdDevTensor,
                                       int batchSize)
{
    for(int batchCount = 0; batchCount < 1; batchCount++)
    {
        Rpp32f* filter = &filterTensor[batchCount * 9];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32f mulFactor = 1 / (2 * stdDev * stdDev);
        int rowIdx = 0;

        // compute values for only top left quarter and replicate the values
        for (int i = -1; i <= 0; i++, rowIdx += 3)
        {
            int iSquare = i * i;
            filter[rowIdx + 2] = filter[rowIdx] = gaussian(iSquare, -1, mulFactor);
            filter[rowIdx + 1] = gaussian(iSquare, 0, mulFactor);

            if ((6 - rowIdx) != rowIdx)
                std::memcpy(&filter[6 - rowIdx], &filter[rowIdx], 3 * sizeof(float));
        }

        Rpp32f kernelSum = 0.0f;
        for (int i = 0; i < 9; i++)
            kernelSum += filter[i];
        kernelSum = (1.0f / kernelSum);

        for (int i = 0; i < 9; i++)
            filter[i] *= kernelSum;
    }
}

template<typename T>
inline void gaussian_filter_generic_tensor(T **srcPtrTemp, T *dstPtrTemp, Rpp32s columnIndex,
                                      Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                      Rpp32f *filterTensor, Rpp32u channels = 1)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    if constexpr (std::is_same<T, Rpp8s>::value)
    {
        for (int i = 0; i < rowKernelLoopLimit; i++)
            for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
                accum += static_cast<Rpp32f>(srcPtrTemp[i][k] + 128) * filterTensor[i * kernelSize + j];
    }
    else
    {
        for (int i = 0; i < rowKernelLoopLimit; i++)
            for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
                accum += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterTensor[i * kernelSize + j];

    }

    saturate_pixel(accum, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them separately
template<typename T>
inline void process_left_border_columns_pln_pln(T **srcPtrTemp, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterTensor)
{
    for (int k = 0; k < padLength; k++)
    {
        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
        dstPtrTemp++;
    }
}

template<typename T>
inline void process_left_border_columns_pkd_pkd(T **srcPtrTemp, T **srcPtrRow, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterTensor)
{
    for (int c = 0; c < 3; c++)
    {
        T *dstPtrTempChannel = dstPtrTemp + c;
        for (int k = 0; k < padLength; k++)
        {
            gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
            dstPtrTempChannel += 3;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }
    // reset source to initial position
    for (int k = 0; k < kernelSize; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}

template<typename T>
inline void process_left_border_columns_pkd_pln(T **srcPtrTemp, T **srcPtrRow, T **dstPtrTempChannels, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int c = 0; c < 3; c++)
    {
        for (int k = 0; k < padLength; k++)
        {
            gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}

inline void unpacklo_and_add_3x3_host(__m256i *pxRow, __m256i *pxDst, __m256i *pxFilterRow)
{
    __m256i pxTemp[3];
    pxTemp[0] = _mm256_mulhi_epi16(_mm256_unpacklo_epi8(pxRow[0], avx_px0), pxFilterRow[0]);
    pxTemp[1] = _mm256_mulhi_epi16(_mm256_unpacklo_epi8(pxRow[1], avx_px0), pxFilterRow[1]);
    pxTemp[2] = _mm256_mulhi_epi16(_mm256_unpacklo_epi8(pxRow[2], avx_px0), pxFilterRow[2]);
    pxDst[0] = _mm256_add_epi16(pxTemp[0], pxTemp[1]);
    pxDst[0] = _mm256_add_epi16(pxDst[0], pxTemp[2]);
}

inline void unpackhi_and_add_3x3_host(__m256i *pxRow, __m256i *pxDst, __m256i *pxFilterRow)
{
    __m256i pxTemp[3];
    pxTemp[0] = _mm256_mulhi_epi16(_mm256_unpackhi_epi8(pxRow[0], avx_px0), pxFilterRow[0]);;
    pxTemp[1] = _mm256_mulhi_epi16(_mm256_unpackhi_epi8(pxRow[1], avx_px0), pxFilterRow[1]);
    pxTemp[2] = _mm256_mulhi_epi16(_mm256_unpackhi_epi8(pxRow[2], avx_px0), pxFilterRow[2]);
    pxDst[0] = _mm256_add_epi16(pxTemp[0], pxTemp[1]);
    pxDst[0] = _mm256_add_epi16(pxDst[0], pxTemp[2]);
}

inline void add_rows_3x3(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(pRow[0], pRow[1]);
    pDst[0] = _mm256_add_ps(pDst[0], pRow[2]);
}

template<typename T>
RppStatus gaussian_filter_char_host_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *stdDevTensor,
                                      Rpp32u kernelSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();
    // static_assert((std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value), "T must be Rpp8u or Rpp8s");

    Rpp32f *filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
    create_gaussian_kernel_3x3_host(filterTensor, stdDevTensor, dstDescPtr->n);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            T *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude 2 * padLength number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                __m256 pFilterRow1[3];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7]);
                __m256 pFilterRow2[3];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6]);
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
                        T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp += padLength;

#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[3], pTemp[3], pDst[2];
                            rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                            add_rows_3x3(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                            add_rows_3x3(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;

                            gaussian_filter_blend_permute_add_mul_3x3_pln(&pTemp[0], &pDst[0]);
                            gaussian_filter_blend_permute_add_mul_3x3_pln(&pTemp[1], &pDst[1]);
                            // rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
                            rpp_store_gaussian_filter_3x3_pln_host(dstPtrTemp, pDst);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 6);
                            dstPtrTemp += 14;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                    since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                __m256 pFilterRow1[3];
                // std::cerr<<"\n aligned Length"<<alignedLength;
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7]);
                __m256 pFilterRow2[3];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6]);
                __m256 pFilterRow3[3];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8]);
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength * 3;
    #if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m256 pRow[3], pTemp[3], pDst[2];
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_3x3(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_3x3(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_3x3(pRow, &pTemp[2]);

                        gaussian_filter_blend_permute_add_mul_3x3_pkd(&pTemp[0], &pDst[0]);
                        gaussian_filter_blend_permute_add_mul_3x3_pkd(&pTemp[1], &pDst[1]);

                        rpp_store_gaussian_filter_3x3_pln_host(dstPtrTemp, pDst);
                        dstPtrTemp += 16;
                    }
    #endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        // std::cerr<<"\n in c condition";
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
    //         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    //         {
    //             /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
    //                 since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
    //             Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
    //             __m256 pFilterRow1[3];
    //             pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1]);
    //             pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4]);
    //             pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7]);
    //             __m256 pFilterRow2[3];
    //             pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0]);
    //             pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3]);
    //             pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6]);
    //             __m256 pFilterRow3[3];
    //             pFilterRow3[0] = _mm256_setr_ps(filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2]);
    //             pFilterRow3[1] = _mm256_setr_ps(filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5]);
    //             pFilterRow3[2] = _mm256_setr_ps(filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8]);
    //             T *dstPtrChannels[3];
    //             for (int i = 0; i < 3; i++)
    //                 dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
    //             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
    //             {
    //                 int vectorLoopCount = 0;
    //                 bool padLengthRows = (i < padLength) ? 1: 0;
    //                 T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
    //                 T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

    //                 Rpp32s rowKernelLoopLimit = kernelSize;
    //                 get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
    //                 process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
    // #if __AVX2__
    //                 // process remaining columns in each row
    //                 for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
    //                 {
    //                     __m256 pRow[3], pTemp[3], pDst[2];
    //                     rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
    //                     add_rows_3x3(pRow, &pTemp[0]);

    //                     increment_row_ptrs(srcPtrTemp, kernelSize, 8);
    //                     rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
    //                     add_rows_3x3(pRow, &pTemp[1]);

    //                     increment_row_ptrs(srcPtrTemp, kernelSize, 8);
    //                     rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
    //                     add_rows_3x3(pRow, &pTemp[2]);

    //                     blend_permute_add_mul_3x3_pkd(&pTemp[0], &pDst[0], pConvolutionFactor);
    //                     blend_permute_add_mul_3x3_pkd(&pTemp[1], &pDst[1], pConvolutionFactor);

    //                     __m128 pDstPln[3];
    //                     rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
    //                     rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

    //                     increment_row_ptrs(srcPtrTemp, kernelSize, -4);
    //                     increment_row_ptrs(dstPtrTempChannels, kernelSize, 4);
    //                 }
    // #endif
    //                 vectorLoopCount += padLength * 3;
    //                 for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
    //                 {
    //                     int channel = c % 3;
    //                     gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
    //                     increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    //                     dstPtrTempChannels[channel]++;
    //                 }
    //                 // for the first padLength rows, we need not increment the src row pointers to next rows
    //                 increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
    //                 increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
    //             }
    //         }
    //         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    //         {
    //             /* exclude (2 * padLength) number of columns from alignedLength calculation
    //                 since padLength number of columns from the beginning and end of each row will be computed using raw c code */
    //             Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
    //             __m256 pFilterRow1[3];
    //             pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1]);
    //             pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4]);
    //             pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7]);
    //             __m256 pFilterRow2[3];
    //             pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0]);
    //             pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3]);
    //             pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6]);
    //             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
    //             {
    //                 int vectorLoopCount = 0;
    //                 bool padLengthRows = (i < padLength) ? 1: 0;
    //                 T *srcPtrTemp[3][3] = {
    //                                             {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
    //                                             {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
    //                                             {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
    //                                             };

    //                 T *dstPtrTemp = dstPtrRow;
    //                 // get the number of rows needs to be loaded for the corresponding row
    //                 Rpp32s rowKernelLoopLimit = kernelSize;
    //                 get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

    //                 // process padLength number of columns in each row
    //                 // left border pixels in image which does not have required pixels in 3x3 box, process them separately
    //                 for (int k = 0; k < padLength; k++)
    //                 {
    //                     for (int c = 0; c < 3; c++)
    //                     {
    //                         gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
    //                         dstPtrTemp++;
    //                     }
    //                 }
    // #if __AVX2__
    //                 // process alignedLength number of columns in each row
    //                 for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
    //                 {
    //                     __m256 pResult[6];
    //                     for (int c = 0; c < 3; c++)
    //                     {
    //                         int channelStride = c * 2;
    //                         __m256 pRow[3], pTemp[3];
    //                         rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
    //                         add_rows_3x3(pRow, &pTemp[0]);

    //                         increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
    //                         rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
    //                         add_rows_3x3(pRow, &pTemp[1]);
    //                         pTemp[2] = avx_p0;

    //                         blend_permute_add_mul_3x3_pln(&pTemp[0], &pResult[channelStride]);
    //                         blend_permute_add_mul_3x3_pln(&pTemp[1], &pResult[channelStride + 1]);
    //                         increment_row_ptrs(srcPtrTemp[c], kernelSize, 6);
    //                     }

    //                     // convert result from pln to pkd format and store in output buffer
    //                     if constexpr (std::is_same<T, Rpp32f>::value)
    //                         rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResult);
    //                     else if constexpr (std::is_same<T, Rpp16f>::value)
    //                         rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResult);

    //                     dstPtrTemp += 42;
    //                 }
    // #endif
    //                 vectorLoopCount += padLength;
    //                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
    //                 {
    //                     for (int c = 0; c < 3; c++)
    //                     {
    //                         gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
    //                         increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
    //                         dstPtrTemp++;
    //                     }
    //                 }
    //                 // for the first padLength rows, we need not increment the src row pointers to next rows
    //                 increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
    //                 dstPtrRow += dstDescPtr->strides.hStride;
    //             }
    //         }
        }
    }
//     else if (kernelSize == 5)
//     {
//         T *srcPtrRow[5], *dstPtrRow;
//         for (int i = 0; i < 5; i++)
//             srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
//         dstPtrRow = dstPtrChannel;

//         // box filter without fused output-layout toggle (NCHW -> NCHW)
//         if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             /* exclude (2 * padLength) number of columns from alignedLength calculation
//                 since padLength number of columns from the beginning and end of each row will be computed using raw c code */
//             Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
//             for (int c = 0; c < srcDescPtr->c; c++)
//             {
//                 srcPtrRow[0] = srcPtrChannel;
//                 for (int k = 1; k < 5; k++)
//                     srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;

//                 dstPtrRow = dstPtrChannel;
//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     int vectorLoopCount = 0;
//                     bool padLengthRows = (i < padLength) ? 1: 0;
//                     T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
//                     T *dstPtrTemp = dstPtrRow;

//                     // get the number of rows needs to be loaded for the corresponding row
//                     Rpp32s rowKernelLoopLimit = kernelSize;
//                     get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
//                     process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
//                     dstPtrTemp += padLength;
// #if __AVX2__
//                     // process alignedLength number of columns in each row
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
//                     {
//                         __m256 pRow[5], pDst[2], pTemp[3];
//                         rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                         add_rows_5x5(pRow, &pTemp[0]);

//                         increment_row_ptrs(srcPtrTemp, kernelSize, 8);
//                         rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                         add_rows_5x5(pRow, &pTemp[1]);
//                         pTemp[2] = avx_p0;

//                         blend_permute_add_mul_5x5_pln(&pTemp[0], &pDst[0], pConvolutionFactor);
//                         blend_permute_add_mul_5x5_pln(&pTemp[1], &pDst[1], pConvolutionFactor);

//                         rpp_store16_float(dstPtrTemp, pDst);
//                         increment_row_ptrs(srcPtrTemp, kernelSize, 4);
//                         dstPtrTemp += 12;
//                     }
// #endif
//                     vectorLoopCount += padLength;
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
//                         increment_row_ptrs(srcPtrTemp, kernelSize, 1);
//                         dstPtrTemp++;
//                     }
//                     // for the first padLength rows, we need not increment the src row pointers to next rows
//                     increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }
//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }
//         else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
//                 since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
//             Rpp32u alignedLength = ((bufferLength - (2 * padLength * 3)) / 24) * 24;
//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 int vectorLoopCount = 0;
//                 bool padLengthRows = (i < padLength) ? 1: 0;
//                 T *srcPtrTemp[5];
//                 for (int k = 0; k < 5; k++)
//                     srcPtrTemp[k] = srcPtrRow[k];
//                 T *dstPtrTemp = dstPtrRow;

//                 Rpp32s rowKernelLoopLimit = kernelSize;
//                 get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
//                 process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
//                 dstPtrTemp += padLength * 3;
// #if __AVX2__
//                 // process remaining columns in each row
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
//                 {
//                     // add loaded values from 9 rows
//                     __m256 pRow[5], pDst[2], pTemp[4];
//                     rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                     add_rows_5x5(pRow, &pTemp[0]);

//                     increment_row_ptrs(srcPtrTemp, kernelSize, 8);
//                     rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                     add_rows_5x5(pRow, &pTemp[1]);

//                     increment_row_ptrs(srcPtrTemp, kernelSize, 8);
//                     rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                     add_rows_5x5(pRow, &pTemp[2]);
//                     pTemp[3] = avx_p0;

//                     blend_permute_add_mul_5x5_pkd(&pTemp[0], &pDst[0], pConvolutionFactor);
//                     blend_permute_add_mul_5x5_pkd(&pTemp[1], &pDst[1], pConvolutionFactor);

//                     rpp_store16_float(dstPtrTemp, pDst);
//                     increment_row_ptrs(srcPtrTemp, kernelSize, -4);
//                     dstPtrTemp += 12;
//                 }
// #endif
//                 vectorLoopCount += padLength * 3;
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
//                     increment_row_ptrs(srcPtrTemp, kernelSize, 1);
//                     dstPtrTemp++;
//                 }
//                 // for the first padLength rows, we need not increment the src row pointers to next rows
//                 increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             /* exclude (2 * padLength) number of columns from alignedLength calculation
//                 since padLength number of columns from the beginning and end of each row will be computed using raw c code */
//             Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 int vectorLoopCount = 0;
//                 bool padLengthRows = (i < padLength) ? 1: 0;
//                 T *srcPtrTemp[3][5];
//                 for (int c = 0; c < 3; c++)
//                 {
//                     Rpp32u channelStride = c * srcDescPtr->strides.cStride;
//                     for (int k = 0; k < 5; k++)
//                         srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
//                 }
//                 T *dstPtrTemp = dstPtrRow;

//                 // get the number of rows needs to be loaded for the corresponding row
//                 Rpp32s rowKernelLoopLimit = kernelSize;
//                 get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

//                 // process padLength number of columns in each row
//                 for (int k = 0; k < padLength; k++)
//                 {
//                     for (int c = 0; c < 3; c++)
//                     {
//                         gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
//                         dstPtrTemp++;
//                     }
//                 }
// #if __AVX2__
//                 // process alignedLength number of columns in each row
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
//                 {
//                     __m256 pResultPln[3];
//                     for (int c = 0; c < 3; c++)
//                     {
//                         __m256 pRow[5], pTemp[2];
//                         rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
//                         add_rows_5x5(pRow, &pTemp[0]);

//                         increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
//                         rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
//                         add_rows_5x5(pRow, &pTemp[1]);
//                         blend_permute_add_mul_5x5_pln(pTemp, &pResultPln[c], pConvolutionFactor);
//                     }

//                     // convert result from pln to pkd format and store in output buffer
//                     if constexpr (std::is_same<T, Rpp32f>::value)
//                         rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResultPln);
//                     else if constexpr (std::is_same<T, Rpp16f>::value)
//                         rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResultPln);

//                     dstPtrTemp += 24;
//                 }
// #endif
//                 vectorLoopCount += padLength;
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     for (int c = 0; c < srcDescPtr->c; c++)
//                     {
//                         gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
//                         increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
//                         dstPtrTemp++;
//                     }
//                 }
//                 // for the first padLength rows, we need not increment the src row pointers to next rows
//                 increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
//                 since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
//             Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
//             T *dstPtrChannels[3];
//             for (int i = 0; i < 3; i++)
//                 dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 int vectorLoopCount = 0;
//                 bool padLengthRows = (i < padLength) ? 1: 0;
//                 T *srcPtrTemp[5];
//                 for (int k = 0; k < 5; k++)
//                     srcPtrTemp[k] = srcPtrRow[k];
//                 T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

//                 Rpp32s rowKernelLoopLimit = kernelSize;
//                 get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
//                 process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
// #if __AVX2__
//                 // process remaining columns in each row
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
//                 {
//                     // add loaded values from 9 rows
//                     __m256 pRow[5], pDst[2], pTemp[4];
//                     rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                     add_rows_5x5(pRow, &pTemp[0]);

//                     increment_row_ptrs(srcPtrTemp, kernelSize, 8);
//                     rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                     add_rows_5x5(pRow, &pTemp[1]);

//                     increment_row_ptrs(srcPtrTemp, kernelSize, 8);
//                     rpp_load_gaussian_filter_float_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
//                     add_rows_5x5(pRow, &pTemp[2]);
//                     pTemp[3] = avx_p0;

//                     blend_permute_add_mul_5x5_pkd(&pTemp[0], &pDst[0], pConvolutionFactor);
//                     blend_permute_add_mul_5x5_pkd(&pTemp[1], &pDst[1], pConvolutionFactor);

//                     __m128 pDstPln[3];
//                     rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
//                     rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

//                     increment_row_ptrs(srcPtrTemp, kernelSize, -4);
//                     increment_row_ptrs(dstPtrTempChannels, 3, 4);
//                 }
// #endif
//                 vectorLoopCount += padLength * 3;
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     int channel = vectorLoopCount % 3;
//                     gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
//                     increment_row_ptrs(srcPtrTemp, kernelSize, 1);
//                     dstPtrTempChannels[channel]++;
//                 }
//                 // for the first padLength rows, we need not increment the src row pointers to next rows
//                 increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
//                 increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
//             }
//         }
    // }
    return RPP_SUCCESS;
}