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

inline void rpp_store_gaussian_filter_3x3_host(Rpp8u *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
}

inline void rpp_store_gaussian_filter_3x3_host(Rpp8s *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_f32_to_i8_avx(dstPtrTemp, pDst);
}

inline void rpp_store_gaussian_filter_3x3_host(Rpp32f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_float(dstPtrTemp, pDst);
}

inline void rpp_store_gaussian_filter_3x3_host(Rpp16f *dstPtrTemp, __m256 *pDst)
{
    rpp_store16_float(dstPtrTemp, pDst);
}

inline Rpp32f gaussian(int iSquare, int j, Rpp32f mulFactor)
{
    Rpp32f expFactor = - (iSquare + (j * j)) * mulFactor;
    expFactor = std::exp(expFactor);
    return expFactor;
}

inline void create_gaussian_kernel_3x3_host(Rpp32f* filter,
                                            Rpp32f stdDev)
{
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

inline void create_gaussian_kernel_5x5_host(Rpp32f* filter,
                                            Rpp32f stdDev)
{
    Rpp32f mulFactor = 1 / (2 * stdDev * stdDev);
    int rowIdx = 0;

    // compute values for only top left quarter and replicate the values
    for (int i = -2; i <= 0; i++, rowIdx += 5)
    {
        int iSquare = i * i;
        filter[rowIdx + 4] = filter[rowIdx] = gaussian(iSquare, -2, mulFactor);
        filter[rowIdx + 3] = filter[rowIdx + 1] = gaussian(iSquare, -1, mulFactor);
        filter[rowIdx + 2] = gaussian(iSquare, 0, mulFactor);

        if ((20 - rowIdx) != rowIdx)
            std::memcpy(&filter[20 - rowIdx], &filter[rowIdx], 5 * sizeof(float));
    }

    Rpp32f kernelSum = 0.0f;
    for (int i = 0; i < 25; i++)
        kernelSum += filter[i];
    kernelSum = (1.0f / kernelSum);

    for (int i = 0; i < 25; i++)
        filter[i] *= kernelSum;
}

inline void create_gaussian_kernel_7x7_host(Rpp32f* filter,
                                            Rpp32f stdDev)
{
    Rpp32f mulFactor = 1 / (2 * stdDev * stdDev);
    int rowIdx = 0;

    // compute values for only top left quarter and replicate the values
    for (int i = -3; i <= 0; i++, rowIdx += 7)
    {
        int iSquare = i * i;
        filter[rowIdx + 6] = filter[rowIdx] = gaussian(iSquare, -3, mulFactor);
        filter[rowIdx + 5] = filter[rowIdx + 1] = gaussian(iSquare, -2, mulFactor);
        filter[rowIdx + 4] = filter[rowIdx + 2] = gaussian(iSquare, -1, mulFactor);
        filter[rowIdx + 3] = gaussian(iSquare, 0, mulFactor);

        if ((42 - rowIdx) != rowIdx)
            std::memcpy(&filter[42 - rowIdx], &filter[rowIdx], 7 * sizeof(float));
    }

    Rpp32f kernelSum = 0.0f;
    for (int i = 0; i < 49; i++)
        kernelSum += filter[i];
    kernelSum = (1.0f / kernelSum);

    for (int i = 0; i < 49; i++)
        filter[i] *= kernelSum;
    }

inline void create_gaussian_kernel_9x9_host(Rpp32f* filter,
                                            Rpp32f stdDev)
{
    Rpp32f mulFactor = 1 / (2 * stdDev * stdDev);
    int rowIdx = 0;

    // compute values for only top left quarter and replicate the values
    for (int i = -4; i <= 0; i++, rowIdx += 9)
    {
        int iSquare = i * i;
        filter[rowIdx + 8] = filter[rowIdx] = gaussian(iSquare, -4, mulFactor);
        filter[rowIdx + 7] = filter[rowIdx + 1] = gaussian(iSquare, -3, mulFactor);
        filter[rowIdx + 6] = filter[rowIdx + 2] = gaussian(iSquare, -2, mulFactor);
        filter[rowIdx + 5] = filter[rowIdx + 3] = gaussian(iSquare, -1, mulFactor);
        filter[rowIdx + 4] = gaussian(iSquare, 0, mulFactor);

        if ((72 - rowIdx) != rowIdx)
            std::memcpy(&filter[72 - rowIdx], &filter[rowIdx], 9 * sizeof(float));
    }

    Rpp32f kernelSum = 0.0f;
    for (int i = 0; i < 81; i++)
        kernelSum += filter[i];
    kernelSum = (1.0f / kernelSum);

    for (int i = 0; i < 81; i++)
        filter[i] *= kernelSum;
}

// Generic function to create a Gaussian kernel of any size
inline void create_gaussian_kernel_host(Rpp32f* filter, Rpp32f stdDev, int kernelSize)
{
    int kernelHalfSize  = kernelSize / 2;
    Rpp32f mulFactor = 1.0f / (2.0f * stdDev * stdDev);
    int rowIdx = 0;

    // Compute values for only the top left quarter and replicate the values
    for (int i = -kernelHalfSize; i <= 0; i++, rowIdx += kernelSize)
    {
        int iSquare = i * i;
        for (int j = -kernelHalfSize; j <= 0; j++)
        {
            int index = rowIdx + (j + kernelHalfSize);
            filter[index] = gaussian(iSquare, j * j, mulFactor);

            // Replicate the values to the other quadrants
            filter[rowIdx + (kernelSize - 1) - (j + kernelHalfSize)] = filter[index];
            filter[(kernelSize - 1 - rowIdx) * kernelSize + (j + kernelHalfSize)] = filter[index];
            filter[(kernelSize - 1 - rowIdx) * kernelSize + (kernelSize - 1) - (j + kernelHalfSize)] = filter[index];
        }
    }

    Rpp32f kernelSum = 0.0f;
    for (int i = 0; i < kernelSize * kernelSize; i++)
        kernelSum += filter[i];
    kernelSum = 1.0f / kernelSum;

    for (int i = 0; i < kernelSize * kernelSize; i++)
        filter[i] *= kernelSum;
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

    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        accum = round(accum);
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
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterTensor)
{
    for (int c = 0; c < 3; c++)
    {
        for (int k = 0; k < padLength; k++)
        {
            gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < kernelSize; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}

template<typename T>
RppStatus gaussian_filter_host_tensor(T *srcPtr,
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

    if ((kernelSize != 3) && (kernelSize != 5) && (kernelSize != 7) && (kernelSize != 9))
        return gaussian_filter_generic_host_tensor(srcPtr, srcDescPtr, dstPtr, dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc, roiType, layoutParams, handle);

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

        Rpp32f *filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * kernelSize * kernelSize;
        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            create_gaussian_kernel_3x3_host(filterTensor, stdDevTensor[batchCount]);
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
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
#if __AVX2__
                    __m256 pFilter[9];
                    for (int i = 0; i < 9; i++)
                        pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
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
                            __m256 pRow[6], pDst[2];
                            rpp_load_filter_3x3_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst[0] = avx_p0;
                            pDst[1] = avx_p0;
                            for (int k = 0; k < 3; k++)
                            {
                                __m256 pTemp[3];
                                Rpp32s filterIndex =  k * 3;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                            }

                            rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 14);
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
#if __AVX2__
                __m256 pFilter[9];
                for (int i = 0; i < 9; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
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
                        __m256 pRow[9], pDst[2];
                        rpp_load_filter_3x3_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);

                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0; k < 3; k++)
                        {
                            __m256 pTemp[3];
                            Rpp32s filterIndex = k * 3;
                            Rpp32s rowIndex = k * 3;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                        }

                        increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                        rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                        dstPtrTemp += 16;
                    }
    #endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                    since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
#if __AVX2__
                __m256 pFilter[9];
                for (int i = 0; i < 9; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
    #if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[9], pDst[2];
                        rpp_load_filter_3x3_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);

                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0; k < 3; k++)
                        {
                            __m256 pTemp[3];
                            Rpp32s filterIndex = k * 3;
                            Rpp32s rowIndex = k * 3;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                        }

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 4);
                    }
    #endif
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                __m256 pFilter[9];
                for (int i = 0; i < 9; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][3] = {
                                                {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                                {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
                                                {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
                                                };

                    T *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            dstPtrTemp++;
                        }
                    }
    #if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                    {
                        __m256 pResult[6];
                        for (int c = 0; c < 3; c++)
                        {
                            int channelStride = c * 2;
                            __m256 pRow[6];
                            rpp_load_filter_3x3_pln_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            pResult[channelStride] = avx_p0;
                            pResult[channelStride + 1] = avx_p0;
                            for (int k = 0; k < 3; k++)
                            {
                                __m256 pTemp[3];
                                Rpp32s filterIndex =  k * 3;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pResult[channelStride] = _mm256_add_ps(pResult[channelStride], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pResult[channelStride + 1] = _mm256_add_ps(pResult[channelStride + 1], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                            }
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 14);
                        }

                        // convert result from pln to pkd format and store in output buffer
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResult);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResult);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResult);
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResult);

                        dstPtrTemp += 42;
                    }
    #endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if (kernelSize == 5)
        {
            create_gaussian_kernel_5x5_host(filterTensor, stdDevTensor[batchCount]);
            T *srcPtrRow[5], *dstPtrRow;
            for (int i = 0; i < 5; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                    __m256 pFilter[25];
                    for (int i = 0; i < 25; i++)
                        pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 5; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;

                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp += padLength;
    #if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                        {
                            __m256 pRow[10], pDst[2];
                            rpp_load_filter_5x5_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst[0] = avx_p0;
                            pDst[1] = avx_p0;
                            for (int k = 0; k < 5; k++)
                            {
                                __m256 pTemp[5];
                                Rpp32s filterIndex =  k * 5;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], avx_p0, 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));
                            }

                            rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                            dstPtrTemp += 12;
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
                Rpp32u alignedLength = ((bufferLength - (2 * padLength * 3)) / 32) * 32;
#if __AVX2__
                __m256 pFilter[25];
                for (int i = 0; i < 25; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[5];
                    for (int k = 0; k < 5; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength * 3;
    #if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m256 pRow[20], pDst[2];
                        rpp_load_filter_5x5_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0; k < 5; k++)
                        {
                            __m256 pTemp[5];
                            Rpp32s filterIndex =  k * 5;
                            Rpp32s rowIndex = k * 4;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));
                        }

                        increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                        rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                        dstPtrTemp += 16;
                    }
    #endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                    since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                __m256 pFilter[25];
                for (int i = 0; i < 25; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][5];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 5; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    T *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            dstPtrTemp++;
                        }
                    }
    #if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256 pRow[10];
                            rpp_load_filter_5x5_pln_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            pResultPln[c] = avx_p0;
                            for (int k = 0; k < 5; k++)
                            {
                                __m256 pTemp[5];
                                Rpp32s filterIndex =  k * 5;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pResultPln[c] = _mm256_add_ps(pResultPln[c], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));
                            }
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        }

                        // convert result from pln to pkd format and store in output buffer
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResultPln);

                        dstPtrTemp += 24;
                    }
    #endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                    since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
#if __AVX2__
                __m256 pFilter[25];
                for (int i = 0; i < 25; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[5];
                    for (int k = 0; k < 5; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
    #if __AVX2__
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[20], pDst[2];
                        rpp_load_filter_5x5_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0; k < 5; k++)
                        {
                            __m256 pTemp[5];
                            Rpp32s filterIndex =  k * 5;
                            Rpp32s rowIndex = k * 4;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(pTemp[3], pTemp[4])));
                        }

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        increment_row_ptrs(dstPtrTempChannels, 3, 4);
                    }
    #endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
        else if (kernelSize == 7)
        {
            create_gaussian_kernel_7x7_host(filterTensor, stdDevTensor[batchCount]);
            T *srcPtrRow[7], *dstPtrRow;
            for (int i = 0; i < 7; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                __m256 pFilter[49];
                for (int i = 0; i < 49; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 7; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;

                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[7];
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[14], pDst;
                            rpp_load_filter_7x7_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst = avx_p0;
                            for (int k = 0; k < 7; k++)
                            {
                                __m256 pTemp[7];
                                Rpp32s filterIndex =  k * 7;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 6]);
                                pDst =  _mm256_add_ps(pDst, _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));
                            }

                            // convert result from pln to pkd format and store in output buffer
                            if constexpr (std::is_same<T, Rpp32f>::value)
                                _mm256_storeu_ps(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
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
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
#if __AVX2__
                __m256 pFilter[49];
                for (int i = 0; i < 49; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength * 3;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pRow[28], pDst;
                        rpp_load_filter_7x7_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst = avx_p0;
                        for (int k = 0; k < 7; k++)
                        {
                            __m256 pTemp[7];
                            Rpp32s filterIndex =  k * 7;
                            Rpp32s rowIndex = k * 4;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 5]);
                            pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 6]);
                            pDst =  _mm256_add_ps(pDst, _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));
                        }

                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        dstPtrTemp += 8;
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                __m256 pFilter[49];
                for (int i = 0; i < 49; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][7];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    T *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            dstPtrTemp++;
                        }
                    }
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256 pRow[14];
                            rpp_load_filter_7x7_pln_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            pResultPln[c] = avx_p0;
                            for (int k = 0; k < 7; k++)
                            {
                                __m256 pTemp[7];
                                Rpp32s filterIndex =  k * 7;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 6]);
                                pResultPln[c] = _mm256_add_ps(pResultPln[c], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));
                            }
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        }
                        // convert result from pln to pkd format and store in output buffer
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            rpp_store24_f32pln3_to_f16pkd3_avx(dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResultPln);

                        dstPtrTemp += 24;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
#if __AVX2__
                __m256 pFilter[49];
                for (int i = 0; i < 49; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[28], pDst[2];
                        rpp_load_filter_7x7_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0; k < 7; k++)
                        {
                            __m256 pTemp[7];
                            Rpp32s filterIndex =  k * 7;
                            Rpp32s rowIndex = k * 4;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 5]);
                            pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 6]);
                            pDst[0] =  _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 5]);
                            pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 3], avx_p0, 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 6]);
                            pDst[1] =  _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(pTemp[0], _mm256_add_ps(pTemp[1], pTemp[2])), _mm256_add_ps(_mm256_add_ps(pTemp[3], pTemp[4]), _mm256_add_ps(pTemp[5], pTemp[6]))));
                        }

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        increment_row_ptrs(dstPtrTempChannels, 3, 4);
                    }
#endif
                    vectorLoopCount += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
        else if (kernelSize == 9)
        {
            create_gaussian_kernel_9x9_host(filterTensor, stdDevTensor[batchCount]);
            T *srcPtrRow[9], *dstPtrRow;
            for (int i = 0; i < 9; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                __m256 pFilter[81];
                for (int i = 0; i < 81; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 9; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        T *srcPtrTemp[9];
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        T *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[18], pDst;
                            rpp_load_filter_9x9_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            pDst = avx_p0;
                            for (int k = 0; k < 9; k++)
                            {
                                __m256 pTemp[9];
                                Rpp32s filterIndex =  k * 9;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 6]);
                                pTemp[7] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 7]);
                                pTemp[8] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex + 8]);
                                pDst = _mm256_add_ps(pDst, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), _mm256_add_ps(pTemp[2], pTemp[3])), _mm256_add_ps(_mm256_add_ps(pTemp[4], pTemp[5]), _mm256_add_ps(pTemp[6], _mm256_add_ps(pTemp[7], pTemp[8])))));
                            }

                            if constexpr (std::is_same<T, Rpp32f>::value)
                                _mm256_storeu_ps(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
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
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
#if __AVX2__
                __m256 pFilter[81];
                for (int i = 0; i < 81; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength * 3;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pRow[36], pDst;
                        rpp_load_filter_9x9_pkd_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst = avx_p0;
                        for (int k = 0; k < 9; k++)
                        {
                            __m256 pTemp[9];
                            Rpp32s filterIndex =  k * 9;
                            Rpp32s rowIndex = k * 4;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 5]);
                            pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 6]);
                            pTemp[7] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 7]);
                            pTemp[8] = _mm256_mul_ps(pRow[rowIndex + 3], pFilter[filterIndex + 8]);
                            pDst = _mm256_add_ps(pDst, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), _mm256_add_ps(pTemp[2], pTemp[3])), _mm256_add_ps(_mm256_add_ps(pTemp[4], pTemp[5]), _mm256_add_ps(pTemp[6], _mm256_add_ps(pTemp[7], pTemp[8])))));
                        }

                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        dstPtrTemp += 8;
                    }
#endif
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            // gaussian filter with fused output-layout toggle (NCHW -> NHWC)
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
#if __AVX2__
                __m256 pFilter[81];
                for (int i = 0; i < 81; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[3][9];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    T *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            dstPtrTemp++;
                        }
                    }
#if __AVX2__
                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256 pRow[18];
                            rpp_load_filter_9x9_pln_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            pResultPln[c] = avx_p0;
                            for (int k = 0; k < 9; k++)
                            {
                                __m256 pTemp[9];
                                Rpp32s filterIndex =  k * 9;
                                Rpp32s rowIndex = k * 2;

                                pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 3]);
                                pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                                pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 5]);
                                pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 6]);
                                pTemp[7] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 7]);
                                pTemp[8] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex + 8]);
                                pResultPln[c] = _mm256_add_ps(pResultPln[c], _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), _mm256_add_ps(pTemp[2], pTemp[3])), _mm256_add_ps(_mm256_add_ps(pTemp[4], pTemp[5]), _mm256_add_ps(pTemp[6], _mm256_add_ps(pTemp[7], pTemp[8])))));
                            }
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                        }

                        if constexpr (std::is_same<T, Rpp32f>::value)
                           rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                           rpp_store24_f32pln3_to_f16pkd3_avx(dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pResultPln);
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pResultPln);

                        dstPtrTemp += 24;
                    }
#endif
                    vectorLoopCount += padLength;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude ((2 * padLength) * 3) number of columns from alignedLength calculation
                   since (padLength * 3) number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 40) * 40;
#if __AVX2__
                __m256 pFilter[81];
                for (int i = 0; i < 81; i++)
                    pFilter[i] = _mm256_set1_ps(filterTensor[i]);
#endif
                T *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[45], pDst[2];
                        rpp_load_gaussian_filter_9x9_pkd_pln_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        pDst[0] = avx_p0;
                        pDst[1] = avx_p0;
                        for (int k = 0; k < 9; k++)
                        {
                            __m256 pTemp[9];
                            Rpp32s filterIndex =  k * 9;
                            Rpp32s rowIndex = k * 4;

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 5]);
                            pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 6]);
                            pTemp[7] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 7]);
                            pTemp[8] = _mm256_mul_ps(pRow[rowIndex + 3], pFilter[filterIndex + 8]);
                            pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), _mm256_add_ps(pTemp[2], pTemp[3])), _mm256_add_ps(_mm256_add_ps(pTemp[4], pTemp[5]), _mm256_add_ps(pTemp[6], _mm256_add_ps(pTemp[7], pTemp[8])))));

                            pTemp[0] = _mm256_mul_ps(pRow[rowIndex + 1], pFilter[filterIndex]);
                            pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 7), avx_pxMaskRotate0To3), pFilter[filterIndex + 1]);
                            pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 1], pRow[rowIndex + 2], 63), avx_pxMaskRotate0To6), pFilter[filterIndex + 2]);
                            pTemp[3] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 3]);
                            pTemp[4] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 15), avx_pxMaskRotate0To4), pFilter[filterIndex + 4]);
                            pTemp[5] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 2], pRow[rowIndex + 3], 127), avx_pxMaskRotate0To7), pFilter[filterIndex + 5]);
                            pTemp[6] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 3], pRow[rowIndex + 4], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 6]);
                            pTemp[7] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex + 3], pRow[rowIndex + 4], 31), avx_pxMaskRotate0To5), pFilter[filterIndex + 7]);
                            pTemp[8] = _mm256_mul_ps(pRow[rowIndex + 4], pFilter[filterIndex + 8]);
                            pDst[1] = _mm256_add_ps(pDst[1], _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), _mm256_add_ps(pTemp[2], pTemp[3])), _mm256_add_ps(_mm256_add_ps(pTemp[4], pTemp[5]), _mm256_add_ps(pTemp[6], _mm256_add_ps(pTemp[7], pTemp[8])))));
                        }

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 12);
                        increment_row_ptrs(dstPtrTempChannels, 3, 4);
                    }
#endif
                    vectorLoopCount += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    // for the first padLength rows, we need not increment the src row pointers to next rows
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
    }
    return RPP_SUCCESS;
}

template<typename T>
RppStatus gaussian_filter_generic_host_tensor(T *srcPtr,
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

        Rpp32f *filterTensor = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * kernelSize * kernelSize;

        Rpp32u padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        T *srcPtrRow[kernelSize], *dstPtrRow;
        for (int k = 0; k < kernelSize; k++)
            srcPtrRow[k] = srcPtrChannel + k * srcDescPtr->strides.hStride;
        dstPtrRow = dstPtrChannel;
        create_gaussian_kernel_host(filterTensor, stdDevTensor[batchCount], kernelSize);
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (int c = 0; c < srcDescPtr->c; c++)
            {
                srcPtrRow[0] = srcPtrChannel;
                for (int k = 1; k < kernelSize; k++)
                    srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    T *srcPtrTemp[kernelSize];
                    for (int k = 0; k < kernelSize; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    T *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                    process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                    dstPtrTemp += padLength;
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
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
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[kernelSize];
                for (int k = 0; k < kernelSize; k++)
                    srcPtrTemp[k] = srcPtrRow[k];
                T *dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                dstPtrTemp += padLength * 3;
                vectorLoopCount += padLength * 3;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTemp++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[3][kernelSize];
                for (int c = 0; c < 3; c++)
                {
                    Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                    for (int k = 0; k < kernelSize; k++)
                        srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                }
                T *dstPtrTemp = dstPtrRow;

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);

                // process padLength number of columns in each row
                for (int k = 0; k < padLength; k++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        dstPtrTemp++;
                    }
                }
                vectorLoopCount += padLength;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        gaussian_filter_generic_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                        increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                        dstPtrTemp++;
                    }
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *dstPtrChannels[3];
            for (int c = 0; c < 3; c++)
                dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                int vectorLoopCount = 0;
                bool padLengthRows = (i < padLength) ? 1: 0;
                T *srcPtrTemp[kernelSize];
                for (int k = 0; k < kernelSize; k++)
                    srcPtrTemp[k] = srcPtrRow[k];
                T *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                Rpp32s rowKernelLoopLimit = kernelSize;
                get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor);
                vectorLoopCount += padLength * 3;

                // process remaining columns in each row
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    int channel = vectorLoopCount % 3;
                    gaussian_filter_generic_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 3);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    dstPtrTempChannels[channel]++;
                }
                // for the first padLength rows, we need not increment the src row pointers to next rows
                increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
            }
        }
    }
    return RPP_SUCCESS;
}
