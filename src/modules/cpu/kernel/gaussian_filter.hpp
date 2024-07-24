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

inline void rpp_convert12_f32pkd3_to_f32pln3(__m256 *pSrc, __m128 *pDst)
{
    __m128 pSrcPkd[3], pTemp;
    pSrcPkd[0] = _mm256_castps256_ps128(pSrc[0]);
    pSrcPkd[1] = _mm256_extractf128_ps(pSrc[0], 1);
    pSrcPkd[2] = _mm256_castps256_ps128(pSrc[1]);

    pTemp = _mm_blend_ps(pSrcPkd[0], pSrcPkd[1], 4);
    pTemp = _mm_blend_ps(pTemp, pSrcPkd[2], 2);
    pDst[0] = _mm_shuffle_ps(pTemp, pTemp, 108);

    pTemp = _mm_blend_ps(pSrcPkd[0], pSrcPkd[1], 9);
    pTemp = _mm_blend_ps(pTemp, pSrcPkd[2], 4);
    pDst[1] = _mm_shuffle_ps(pTemp, pTemp, 177);

    pTemp = _mm_blend_ps(pSrcPkd[0], pSrcPkd[1], 2);
    pTemp = _mm_blend_ps(pTemp, pSrcPkd[2], 9);
    pDst[2] = _mm_shuffle_ps(pTemp, pTemp, 198);
}

inline void rpp_store12_float_pkd_pln(Rpp32f **dstPtrTempChannels, __m128 *pDst)
{
    _mm_storeu_ps(dstPtrTempChannels[0], pDst[0]);
    _mm_storeu_ps(dstPtrTempChannels[1], pDst[1]);
    _mm_storeu_ps(dstPtrTempChannels[2], pDst[2]);
}

inline void rpp_store12_float_pkd_pln(Rpp16f **dstPtrTempChannels, __m128 *pDst)
{
    __m128i pxDst[3];
    pxDst[0] = _mm_cvtps_ph(pDst[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    pxDst[1] = _mm_cvtps_ph(pDst[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    pxDst[2] = _mm_cvtps_ph(pDst[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtrTempChannels[0]), pxDst[0]);
    _mm_storeu_si128((__m128i *)(dstPtrTempChannels[1]), pxDst[1]);
    _mm_storeu_si128((__m128i *)(dstPtrTempChannels[2]), pxDst[2]);
}

inline void rpp_store12_float_pkd_pln(Rpp8u **dstPtrTempChannels, __m128 *pDst)
{
    __m128i px[4];
    for(int i = 0; i < 3; i++)
    {
        px[0] = _mm_cvtps_epi32(pDst[i]);    /* pixels 0-3 */
        px[1] = _mm_cvtps_epi32(xmm_p0);    /* pixels 4-7 */
        px[2] = _mm_cvtps_epi32(xmm_p0);    /* pixels 8-11 */
        px[3] = _mm_cvtps_epi32(xmm_p0);    /* pixels 12-15 */
        px[0] = _mm_packus_epi32(px[0], px[1]);    /* pixels 0-7 */
        px[1] = _mm_packus_epi32(px[2], px[3]);    /* pixels 8-15 */
        px[0] = _mm_packus_epi16(px[0], px[1]);    /* pixels 0-15 */
        _mm_storeu_si32((__m128i *)dstPtrTempChannels[i], px[0]);    /* store pixels 0-15 */
    }
}

inline void rpp_store12_float_pkd_pln(Rpp8s **dstPtrTempChannels, __m128 *pDst)
{
    __m128i px[4];
    for(int i = 0; i < 3; i++)
    {
        px[0] = _mm_cvtps_epi32(pDst[i]);    /* pixels 0-3 */
        px[1] = _mm_cvtps_epi32(xmm_p0);    /* pixels 4-7 */
        px[2] = _mm_cvtps_epi32(xmm_p0);    /* pixels 8-11 */
        px[3] = _mm_cvtps_epi32(xmm_p0);    /* pixels 12-15 */
        px[0] = _mm_packus_epi32(px[0], px[1]);    /* pixels 0-7 */
        px[1] = _mm_packus_epi32(px[2], px[3]);    /* pixels 8-15 */
        px[0] = _mm_packus_epi16(px[0], px[1]);    /* pixels 0-15 */
        px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
        _mm_storeu_si32((__m128i *)dstPtrTempChannels[i], px[0]);    /* store pixels 0-15 */
    }
}

inline void rpp_store8_f32_to_u8_avx(Rpp8u *dstPtrTemp, __m256 pDst)
{
    __m256i px1 = _mm256_cvtps_epi32(pDst);
    // Pack int32 values to uint16
    __m128i px2 = _mm_packus_epi32(_mm256_castsi256_si128(px1), _mm256_extracti128_si256(px1, 1));
    // Pack uint16 values to uint8
    __m128i px3 = _mm_packus_epi16(px2, _mm_setzero_si128());
    // Store the result to dst
    _mm_storeu_si64((__m128i*)dstPtrTemp, px3);
}

inline void rpp_store8_f32_to_i8_avx(Rpp8s *dstPtrTemp, __m256 pDst)
{
    __m256i px1 = _mm256_cvtps_epi32(pDst);
    __m128i px2 = _mm_packus_epi32(_mm256_castsi256_si128(px1), _mm256_extracti128_si256(px1, 1));
    __m128i px3 = _mm_packus_epi16(px2, _mm_setzero_si128());
    px3 = _mm_sub_epi8(px3, xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    // Store the result to dst
    _mm_storeu_si64((__m128i*)dstPtrTemp, px3);
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
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
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

inline void create_gaussian_kernel_5x5_host(Rpp32f* filterTensor,
                                       Rpp32f* stdDevTensor,
                                       int batchSize)
{
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f* filter = &filterTensor[batchCount * 25];
        Rpp32f stdDev = stdDevTensor[batchCount];
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
}

inline void create_gaussian_kernel_7x7_host(Rpp32f* filterTensor,
                                       Rpp32f* stdDevTensor,
                                       int batchSize)
{
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f* filter = &filterTensor[batchCount * 49];
        Rpp32f stdDev = stdDevTensor[batchCount];
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
}

inline void create_gaussian_kernel_9x9_host(Rpp32f* filterTensor,
                                            Rpp32f* stdDevTensor,
                                            int batchSize)
{       
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f* filter = &filterTensor[batchCount * 81];
        Rpp32f stdDev = stdDevTensor[batchCount];
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

// -------------------- 3x3 kernel size - F32/F16 bitdepth compute functions --------------------

inline void add_rows_3x3(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(pRow[0], pRow[1]);
    pDst[0] = _mm256_add_ps(pDst[0], pRow[2]);
}

// -------------------- 5x5 kernel size - F32/F16 bitdepth compute functions --------------------

inline void add_rows_5x5(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(pRow[3], pRow[4]));
}

// -------------------- 7x7 kernel size - F32/F16 bitdepth compute functions --------------------

inline void add_rows_7x7(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pRow[3], pRow[4]), pRow[5]));
    pDst[0] = _mm256_add_ps(pDst[0], pRow[6]);
}

// -------------------- 9x9 kernel size - F32/F16 bitdepth compute functions --------------------

inline void add_rows_9x9(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pRow[3], pRow[4]), pRow[5]));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pRow[6], pRow[7]), pRow[8]));
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
    if(kernelSize == 3)
        create_gaussian_kernel_3x3_host(filterTensor, stdDevTensor, dstDescPtr->n);
    else if(kernelSize == 5)
        create_gaussian_kernel_5x5_host(filterTensor, stdDevTensor, dstDescPtr->n);
    else if(kernelSize == 7)
        create_gaussian_kernel_7x7_host(filterTensor, stdDevTensor, dstDescPtr->n);
    else if(kernelSize == 9)
        create_gaussian_kernel_9x9_host(filterTensor, stdDevTensor, dstDescPtr->n);

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
                // for(int i = 0; i < 9 ; i++)
                // {
                //     printf("\n filter value before setr %0.10f", filterTensor[i]);
                // }
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
                                for(int i = 0; i < 8; i++)
                                {
                                    printf("\n value %0.10f ", (float)srcPtrTemp[0][i] * filterTensor[i]);
                                }
                            rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                            add_rows_3x3(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                            add_rows_3x3(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;

                            gaussian_filter_blend_permute_add_mul_3x3_pln(&pTemp[0], &pDst[0]);
                            gaussian_filter_blend_permute_add_mul_3x3_pln(&pTemp[1], &pDst[1]);
                            // rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
                            std::cout<<"\n before store ";
                            rpp_mm256_print_ps(pDst[0]);
                            rpp_mm256_print_ps(pDst[1]);
                            rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                            for(int i = 0; i < 16; i++)
                                {
                                    printf("\n after store %d ", (int)*(dstPtrTemp + i));
                                }

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
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5], filterTensor[5]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7], filterTensor[8], filterTensor[8]);
                __m256 pFilterRow2[3];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7], filterTensor[8]);
                __m256 pFilterRow3[3];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[2], filterTensor[2], filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[8], filterTensor[8], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7]);
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
                __m256 pFilterRow1[3];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5], filterTensor[5]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7], filterTensor[8], filterTensor[8]);
                __m256 pFilterRow2[3];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7], filterTensor[8]);
                __m256 pFilterRow3[3];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[2], filterTensor[2], filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[8], filterTensor[8], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7]);
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
                        __m256 pRow[3], pTemp[3], pDst[2];
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_3x3(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_3x3(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_3x3(pRow, &pTemp[2]);

                        gaussian_filter_blend_permute_add_mul_3x3_pkd(&pTemp[0], &pDst[0]);
                        gaussian_filter_blend_permute_add_mul_3x3_pkd(&pTemp[1], &pDst[1]);

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, -4);
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
                __m256 pFilterRow1[3];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7]);
                __m256 pFilterRow2[3];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[3]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[6]);
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
                            __m256 pRow[3], pTemp[3];
                            rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow1);
                            add_rows_3x3(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow2);
                            add_rows_3x3(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;

                            gaussian_filter_blend_permute_add_mul_3x3_pln(&pTemp[0], &pResult[channelStride]);
                            gaussian_filter_blend_permute_add_mul_3x3_pln(&pTemp[1], &pResult[channelStride + 1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 6);
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
                __m256 pFilterRow1[5];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[0], filterTensor[1], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[5], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[9], filterTensor[5], filterTensor[6], filterTensor[7]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[10], filterTensor[11], filterTensor[12]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[15], filterTensor[16], filterTensor[27]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[20], filterTensor[21], filterTensor[22]);
                __m256 pFilterRow2[5];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[8], filterTensor[9], filterTensor[5], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[9], filterTensor[5]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[13], filterTensor[14], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[10]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[18], filterTensor[19], filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[15]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[23], filterTensor[24], filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[20]);

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
                            __m256 pRow[5], pDst[2], pTemp[3];
                            rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                            add_rows_5x5(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                            add_rows_5x5(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;

                            gaussian_filter_blend_permute_add_5x5_pln(&pTemp[0], &pDst[0]);
                            gaussian_filter_blend_permute_add_5x5_pln(&pTemp[1], &pDst[1]);

                            rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 4);
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
                __m256 pFilterRow1[5];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[10], filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12], filterTensor[12]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16], filterTensor[17], filterTensor[17]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[20], filterTensor[20], filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22]);
                __m256 pFilterRow2[5];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[7], filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[5]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[12], filterTensor[13], filterTensor[13], filterTensor[13], filterTensor[14], filterTensor[14], filterTensor[14], filterTensor[10]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[17], filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19], filterTensor[19], filterTensor[19], filterTensor[15]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[22], filterTensor[23], filterTensor[23], filterTensor[23], filterTensor[24], filterTensor[24], filterTensor[24], filterTensor[20]);
                __m256 pFilterRow3[5];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2], filterTensor[2]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12], filterTensor[12], filterTensor[12]);
                pFilterRow3[3] = _mm256_setr_ps(filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16], filterTensor[17], filterTensor[17], filterTensor[17]);
                pFilterRow3[4] = _mm256_setr_ps(filterTensor[20], filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22], filterTensor[22]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength * 3)) / 24) * 24;
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
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        // add loaded values from 9 rows
                        __m256 pRow[5], pDst[2], pTemp[4];
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_5x5(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_5x5(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_5x5(pRow, &pTemp[2]);
                        pTemp[3] = avx_p0;

                        gaussian_filter_blend_permute_add_5x5_pkd(&pTemp[0], &pDst[0]);
                        gaussian_filter_blend_permute_add_5x5_pkd(&pTemp[1], &pDst[1]);

                        rpp_store_gaussian_filter_3x3_host(dstPtrTemp, pDst);
                        increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                        dstPtrTemp += 12;
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
                __m256 pFilterRow1[5];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[0], filterTensor[1], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[5], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[9], filterTensor[5], filterTensor[6], filterTensor[7]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[10], filterTensor[11], filterTensor[12]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[15], filterTensor[16], filterTensor[27]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[20], filterTensor[21], filterTensor[22]);
                __m256 pFilterRow2[5];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[3], filterTensor[4], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[8], filterTensor[9], filterTensor[5], filterTensor[6], filterTensor[7], filterTensor[8], filterTensor[9], filterTensor[5]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[13], filterTensor[14], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[10]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[18], filterTensor[19], filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[15]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[23], filterTensor[24], filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[20]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
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
                            __m256 pRow[5], pTemp[2];
                            rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow1);
                            add_rows_5x5(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow2);
                            add_rows_5x5(pRow, &pTemp[1]);

                            gaussian_filter_blend_permute_add_5x5_pln(pTemp, &pResultPln[c]);
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
                 __m256 pFilterRow1[5];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[10], filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12], filterTensor[12]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16], filterTensor[17], filterTensor[17]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[20], filterTensor[20], filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22]);
                __m256 pFilterRow2[5];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[0]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[7], filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[5]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[12], filterTensor[13], filterTensor[13], filterTensor[13], filterTensor[14], filterTensor[14], filterTensor[14], filterTensor[10]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[17], filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19], filterTensor[19], filterTensor[19], filterTensor[15]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[22], filterTensor[23], filterTensor[23], filterTensor[23], filterTensor[24], filterTensor[24], filterTensor[24], filterTensor[20]);
                __m256 pFilterRow3[5];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2], filterTensor[2]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12], filterTensor[12], filterTensor[12]);
                pFilterRow3[3] = _mm256_setr_ps(filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16], filterTensor[17], filterTensor[17], filterTensor[17]);
                pFilterRow3[4] = _mm256_setr_ps(filterTensor[20], filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22], filterTensor[22]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
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
                        // add loaded values from 9 rows
                        __m256 pRow[5], pDst[2], pTemp[4];
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_5x5(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_5x5(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_5x5(pRow, &pTemp[2]);
                        pTemp[3] = avx_p0;

                        gaussian_filter_blend_permute_add_5x5_pkd(&pTemp[0], &pDst[0]);
                        gaussian_filter_blend_permute_add_5x5_pkd(&pTemp[1], &pDst[1]);

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, -4);
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
            T *srcPtrRow[7], *dstPtrRow;
            for (int i = 0; i < 7; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // gaussian filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                __m256 pFilterRow1[7];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6], filterTensor[0]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[7], filterTensor[8], filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[7]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[14], filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[14]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[25], filterTensor[26], filterTensor[27], filterTensor[21]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[28], filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33], filterTensor[34], filterTensor[28]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[35], filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[35]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[42], filterTensor[43], filterTensor[44], filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[42]);

                __m256 pFilterRow2[7];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6], filterTensor[0], filterTensor[1]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[8], filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[7], filterTensor[8]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[14], filterTensor[15]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[25], filterTensor[26], filterTensor[27], filterTensor[21], filterTensor[22]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33], filterTensor[34], filterTensor[28], filterTensor[29]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[35], filterTensor[36]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[43], filterTensor[44], filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[42], filterTensor[43]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
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
                            __m256 pRow[7], pTemp[2], pDst;
                            rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                            add_rows_7x7(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                            add_rows_7x7(pRow, &pTemp[1]);
                            gaussian_filter_blend_permute_add_7x7_pln(&pTemp[0], &pDst);

                            // convert result from pln to pkd format and store in output buffer
                            if constexpr (std::is_same<T, Rpp32f>::value)
                                _mm256_storeu_ps(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

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
                __m256 pFilterRow1[7];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[7], filterTensor[7], filterTensor[7], filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[9], filterTensor[9]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[14], filterTensor[14], filterTensor[14], filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22], filterTensor[22], filterTensor[23], filterTensor[23]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[28], filterTensor[28], filterTensor[28], filterTensor[29], filterTensor[29], filterTensor[29], filterTensor[30], filterTensor[30]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[35], filterTensor[35], filterTensor[35], filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[42], filterTensor[42], filterTensor[42], filterTensor[43], filterTensor[43], filterTensor[43], filterTensor[44], filterTensor[44]);
                __m256 pFilterRow2[7];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[9], filterTensor[10], filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[16], filterTensor[17], filterTensor[17], filterTensor[17], filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[23], filterTensor[24], filterTensor[24], filterTensor[24], filterTensor[25], filterTensor[25], filterTensor[25], filterTensor[26]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[30], filterTensor[31], filterTensor[31], filterTensor[31], filterTensor[32], filterTensor[32], filterTensor[32], filterTensor[33]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[37], filterTensor[38], filterTensor[38], filterTensor[38], filterTensor[39], filterTensor[39], filterTensor[39], filterTensor[40]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[44], filterTensor[45], filterTensor[45], filterTensor[45], filterTensor[46], filterTensor[46], filterTensor[46], filterTensor[47]);
                __m256 pFilterRow3[7];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[0], filterTensor[0], filterTensor[0]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[12], filterTensor[12], filterTensor[13], filterTensor[13], filterTensor[13], filterTensor[7], filterTensor[7], filterTensor[7]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[19], filterTensor[19], filterTensor[20], filterTensor[20], filterTensor[20], filterTensor[14], filterTensor[14], filterTensor[14]);
                pFilterRow3[3] = _mm256_setr_ps(filterTensor[26], filterTensor[26], filterTensor[27], filterTensor[27], filterTensor[27], filterTensor[21], filterTensor[21], filterTensor[21]);
                pFilterRow3[4] = _mm256_setr_ps(filterTensor[33], filterTensor[33], filterTensor[34], filterTensor[34], filterTensor[34], filterTensor[28], filterTensor[28], filterTensor[28]);
                pFilterRow3[5] = _mm256_setr_ps(filterTensor[40], filterTensor[40], filterTensor[41], filterTensor[41], filterTensor[41], filterTensor[35], filterTensor[35], filterTensor[35]);
                pFilterRow3[6] = _mm256_setr_ps(filterTensor[47], filterTensor[47], filterTensor[48], filterTensor[48], filterTensor[48], filterTensor[42], filterTensor[42], filterTensor[42]);
                __m256 pFilterRow4[7];
                pFilterRow4[0] = _mm256_setr_ps(filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2], filterTensor[2], filterTensor[3], filterTensor[3]);
                pFilterRow4[1] = _mm256_setr_ps(filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[10], filterTensor[10]);
                pFilterRow4[2] = _mm256_setr_ps(filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16], filterTensor[17], filterTensor[17]);
                pFilterRow4[3] = _mm256_setr_ps(filterTensor[22], filterTensor[22], filterTensor[22], filterTensor[23], filterTensor[23], filterTensor[23], filterTensor[24], filterTensor[24]);
                pFilterRow4[4] = _mm256_setr_ps(filterTensor[29], filterTensor[29], filterTensor[29], filterTensor[30], filterTensor[30], filterTensor[30], filterTensor[31], filterTensor[31]);
                pFilterRow4[5] = _mm256_setr_ps(filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37], filterTensor[37], filterTensor[38], filterTensor[38]);
                pFilterRow4[6] = _mm256_setr_ps(filterTensor[43], filterTensor[43], filterTensor[43], filterTensor[44], filterTensor[44], filterTensor[44], filterTensor[45], filterTensor[45]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
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
                    __m256 pRow[7], pTemp[4];
                    if (alignedLength)
                    {
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_7x7(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_7x7(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_7x7(pRow, &pTemp[2]);
                    }

                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        // add loaded values from 7 rows
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow4);
                        add_rows_7x7(pRow, &pTemp[3]);

                        __m256 pDst;
                        gaussian_filter_blend_permute_add_7x7_pkd(pTemp, &pDst);

                        // convert result from pln to pkd format and store in output buffer
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

                        dstPtrTemp += 8;
                        pTemp[0] = pTemp[1];
                        pTemp[1] = pTemp[2];
                        pTemp[2] = pTemp[3];
                    }
                    increment_row_ptrs(srcPtrTemp, kernelSize, -16);
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
                __m256 pFilterRow1[7];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6], filterTensor[0]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[7], filterTensor[8], filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[7]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[14], filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[14]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[25], filterTensor[26], filterTensor[27], filterTensor[21]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[28], filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33], filterTensor[34], filterTensor[28]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[35], filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[35]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[42], filterTensor[43], filterTensor[44], filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[42]);

                __m256 pFilterRow2[7];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6], filterTensor[0], filterTensor[1]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[8], filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[7], filterTensor[8]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[15], filterTensor[16], filterTensor[17], filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[14], filterTensor[15]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[25], filterTensor[26], filterTensor[27], filterTensor[21], filterTensor[22]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33], filterTensor[34], filterTensor[28], filterTensor[29]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[35], filterTensor[36]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[43], filterTensor[44], filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[42], filterTensor[43]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
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
                            __m256 pRow[7], pTemp[2];
                            rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow1);
                            add_rows_7x7(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow2);
                            add_rows_7x7(pRow, &pTemp[1]);
                            gaussian_filter_blend_permute_add_7x7_pln(pTemp, &pResultPln[c]);
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
                __m256 pFilterRow1[7];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[7], filterTensor[7], filterTensor[7], filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[9], filterTensor[9]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[14], filterTensor[14], filterTensor[14], filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22], filterTensor[22], filterTensor[23], filterTensor[23]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[28], filterTensor[28], filterTensor[28], filterTensor[29], filterTensor[29], filterTensor[29], filterTensor[30], filterTensor[30]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[35], filterTensor[35], filterTensor[35], filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[42], filterTensor[42], filterTensor[42], filterTensor[43], filterTensor[43], filterTensor[43], filterTensor[44], filterTensor[44]);
                __m256 pFilterRow2[7];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[9], filterTensor[10], filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[16], filterTensor[17], filterTensor[17], filterTensor[17], filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[23], filterTensor[24], filterTensor[24], filterTensor[24], filterTensor[25], filterTensor[25], filterTensor[25], filterTensor[26]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[30], filterTensor[31], filterTensor[31], filterTensor[31], filterTensor[32], filterTensor[32], filterTensor[32], filterTensor[33]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[37], filterTensor[38], filterTensor[38], filterTensor[38], filterTensor[39], filterTensor[39], filterTensor[39], filterTensor[40]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[44], filterTensor[45], filterTensor[45], filterTensor[45], filterTensor[46], filterTensor[46], filterTensor[46], filterTensor[47]);
                __m256 pFilterRow3[7];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[0], filterTensor[0], filterTensor[0]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[12], filterTensor[12], filterTensor[13], filterTensor[13], filterTensor[13], filterTensor[7], filterTensor[7], filterTensor[7]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[19], filterTensor[19], filterTensor[20], filterTensor[20], filterTensor[20], filterTensor[14], filterTensor[14], filterTensor[14]);
                pFilterRow3[3] = _mm256_setr_ps(filterTensor[26], filterTensor[26], filterTensor[27], filterTensor[27], filterTensor[27], filterTensor[21], filterTensor[21], filterTensor[21]);
                pFilterRow3[4] = _mm256_setr_ps(filterTensor[33], filterTensor[33], filterTensor[34], filterTensor[34], filterTensor[34], filterTensor[28], filterTensor[28], filterTensor[28]);
                pFilterRow3[5] = _mm256_setr_ps(filterTensor[40], filterTensor[40], filterTensor[41], filterTensor[41], filterTensor[41], filterTensor[35], filterTensor[35], filterTensor[35]);
                pFilterRow3[6] = _mm256_setr_ps(filterTensor[47], filterTensor[47], filterTensor[48], filterTensor[48], filterTensor[48], filterTensor[42], filterTensor[42], filterTensor[42]);
                __m256 pFilterRow4[7];
                pFilterRow4[0] = _mm256_setr_ps(filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2], filterTensor[2], filterTensor[3], filterTensor[3]);
                pFilterRow4[1] = _mm256_setr_ps(filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[10], filterTensor[10]);
                pFilterRow4[2] = _mm256_setr_ps(filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16], filterTensor[17], filterTensor[17]);
                pFilterRow4[3] = _mm256_setr_ps(filterTensor[22], filterTensor[22], filterTensor[22], filterTensor[23], filterTensor[23], filterTensor[23], filterTensor[24], filterTensor[24]);
                pFilterRow4[4] = _mm256_setr_ps(filterTensor[29], filterTensor[29], filterTensor[29], filterTensor[30], filterTensor[30], filterTensor[30], filterTensor[31], filterTensor[31]);
                pFilterRow4[5] = _mm256_setr_ps(filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37], filterTensor[37], filterTensor[38], filterTensor[38]);
                pFilterRow4[6] = _mm256_setr_ps(filterTensor[43], filterTensor[43], filterTensor[43], filterTensor[44], filterTensor[44], filterTensor[44], filterTensor[45], filterTensor[45]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
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
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[7], pTemp[5];
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_7x7(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_7x7(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_7x7(pRow, &pTemp[2]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow4);
                        add_rows_7x7(pRow, &pTemp[3]);
                        pTemp[4] = avx_p0;

                        __m256 pDst[2];
                        gaussian_filter_blend_permute_add_7x7_pkd(&pTemp[0], &pDst[0]);
                        gaussian_filter_blend_permute_add_7x7_pkd(&pTemp[1], &pDst[1]);

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, -12);
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
            T *srcPtrRow[9], *dstPtrRow;
            for (int i = 0; i < 9; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                __m256 pFilterRow1[9];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6], filterTensor[7]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[15], filterTensor[16]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[25]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[27], filterTensor[28], filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33], filterTensor[34]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[42], filterTensor[43]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[49], filterTensor[50], filterTensor[51], filterTensor[52]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[54], filterTensor[55], filterTensor[56], filterTensor[57], filterTensor[58], filterTensor[59], filterTensor[60], filterTensor[61]);
                pFilterRow1[7] = _mm256_setr_ps(filterTensor[63], filterTensor[64], filterTensor[65], filterTensor[66], filterTensor[67], filterTensor[68], filterTensor[69], filterTensor[70]);
                pFilterRow1[8] = _mm256_setr_ps(filterTensor[72], filterTensor[73], filterTensor[74], filterTensor[75], filterTensor[76], filterTensor[77], filterTensor[78], filterTensor[79]);

                __m256 pFilterRow2[9];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[8], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[17], filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[15]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[26], filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[35], filterTensor[27], filterTensor[28], filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[44], filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[42]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[53], filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[49], filterTensor[50], filterTensor[51]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[62], filterTensor[54], filterTensor[55], filterTensor[56], filterTensor[57], filterTensor[58], filterTensor[59], filterTensor[60]);
                pFilterRow2[7] = _mm256_setr_ps(filterTensor[71], filterTensor[63], filterTensor[64], filterTensor[65], filterTensor[66], filterTensor[67], filterTensor[68], filterTensor[69]);
                pFilterRow2[8] = _mm256_setr_ps(filterTensor[80], filterTensor[72], filterTensor[73], filterTensor[74], filterTensor[75], filterTensor[76], filterTensor[77], filterTensor[78]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
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
                        __m256 pRow[9];
                        if (alignedLength)
                            rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);

                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            // add loaded values from 9 rows
                            __m256 pTemp[2], pDst;
                            add_rows_9x9(pRow, &pTemp[0]);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);

                            rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                            add_rows_9x9(pRow, &pTemp[1]);
                            gaussian_filter_blend_permute_add_9x9_pln(pTemp, &pDst);

                            if constexpr (std::is_same<T, Rpp32f>::value)
                                _mm256_storeu_ps(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp16f>::value)
                                _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                            else if constexpr (std::is_same<T, Rpp8s>::value)
                                rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                            else if constexpr (std::is_same<T, Rpp8u>::value)
                                rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

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
                __m256 pFilterRow1[9];
                // for(int i = 0; i < 9; i++)
                // {
                //     pFilterRow1[i] = _mm256_set1_ps((float)1/(float)81);
                // }
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[10], filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19], filterTensor[19], filterTensor[19], filterTensor[20], filterTensor[20]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[27], filterTensor[27], filterTensor[27], filterTensor[28], filterTensor[28], filterTensor[28], filterTensor[29], filterTensor[29]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37], filterTensor[37], filterTensor[38], filterTensor[38]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[45], filterTensor[45], filterTensor[45], filterTensor[46], filterTensor[46], filterTensor[46], filterTensor[47], filterTensor[47]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[54], filterTensor[54], filterTensor[54], filterTensor[55], filterTensor[55], filterTensor[55], filterTensor[56], filterTensor[56]);
                pFilterRow1[7] = _mm256_setr_ps(filterTensor[63], filterTensor[63], filterTensor[63], filterTensor[64], filterTensor[64], filterTensor[64], filterTensor[65], filterTensor[65]);
                pFilterRow1[8] = _mm256_setr_ps(filterTensor[72], filterTensor[72], filterTensor[72], filterTensor[73], filterTensor[73], filterTensor[73], filterTensor[74], filterTensor[74]);
                __m256 pFilterRow2[9];
                // for(int i = 0; i < 9; i++)
                // {
                //     pFilterRow2[i] = _mm256_set1_ps((float)1/(float)81);
                // }
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[11], filterTensor[12], filterTensor[12], filterTensor[12], filterTensor[13], filterTensor[13], filterTensor[13], filterTensor[14]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22], filterTensor[22], filterTensor[23]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[29], filterTensor[30], filterTensor[30], filterTensor[30], filterTensor[31], filterTensor[31], filterTensor[31], filterTensor[32]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[38], filterTensor[39], filterTensor[39], filterTensor[39], filterTensor[40], filterTensor[40], filterTensor[40], filterTensor[41]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[47], filterTensor[48], filterTensor[48], filterTensor[48], filterTensor[49], filterTensor[49], filterTensor[49], filterTensor[50]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[56], filterTensor[57], filterTensor[57], filterTensor[57], filterTensor[58], filterTensor[58], filterTensor[58], filterTensor[59]);
                pFilterRow2[7] = _mm256_setr_ps(filterTensor[65], filterTensor[66], filterTensor[66], filterTensor[66], filterTensor[67], filterTensor[67], filterTensor[67], filterTensor[68]);
                pFilterRow2[8] = _mm256_setr_ps(filterTensor[74], filterTensor[75], filterTensor[75], filterTensor[75], filterTensor[76], filterTensor[76], filterTensor[76], filterTensor[77]);
                __m256 pFilterRow3[9];
                // for(int i = 0; i < 9; i++)
                // {
                //     pFilterRow3[i] = _mm256_set1_ps((float)1/(float)81);
                // }
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[14], filterTensor[14], filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[23], filterTensor[23], filterTensor[24], filterTensor[24], filterTensor[24], filterTensor[25], filterTensor[25], filterTensor[25]);
                pFilterRow3[3] = _mm256_setr_ps(filterTensor[32], filterTensor[32], filterTensor[33], filterTensor[33], filterTensor[33], filterTensor[34], filterTensor[34], filterTensor[34]);
                pFilterRow3[4] = _mm256_setr_ps(filterTensor[41], filterTensor[41], filterTensor[42], filterTensor[42], filterTensor[42], filterTensor[43], filterTensor[43], filterTensor[43]);
                pFilterRow3[5] = _mm256_setr_ps(filterTensor[50], filterTensor[50], filterTensor[51], filterTensor[51], filterTensor[51], filterTensor[52], filterTensor[52], filterTensor[52]);
                pFilterRow3[6] = _mm256_setr_ps(filterTensor[59], filterTensor[59], filterTensor[60], filterTensor[60], filterTensor[60], filterTensor[61], filterTensor[61], filterTensor[61]);
                pFilterRow3[7] = _mm256_setr_ps(filterTensor[68], filterTensor[68], filterTensor[69], filterTensor[69], filterTensor[69], filterTensor[70], filterTensor[70], filterTensor[70]);
                pFilterRow3[8] = _mm256_setr_ps(filterTensor[77], filterTensor[77], filterTensor[78], filterTensor[78], filterTensor[78], filterTensor[79], filterTensor[79], filterTensor[79]);
                __m256 pFilterRow4[9];
                // for(int i = 0; i < 9; i++)
                // {
                //     pFilterRow4[i] = _mm256_set1_ps(filterTensor[80]);
                // }
                // pFilterRow4[0] = _mm256_setr_ps(filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1]);
                // pFilterRow4[1] = _mm256_setr_ps(filterTensor[17], filterTensor[17], filterTensor[17], filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[10], filterTensor[10]);
                // pFilterRow4[2] = _mm256_setr_ps(filterTensor[26], filterTensor[26], filterTensor[26], filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19], filterTensor[19]);
                // pFilterRow4[3] = _mm256_setr_ps(filterTensor[35], filterTensor[35], filterTensor[35], filterTensor[27], filterTensor[27], filterTensor[27], filterTensor[28], filterTensor[28]);
                // pFilterRow4[4] = _mm256_setr_ps(filterTensor[44], filterTensor[44], filterTensor[44], filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37]);
                // pFilterRow4[5] = _mm256_setr_ps(filterTensor[53], filterTensor[53], filterTensor[53], filterTensor[45], filterTensor[45], filterTensor[45], filterTensor[46], filterTensor[46]);
                // pFilterRow4[6] = _mm256_setr_ps(filterTensor[62], filterTensor[62], filterTensor[62], filterTensor[54], filterTensor[54], filterTensor[54], filterTensor[55], filterTensor[55]);
                // pFilterRow4[7] = _mm256_setr_ps(filterTensor[71], filterTensor[71], filterTensor[71], filterTensor[63], filterTensor[63], filterTensor[63], filterTensor[64], filterTensor[64]);
                // pFilterRow4[8] = _mm256_setr_ps(filterTensor[80], filterTensor[80], filterTensor[80], filterTensor[72], filterTensor[72], filterTensor[72], filterTensor[73], filterTensor[73]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
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
                    __m256 pRow[9], pTemp[4];
                    if (alignedLength)
                    {
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_9x9(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_9x9(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_9x9(pRow, &pTemp[2]);
                    }

                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        // add loaded values from 9 rows
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow4);
                        add_rows_9x9(pRow, &pTemp[3]);

                        __m256 pDst;
                        gaussian_filter_blend_permute_add_9x9_pkd(pTemp, &pDst);
                        if constexpr (std::is_same<T, Rpp32f>::value)
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp16f>::value)
                            _mm_storeu_si128((__m128i *)dstPtrTemp, _mm256_cvtps_ph(pDst, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
                        else if constexpr (std::is_same<T, Rpp8s>::value)
                            rpp_store8_f32_to_i8_avx(dstPtrTemp, pDst);
                        else if constexpr (std::is_same<T, Rpp8u>::value)
                            rpp_store8_f32_to_u8_avx(dstPtrTemp, pDst);

                        dstPtrTemp += 8;
                        pTemp[0] = pTemp[1];
                        pTemp[1] = pTemp[2];
                        pTemp[2] = pTemp[3];
                    }
                    increment_row_ptrs(srcPtrTemp, kernelSize, -16);
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
            // box filter with fused output-layout toggle (NCHW -> NHWC)
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                /* exclude (2 * padLength) number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                __m256 pFilterRow1[9];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6], filterTensor[7]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[15], filterTensor[16]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24], filterTensor[25]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[27], filterTensor[28], filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33], filterTensor[34]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[42], filterTensor[43]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[49], filterTensor[50], filterTensor[51], filterTensor[52]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[54], filterTensor[55], filterTensor[56], filterTensor[57], filterTensor[58], filterTensor[59], filterTensor[60], filterTensor[61]);
                pFilterRow1[7] = _mm256_setr_ps(filterTensor[63], filterTensor[64], filterTensor[65], filterTensor[66], filterTensor[67], filterTensor[68], filterTensor[69], filterTensor[70]);
                pFilterRow1[8] = _mm256_setr_ps(filterTensor[72], filterTensor[73], filterTensor[74], filterTensor[75], filterTensor[76], filterTensor[77], filterTensor[78], filterTensor[79]);

                __m256 pFilterRow2[9];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[8], filterTensor[0], filterTensor[1], filterTensor[2], filterTensor[3], filterTensor[4], filterTensor[5], filterTensor[6]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[17], filterTensor[9], filterTensor[10], filterTensor[11], filterTensor[12], filterTensor[13], filterTensor[14], filterTensor[15]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[26], filterTensor[18], filterTensor[19], filterTensor[20], filterTensor[21], filterTensor[22], filterTensor[23], filterTensor[24]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[35], filterTensor[27], filterTensor[28], filterTensor[29], filterTensor[30], filterTensor[31], filterTensor[32], filterTensor[33]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[44], filterTensor[36], filterTensor[37], filterTensor[38], filterTensor[39], filterTensor[40], filterTensor[41], filterTensor[42]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[53], filterTensor[45], filterTensor[46], filterTensor[47], filterTensor[48], filterTensor[49], filterTensor[50], filterTensor[51]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[62], filterTensor[54], filterTensor[55], filterTensor[56], filterTensor[57], filterTensor[58], filterTensor[59], filterTensor[60]);
                pFilterRow2[7] = _mm256_setr_ps(filterTensor[71], filterTensor[63], filterTensor[64], filterTensor[65], filterTensor[66], filterTensor[67], filterTensor[68], filterTensor[69]);
                pFilterRow2[8] = _mm256_setr_ps(filterTensor[80], filterTensor[72], filterTensor[73], filterTensor[74], filterTensor[75], filterTensor[76], filterTensor[77], filterTensor[78]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
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
                            // add loaded values from 9 rows
                            __m256 pRow[9], pTemp[2];
                            rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow1);
                            add_rows_9x9(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp[c], rowKernelLoopLimit, pFilterRow2);
                            add_rows_9x9(pRow, &pTemp[1]);

                            gaussian_filter_blend_permute_add_9x9_pln(pTemp, &pResultPln[c]);
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
                __m256 pFilterRow1[9];
                pFilterRow1[0] = _mm256_setr_ps(filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1], filterTensor[1], filterTensor[2], filterTensor[2]);
                pFilterRow1[1] = _mm256_setr_ps(filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[10], filterTensor[10], filterTensor[10], filterTensor[11], filterTensor[11]);
                pFilterRow1[2] = _mm256_setr_ps(filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19], filterTensor[19], filterTensor[19], filterTensor[20], filterTensor[20]);
                pFilterRow1[3] = _mm256_setr_ps(filterTensor[27], filterTensor[27], filterTensor[27], filterTensor[28], filterTensor[28], filterTensor[28], filterTensor[29], filterTensor[29]);
                pFilterRow1[4] = _mm256_setr_ps(filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37], filterTensor[37], filterTensor[38], filterTensor[38]);
                pFilterRow1[5] = _mm256_setr_ps(filterTensor[45], filterTensor[45], filterTensor[45], filterTensor[46], filterTensor[46], filterTensor[46], filterTensor[47], filterTensor[47]);
                pFilterRow1[6] = _mm256_setr_ps(filterTensor[54], filterTensor[54], filterTensor[54], filterTensor[55], filterTensor[55], filterTensor[55], filterTensor[56], filterTensor[56]);
                pFilterRow1[7] = _mm256_setr_ps(filterTensor[63], filterTensor[63], filterTensor[63], filterTensor[64], filterTensor[64], filterTensor[64], filterTensor[65], filterTensor[65]);
                pFilterRow1[8] = _mm256_setr_ps(filterTensor[72], filterTensor[72], filterTensor[72], filterTensor[73], filterTensor[73], filterTensor[73], filterTensor[74], filterTensor[74]);
                __m256 pFilterRow2[9];
                pFilterRow2[0] = _mm256_setr_ps(filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4], filterTensor[4], filterTensor[4], filterTensor[5]);
                pFilterRow2[1] = _mm256_setr_ps(filterTensor[11], filterTensor[12], filterTensor[12], filterTensor[12], filterTensor[13], filterTensor[13], filterTensor[13], filterTensor[14]);
                pFilterRow2[2] = _mm256_setr_ps(filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22], filterTensor[22], filterTensor[22], filterTensor[23]);
                pFilterRow2[3] = _mm256_setr_ps(filterTensor[29], filterTensor[30], filterTensor[30], filterTensor[30], filterTensor[31], filterTensor[31], filterTensor[31], filterTensor[32]);
                pFilterRow2[4] = _mm256_setr_ps(filterTensor[38], filterTensor[39], filterTensor[39], filterTensor[39], filterTensor[40], filterTensor[40], filterTensor[40], filterTensor[41]);
                pFilterRow2[5] = _mm256_setr_ps(filterTensor[47], filterTensor[48], filterTensor[48], filterTensor[48], filterTensor[49], filterTensor[49], filterTensor[49], filterTensor[50]);
                pFilterRow2[6] = _mm256_setr_ps(filterTensor[56], filterTensor[57], filterTensor[57], filterTensor[57], filterTensor[58], filterTensor[58], filterTensor[58], filterTensor[59]);
                pFilterRow2[7] = _mm256_setr_ps(filterTensor[65], filterTensor[66], filterTensor[66], filterTensor[66], filterTensor[67], filterTensor[67], filterTensor[67], filterTensor[68]);
                pFilterRow2[8] = _mm256_setr_ps(filterTensor[74], filterTensor[75], filterTensor[75], filterTensor[75], filterTensor[76], filterTensor[76], filterTensor[76], filterTensor[77]);
                __m256 pFilterRow3[9];
                pFilterRow3[0] = _mm256_setr_ps(filterTensor[5], filterTensor[5], filterTensor[6], filterTensor[6], filterTensor[6], filterTensor[7], filterTensor[7], filterTensor[7]);
                pFilterRow3[1] = _mm256_setr_ps(filterTensor[14], filterTensor[14], filterTensor[15], filterTensor[15], filterTensor[15], filterTensor[16], filterTensor[16], filterTensor[16]);
                pFilterRow3[2] = _mm256_setr_ps(filterTensor[23], filterTensor[23], filterTensor[24], filterTensor[24], filterTensor[24], filterTensor[25], filterTensor[25], filterTensor[25]);
                pFilterRow3[3] = _mm256_setr_ps(filterTensor[32], filterTensor[32], filterTensor[33], filterTensor[33], filterTensor[33], filterTensor[34], filterTensor[34], filterTensor[34]);
                pFilterRow3[4] = _mm256_setr_ps(filterTensor[41], filterTensor[41], filterTensor[42], filterTensor[42], filterTensor[42], filterTensor[43], filterTensor[43], filterTensor[43]);
                pFilterRow3[5] = _mm256_setr_ps(filterTensor[50], filterTensor[50], filterTensor[51], filterTensor[51], filterTensor[51], filterTensor[52], filterTensor[52], filterTensor[52]);
                pFilterRow3[6] = _mm256_setr_ps(filterTensor[59], filterTensor[59], filterTensor[60], filterTensor[60], filterTensor[60], filterTensor[61], filterTensor[61], filterTensor[61]);
                pFilterRow3[7] = _mm256_setr_ps(filterTensor[68], filterTensor[68], filterTensor[69], filterTensor[69], filterTensor[69], filterTensor[70], filterTensor[70], filterTensor[70]);
                pFilterRow3[8] = _mm256_setr_ps(filterTensor[77], filterTensor[77], filterTensor[78], filterTensor[78], filterTensor[78], filterTensor[79], filterTensor[79], filterTensor[79]);
                __m256 pFilterRow4[9];
                pFilterRow4[0] = _mm256_setr_ps(filterTensor[8], filterTensor[8], filterTensor[8], filterTensor[0], filterTensor[0], filterTensor[0], filterTensor[1], filterTensor[1]);
                pFilterRow4[1] = _mm256_setr_ps(filterTensor[17], filterTensor[17], filterTensor[17], filterTensor[9], filterTensor[9], filterTensor[9], filterTensor[10], filterTensor[10]);
                pFilterRow4[2] = _mm256_setr_ps(filterTensor[26], filterTensor[26], filterTensor[26], filterTensor[18], filterTensor[18], filterTensor[18], filterTensor[19], filterTensor[19]);
                pFilterRow4[3] = _mm256_setr_ps(filterTensor[35], filterTensor[35], filterTensor[35], filterTensor[27], filterTensor[27], filterTensor[27], filterTensor[28], filterTensor[28]);
                pFilterRow4[4] = _mm256_setr_ps(filterTensor[44], filterTensor[44], filterTensor[44], filterTensor[36], filterTensor[36], filterTensor[36], filterTensor[37], filterTensor[37]);
                pFilterRow4[5] = _mm256_setr_ps(filterTensor[53], filterTensor[53], filterTensor[53], filterTensor[45], filterTensor[45], filterTensor[45], filterTensor[46], filterTensor[46]);
                pFilterRow4[6] = _mm256_setr_ps(filterTensor[62], filterTensor[62], filterTensor[62], filterTensor[54], filterTensor[54], filterTensor[54], filterTensor[55], filterTensor[55]);
                pFilterRow4[7] = _mm256_setr_ps(filterTensor[71], filterTensor[71], filterTensor[71], filterTensor[63], filterTensor[63], filterTensor[63], filterTensor[64], filterTensor[64]);
                pFilterRow4[8] = _mm256_setr_ps(filterTensor[80], filterTensor[80], filterTensor[80], filterTensor[72], filterTensor[72], filterTensor[72], filterTensor[73], filterTensor[73]);
                __m256 pFilterRow5[9];
                pFilterRow5[0] = _mm256_setr_ps(filterTensor[1], filterTensor[2], filterTensor[2], filterTensor[2], filterTensor[3], filterTensor[3], filterTensor[3], filterTensor[4]);
                pFilterRow5[1] = _mm256_setr_ps(filterTensor[10], filterTensor[11], filterTensor[11], filterTensor[11], filterTensor[12], filterTensor[12], filterTensor[12], filterTensor[13]);
                pFilterRow5[2] = _mm256_setr_ps(filterTensor[19], filterTensor[20], filterTensor[20], filterTensor[20], filterTensor[21], filterTensor[21], filterTensor[21], filterTensor[22]);
                pFilterRow5[3] = _mm256_setr_ps(filterTensor[28], filterTensor[29], filterTensor[29], filterTensor[29], filterTensor[30], filterTensor[30], filterTensor[30], filterTensor[31]);
                pFilterRow5[4] = _mm256_setr_ps(filterTensor[37], filterTensor[38], filterTensor[38], filterTensor[38], filterTensor[39], filterTensor[39], filterTensor[39], filterTensor[40]);
                pFilterRow5[5] = _mm256_setr_ps(filterTensor[46], filterTensor[47], filterTensor[47], filterTensor[47], filterTensor[48], filterTensor[48], filterTensor[48], filterTensor[49]);
                pFilterRow5[6] = _mm256_setr_ps(filterTensor[55], filterTensor[56], filterTensor[56], filterTensor[56], filterTensor[57], filterTensor[57], filterTensor[57], filterTensor[58]);
                pFilterRow5[7] = _mm256_setr_ps(filterTensor[64], filterTensor[65], filterTensor[65], filterTensor[65], filterTensor[66], filterTensor[66], filterTensor[66], filterTensor[67]);
                pFilterRow5[8] = _mm256_setr_ps(filterTensor[73], filterTensor[74], filterTensor[74], filterTensor[74], filterTensor[75], filterTensor[75], filterTensor[75], filterTensor[76]);
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 40) * 40;
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
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[9], pTemp[5];
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow1);
                        add_rows_9x9(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow2);
                        add_rows_9x9(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow3);
                        add_rows_9x9(pRow, &pTemp[2]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow4);
                        add_rows_9x9(pRow, &pTemp[3]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit, pFilterRow5);
                        add_rows_9x9(pRow, &pTemp[4]);

                        __m256 pDst[2];
                        gaussian_filter_blend_permute_add_9x9_pkd(&pTemp[0], &pDst[0]);
                        gaussian_filter_blend_permute_add_9x9_pkd(&pTemp[1], &pDst[1]);

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        rpp_store12_float_pkd_pln(dstPtrTempChannels, pDstPln);

                        increment_row_ptrs(srcPtrTemp, kernelSize, -20);
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