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

    if constexpr (std::is_same<T, Rpp8u>::value || std::is_same<T, Rpp8s>::value)
        accum = round(accum);
    saturate_pixel(accum, dstPtrTemp);
}

// load function for 3x3 kernel size
inline void rpp_load_filter_3x3_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

inline void rpp_load_filter_3x3_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

inline void rpp_load_filter_3x3_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

inline void rpp_load16_f16_to_f32_avx(Rpp16f *srcPtr, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 8))));
}

// load function for 3x3 kernel size
inline void rpp_load_filter_3x3_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    else
    {
        pRow[4] = avx_p0;
        pRow[5] = avx_p0;
    }
}

// load function for 5x5 kernel size
inline void rpp_load_filter_5x5_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_5x5_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_5x5_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_5x5_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);

    for (int k = 3; k < rowKernelLoopLimit; k++)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

// load function for 7x7 kernel size
inline void rpp_load_filter_7x7_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_7x7_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_7x7_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_7x7_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

// load function for 9x9 kernel size
inline void rpp_load_filter_9x9_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    rpp_load16_u8_to_f32_avx(srcPtrTemp[4], &pRow[8]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        rpp_load16_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_9x9_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    rpp_load16_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    rpp_load16_i8_to_f32_avx(srcPtrTemp[4], &pRow[8]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        rpp_load16_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_9x9_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    rpp_load16_f32_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    rpp_load16_f32_to_f32_avx(srcPtrTemp[4], &pRow[8]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        rpp_load16_f32_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
}

inline void rpp_load_filter_9x9_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load16_f16_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[1], &pRow[2]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[2], &pRow[4]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[3], &pRow[6]);
    rpp_load16_f16_to_f32_avx(srcPtrTemp[4], &pRow[8]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        rpp_load16_f16_to_f32_avx(srcPtrTemp[k], &pRow[k * 2]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 2] = avx_p0;
        pRow[k * 2 + 1] = avx_p0;
    }
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
#pragma omp parallel for num_threads(1)
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
                            rpp_load_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);

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
                        rpp_load_gaussian_filter_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);

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
                            rpp_load_filter_3x3_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
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
                            rpp_load_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                            rpp_load_filter_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
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
                        rpp_load_gaussian_filter_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                            rpp_load_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                            rpp_load_filter_7x7_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
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
                        rpp_load_gaussian_filter_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                            rpp_load_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
                        rpp_load_gaussian_filter_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit);
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
            // box filter with fused output-layout toggle (NCHW -> NHWC)
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
                            rpp_load_filter_9x9_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
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