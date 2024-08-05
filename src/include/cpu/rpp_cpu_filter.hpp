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

#ifndef AMD_RPP_RPP_CPU_FILTER_HPP
#define AMD_RPP_RPP_CPU_FILTER_HPP

#include "stdio.h"
#include "rppdefs.h"
#include <half/half.hpp>
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;
#include "rpp_cpu_simd.hpp"

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

const __m128i xmm_pxMaskRotate0To1 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
const __m128i xmm_pxMaskRotate0To3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
const __m128i xmm_pxMaskRotate0To5 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5);
const __m128i xmm_pxMaskRotate0To7 = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
const __m128i xmm_pxMaskRotate0To9 = _mm_setr_epi8(10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
const __m128i xmm_pxMaskRotate0To11 = _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
const __m128i xmm_pxMaskRotate0To13 = _mm_setr_epi8(14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);

const __m256i avx_pxMaskRotate0To1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
const __m256i avx_pxMaskRotate0To2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
const __m256i avx_pxMaskRotate0To3 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);
const __m256i avx_pxMaskRotate0To4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
const __m256i avx_pxMaskRotate0To5 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);
const __m256i avx_pxMaskRotate0To6 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
const __m256i avx_pxMaskRotate0To7 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);

template<typename T>
inline void increment_row_ptrs(T **srcPtrTemp, Rpp32u kernelSize, Rpp32s increment)
{
    for (int i = 0; i < kernelSize; i++)
        srcPtrTemp[i] += increment;
}

// get the kernel loop limit based on index
inline void get_kernel_loop_limit(Rpp32s &index, Rpp32s &loopLimit, Rpp32u &padLength, Rpp32u &unpaddedLength)
{
    if ((index < padLength) || (index >= unpaddedLength))
    {
        Rpp32u factor = (index < padLength) ? (index - padLength) : (unpaddedLength - 1 - index);
        loopLimit += factor;
    }
}

// -------------------- Filter load functions for U8 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_filter_3x3_pln_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

// load function for 5x5 kernel size
inline void rpp_load_filter_5x5_pln_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

// load function for 7x7 kernel size
inline void rpp_load_filter_7x7_pln_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

// load function for 9x9 kernel size
inline void rpp_load_filter_9x9_pln_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_3x3_pkd_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load8_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load8_u8_to_f32_avx(srcPtrTemp[0] + 8, &pRow[1]);
    rpp_load8_u8_to_f32_avx(srcPtrTemp[0] + 16, &pRow[2]);
    rpp_load8_u8_to_f32_avx(srcPtrTemp[1], &pRow[3]);
    rpp_load8_u8_to_f32_avx(srcPtrTemp[1] + 8, &pRow[4]);
    rpp_load8_u8_to_f32_avx(srcPtrTemp[1] + 16, &pRow[5]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
    {
        rpp_load8_u8_to_f32_avx(srcPtrTemp[2], &pRow[6]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[2] + 8, &pRow[7]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[2] + 16, &pRow[8]);
    }
    else
    {
        pRow[6] = avx_px0;
        pRow[7] = avx_px0;
        pRow[8] = avx_px0;
    }
}

// load function for 5x5 kernel size
inline void rpp_load_filter_5x5_pkd_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    rpp_load32_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[1], &pRow[4]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[2], &pRow[8]);
    for (int k = 3; k < rowKernelLoopLimit; k++)
    {
        rpp_load32_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
    }
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_filter_7x7_pkd_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    rpp_load32_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[1], &pRow[4]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[2], &pRow[8]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[3], &pRow[12]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
    {
        rpp_load32_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
    }
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_filter_9x9_pkd_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    rpp_load32_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[1], &pRow[4]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[2], &pRow[8]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[3], &pRow[12]);
    rpp_load32_u8_to_f32_avx(srcPtrTemp[4], &pRow[16]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        rpp_load32_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_gaussian_filter_9x9_pkd_pln_host(__m256 *pRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        rpp_load8_u8_to_f32_avx(srcPtrTemp[i], &pRow[i * 4]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[i] + 8, &pRow[i * 4 + 1]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[i] + 16, &pRow[i * 4 + 2]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[i] + 24, &pRow[i * 4 + 3]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[i] + 32, &pRow[i * 4 + 4]);
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        rpp_load8_u8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[k] + 8, &pRow[k * 4 + 1]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[k] + 16, &pRow[k * 4 + 2]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[k] + 24, &pRow[k * 4 + 3]);
        rpp_load8_u8_to_f32_avx(srcPtrTemp[k] + 32, &pRow[k * 4 + 4]);
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
        pRow[k * 4 + 4] = avx_p0;
    }
}

// -------------------- Filter load functions for I8 bitdepth --------------------

inline void rpp_load_filter_3x3_pln_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_5x5_pln_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_7x7_pln_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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


inline void rpp_load_filter_9x9_pln_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_3x3_pkd_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    rpp_load8_i8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
    rpp_load8_i8_to_f32_avx(srcPtrTemp[0] + 8, &pRow[1]);
    rpp_load8_i8_to_f32_avx(srcPtrTemp[0] + 16, &pRow[2]);
    rpp_load8_i8_to_f32_avx(srcPtrTemp[1], &pRow[3]);
    rpp_load8_i8_to_f32_avx(srcPtrTemp[1] + 8, &pRow[4]);
    rpp_load8_i8_to_f32_avx(srcPtrTemp[1] + 16, &pRow[5]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[2], &pRow[6]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[2] + 8, &pRow[7]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[2] + 16, &pRow[8]);
    }
    else
    {
        pRow[6] = avx_px0;
        pRow[7] = avx_px0;
        pRow[8] = avx_px0;
    }
}

// load function for 5x5 kernel size
inline void rpp_load_filter_5x5_pkd_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    for(int i = 0; i < 3; i++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i], &pRow[i * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 8, &pRow[i * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 16, &pRow[i * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 24, &pRow[i * 4 + 3]);
    }
    for (int k = 3; k < rowKernelLoopLimit; k++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 8, &pRow[k * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 16, &pRow[k * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 24, &pRow[k * 4 + 3]);
    }
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_filter_7x7_pkd_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    for(int i = 0; i < 4; i++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i], &pRow[i * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 8, &pRow[i * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 16, &pRow[i * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 24, &pRow[i * 4 + 3]);
    }
    for (int k = 4; k < rowKernelLoopLimit; k++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 8, &pRow[k * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 16, &pRow[k * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 24, &pRow[k * 4 + 3]);
    }
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_filter_9x9_pkd_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i], &pRow[i * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 8, &pRow[i * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 16, &pRow[i * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 24, &pRow[i * 4 + 3]);
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 8, &pRow[k * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 16, &pRow[k * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 24, &pRow[k * 4 + 3]);
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_gaussian_filter_9x9_pkd_pln_host(__m256 *pRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i], &pRow[i * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 8, &pRow[i * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 16, &pRow[i * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 24, &pRow[i * 4 + 3]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[i] + 32, &pRow[i * 4 + 4]);
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k], &pRow[k * 4]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 8, &pRow[k * 4 + 1]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 16, &pRow[k * 4 + 2]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 24, &pRow[k * 4 + 3]);
        rpp_load8_i8_to_f32_avx(srcPtrTemp[k] + 32, &pRow[k * 4 + 4]);
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
        pRow[k * 4 + 4] = avx_p0;
    }
}

// -------------------- Filter load functions for F32 bitdepth --------------------

inline void rpp_load_filter_3x3_pln_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_5x5_pln_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_7x7_pln_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_9x9_pln_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_3x3_pkd_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[0] + 8);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[0] + 16);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[4] = _mm256_loadu_ps(srcPtrTemp[1] + 8);
    pRow[5] = _mm256_loadu_ps(srcPtrTemp[1] + 16);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
    {
        pRow[6] = _mm256_loadu_ps(srcPtrTemp[2]);
        pRow[7] = _mm256_loadu_ps(srcPtrTemp[2] + 8);
        pRow[8] = _mm256_loadu_ps(srcPtrTemp[2] + 16);
    }
    else
    {
        pRow[6] = avx_px0;
        pRow[7] = avx_px0;
        pRow[8] = avx_px0;
    }
}

// load function for 5x5 kernel size
inline void rpp_load_filter_5x5_pkd_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    for(int i = 0; i < 3; i++)
    {
        pRow[i * 4] = _mm256_loadu_ps(srcPtrTemp[i]);
        pRow[i * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[i] + 8);
        pRow[i * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[i] + 16);
        pRow[i * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[i] + 24);
    }
    for (int k = 3; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_loadu_ps(srcPtrTemp[k]);
        pRow[k * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[k] + 8);
        pRow[k * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[k] + 16);
        pRow[k * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[k] + 24);
    }
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

// load function for 7x7 kernel size
inline void rpp_load_filter_7x7_pkd_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    for(int i = 0; i < 4; i++)
    {
        pRow[i * 4] = _mm256_loadu_ps(srcPtrTemp[i]);
        pRow[i * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[i] + 8);
        pRow[i * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[i] + 16);
        pRow[i * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[i] + 24);
    }
    for (int k = 4; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_loadu_ps(srcPtrTemp[k]);
        pRow[k * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[k] + 8);
        pRow[k * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[k] + 16);
        pRow[k * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[k] + 24);
    }
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

// load function for 9x9 kernel size
inline void rpp_load_filter_9x9_pkd_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        pRow[i * 4] = _mm256_loadu_ps(srcPtrTemp[i]);
        pRow[i * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[i] + 8);
        pRow[i * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[i] + 16);
        pRow[i * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[i] + 24);
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_loadu_ps(srcPtrTemp[k]);
        pRow[k * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[k] + 8);
        pRow[k * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[k] + 16);
        pRow[k * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[k] + 24);
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_gaussian_filter_9x9_pkd_pln_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        pRow[i * 4] = _mm256_loadu_ps(srcPtrTemp[i]);
        pRow[i * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[i] + 8);
        pRow[i * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[i] + 16);
        pRow[i * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[i] + 24);
        pRow[i * 4 + 4] = _mm256_loadu_ps(srcPtrTemp[i] + 32);
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_loadu_ps(srcPtrTemp[k]);
        pRow[k * 4 + 1] = _mm256_loadu_ps(srcPtrTemp[k] + 8);
        pRow[k * 4 + 2] = _mm256_loadu_ps(srcPtrTemp[k] + 16);
        pRow[k * 4 + 3] = _mm256_loadu_ps(srcPtrTemp[k] + 24);
        pRow[k * 4 + 4] = _mm256_loadu_ps(srcPtrTemp[k] + 32);
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
        pRow[k * 4 + 4] = avx_p0;
    }
}

// -------------------- Filter load functions for F16 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_filter_3x3_pln_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_5x5_pln_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_7x7_pln_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_9x9_pln_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

inline void rpp_load_filter_3x3_pkd_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0]))));
    pRow[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0] + 8))));
    pRow[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0] + 16))));
    pRow[3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]))));
    pRow[4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1] + 8))));
    pRow[5] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]+ 16))));

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
    {
        pRow[6] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2]))));
        pRow[7] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2] + 8))));
        pRow[8] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2] + 16))));
    }
    else
    {
        pRow[6] = avx_px0;
        pRow[7] = avx_px0;
        pRow[8] = avx_px0;
    }
}

// load function for 5x5 kernel size
inline void rpp_load_filter_5x5_pkd_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    for(int i = 0; i < 3; i++)
    {
        pRow[i * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i]))));
        pRow[i * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 8))));
        pRow[i * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 16))));
        pRow[i * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 24))));
    }
    for (int k = 3; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
        pRow[k * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 8))));
        pRow[k * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 16))));
        pRow[k * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 24))));
    }
    for (int k = rowKernelLoopLimit; k < 5; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

// load function for 7x7 kernel size
inline void rpp_load_filter_7x7_pkd_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    for(int i = 0; i < 4; i++)
    {
        pRow[i * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i]))));
        pRow[i * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 8))));
        pRow[i * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 16))));
        pRow[i * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 24))));
    }
    for (int k = 4; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
        pRow[k * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 8))));
        pRow[k * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 16))));
        pRow[k * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 24))));
    }
    for (int k = rowKernelLoopLimit; k < 7; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

// load function for 9x9 kernel size
inline void rpp_load_filter_9x9_pkd_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        pRow[i * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i]))));
        pRow[i * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 8))));
        pRow[i * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 16))));
        pRow[i * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 24))));
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
        pRow[k * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 8))));
        pRow[k * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 16))));
        pRow[k * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 24))));
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
    }
}

inline void rpp_load_gaussian_filter_9x9_pkd_pln_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    for(int i = 0; i < 5; i++)
    {
        pRow[i * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i]))));
        pRow[i * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 8))));
        pRow[i * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 16))));
        pRow[i * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 24))));
        pRow[i * 4 + 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[i] + 32))));
    }
    for (int k = 5; k < rowKernelLoopLimit; k++)
    {
        pRow[k * 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
        pRow[k * 4 + 1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 8))));
        pRow[k * 4 + 2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 16))));
        pRow[k * 4 + 3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 24))));
        pRow[k * 4 + 4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k] + 32))));
    }
    for (int k = rowKernelLoopLimit; k < 9; k++)
    {
        pRow[k * 4] = avx_p0;
        pRow[k * 4 + 1] = avx_p0;
        pRow[k * 4 + 2] = avx_p0;
        pRow[k * 4 + 3] = avx_p0;
        pRow[k * 4 + 4] = avx_p0;
    }
}

#endif //RPP_CPU_FILTER_HPP