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

inline void get_kernel_loop_limit(Rpp32s index, Rpp32s &loopLimit, Rpp32u kernelSize, Rpp32u padLength, Rpp32u length)
{
    if ((index < padLength) || (index >= (length - padLength)))
    {
        Rpp32u factor = (index < padLength) ? index : (length - 1 - index);
        loopLimit = kernelSize - padLength + factor;
    }
}

inline void extract_4sse_registers(__m256i &pxLower, __m256i &pxUpper, __m128i *px128)
{
    px128[0] =  _mm256_castsi256_si128(pxLower);
    px128[1] =  _mm256_castsi256_si128(pxUpper);
    px128[2] =  _mm256_extracti128_si256(pxLower, 1);
    px128[3] =  _mm256_extracti128_si256(pxUpper, 1);
}

inline void extract_3sse_registers(__m256i &pxLower, __m256i &pxUpper, __m128i *px128)
{
    px128[0] =  _mm256_castsi256_si128(pxLower);
    px128[1] =  _mm256_castsi256_si128(pxUpper);
    px128[2] =  _mm256_extracti128_si256(pxLower, 1);
}

inline void blend_shuffle_add_3x3_pln_host(__m128i &pxLower1, __m128i &pxLower2)
{
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
}

inline void blend_shuffle_add_3x3_pkd_host(__m128i &pxLower1, __m128i &pxLower2)
{
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
}

inline void blend_shuffle_add_5x5_pln_host(__m128i *px128)
{
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 15), xmm_pxMaskRotate0To7);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
}

inline void blend_shuffle_add_5x5_pkd_host(__m128i *px128)
{
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 1), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 15), xmm_pxMaskRotate0To7);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
}

inline void blend_shuffle_add_7x7_pln_host(__m128i *px128)
{
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 63), xmm_pxMaskRotate0To11);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[4]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[5]);
}

inline void blend_shuffle_add_7x7_pkd_host(__m128i *px128)
{
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 1), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 127), xmm_pxMaskRotate0To13);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[2], px128[3], 3), xmm_pxMaskRotate0To3);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[4]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[5]);
}

inline void blend_shuffle_add_9x9_pln_host(__m128i *px128)
{
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 63), xmm_pxMaskRotate0To11);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 127), xmm_pxMaskRotate0To13);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[4]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[5]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[6]);
    px128[0] = _mm_add_epi16(px128[0], px128[1]);
}
inline void blend_shuffle_add_9x9_pkd_host(__m128i *px128)
{
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 1), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 127), xmm_pxMaskRotate0To13);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[2], px128[3], 3), xmm_pxMaskRotate0To3);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[2], px128[3], 31), xmm_pxMaskRotate0To9);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[4]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[5]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[6]);
    px128[0] = _mm_add_epi16(px128[0], px128[3]);
}

inline void blend_permute_add_3x3_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_3x3_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_5x5_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_5x5_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 1), avx_pxMaskRotate0To1));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_7x7_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 31), avx_pxMaskRotate0To5));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_7x7_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 1), avx_pxMaskRotate0To1));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 127), avx_pxMaskRotate0To7));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[2], pSrc[3], 3), avx_pxMaskRotate0To2));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_9x9_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    __m256 pTemp;
    pTemp = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 15), avx_pxMaskRotate0To4));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 31), avx_pxMaskRotate0To5));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 127), avx_pxMaskRotate0To7));
    pTemp = _mm256_add_ps(pTemp, pSrc[1]);
    pDst[0] = _mm256_mul_ps(pTemp, pConvolutionFactor);
}

inline void blend_permute_add_9x9_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    __m256 pTemp;
    pTemp = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 1), avx_pxMaskRotate0To1));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 15), avx_pxMaskRotate0To4));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 127), avx_pxMaskRotate0To7));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[2], pSrc[3], 3), avx_pxMaskRotate0To2));
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[2], pSrc[3], 31), avx_pxMaskRotate0To5));
    pTemp = _mm256_add_ps(pTemp, pSrc[3]);
    pDst[0] = _mm256_mul_ps(pTemp, pConvolutionFactor);
}

// -------------------- Set 1 box_filter load functions --------------------

// 3x3 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_char_3x3_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
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

// 5x5 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_char_5x5_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
        pxRow[k] = avx_px0;
}

// 7x7 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_char_7x7_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    pxRow[3] = _mm256_loadu_si256((__m256i *)srcPtrTemp[3]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
        pxRow[k] = avx_px0;
}

// 9x9 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_char_9x9_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    pxRow[3] = _mm256_loadu_si256((__m256i *)srcPtrTemp[3]);
    pxRow[4] = _mm256_loadu_si256((__m256i *)srcPtrTemp[4]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
        pxRow[k] = avx_px0;
}

// 3x3 kernel loads for I8 bitdepth
inline void rpp_load_box_filter_char_3x3_host(__m256i *pxRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pxRow[0] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[0]));
    pxRow[1] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[1]));

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        pxRow[2] =  _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[2]));
    else
        pxRow[2] = avx_p0;
}

// 5x5 kernel loads for I8 bitdepth
inline void rpp_load_box_filter_char_5x5_host(__m256i *pxRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pxRow[0] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[0]));
    pxRow[1] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[1]));
    pxRow[2] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[2]));
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[k]));
    for (int k = rowKernelLoopLimit; k < 5; k++)
        pxRow[k] = avx_p0;
}

// 7x7 kernel loads for I8 bitdepth
inline void rpp_load_box_filter_char_7x7_host(__m256i *pxRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pxRow[0] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[0]));
    pxRow[1] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[1]));
    pxRow[2] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[2]));
    pxRow[3] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[3]));
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[k]));
    for (int k = rowKernelLoopLimit; k < 7; k++)
        pxRow[k] = avx_p0;
}

// 9x9 kernel loads for I8 bitdepth
inline void rpp_load_box_filter_char_9x9_host(__m256i *pxRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pxRow[0] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[0]));
    pxRow[1] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[1]));
    pxRow[2] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[2]));
    pxRow[3] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[3]));
    pxRow[4] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[4]));
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[k]));
    for (int k = rowKernelLoopLimit; k < 9; k++)
        pxRow[k] = avx_p0;
}

// 3x3 kernel loads for F32 bitdepth
inline void rpp_load_box_filter_float_3x3_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    else
        pRow[2] = avx_px0;
}

// 5x5 kernel loads for F32 bitdepth
inline void rpp_load_box_filter_float_5x5_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
        pRow[k] = avx_p0;
}

// 7x7 kernel loads for F32 bitdepth
inline void rpp_load_box_filter_float_7x7_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[3]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
        pRow[k] = avx_p0;
}

// 9x9 kernel loads for F32 bitdepth
inline void rpp_load_box_filter_float_9x9_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[3]);
    pRow[4] = _mm256_loadu_ps(srcPtrTemp[4]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
        pRow[k] = avx_p0;
}

// 3x3 kernel loads for F16 bitdepth
inline void rpp_load_box_filter_float_3x3_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0]))));
    pRow[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]))));

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2]))));
    else
        pRow[2] = avx_px0;
}

// 5x5 kernel loads for F16 bitdepth
inline void rpp_load_box_filter_float_5x5_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pRow[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0]))));
    pRow[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]))));
    pRow[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2]))));
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
    for (int k = rowKernelLoopLimit; k < 5; k++)
        pRow[k] = avx_p0;
}

// 7x7 kernel loads for F16 bitdepth
inline void rpp_load_box_filter_float_7x7_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pRow[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0]))));
    pRow[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]))));
    pRow[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2]))));
    pRow[3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[3]))));
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
    for (int k = rowKernelLoopLimit; k < 7; k++)
        pRow[k] = avx_p0;
}

// 9x9 kernel loads for F16 bitdepth
inline void rpp_load_box_filter_float_9x9_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pRow[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0]))));
    pRow[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]))));
    pRow[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2]))));
    pRow[3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[3]))));
    pRow[4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[4]))));
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[k]))));
    for (int k = rowKernelLoopLimit; k < 9; k++)
        pRow[k] = avx_p0;
}

#endif //RPP_CPU_FILTER_HPP