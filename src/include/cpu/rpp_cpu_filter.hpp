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

#include "rpp_cpu_simd.hpp"

// declare masks used for shuffle and permute operations used in filter functions
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

// increment the kernel size number of row pointers with increment value
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

// extract 4 SSE registers from 2 AVX registers
inline void extract_4sse_registers(__m256i *pxRowHalf, __m128i *px128)
{
    px128[0] =  _mm256_castsi256_si128(pxRowHalf[0]);
    px128[1] =  _mm256_castsi256_si128(pxRowHalf[1]);
    px128[2] =  _mm256_extracti128_si256(pxRowHalf[0], 1);
    px128[3] =  _mm256_extracti128_si256(pxRowHalf[1], 1);
}

// extract 3 SSE registers from 2 AVX registers
inline void extract_3sse_registers(__m256i *pxRowHalf, __m128i *px128)
{
    px128[0] =  _mm256_castsi256_si128(pxRowHalf[0]);
    px128[1] =  _mm256_castsi256_si128(pxRowHalf[1]);
    px128[2] =  _mm256_extracti128_si256(pxRowHalf[0], 1);
}

// -------------------- U8/I8 bitdepth compute functions for kernel size (3/5/7/9) --------------------

// perform required blend shuffle add operations for 3x3 kernel size
template <int blendMask1, int blendMask2> 
inline void blend_shuffle_add_3x3_host(__m128i *px128, __m128i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                             | For PKD inputs
        px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] - [B03|R04|G04|B04|R05|G05|B05|R06]
        pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle)
        pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) */ 
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1), pxMask[0]);    
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2), pxMask[1]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]);
}

// perform required blend shuffle add operations for 5x5 kernel size
template<int blendMask1, int blendMask2, int blendMask3, int blendMask4> 
inline void blend_shuffle_add_5x5_host(__m128i *px128, __m128i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                             | For PKD inputs
        px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] - [B03|R04|G04|B04|R05|G05|B05|R06]
        pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08]
        pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle)
        pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) 
        pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and shuffle)
                                                                                                   | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and shuffle) */ 
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1), pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2), pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3), pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4), pxMask[3]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),  pxTemp[3]);
}

// perform required blend shuffle add operations for 7x7 kernel size
template<int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5, int blendMask6>
inline void blend_shuffle_add_7x7_host(__m128i *px128, __m128i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                             | For PKD inputs
        px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] - [B03|R04|G04|B04|R05|G05|B05|R06],
        pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] - [R09|G09|B09|R10|G10|B10|R11|G11]
        pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle)
        pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) 
        pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and shuffle)
        pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and shuffle)
        pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and shuffle)
                                                                                                   | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and shuffle) */ 
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1), pxMask[0]);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2), pxMask[1]);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3), pxMask[2]);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4), pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5), pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6), pxMask[5]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[3]), pxTemp[4]), pxTemp[5]);
}

// perform required blend shuffle add operations for 9x9 kernel size
template<int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5, int blendMask6, int blendMask7>
inline void blend_shuffle_add_9x9_host(__m128i *px128, __m128i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                             | For PKD inputs
        px128[0] -  [X01|X02|X03|X04|X05|X06|X07|X08], px128[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| px128[0]  - [R01|G01|B01|R02|G02|B02|R03|G03], px128[1] - [B03|R04|G04|B04|R05|G05|B05|R06],
        pxTemp[0] - [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and shuffle)    | px128[2] -  [G06|B06|R07|G07|B07|R08|G08|B08], px128[3] - [R09|G09|B09|R10|G10|B10|R11|G11]
        pxTemp[1] - [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and shuffle)    | pxTemp[0] - [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and shuffle)
        pxTemp[2] - [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and shuffle)    | pxTemp[1] - [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and shuffle) 
        pxTemp[3] - [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and shuffle)    | pxTemp[2] - [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and shuffle)
        pxTemp[4] - [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and shuffle)    | pxTemp[3] - [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and shuffle)
        pxTemp[5] - [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and shuffle)    | pxTemp[4] - [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and shuffle)
        pxTemp[6] - [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and shuffle)    | pxTemp[5] - [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and shuffle)
                                                                                                   | pxTemp[6] - [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and shuffle) */ 
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[0]], px128[index[0] + 1], blendMask1), pxMask[0]);    
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[1]], px128[index[1] + 1], blendMask2), pxMask[1]);    
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[2]], px128[index[2] + 1], blendMask3), pxMask[2]);    
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[3]], px128[index[3] + 1], blendMask4), pxMask[3]);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[4]], px128[index[4] + 1], blendMask5), pxMask[4]);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[5]], px128[index[5] + 1], blendMask6), pxMask[5]);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[index[6]], px128[index[6] + 1], blendMask7), pxMask[6]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[0]), pxTemp[1]), pxTemp[2]),  pxTemp[3]);
    px128[0] = _mm_add_epi16(_mm_add_epi16(_mm_add_epi16(_mm_add_epi16(px128[0], pxTemp[4]), pxTemp[5]), pxTemp[6]),  px128[index[6] + 1]);
}

// -------------------- F32/F16 bitdepth compute functions for kernel size (3/5/7/9) --------------------

// perform required blend permute add multiplication operations for 3x3 kernel size
template <int blendMask1, int blendMask2> 
inline void blend_permute_add_mul_3x3_host(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor, __m256i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                          | For PKD inputs
        pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] - [B03|R04|G04|B04|R05|G05|B05|R06]
                  [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                  [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute) */ 
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));   
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// perform required blend permute add multiplication operations for 5x5 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4> 
inline void blend_permute_add_mul_5x5_host(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor, __m256i *pxMask, Rpp32u *index)
{
   /*   For PLN inputs                                                                          | For PKD inputs
        pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] - [B03|R04|G04|B04|R05|G05|B05|R06]
                  [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08]
                  [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                  [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute) 
                  [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |           [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                                                                                                |           [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute) */ 
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));   // blend with mask [0000 0001] and permute - [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));   // blend with mask [0000 0011] and permute - [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3), pxMask[2]));   // blend with mask [0000 0111] and permute - [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4), pxMask[3]));  // blend with mask [0000 1111] and permute - [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// perform required blend permute add multiplication operations for 7x7 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5, int blendMask6>  
inline void blend_permute_add_mul_7x7_host(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor, __m256i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                          | For PKD inputs
        pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] - [B03|R04|G04|B04|R05|G05|B05|R06],
                  [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and permute)   | pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] - [R09|G09|B09|R10|G10|B10|R11|G11]
                  [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                  [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute) 
                  [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |           [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |           [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |           [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute)
                                                                                                |           [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute) */ 
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));   // blend with mask [0000 0001] and permute - [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));   // blend with mask [0000 0011] and permute - [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3), pxMask[2]));   // blend with mask [0000 0111] and permute - [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4), pxMask[3]));  // blend with mask [0000 1111] and permute - [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5), pxMask[4]));  // blend with mask [0001 1111] and permute - [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6), pxMask[5]));  // blend with mask [0011 1111] and permute - [X07|X08|X09|X10|X11|X12|X13|X14]
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// perform required blend permute add multiplication operations for 9x9 kernel size
template <int blendMask1, int blendMask2, int blendMask3, int blendMask4, int blendMask5, int blendMask6, int blendMask7>  
inline void blend_permute_add_mul_9x9_host(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor, __m256i *pxMask, Rpp32u *index)
{
    /*  For PLN inputs                                                                          | For PKD inputs
        pSrc[0] - [X01|X02|X03|X04|X05|X06|X07|X08], pSrc[1] - [X09|X10|X11|X12|X13|X14|X15|X16]| pSrc[0] - [R01|G01|B01|R02|G02|B02|R03|G03], pSrc[1] - [B03|R04|G04|B04|R05|G05|B05|R06],
                  [X02|X03|X04|X05|X06|X07|X08|X09] (blend with mask [0000 0001] and permute)     pSrc[2] - [G06|B06|R07|G07|B07|R08|G08|B08], pSrc[3] - [R09|G09|B09|R10|G10|B10|R11|G11]
                  [X03|X04|X05|X06|X07|X08|X09|X10] (blend with mask [0000 0011] and permute)   |           [R02|G02|B02|R03|G03|B03|R04|G04] (blend with mask [0000 0111] and permute)
                  [X04|X05|X06|X07|X08|X09|X10|X11] (blend with mask [0000 0111] and permute)   |           [R03|G03|B03|R04|G04|B04|R05|G05] (blend with mask [0011 1111] and permute) 
                  [X05|X06|X07|X08|X09|X10|X11|X12] (blend with mask [0000 1111] and permute)   |           [R04|G04|B04|R05|G05|B05|R06|G06] (blend with mask [0000 0001] and permute)
                  [X06|X07|X08|X09|X10|X11|X12|X13] (blend with mask [0001 1111] and permute)   |           [R05|G05|B05|R06|G06|B06|R07|G07] (blend with mask [0000 1111] and permute)
                  [X07|X08|X09|X10|X11|X12|X13|X14] (blend with mask [0011 1111] and permute)   |           [R06|G06|B06|R07|G07|B07|R08|G08] (blend with mask [0111 1111] and permute)
                  [X08|X09|X10|X11|X12|X13|X14|X15] (blend with mask [0111 1111] and permute)   |           [R07|G07|B07|R08|G08|B08|R09|G09] (blend with mask [0000 0011] and permute)
                                                                                                |           [R08|G08|B08|R09|G09|B09|R10|G10] (blend with mask [0001 1111] and permute)
    */
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[0]], pSrc[index[0] + 1], blendMask1), pxMask[0]));   // blend with mask [0000 0001] and permute - [X02|X03|X04|X05|X06|X07|X08|X09]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[1]], pSrc[index[1] + 1], blendMask2), pxMask[1]));   // blend with mask [0000 0011] and permute - [X03|X04|X05|X06|X07|X08|X09|X10]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[2]], pSrc[index[2] + 1], blendMask3), pxMask[2]));   // blend with mask [0000 0111] and permute - [X04|X05|X06|X07|X08|X09|X10|X11]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[3]], pSrc[index[3] + 1], blendMask4), pxMask[3]));   // blend with mask [0000 1111] and permute - [X05|X06|X07|X08|X09|X10|X11|X12]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[4]], pSrc[index[4] + 1], blendMask5), pxMask[4]));   // blend with mask [0001 1111] and permute - [X06|X07|X08|X09|X10|X11|X12|X13]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[5]], pSrc[index[5] + 1], blendMask6), pxMask[5]));   // blend with mask [0011 1111] and permute - [X07|X08|X09|X10|X11|X12|X13|X14]
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[index[6]], pSrc[index[6] + 1], blendMask7), pxMask[6]));   // blend with mask [0111 1111] and permute - [X08|X09|X10|X11|X12|X13|X14|X15]
    pDst[0] = _mm256_add_ps(pDst[0], pSrc[index[6] + 1]);
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// -------------------- Filter load functions for U8 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_box_filter_char_3x3_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    if (rowKernelLoopLimit == 3)
        pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    else
        pxRow[2] = avx_px0;
}

// load function for 5x5 kernel size
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

// load function for 7x7 kernel size
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

// load function for 9x9 kernel size
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

// -------------------- Filter load functions for I8 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_box_filter_char_3x3_host(__m256i *pxRow, Rpp8s **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pxRow[0] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[0]));
    pxRow[1] = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[1]));
    if (rowKernelLoopLimit == 3)
        pxRow[2] =  _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtrTemp[2]));
    else
        pxRow[2] = avx_p0;
}

// load function for 5x5 kernel size
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

// load function for 7x7 kernel size
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

// load function for 9x9 kernel size
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

// -------------------- Filter load functions for F32 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_box_filter_float_3x3_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    else
        pRow[2] = avx_px0;
}

// load function for 5x5 kernel size
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

// load function for 7x7 kernel size
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

// load function for 9x9 kernel size
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

// -------------------- Filter load functions for F16 bitdepth --------------------

// load function for 3x3 kernel size
inline void rpp_load_box_filter_float_3x3_host(__m256 *pRow, Rpp16f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[0]))));
    pRow[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[1]))));
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtrTemp[2]))));
    else
        pRow[2] = avx_px0;
}

// load function for 5x5 kernel size
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

// load function for 7x7 kernel size
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

// load function for 9x9 kernel size
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