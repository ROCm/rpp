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

#ifndef AMD_RPP_RPP_CPU_SIMD_HPP
#define AMD_RPP_RPP_CPU_SIMD_HPP

#include "stdio.h"
#include "rppdefs.h"

#if _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

#define M256I(m256i_register) (*((_m256i_union*)&m256i_register))
typedef union
{
    char m256i_i8[32];
    short m256i_i16[16];
    int m256i_i32[8];
    long long m256i_i64[4];
    __m128i m256i_i128[2];
} _m256i_union;

#if defined(_MSC_VER)
#define SIMD_ALIGN_VAR(type, name, alignment) \
    __declspec(align(alignment)) type name
#else
#define SIMD_ALIGN_VAR(type, name, alignment) \
    type __attribute__((__aligned__(alignment))) name
#endif // _MSC_VER

#define SIMD_CONST_PI(name, val0, val1, val2, val3) \
    SIMD_ALIGN_VAR(static const int, _xmm_const_##name[4], 16) = { \
        static_cast<int>(val3), \
        static_cast<int>(val2), \
        static_cast<int>(val1), \
        static_cast<int>(val0)  \
    }

#define SIMD_CONST_PS(name, val0, val1, val2, val3) \
    SIMD_ALIGN_VAR(static const float, _xmm_const_##name[4], 16) = { \
        static_cast<float>(val3), \
        static_cast<float>(val2), \
        static_cast<float>(val1), \
        static_cast<float>(val0)  \
    }

#define SIMD_GET_PS(name) (*(const __m128  *)_xmm_const_##name)

const __m128 xmm_p0 = _mm_setzero_ps();
const __m128 xmm_p1 = _mm_set1_ps(1.0f);
const __m128 xmm_p2 = _mm_set1_ps(2.0f);
const __m128 xmm_pm2 = _mm_set1_ps(-2.0f);
const __m128 xmm_p3 = _mm_set1_ps(3.0f);
const __m128 xmm_p4 = _mm_set1_ps(4.0f);
const __m128 xmm_p6 = _mm_set1_ps(6.0f);
const __m128 xmm_p16 = _mm_set1_ps(16.0f);
const __m128 xmm_p255 = _mm_set1_ps(255.0f);
const __m128 xmm_p1op255 = _mm_set1_ps(1.0f / 255.0f);
const __m128 xmm_p1op3 = _mm_set1_ps(1.0f / 3.0f);
const __m128 xmm_p2op3 = _mm_set1_ps(2.0f / 3.0f);

const __m128 xmm_cephesSQRTHF = _mm_set1_ps(0.707106781186547524);
const __m128 xmm_cephesLogP0 = _mm_set1_ps(7.0376836292E-2);
const __m128 xmm_cephesLogP1 = _mm_set1_ps(-1.1514610310E-1);
const __m128 xmm_cephesLogP2 = _mm_set1_ps(1.1676998740E-1);
const __m128 xmm_cephesLogP3 = _mm_set1_ps(-1.2420140846E-1);
const __m128 xmm_cephesLogP4 = _mm_set1_ps(1.4249322787E-1);
const __m128 xmm_cephesLogP5 = _mm_set1_ps(-1.6668057665E-1);
const __m128 xmm_cephesLogP6 = _mm_set1_ps(2.0000714765E-1);
const __m128 xmm_cephesLogP7 = _mm_set1_ps(-2.4999993993E-1);
const __m128 xmm_cephesLogP8 = _mm_set1_ps(3.3333331174E-1);
const __m128 xmm_cephesLogQ1 = _mm_set1_ps(-2.12194440e-4);
const __m128 xmm_cephesLogQ2 = _mm_set1_ps(0.693359375);

const __m128i xmm_px0 = _mm_set1_epi32(0);
const __m128i xmm_px1 = _mm_set1_epi32(1);
const __m128i xmm_px2 = _mm_set1_epi32(2);
const __m128i xmm_px3 = _mm_set1_epi32(3);
const __m128i xmm_px4 = _mm_set1_epi32(4);
const __m128i xmm_px5 = _mm_set1_epi32(5);
const __m128i xmm_pxConvertI8 = _mm_set1_epi8((char)128);
const __m128 xmm_pDstLocInit = _mm_setr_ps(0, 1, 2, 3);

const __m256 avx_p0 = _mm256_set1_ps(0.0f);
const __m256 avx_p1 = _mm256_set1_ps(1.0f);
const __m256 avx_p2 = _mm256_set1_ps(2.0f);
const __m256 avx_pm2 = _mm256_set1_ps(-2.0f);
const __m256 avx_p3 = _mm256_set1_ps(3.0f);
const __m256 avx_p4 = _mm256_set1_ps(4.0f);
const __m256 avx_p6 = _mm256_set1_ps(6.0f);
const __m256 avx_p8 = _mm256_set1_ps(8.0f);
const __m256 avx_p128 = _mm256_set1_ps(128.0f);
const __m256 avx_p255 = _mm256_set1_ps(255.0f);
const __m256 avx_p1op255 = _mm256_set1_ps(1.0f / 255.0f);
const __m256 avx_p1op3 = _mm256_set1_ps(1.0f / 3.0f);
const __m256 avx_p2op3 = _mm256_set1_ps(2.0f / 3.0f);

const __m256i avx_cephesSQRTHF = _mm256_set1_ps(0.707106781186547524);
const __m256i avx_cephesLogP0 = _mm256_set1_ps(7.0376836292E-2);
const __m256i avx_cephesLogP1 = _mm256_set1_ps(-1.1514610310E-1);
const __m256i avx_cephesLogP2 = _mm256_set1_ps(1.1676998740E-1);
const __m256i avx_cephesLogP3 = _mm256_set1_ps(-1.2420140846E-1);
const __m256i avx_cephesLogP4 = _mm256_set1_ps(1.4249322787E-1);
const __m256i avx_cephesLogP5 = _mm256_set1_ps(-1.6668057665E-1);
const __m256i avx_cephesLogP6 = _mm256_set1_ps(2.0000714765E-1);
const __m256i avx_cephesLogP7 = _mm256_set1_ps(-2.4999993993E-1);
const __m256i avx_cephesLogP8 = _mm256_set1_ps(3.3333331174E-1);
const __m256i avx_cephesLogQ1 = _mm256_set1_ps(-2.12194440e-4);
const __m256i avx_cephesLogQ2 = _mm256_set1_ps(0.693359375);

const __m256i avx_px0 = _mm256_set1_epi32(0);
const __m256i avx_px1 = _mm256_set1_epi32(1);
const __m256i avx_px2 = _mm256_set1_epi32(2);
const __m256i avx_px3 = _mm256_set1_epi32(3);
const __m256i avx_px4 = _mm256_set1_epi32(4);
const __m256i avx_px5 = _mm256_set1_epi32(5);
const __m256i avx_pxConvertI8 = _mm256_set1_epi8((char)128);
const __m256 avx_pDstLocInit = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);

const __m128i xmm_pxMask00To03 = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask04To07 = _mm_setr_epi8(4, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask08To11 = _mm_setr_epi8(8, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 11, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask12To15 = _mm_setr_epi8(12, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 15, 0x80, 0x80, 0x80);

const __m128i xmm_pxMask00To02 = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask03To05 = _mm_setr_epi8(3, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask06To08 = _mm_setr_epi8(6, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask09To11 = _mm_setr_epi8(9, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask08To15 = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

const __m128i xmm_pxMask03To00 = _mm_setr_epi8(3, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80, 0, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask07To04 = _mm_setr_epi8(7, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask11To08 = _mm_setr_epi8(11, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80);
const __m128i xmm_pxMask15To12 = _mm_setr_epi8(15, 0x80, 0x80, 0x80, 14, 0x80, 0x80, 0x80, 13, 0x80, 0x80, 0x80, 12, 0x80, 0x80, 0x80);

const __m128i xmm_pxMaskR = _mm_setr_epi8(0, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 9, 0x80, 0x80, 0x80);
const __m128i xmm_pxMaskG = _mm_setr_epi8(1, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 10, 0x80, 0x80, 0x80);
const __m128i xmm_pxMaskB = _mm_setr_epi8(2, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 11, 0x80, 0x80, 0x80);

const __m128i xmm_pxMaskRMirror = _mm_setr_epi8(9, 0x80, 0x80, 0x80, 6, 0x80, 0x80, 0x80, 3, 0x80, 0x80, 0x80, 0, 0x80, 0x80, 0x80);
const __m128i xmm_pxMaskGMirror = _mm_setr_epi8(10, 0x80, 0x80, 0x80, 7, 0x80, 0x80, 0x80, 4, 0x80, 0x80, 0x80, 1, 0x80, 0x80, 0x80);
const __m128i xmm_pxMaskBMirror = _mm_setr_epi8(11, 0x80, 0x80, 0x80, 8, 0x80, 0x80, 0x80, 5, 0x80, 0x80, 0x80, 2, 0x80, 0x80, 0x80);

const __m128i xmm_char_maskR = _mm_setr_epi8(0, 3, 6, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_char_maskG = _mm_setr_epi8(1, 4, 7, 10, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_char_maskB = _mm_setr_epi8(2, 5, 8, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_pkd_mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_store4_pkd_pixels = _mm_setr_epi8(0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11, 0x80, 0x80, 0x80, 0x80);
const __m256i avx_store8_pkd_pixels = _mm256_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 23, 15, 14, 22, 13, 12, 21, 11, 10, 20, 9, 8, 19, 7, 6, 18, 5, 4, 17, 3, 2, 16, 1, 0);

const __m128i xmm_pxStore4Pkd = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 0x80, 0x80, 0x80, 0x80);
const __m256i avx_pxPermPkd = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 3);
const __m256i avx_pxShufflePkd = _mm256_setr_m128(xmm_pxStore4Pkd, xmm_pxStore4Pkd);

const __m128i xmm_pxMask00 = _mm_setr_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0, 1, 2, 3);
const __m128i xmm_pxMask04To11 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

const __m256i avx_pxMaskR = _mm256_setr_epi8(0, 0x80, 0x80, 3, 0x80, 0x80, 6, 0x80, 0x80, 9, 0x80, 0x80, 12, 0x80, 0x80, 15, 0x80, 0x80, 18, 0x80, 0x80, 21, 0x80, 0x80, 24, 0x80, 0x80, 27, 0x80, 0x80, 0x80, 0x80);
const __m256i avx_pxMaskG = _mm256_setr_epi8(0x80, 1, 0x80, 0x80, 4, 0x80, 0x80, 7, 0x80, 0x80, 10, 0x80, 0x80, 13, 0x80, 0x80, 16, 0x80, 0x80, 19, 0x80, 0x80, 22, 0x80, 0x80, 25, 0x80, 0x80, 28, 0x80, 0x80, 0x80);
const __m256i avx_pxMaskB = _mm256_setr_epi8(0x80, 0x80, 2, 0x80, 0x80, 5, 0x80, 0x80, 8, 0x80, 0x80, 11, 0x80, 0x80, 14, 0x80, 0x80, 17, 0x80, 0x80, 20, 0x80, 0x80, 23, 0x80, 0x80, 26, 0x80, 0x80, 29, 0x80, 0x80);

// Print helpers

inline void rpp_mm_print_epi8(__m128i vPrintArray)
{
    char printArray[16];
    _mm_storeu_si128((__m128i *)printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 16; ct++)
    {
        printf("%d ", printArray[ct]);
    }
}

inline void rpp_storeu_si32(void *__p,
                        	__m128i __b) {
  struct __storeu_si32 {
    int __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_si32 *)__p)->__v = ((__v4si)__b)[0];
}

inline void rpp_storeu_si64(void *__p,
                            __m128i __b) {
  struct __storeu_si64 {
    long long __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_si64 *)__p)->__v = ((__v2di)__b)[0];
}

inline void rpp_mm_print_epi32(__m128i vPrintArray)
{
    int printArray[4];
    _mm_storeu_si128((__m128i *)printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 4; ct++)
    {
        printf("%d ", printArray[ct]);
    }
}

inline void rpp_mm_print_epi16(__m128i vPrintArray)
{
    unsigned short int printArray[8];
    _mm_storeu_si128((__m128i *)printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 8; ct++)
    {
        printf("%hu ", printArray[ct]);
    }
}

inline void rpp_mm_print_ps(__m128 vPrintArray)
{
    float printArray[4];
    _mm_storeu_ps(printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 4; ct++)
    {
        printf("%0.6f ", printArray[ct]);
    }
}

inline void rpp_mm256_print_epi8(__m256i vPrintArray)
{
    unsigned char printArray[32];
    _mm256_storeu_si256((__m256i *)printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 32; ct++)
    {
        printf("%d ", (unsigned char)printArray[ct]);
    }
}

inline void rpp_mm256_print_epi32(__m256i vPrintArray)
{
    int printArray[8];
    _mm256_storeu_si256((__m256i *)printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 8; ct++)
    {
        printf("%d ", printArray[ct]);
    }
}

inline void rpp_mm256_print_epi16(__m256i vPrintArray)
{
    unsigned short int printArray[8];
    _mm256_storeu_si256((__m256i *)printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 16; ct++)
    {
        printf("%hu ", printArray[ct]);
    }
}

inline void rpp_mm256_print_ps(__m256 vPrintArray)
{
    float printArray[8];
    _mm256_storeu_ps(printArray, vPrintArray);
    printf("\n");
    for (int ct = 0; ct < 8; ct++)
    {
        printf("%0.6f ", printArray[ct]);
    }
}

inline void rpp_saturate64_0to1_avx(__m256 *p)
{
    p[0] = _mm256_min_ps(_mm256_max_ps(p[0], avx_p0), avx_p1);
    p[1] = _mm256_min_ps(_mm256_max_ps(p[1], avx_p0), avx_p1);
    p[2] = _mm256_min_ps(_mm256_max_ps(p[2], avx_p0), avx_p1);
    p[3] = _mm256_min_ps(_mm256_max_ps(p[3], avx_p0), avx_p1);
    p[4] = _mm256_min_ps(_mm256_max_ps(p[4], avx_p0), avx_p1);
    p[5] = _mm256_min_ps(_mm256_max_ps(p[5], avx_p0), avx_p1);
    p[6] = _mm256_min_ps(_mm256_max_ps(p[6], avx_p0), avx_p1);
    p[7] = _mm256_min_ps(_mm256_max_ps(p[7], avx_p0), avx_p1);
}

inline void rpp_saturate48_0to1_avx(__m256 *p)
{
    p[0] = _mm256_min_ps(_mm256_max_ps(p[0], avx_p0), avx_p1);
    p[1] = _mm256_min_ps(_mm256_max_ps(p[1], avx_p0), avx_p1);
    p[2] = _mm256_min_ps(_mm256_max_ps(p[2], avx_p0), avx_p1);
    p[3] = _mm256_min_ps(_mm256_max_ps(p[3], avx_p0), avx_p1);
    p[4] = _mm256_min_ps(_mm256_max_ps(p[4], avx_p0), avx_p1);
    p[5] = _mm256_min_ps(_mm256_max_ps(p[5], avx_p0), avx_p1);
}

inline void rpp_saturate24_0to1_avx(__m256 *p)
{
    p[0] = _mm256_min_ps(_mm256_max_ps(p[0], avx_p0), avx_p1);
    p[1] = _mm256_min_ps(_mm256_max_ps(p[1], avx_p0), avx_p1);
    p[2] = _mm256_min_ps(_mm256_max_ps(p[2], avx_p0), avx_p1);
}

inline void rpp_saturate16_0to1_avx(__m256 *p)
{
    p[0] = _mm256_min_ps(_mm256_max_ps(p[0], avx_p0), avx_p1);
    p[1] = _mm256_min_ps(_mm256_max_ps(p[1], avx_p0), avx_p1);
}

inline void rpp_saturate8_0to1_avx(__m256 *p)
{
    p[0] = _mm256_min_ps(_mm256_max_ps(p[0], avx_p0), avx_p1);
}

// SSE loads and stores

inline void rpp_load48_u8pkd3_to_f32pln3(Rpp8u *srcPtr, __m128 *p)
{
    __m128i px[4];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMaskR));    /* Contains R01-04 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMaskR));    /* Contains R05-08 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMaskR));    /* Contains R09-12 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[3], xmm_pxMaskR));    /* Contains R13-16 */
    p[4] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMaskG));    /* Contains G01-04 */
    p[5] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMaskG));    /* Contains G05-08 */
    p[6] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMaskG));    /* Contains G09-12 */
    p[7] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[3], xmm_pxMaskG));    /* Contains G13-16 */
    p[8] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMaskB));    /* Contains B01-04 */
    p[9] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMaskB));    /* Contains B05-08 */
    p[10] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMaskB));    /* Contains B09-12 */
    p[11] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[3], xmm_pxMaskB));    /* Contains B13-16 */
}

inline void rpp_store48_f32pln3_to_u8pln3(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m128 *p)
{
    __m128i px[8];

    px[4] = _mm_cvtps_epi32(p[0]);    /* convert to int32 for R */
    px[5] = _mm_cvtps_epi32(p[1]);    /* convert to int32 for R */
    px[6] = _mm_cvtps_epi32(p[2]);    /* convert to int32 for R */
    px[7] = _mm_cvtps_epi32(p[3]);    /* convert to int32 for R */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 for R */
    px[5] = _mm_packus_epi32(px[6], px[7]);    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 for R */
    px[4] = _mm_cvtps_epi32(p[4]);    /* convert to int32 for G */
    px[5] = _mm_cvtps_epi32(p[5]);    /* convert to int32 for G */
    px[6] = _mm_cvtps_epi32(p[6]);    /* convert to int32 for G */
    px[7] = _mm_cvtps_epi32(p[7]);    /* convert to int32 for G */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 for G */
    px[5] = _mm_packus_epi32(px[6], px[7]);    /* pack pixels 8-15 for G */
    px[1] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 for G */
    px[4] = _mm_cvtps_epi32(p[8]);    /* convert to int32 for B */
    px[5] = _mm_cvtps_epi32(p[9]);    /* convert to int32 for B */
    px[6] = _mm_cvtps_epi32(p[10]);    /* convert to int32 for B */
    px[7] = _mm_cvtps_epi32(p[11]);    /* convert to int32 for B */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 for B */
    px[5] = _mm_packus_epi32(px[6], px[7]);    /* pack pixels 8-15 for B */
    px[2] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 for B */
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_u8pln3_to_f32pln3(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m128 *p)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask00To03));    /* Contains R01-04 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask04To07));    /* Contains R05-08 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask08To11));    /* Contains R09-12 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask12To15));    /* Contains R13-16 */
    p[4] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask00To03));    /* Contains G01-04 */
    p[5] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask04To07));    /* Contains G05-08 */
    p[6] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask08To11));    /* Contains G09-12 */
    p[7] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask12To15));    /* Contains G13-16 */
    p[8] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask00To03));    /* Contains B01-04 */
    p[9] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask04To07));    /* Contains B05-08 */
    p[10] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask08To11));    /* Contains B09-12 */
    p[11] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask12To15));    /* Contains B13-16 */
}

inline void rpp_store48_f32pln3_to_u8pkd3(Rpp8u *dstPtr, __m128 *p)
{
    __m128i px[7];
    __m128i pxMask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);
    __m128i pxZero = _mm_setzero_si128();

    px[4] = _mm_cvtps_epi32(p[0]);    /* convert to int32 for R01-04 */
    px[5] = _mm_cvtps_epi32(p[4]);    /* convert to int32 for G01-04 */
    px[6] = _mm_cvtps_epi32(p[8]);    /* convert to int32 for B01-04 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R01-04|G01-04 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B01-04|X01-04 */
    px[0] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[1]);    /* convert to int32 for R05-08 */
    px[5] = _mm_cvtps_epi32(p[5]);    /* convert to int32 for G05-08 */
    px[6] = _mm_cvtps_epi32(p[9]);    /* convert to int32 for B05-08 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R05-08|G05-08 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B05-08|X01-04 */
    px[1] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[2]);    /* convert to int32 for R09-12 */
    px[5] = _mm_cvtps_epi32(p[6]);    /* convert to int32 for G09-12 */
    px[6] = _mm_cvtps_epi32(p[10]);    /* convert to int32 for B09-12 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R09-12|G09-12 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B09-12|X01-04 */
    px[2] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[3]);    /* convert to int32 for R13-16 */
    px[5] = _mm_cvtps_epi32(p[7]);    /* convert to int32 for G13-16 */
    px[6] = _mm_cvtps_epi32(p[11]);    /* convert to int32 for B13-16 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R13-16|G13-16 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B13-16|X01-04 */
    px[3] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00] */
    px[0] = _mm_shuffle_epi8(px[0], pxMask);    /* shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    px[1] = _mm_shuffle_epi8(px[1], pxMask);    /* shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    px[2] = _mm_shuffle_epi8(px[2], pxMask);    /* shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    px[3] = _mm_shuffle_epi8(px[3], pxMask);    /* shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px[1]);    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px[2]);    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px[3]);    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_load48_u8pkd3_to_u8pln3(Rpp8u *srcPtr, __m128i *px)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);    /* shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04 */
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);    /* shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08 */
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);    /* shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12 */
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);    /* shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16 */
    pxSrc[4] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 lo-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[5] = _mm_unpacklo_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 lo-pixels of pxSrc[2] and pxSrc[3] */
    pxSrc[6] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 hi-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[7] = _mm_unpackhi_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 hi-pixels of pxSrc[2] and pxSrc[3] */
    px[0] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);    /* unpack 8 lo-pixels of pxSrc[4] and pxSrc[5] to get R01-16 */
    px[1] = _mm_shuffle_epi8(_mm_unpackhi_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);    /* unpack 8 hi-pixels of pxSrc[4] and pxSrc[5] to get G01-16 */
    px[2] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[6], pxSrc[7]), pxMaskRGB);    /* unpack 8 lo-pixels of pxSrc[6] and pxSrc[7] to get B01-16 */
}

inline void rpp_store48_u8pln3_to_u8pln3(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m128i *px)
{
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_u8pln3_to_u8pln3(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m128i *px)
{
    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_store48_u8pln3_to_u8pkd3(Rpp8u *dstPtr, __m128i *px)
{
    __m128i pxDst[4];
    __m128i pxZero = _mm_setzero_si128();
    __m128i pxMaskRGBAtoRGB = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
    pxDst[0] = _mm_unpacklo_epi8(px[1], pxZero);
    pxDst[1] = _mm_unpackhi_epi8(px[1], pxZero);
    pxDst[2] = _mm_unpacklo_epi8(px[0], px[2]);
    pxDst[3] = _mm_unpackhi_epi8(px[0], px[2]);
    _mm_storeu_si128((__m128i *)dstPtr, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_load16_u8_to_f32(Rpp8u *srcPtr, __m128 *p)
{
    __m128i px = _mm_loadu_si128((__m128i *)srcPtr);    /* load pixels 0-15 */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask00To03));    /* pixels 0-3 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask04To07));    /* pixels 4-7 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask08To11));    /* pixels 8-11 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask12To15));    /* pixels 12-15 */
}

inline void rpp_store16_f32_to_u8(Rpp8u *dstPtr, __m128 *p)
{
    __m128i px[4];

    px[0] = _mm_cvtps_epi32(p[0]);    /* pixels 0-3 */
    px[1] = _mm_cvtps_epi32(p[1]);    /* pixels 4-7 */
    px[2] = _mm_cvtps_epi32(p[2]);    /* pixels 8-11 */
    px[3] = _mm_cvtps_epi32(p[3]);    /* pixels 12-15 */
    px[0] = _mm_packus_epi32(px[0], px[1]);    /* pixels 0-7 */
    px[1] = _mm_packus_epi32(px[2], px[3]);    /* pixels 8-15 */
    px[0] = _mm_packus_epi16(px[0], px[1]);    /* pixels 0-15 */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);    /* store pixels 0-15 */
}

inline void rpp_load16_f32_to_f32(Rpp32f *srcPtr, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtr);
    p[1] = _mm_loadu_ps(srcPtr + 4);
    p[2] = _mm_loadu_ps(srcPtr + 8);
    p[3] = _mm_loadu_ps(srcPtr + 12);
}

inline void rpp_load12_f32pkd3_to_f32pln3(Rpp32f *srcPtr, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtr);
    p[1] = _mm_loadu_ps(srcPtr + 3);
    p[2] = _mm_loadu_ps(srcPtr + 6);
    p[3] = _mm_loadu_ps(srcPtr + 9);
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], p[3]);
}

inline void rpp_store12_f32pln3_to_f32pln3(Rpp32f *dstPtrR, Rpp32f *dstPtrG, Rpp32f *dstPtrB, __m128 *p)
{
    _mm_storeu_ps(dstPtrR, p[0]);
    _mm_storeu_ps(dstPtrG, p[1]);
    _mm_storeu_ps(dstPtrB, p[2]);
}

inline void rpp_load12_f32pln3_to_f32pln3(Rpp32f *srcPtrR, Rpp32f *srcPtrG, Rpp32f *srcPtrB, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtrR);
    p[1] = _mm_loadu_ps(srcPtrG);
    p[2] = _mm_loadu_ps(srcPtrB);
}

inline void rpp_store12_f32pln3_to_f32pkd3(Rpp32f *dstPtr, __m128 *p)
{
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], p[3]);
    _mm_storeu_ps(dstPtr, p[0]);
    _mm_storeu_ps(dstPtr + 3, p[1]);
    _mm_storeu_ps(dstPtr + 6, p[2]);
    _mm_storeu_ps(dstPtr + 9, p[3]);
}

inline void rpp_load8_f32_to_f32(Rpp32f *srcPtr, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtr);
    p[1] = _mm_loadu_ps(srcPtr + 4);
}

inline void rpp_store8_f32_to_f32(Rpp32f *dstPtr, __m128 *p)
{
    _mm_storeu_ps(dstPtr, p[0]);
    _mm_storeu_ps(dstPtr + 4, p[1]);
}

inline void rpp_load4_f32_to_f32(Rpp32f *srcPtr, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcPtr);
}

inline void rpp_store4_f32_to_f32(Rpp32f *dstPtr, __m128 *p)
{
    _mm_storeu_ps(dstPtr, p[0]);
}

inline void rpp_load48_i8pkd3_to_f32pln3(Rpp8s *srcPtr, __m128 *p)
{
    __m128i px[4];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));           /* add I8 conversion param to load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 12)));    /* add I8 conversion param to load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 24)));    /* add I8 conversion param to load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 36)));    /* add I8 conversion param to load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMaskR));    /* Contains R01-04 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMaskR));    /* Contains R05-08 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMaskR));    /* Contains R09-12 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[3], xmm_pxMaskR));    /* Contains R13-16 */
    p[4] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMaskG));    /* Contains G01-04 */
    p[5] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMaskG));    /* Contains G05-08 */
    p[6] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMaskG));    /* Contains G09-12 */
    p[7] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[3], xmm_pxMaskG));    /* Contains G13-16 */
    p[8] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMaskB));    /* Contains B01-04 */
    p[9] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMaskB));    /* Contains B05-08 */
    p[10] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMaskB));    /* Contains B09-12 */
    p[11] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[3], xmm_pxMaskB));    /* Contains B13-16 */
}

inline void rpp_store48_f32pln3_to_i8pln3(Rpp8s *dstPtrR, Rpp8s *dstPtrG, Rpp8s *dstPtrB, __m128 *p)
{
    __m128i px[8];

    px[4] = _mm_cvtps_epi32(p[0]);    /* convert to int32 for R */
    px[5] = _mm_cvtps_epi32(p[1]);    /* convert to int32 for R */
    px[6] = _mm_cvtps_epi32(p[2]);    /* convert to int32 for R */
    px[7] = _mm_cvtps_epi32(p[3]);    /* convert to int32 for R */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 for R */
    px[5] = _mm_packus_epi32(px[6], px[7]);    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 for R */
    px[4] = _mm_cvtps_epi32(p[4]);    /* convert to int32 for G */
    px[5] = _mm_cvtps_epi32(p[5]);    /* convert to int32 for G */
    px[6] = _mm_cvtps_epi32(p[6]);    /* convert to int32 for G */
    px[7] = _mm_cvtps_epi32(p[7]);    /* convert to int32 for G */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 for G */
    px[5] = _mm_packus_epi32(px[6], px[7]);    /* pack pixels 8-15 for G */
    px[1] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 for G */
    px[4] = _mm_cvtps_epi32(p[8]);    /* convert to int32 for B */
    px[5] = _mm_cvtps_epi32(p[9]);    /* convert to int32 for B */
    px[6] = _mm_cvtps_epi32(p[10]);    /* convert to int32 for B */
    px[7] = _mm_cvtps_epi32(p[11]);    /* convert to int32 for B */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 for B */
    px[5] = _mm_packus_epi32(px[6], px[7]);    /* pack pixels 8-15 for B */
    px[2] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 for B */
    px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    px[1] = _mm_sub_epi8(px[1], xmm_pxConvertI8);    /* convert back to i8 for px1 store */
    px[2] = _mm_sub_epi8(px[2], xmm_pxConvertI8);    /* convert back to i8 for px2 store */
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_i8pln3_to_f32pln3(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m128 *p)
{
    __m128i px[3];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrR));    /* add I8 conversion param to load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrG));    /* add I8 conversion param to load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrB));    /* add I8 conversion param to load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask00To03));    /* Contains R01-04 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask04To07));    /* Contains R05-08 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask08To11));    /* Contains R09-12 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[0], xmm_pxMask12To15));    /* Contains R13-16 */
    p[4] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask00To03));    /* Contains G01-04 */
    p[5] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask04To07));    /* Contains G05-08 */
    p[6] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask08To11));    /* Contains G09-12 */
    p[7] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[1], xmm_pxMask12To15));    /* Contains G13-16 */
    p[8] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask00To03));    /* Contains B01-04 */
    p[9] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask04To07));    /* Contains B05-08 */
    p[10] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask08To11));    /* Contains B09-12 */
    p[11] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px[2], xmm_pxMask12To15));    /* Contains B13-16 */
}

inline void rpp_store48_f32pln3_to_i8pkd3(Rpp8s *dstPtr, __m128 *p)
{
    __m128i px[7];
    __m128i pxMask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);
    __m128i pxZero = _mm_setzero_si128();

    px[4] = _mm_cvtps_epi32(p[0]);    /* convert to int32 for R01-04 */
    px[5] = _mm_cvtps_epi32(p[4]);    /* convert to int32 for G01-04 */
    px[6] = _mm_cvtps_epi32(p[8]);    /* convert to int32 for B01-04 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R01-04|G01-04 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B01-04|X01-04 */
    px[0] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[1]);    /* convert to int32 for R05-08 */
    px[5] = _mm_cvtps_epi32(p[5]);    /* convert to int32 for G05-08 */
    px[6] = _mm_cvtps_epi32(p[9]);    /* convert to int32 for B05-08 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R05-08|G05-08 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B05-08|X01-04 */
    px[1] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[2]);    /* convert to int32 for R09-12 */
    px[5] = _mm_cvtps_epi32(p[6]);    /* convert to int32 for G09-12 */
    px[6] = _mm_cvtps_epi32(p[10]);    /* convert to int32 for B09-12 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R09-12|G09-12 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B09-12|X01-04 */
    px[2] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00] */
    px[4] = _mm_cvtps_epi32(p[3]);    /* convert to int32 for R13-16 */
    px[5] = _mm_cvtps_epi32(p[7]);    /* convert to int32 for G13-16 */
    px[6] = _mm_cvtps_epi32(p[11]);    /* convert to int32 for B13-16 */
    px[4] = _mm_packus_epi32(px[4], px[5]);    /* pack pixels 0-7 as R13-16|G13-16 */
    px[5] = _mm_packus_epi32(px[6], pxZero);    /* pack pixels 8-15 as B13-16|X01-04 */
    px[3] = _mm_packus_epi16(px[4], px[5]);    /* pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00] */
    px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    px[1] = _mm_sub_epi8(px[1], xmm_pxConvertI8);    /* convert back to i8 for px1 store */
    px[2] = _mm_sub_epi8(px[2], xmm_pxConvertI8);    /* convert back to i8 for px2 store */
    px[3] = _mm_sub_epi8(px[3], xmm_pxConvertI8);    /* convert back to i8 for px3 store */
    px[0] = _mm_shuffle_epi8(px[0], pxMask);    /* shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    px[1] = _mm_shuffle_epi8(px[1], pxMask);    /* shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    px[2] = _mm_shuffle_epi8(px[2], pxMask);    /* shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    px[3] = _mm_shuffle_epi8(px[3], pxMask);    /* shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px[1]);    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px[2]);    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px[3]);    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_load48_i8pln3_to_i32pln3_avx(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m256i *p)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi8_epi32(px[0]);                                        /* Contains R01-08 */
    p[1] = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(px[0], xmm_pxMask08To15));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi8_epi32(px[1]);                                        /* Contains G01-08 */
    p[3] = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(px[1], xmm_pxMask08To15));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi8_epi32(px[2]);                                        /* Contains B01-08 */
    p[5] = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(px[2], xmm_pxMask08To15));    /* Contains B09-16 */
}

inline void rpp_load48_i8pkd3_to_i8pln3(Rpp8s *srcPtr, __m128i *px)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);    /* shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04 */
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);    /* shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08 */
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);    /* shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12 */
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);    /* shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16 */
    pxSrc[4] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 lo-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[5] = _mm_unpacklo_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 lo-pixels of pxSrc[2] and pxSrc[3] */
    pxSrc[6] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 hi-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[7] = _mm_unpackhi_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 hi-pixels of pxSrc[2] and pxSrc[3] */
    px[0] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);    /* unpack 8 lo-pixels of pxSrc[4] and pxSrc[5] to get R01-16 */
    px[1] = _mm_shuffle_epi8(_mm_unpackhi_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB);    /* unpack 8 hi-pixels of pxSrc[4] and pxSrc[5] to get G01-16 */
    px[2] = _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[6], pxSrc[7]), pxMaskRGB);    /* unpack 8 lo-pixels of pxSrc[6] and pxSrc[7] to get B01-16 */
}

inline void rpp_load48_i8pkd3_to_u8pln3(Rpp8s *srcPtr, __m128i *px)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);    /* shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04 */
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);    /* shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08 */
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);    /* shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12 */
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);    /* shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16 */
    pxSrc[4] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 lo-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[5] = _mm_unpacklo_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 lo-pixels of pxSrc[2] and pxSrc[3] */
    pxSrc[6] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);    /* unpack 8 hi-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[7] = _mm_unpackhi_epi8(pxSrc[2], pxSrc[3]);    /* unpack 8 hi-pixels of pxSrc[2] and pxSrc[3] */
    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB));    /* unpack 8 lo-pixels of pxSrc[4] and pxSrc[5] to get R01-16 and add 128 to get u8 from i8 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_shuffle_epi8(_mm_unpackhi_epi8(pxSrc[4], pxSrc[5]), pxMaskRGB));    /* unpack 8 hi-pixels of pxSrc[4] and pxSrc[5] to get G01-16 and add 128 to get u8 from i8 */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxSrc[6], pxSrc[7]), pxMaskRGB));    /* unpack 8 lo-pixels of pxSrc[6] and pxSrc[7] to get B01-16 and add 128 to get u8 from i8 */
}

inline void rpp_store48_i8pln3_to_i8pln3(Rpp8s *dstPtrR, Rpp8s *dstPtrG, Rpp8s *dstPtrB, __m128i *px)
{
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_store48_u8pln3_to_i8pln3(Rpp8s *dstPtrR, Rpp8s *dstPtrG, Rpp8s *dstPtrB, __m128i *px)
{
    _mm_storeu_si128((__m128i *)dstPtrR, _mm_sub_epi8(px[0], xmm_pxConvertI8));    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, _mm_sub_epi8(px[1], xmm_pxConvertI8));    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, _mm_sub_epi8(px[2], xmm_pxConvertI8));    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_i8pkd3_to_i32pln3_avx(Rpp8s *srcPtr, __m256i *p)
{
    __m128i pxSrc[8];
    __m128i pxMask = _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 13, 14, 15);
    __m128i pxMaskRGB = _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15);

    pxSrc[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    pxSrc[0] = _mm_shuffle_epi8(pxSrc[0], pxMask);    /* shuffle to get [R01|R02|R03|R04|G01|G02|G03|G04 || B01|B02|B03|B04|R05|G05|B05|R06] - Need R01-04, G01-04, B01-04 */
    pxSrc[1] = _mm_shuffle_epi8(pxSrc[1], pxMask);    /* shuffle to get [R05|R06|R07|R08|G05|G06|G07|G08 || B05|B06|B07|B08|R09|G09|B09|R10] - Need R05-08, G05-08, B05-08 */
    pxSrc[2] = _mm_shuffle_epi8(pxSrc[2], pxMask);    /* shuffle to get [R09|R10|R11|R12|G09|G10|G11|G12 || B09|B10|B11|B12|R13|G13|B13|R14] - Need R09-12, G09-12, B09-12 */
    pxSrc[3] = _mm_shuffle_epi8(pxSrc[3], pxMask);    /* shuffle to get [R13|R14|R15|R16|G13|G14|G15|G16 || B13|B14|B15|B16|R17|G17|B17|R18] - Need R13-16, G13-16, B13-16 */
    pxSrc[4] = _mm_unpacklo_epi32(pxSrc[0], pxSrc[1]); /* unpack 32 lo-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[5] = _mm_unpackhi_epi32(pxSrc[0], pxSrc[1]); /* unpack 32 hi-pixels of pxSrc[0] and pxSrc[1] */
    pxSrc[6] = _mm_unpacklo_epi32(pxSrc[2], pxSrc[3]); /* unpack 32 lo-pixels of pxSrc[2] and pxSrc[3] */
    pxSrc[7] = _mm_unpackhi_epi32(pxSrc[2], pxSrc[3]); /* unpack 32 hi-pixels of pxSrc[2] and pxSrc[3] */
    p[0] = _mm256_cvtepi8_epi32(pxSrc[4]);                                        /* Contains R01-08 */
    p[1] = _mm256_cvtepi8_epi32(pxSrc[6]);                                        /* Contains R09-16 */
    p[2] = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(pxSrc[4], xmm_pxMask08To15));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(pxSrc[6], xmm_pxMask08To15));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi8_epi32(pxSrc[5]);                                        /* Contains B01-08 */
    p[5] = _mm256_cvtepi8_epi32(pxSrc[7]);                                        /* Contains B09-16 */
}

inline void rpp_load48_i8pln3_to_i8pln3(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m128i *px)
{
    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_i8pln3_to_u8pln3(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m128i *px)
{
    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrR));    /* load and convert to u8 [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrG));    /* load and convert to u8 [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrB));    /* load and convert to u8 [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_store48_i8pln3_to_i8pkd3(Rpp8s *dstPtr, __m128i *px)
{
    __m128i pxDst[4];
    __m128i pxZero = _mm_setzero_si128();
    __m128i pxMaskRGBAtoRGB = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
    pxDst[0] = _mm_unpacklo_epi8(px[1], pxZero);    /* unpack 8 lo-pixels of px[1] and pxZero */
    pxDst[1] = _mm_unpackhi_epi8(px[1], pxZero);    /* unpack 8 hi-pixels of px[1] and pxZero */
    pxDst[2] = _mm_unpacklo_epi8(px[0], px[2]);    /* unpack 8 lo-pixels of px[0] and px[2] */
    pxDst[3] = _mm_unpackhi_epi8(px[0], px[2]);    /* unpack 8 hi-pixels of px[0] and px[2] */
    _mm_storeu_si128((__m128i *)dstPtr, _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB));    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), _mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), _mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB));    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_store48_u8pln3_to_i8pkd3(Rpp8s *dstPtr, __m128i *px)
{
    __m128i pxDst[4];
    __m128i pxZero = _mm_setzero_si128();
    __m128i pxMaskRGBAtoRGB = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 3, 7, 11, 15);
    pxDst[0] = _mm_unpacklo_epi8(px[1], pxZero);    /* unpack 8 lo-pixels of px[1] and pxZero */
    pxDst[1] = _mm_unpackhi_epi8(px[1], pxZero);    /* unpack 8 hi-pixels of px[1] and pxZero */
    pxDst[2] = _mm_unpacklo_epi8(px[0], px[2]);    /* unpack 8 lo-pixels of px[0] and px[2] */
    pxDst[3] = _mm_unpackhi_epi8(px[0], px[2]);    /* unpack 8 hi-pixels of px[0] and px[2] */
    _mm_storeu_si128((__m128i *)dstPtr, _mm_sub_epi8(_mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB), xmm_pxConvertI8));           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_sub_epi8(_mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[2], pxDst[0]), pxMaskRGBAtoRGB), xmm_pxConvertI8));    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), _mm_sub_epi8(_mm_shuffle_epi8(_mm_unpacklo_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB), xmm_pxConvertI8));    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), _mm_sub_epi8(_mm_shuffle_epi8(_mm_unpackhi_epi8(pxDst[3], pxDst[1]), pxMaskRGBAtoRGB), xmm_pxConvertI8));    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_load16_i8_to_f32(Rpp8s *srcPtr, __m128 *p)
{
    __m128i px = _mm_loadu_si128((__m128i *)srcPtr);    /* load pixels 0-15 */
    px = _mm_add_epi8(px, xmm_pxConvertI8);    /* convert to u8 for px compute */
    p[0] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask00To03));    /* pixels 0-3 */
    p[1] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask04To07));    /* pixels 4-7 */
    p[2] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask08To11));    /* pixels 8-11 */
    p[3] = _mm_cvtepi32_ps(_mm_shuffle_epi8(px, xmm_pxMask12To15));    /* pixels 12-15 */
}

inline void rpp_store16_f32_to_i8(Rpp8s *dstPtr, __m128 *p)
{
    __m128i px[4];

    px[0] = _mm_cvtps_epi32(p[0]);    /* pixels 0-3 */
    px[1] = _mm_cvtps_epi32(p[1]);    /* pixels 4-7 */
    px[2] = _mm_cvtps_epi32(p[2]);    /* pixels 8-11 */
    px[3] = _mm_cvtps_epi32(p[3]);    /* pixels 12-15 */
    px[0] = _mm_packus_epi32(px[0], px[1]);    /* pixels 0-7 */
    px[1] = _mm_packus_epi32(px[2], px[3]);    /* pixels 8-15 */
    px[0] = _mm_packus_epi16(px[0], px[1]);    /* pixels 0-15 */
    px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);    /* store pixels 0-15 */
}

inline void rpp_normalize48(__m128 *p)
{
    p[0] = _mm_mul_ps(p[0], xmm_p1op255);
    p[1] = _mm_mul_ps(p[1], xmm_p1op255);
    p[2] = _mm_mul_ps(p[2], xmm_p1op255);
    p[3] = _mm_mul_ps(p[3], xmm_p1op255);
    p[4] = _mm_mul_ps(p[4], xmm_p1op255);
    p[5] = _mm_mul_ps(p[5], xmm_p1op255);
    p[6] = _mm_mul_ps(p[6], xmm_p1op255);
    p[7] = _mm_mul_ps(p[7], xmm_p1op255);
    p[8] = _mm_mul_ps(p[8], xmm_p1op255);
    p[9] = _mm_mul_ps(p[9], xmm_p1op255);
    p[10] = _mm_mul_ps(p[10], xmm_p1op255);
    p[11] = _mm_mul_ps(p[11], xmm_p1op255);
}

// AVX loads and stores

inline void rpp_load48_u8pkd3_to_f32pkd3_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px[4];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */

    p[0] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask00To02), _mm_shuffle_epi8(px[0], xmm_pxMask03To05)));    /* Contains RGB 01-02 */
    p[1] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask06To08), _mm_shuffle_epi8(px[0], xmm_pxMask09To11)));    /* Contains RGB 03-04 */
    p[2] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask00To02), _mm_shuffle_epi8(px[1], xmm_pxMask03To05)));    /* Contains RGB 05-06 */
    p[3] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask06To08), _mm_shuffle_epi8(px[1], xmm_pxMask09To11)));    /* Contains RGB 07-08 */
    p[4] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask00To02), _mm_shuffle_epi8(px[2], xmm_pxMask03To05)));    /* Contains RGB 09-10 */
    p[5] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask06To08), _mm_shuffle_epi8(px[2], xmm_pxMask09To11)));    /* Contains RGB 11-12 */
    p[6] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMask00To02), _mm_shuffle_epi8(px[3], xmm_pxMask03To05)));    /* Contains RGB 13-14 */
    p[7] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMask06To08), _mm_shuffle_epi8(px[3], xmm_pxMask09To11)));    /* Contains RGB 15-16 */
}

inline void rpp_load48_u8pkd3_to_f32pkd3_mirror_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px[4];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */

    p[0] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMask09To11), _mm_shuffle_epi8(px[3], xmm_pxMask06To08)));    /* Contains RGB 16-15 */
    p[1] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMask03To05), _mm_shuffle_epi8(px[3], xmm_pxMask00To02)));    /* Contains RGB 14-13 */
    p[2] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask09To11), _mm_shuffle_epi8(px[2], xmm_pxMask06To08)));    /* Contains RGB 12-11 */
    p[3] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask03To05), _mm_shuffle_epi8(px[2], xmm_pxMask00To02)));    /* Contains RGB 10-09 */
    p[4] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask09To11), _mm_shuffle_epi8(px[1], xmm_pxMask06To08)));    /* Contains RGB 08-07 */
    p[5] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask03To05), _mm_shuffle_epi8(px[1], xmm_pxMask00To02)));    /* Contains RGB 06-05 */
    p[6] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask09To11), _mm_shuffle_epi8(px[0], xmm_pxMask06To08)));    /* Contains RGB 04-03 */
    p[7] =  _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask03To05), _mm_shuffle_epi8(px[0], xmm_pxMask00To02)));    /* Contains RGB 02-01 */
}

inline void rpp_store48_f32pkd3_to_f32pkd3_avx(Rpp32f *dstPtr, __m256 *p)
{
    __m128 p128[4];
    p128[0] = _mm256_extractf128_ps(p[0], 0);
    p128[1] = _mm256_extractf128_ps(p[0], 1);
    p128[2] = _mm256_extractf128_ps(p[1], 0);
    p128[3] = _mm256_extractf128_ps(p[1], 1);
    _mm_storeu_ps(dstPtr, p128[0]);
    _mm_storeu_ps(dstPtr + 3, p128[1]);
    _mm_storeu_ps(dstPtr + 6, p128[2]);
    _mm_storeu_ps(dstPtr + 9, p128[3]);

    p128[0] = _mm256_extractf128_ps(p[2], 0);
    p128[1] = _mm256_extractf128_ps(p[2], 1);
    p128[2] = _mm256_extractf128_ps(p[3], 0);
    p128[3] = _mm256_extractf128_ps(p[3], 1);
    _mm_storeu_ps(dstPtr + 12, p128[0]);
    _mm_storeu_ps(dstPtr + 15, p128[1]);
    _mm_storeu_ps(dstPtr + 18, p128[2]);
    _mm_storeu_ps(dstPtr + 21, p128[3]);

    p128[0] = _mm256_extractf128_ps(p[4], 0);
    p128[1] = _mm256_extractf128_ps(p[4], 1);
    p128[2] = _mm256_extractf128_ps(p[5], 0);
    p128[3] = _mm256_extractf128_ps(p[5], 1);
    _mm_storeu_ps(dstPtr + 24, p128[0]);
    _mm_storeu_ps(dstPtr + 27, p128[1]);
    _mm_storeu_ps(dstPtr + 30, p128[2]);
    _mm_storeu_ps(dstPtr + 33, p128[3]);

    p128[0] = _mm256_extractf128_ps(p[6], 0);
    p128[1] = _mm256_extractf128_ps(p[6], 1);
    p128[2] = _mm256_extractf128_ps(p[7], 0);
    p128[3] = _mm256_extractf128_ps(p[7], 1);
    _mm_storeu_ps(dstPtr + 36, p128[0]);
    _mm_storeu_ps(dstPtr + 39, p128[1]);
    _mm_storeu_ps(dstPtr + 42, p128[2]);
    _mm_storeu_ps(dstPtr + 45, p128[3]);
}

inline void rpp_store48_f32pkd3_to_f16pkd3_avx(Rpp16f *dstPtr, __m256 *p)
{
    __m128 p128[4];
    p128[0] = _mm256_extractf128_ps(p[0], 0);
    p128[1] = _mm256_extractf128_ps(p[0], 1);
    p128[2] = _mm256_extractf128_ps(p[1], 0);
    p128[3] = _mm256_extractf128_ps(p[1], 1);

    __m128i px128[4];
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 3), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 6), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 9), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[2], 0);
    p128[1] = _mm256_extractf128_ps(p[2], 1);
    p128[2] = _mm256_extractf128_ps(p[3], 0);
    p128[3] = _mm256_extractf128_ps(p[3], 1);
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 15), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 18), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 21), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[4], 0);
    p128[1] = _mm256_extractf128_ps(p[4], 1);
    p128[2] = _mm256_extractf128_ps(p[5], 0);
    p128[3] = _mm256_extractf128_ps(p[5], 1);
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 27), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 30), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 33), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[6], 0);
    p128[1] = _mm256_extractf128_ps(p[6], 1);
    p128[2] = _mm256_extractf128_ps(p[7], 0);
    p128[3] = _mm256_extractf128_ps(p[7], 1);
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 39), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 42), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 45), px128[3]);
}

inline void rpp_load48_u8pkd3_to_f32pln3_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px[4];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskR), _mm_shuffle_epi8(px[1], xmm_pxMaskR)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskR), _mm_shuffle_epi8(px[3], xmm_pxMaskR)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskG), _mm_shuffle_epi8(px[1], xmm_pxMaskG)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskG), _mm_shuffle_epi8(px[3], xmm_pxMaskG)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskB), _mm_shuffle_epi8(px[1], xmm_pxMaskB)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskB), _mm_shuffle_epi8(px[3], xmm_pxMaskB)));    /* Contains B09-16 */
}

inline void rpp_glitch_load24_u8pkd3_to_f32pln3_avx(Rpp8u *srcPtr, __m256 *p, int *srcLocs)
{
    __m128i px[2];
    px[0] = _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[0]));      /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need R01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[0] + 12)); /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need R05-08 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskR), _mm_shuffle_epi8(px[1], xmm_pxMaskR)));   /* Contains R01-08 */

    px[0] = _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[1]));      /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need G01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[1] + 12)); /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need G05-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskG), _mm_shuffle_epi8(px[1], xmm_pxMaskG)));   /* Contains G01-08 */

    px[0] = _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[2]));      /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need B01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[2] + 12)); /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need B05-08 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskB), _mm_shuffle_epi8(px[1], xmm_pxMaskB)));   /* Contains B01-08 */
}

inline void rpp_glitch_load24_f32pkd3_to_f32pln3_avx(Rpp32f *srcPtr, __m256 *p, int *srcLocs)
{
    __m128 p128[8];
    Rpp32f *srcPtrTemp = srcPtr + srcLocs[0];
    p[0] = _mm256_setr_ps(*srcPtrTemp, *(srcPtrTemp + 3), *(srcPtrTemp + 6), *(srcPtrTemp + 9), 
                         *(srcPtrTemp + 12), *(srcPtrTemp + 15), *(srcPtrTemp + 18), *(srcPtrTemp + 21));
    srcPtrTemp = srcPtr + srcLocs[1];
    p[1] = _mm256_setr_ps(*(srcPtrTemp + 1), *(srcPtrTemp + 4), *(srcPtrTemp + 7), *(srcPtrTemp + 10), 
                         *(srcPtrTemp + 13), *(srcPtrTemp + 16), *(srcPtrTemp + 19), *(srcPtrTemp + 22));
    srcPtrTemp = srcPtr + srcLocs[2];
    p[2] = _mm256_setr_ps(*(srcPtrTemp + 2), *(srcPtrTemp + 5), *(srcPtrTemp + 8), *(srcPtrTemp + 11), 
                         *(srcPtrTemp + 14), *(srcPtrTemp + 17), *(srcPtrTemp + 20), *(srcPtrTemp + 23));
}

inline void rpp_glitch_load24_i8pkd3_to_f32pln3_avx(Rpp8s *srcPtr, __m256 *p, int *srcLocs)
{
    __m128i px[2];
    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[0])));      /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need R01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[0] + 12))); /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need R05-08 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskR), _mm_shuffle_epi8(px[1], xmm_pxMaskR)));   /* Contains R01-08 */

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[1])));      /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need G01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[1] + 12))); /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need G05-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskG), _mm_shuffle_epi8(px[1], xmm_pxMaskG)));   /* Contains G01-08 */

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[2])));      /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need B01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + srcLocs[2] + 12))); /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need B05-08 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskB), _mm_shuffle_epi8(px[1], xmm_pxMaskB)));   /* Contains B01-08 */
}

inline void rpp_glitch_load30_u8pkd3_to_u8pkd3_avx(Rpp8u *srcPtr, int *srcLocs, __m256i &p)
{
    __m256i px[3];
    px[0] = _mm256_loadu_si256((__m256i *)(srcPtr + srcLocs[0]));   // Load the source location1 values passed
    px[1] = _mm256_loadu_si256((__m256i *)(srcPtr + srcLocs[1]));   // Load the source location2 values passed
    px[2] = _mm256_loadu_si256((__m256i *)(srcPtr + srcLocs[2]));   // Load the source location3 values passed
    px[0] = _mm256_shuffle_epi8(px[0], avx_pxMaskR);    /* Shuffle to obtain R channel values  */
    px[1] = _mm256_shuffle_epi8(px[1], avx_pxMaskG);    /* Shuffle to obtain G channel values  */
    px[2] = _mm256_shuffle_epi8(px[2], avx_pxMaskB);    /* Shuffle to obtain B channel values  */
    px[0] = _mm256_or_si256(px[0], px[1]);  /* Pack R and G channels to obtain RG format */
    p = _mm256_or_si256(px[0], px[2]);      /* Pack RG values and B channel to obtain RGB format */
}

inline void rpp_glitch_load30_i8pkd3_to_i8pkd3_avx(Rpp8s *srcPtr, int * srcLocs, __m256i &p)
{
    __m256i px[3];
    px[0] = _mm256_loadu_si256((__m256i *)(srcPtr + srcLocs[0]));   // Load the source location1 values passed
    px[1] = _mm256_loadu_si256((__m256i *)(srcPtr + srcLocs[1]));   // Load the source location2 values passed
    px[2] = _mm256_loadu_si256((__m256i *)(srcPtr + srcLocs[2]));   // Load the source location3 values passed
    px[0] = _mm256_shuffle_epi8(px[0], avx_pxMaskR);    /* Shuffle to obtain R channel values  */
    px[1] = _mm256_shuffle_epi8(px[1], avx_pxMaskG);    /* Shuffle to obtain G channel values  */
    px[2] = _mm256_shuffle_epi8(px[2], avx_pxMaskB);    /* Shuffle to obtain B channel values  */
    px[0] = _mm256_or_si256(px[0], px[1]);  /* Pack R and G channels to obtain RG format */
    p = _mm256_or_si256(px[0], px[2]);      /* Pack RG values and B channel to obtain RGB format */
}

inline void rpp_glitch_load6_f32pkd3_to_f32pkd3_avx(Rpp32f *srcPtr, int * srcLocs, __m256 &p)
{
    p =_mm256_setr_ps(*(srcPtr + srcLocs[0]), *(srcPtr + srcLocs[1] + 1), *(srcPtr + srcLocs[2] + 2), *(srcPtr + srcLocs[0] + 3), 
                      *(srcPtr + srcLocs[1] + 4), *(srcPtr + srcLocs[2] + 5), 0.0f, 0.0f);
}

inline void rpp_glitch_load48_u8pln3_to_f32pln3_avx(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m256 *p, int *srcLocs)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR + srcLocs[0]);       /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG + srcLocs[1]);       /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB + srcLocs[2]);       /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask00To03), _mm_shuffle_epi8(px[0], xmm_pxMask04To07)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask08To11), _mm_shuffle_epi8(px[0], xmm_pxMask12To15)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask00To03), _mm_shuffle_epi8(px[1], xmm_pxMask04To07)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask08To11), _mm_shuffle_epi8(px[1], xmm_pxMask12To15)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[2], xmm_pxMask04To07)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[2], xmm_pxMask12To15)));    /* Contains B09-16 */
}

inline void rpp_load48_u8pkd3_to_f32pln3_mirror_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px[4];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMaskRMirror), _mm_shuffle_epi8(px[2], xmm_pxMaskRMirror)));    /* Contains R16-09 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMaskRMirror), _mm_shuffle_epi8(px[0], xmm_pxMaskRMirror)));    /* Contains R01-08 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMaskGMirror), _mm_shuffle_epi8(px[2], xmm_pxMaskGMirror)));    /* Contains G16-09 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMaskGMirror), _mm_shuffle_epi8(px[0], xmm_pxMaskGMirror)));    /* Contains G01-08 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMaskBMirror), _mm_shuffle_epi8(px[2], xmm_pxMaskBMirror)));    /* Contains B16-09 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMaskBMirror), _mm_shuffle_epi8(px[0], xmm_pxMaskBMirror)));    /* Contains B01-08 */
}

inline void rpp_load48_u8pkd3_to_u32pln3_avx(Rpp8u *srcPtr, __m256i *p)
{
    __m128i px[4];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_loadu_si128((__m128i *)(srcPtr + 24));    /* load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_loadu_si128((__m128i *)(srcPtr + 36));    /* load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskR), _mm_shuffle_epi8(px[1], xmm_pxMaskR));    /* Contains R01-08 */
    p[1] = _mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskR), _mm_shuffle_epi8(px[3], xmm_pxMaskR));    /* Contains R09-16 */
    p[2] = _mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskG), _mm_shuffle_epi8(px[1], xmm_pxMaskG));    /* Contains G01-08 */
    p[3] = _mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskG), _mm_shuffle_epi8(px[3], xmm_pxMaskG));    /* Contains G09-16 */
    p[4] = _mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskB), _mm_shuffle_epi8(px[1], xmm_pxMaskB));    /* Contains B01-08 */
    p[5] = _mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskB), _mm_shuffle_epi8(px[3], xmm_pxMaskB));    /* Contains B09-16 */
}

inline void rpp_store48_f32pln3_to_u8pln3_avx(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m256 *p)
{
    __m256i pxCvt;
    __m128i px[4];

    pxCvt = _mm256_cvtps_epi32(p[0]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for R */
    pxCvt = _mm256_cvtps_epi32(p[1]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for R */
    pxCvt = _mm256_cvtps_epi32(p[2]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for G */
    pxCvt = _mm256_cvtps_epi32(p[3]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for G */
    px[1] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for G */
    pxCvt = _mm256_cvtps_epi32(p[4]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for B */
    pxCvt = _mm256_cvtps_epi32(p[5]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for B */
    px[2] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for B */
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_store48_f32pln3_to_f32pln3_avx(Rpp32f *dstPtrR, Rpp32f *dstPtrG, Rpp32f *dstPtrB, __m256 *p)
{
    _mm256_storeu_ps(dstPtrR, p[0]);
    _mm256_storeu_ps(dstPtrG, p[2]);
    _mm256_storeu_ps(dstPtrB, p[4]);
    _mm256_storeu_ps(dstPtrR + 8, p[1]);
    _mm256_storeu_ps(dstPtrG + 8, p[3]);
    _mm256_storeu_ps(dstPtrB + 8, p[5]);
}

inline void rpp_store48_f32pln3_to_f16pln3_avx(Rpp16f *dstPtrR, Rpp16f *dstPtrG, Rpp16f *dstPtrB, __m256 *p)
{
    __m128i px128[6];
    px128[0] = _mm256_cvtps_ph(p[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm256_cvtps_ph(p[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm256_cvtps_ph(p[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm256_cvtps_ph(p[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[4] = _mm256_cvtps_ph(p[4], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[5] = _mm256_cvtps_ph(p[5], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtrR, px128[0]);
    _mm_storeu_si128((__m128i *)dstPtrG, px128[2]);
    _mm_storeu_si128((__m128i *)dstPtrB, px128[4]);
    _mm_storeu_si128((__m128i *)(dstPtrR + 8), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtrG + 8), px128[3]);
    _mm_storeu_si128((__m128i *)(dstPtrB + 8), px128[5]);
}

inline void rpp_store48_f32pln3_to_f32pkd3_avx(Rpp32f *dstPtr, __m256 *p)
{
    __m128 p128[4];
    p128[0] = _mm256_castps256_ps128(p[0]);
    p128[1] = _mm256_castps256_ps128(p[2]);
    p128[2] = _mm256_castps256_ps128(p[4]);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    _mm_storeu_ps(dstPtr, p128[0]);
    _mm_storeu_ps(dstPtr + 3, p128[1]);
    _mm_storeu_ps(dstPtr + 6, p128[2]);
    _mm_storeu_ps(dstPtr + 9, p128[3]);
    p128[0] = _mm256_extractf128_ps(p[0], 1);
    p128[1] = _mm256_extractf128_ps(p[2], 1);
    p128[2] = _mm256_extractf128_ps(p[4], 1);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    _mm_storeu_ps(dstPtr + 12, p128[0]);
    _mm_storeu_ps(dstPtr + 15, p128[1]);
    _mm_storeu_ps(dstPtr + 18, p128[2]);
    _mm_storeu_ps(dstPtr + 21, p128[3]);

    p128[0] = _mm256_castps256_ps128(p[1]);
    p128[1] = _mm256_castps256_ps128(p[3]);
    p128[2] = _mm256_castps256_ps128(p[5]);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    _mm_storeu_ps(dstPtr + 24, p128[0]);
    _mm_storeu_ps(dstPtr + 27, p128[1]);
    _mm_storeu_ps(dstPtr + 30, p128[2]);
    _mm_storeu_ps(dstPtr + 33, p128[3]);
    p128[0] = _mm256_extractf128_ps(p[1], 1);
    p128[1] = _mm256_extractf128_ps(p[3], 1);
    p128[2] = _mm256_extractf128_ps(p[5], 1);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    _mm_storeu_ps(dstPtr + 36, p128[0]);
    _mm_storeu_ps(dstPtr + 39, p128[1]);
    _mm_storeu_ps(dstPtr + 42, p128[2]);
    _mm_storeu_ps(dstPtr + 45, p128[3]);
}

inline void rpp_store48_f32pln3_to_f16pkd3_avx(Rpp16f *dstPtr, __m256 *p)
{
    __m128 p128[4];
    p128[0] = _mm256_extractf128_ps(p[0], 0);
    p128[1] = _mm256_extractf128_ps(p[2], 0);
    p128[2] = _mm256_extractf128_ps(p[4], 0);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);

    __m128i px128[4];
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 3), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 6), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 9), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[0], 1);
    p128[1] = _mm256_extractf128_ps(p[2], 1);
    p128[2] = _mm256_extractf128_ps(p[4], 1);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 15), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 18), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 21), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[1], 0);
    p128[1] = _mm256_extractf128_ps(p[3], 0);
    p128[2] = _mm256_extractf128_ps(p[5], 0);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 27), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 30), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 33), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[1], 1);
    p128[1] = _mm256_extractf128_ps(p[3], 1);
    p128[2] = _mm256_extractf128_ps(p[5], 1);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 39), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 42), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 45), px128[3]);
}

inline void rpp_load48_u8pln3_to_f32pln3_avx(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m256 *p)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask00To03), _mm_shuffle_epi8(px[0], xmm_pxMask04To07)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask08To11), _mm_shuffle_epi8(px[0], xmm_pxMask12To15)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask00To03), _mm_shuffle_epi8(px[1], xmm_pxMask04To07)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask08To11), _mm_shuffle_epi8(px[1], xmm_pxMask12To15)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[2], xmm_pxMask04To07)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[2], xmm_pxMask12To15)));    /* Contains B09-16 */
}

inline void rpp_load48_u8pln3_to_f32pln3_mirror_avx(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m256 *p)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask15To12), _mm_shuffle_epi8(px[0], xmm_pxMask11To08)));    /* Contains R16-09 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask07To04), _mm_shuffle_epi8(px[0], xmm_pxMask03To00)));    /* Contains R01-08 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask15To12), _mm_shuffle_epi8(px[1], xmm_pxMask11To08)));    /* Contains G16-09 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask07To04), _mm_shuffle_epi8(px[1], xmm_pxMask03To00)));    /* Contains G01-08 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask15To12), _mm_shuffle_epi8(px[2], xmm_pxMask11To08)));    /* Contains B16-09 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask07To04), _mm_shuffle_epi8(px[2], xmm_pxMask03To00)));    /* Contains B01-08 */
}

inline void rpp_load48_u8pln3_to_u32pln3_avx(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m256i *p)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask00To03), _mm_shuffle_epi8(px[0], xmm_pxMask04To07));    /* Contains R01-08 */
    p[1] = _mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask08To11), _mm_shuffle_epi8(px[0], xmm_pxMask12To15));    /* Contains R09-16 */
    p[2] = _mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask00To03), _mm_shuffle_epi8(px[1], xmm_pxMask04To07));    /* Contains G01-08 */
    p[3] = _mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask08To11), _mm_shuffle_epi8(px[1], xmm_pxMask12To15));    /* Contains G09-16 */
    p[4] = _mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[2], xmm_pxMask04To07));    /* Contains B01-08 */
    p[5] = _mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[2], xmm_pxMask12To15));    /* Contains B09-16 */
}

inline void rpp_store48_f32pln3_to_u8pkd3_avx(Rpp8u *dstPtr, __m256 *p)
{
    __m256i pxCvt[3];
    __m128i px[5];
    __m128i pxMask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);

    pxCvt[0] = _mm256_cvtps_epi32(p[0]);    /* convert to int32 for R01-08 */
    pxCvt[1] = _mm256_cvtps_epi32(p[2]);    /* convert to int32 for G01-08 */
    pxCvt[2] = _mm256_cvtps_epi32(p[4]);    /* convert to int32 for B01-08 */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 0), _mm256_extracti128_si256(pxCvt[1], 0));    /* pack pixels 0-7 as R01-04|G01-04 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 0), xmm_px0);    /* pack pixels 8-15 as B01-04|X01-04 */
    px[0] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00] */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 1), _mm256_extracti128_si256(pxCvt[1], 1));    /* pack pixels 0-7 as R05-08|G05-08 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 1), xmm_px0);    /* pack pixels 8-15 as B05-08|X05-08 */
    px[1] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00] */
    pxCvt[0] = _mm256_cvtps_epi32(p[1]);    /* convert to int32 for R09-16 */
    pxCvt[1] = _mm256_cvtps_epi32(p[3]);    /* convert to int32 for G09-16 */
    pxCvt[2] = _mm256_cvtps_epi32(p[5]);    /* convert to int32 for B09-16 */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 0), _mm256_extracti128_si256(pxCvt[1], 0));    /* pack pixels 0-7 as R09-12|G09-12 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 0), xmm_px0);    /* pack pixels 8-15 as B09-12|X09-12 */
    px[2] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00] */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 1), _mm256_extracti128_si256(pxCvt[1], 1));    /* pack pixels 0-7 as R13-16|G13-16 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 1), xmm_px0);    /* pack pixels 8-15 as B13-16|X13-16 */
    px[3] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00] */
    px[0] = _mm_shuffle_epi8(px[0], pxMask);    /* shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    px[1] = _mm_shuffle_epi8(px[1], pxMask);    /* shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    px[2] = _mm_shuffle_epi8(px[2], pxMask);    /* shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    px[3] = _mm_shuffle_epi8(px[3], pxMask);    /* shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px[1]);    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px[2]);    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px[3]);    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_load24_u8pln3_to_f64pln3_avx(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m256d *p)
{
    __m128i px[3];

    px[0] = _mm_loadu_si128((__m128i *)srcPtrR);    /* load [R00|R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_loadu_si128((__m128i *)srcPtrG);    /* load [G00|G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_loadu_si128((__m128i *)srcPtrB);    /* load [B00|B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMask00To03));    /* Contains R00-03 */
    p[1] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMask04To07));    /* Contains R04-07 */
    p[2] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMask00To03));    /* Contains G00-03 */
    p[3] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMask04To07));    /* Contains G04-07 */
    p[4] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[2], xmm_pxMask00To03));    /* Contains B00-03 */
    p[5] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[2], xmm_pxMask04To07));    /* Contains B04-07 */
}

inline void rpp_load24_u8pkd3_to_f64pln3_avx(Rpp8u *srcPtr, __m256d *p)
{
    __m128i px[2];

    px[0] = _mm_loadu_si128((__m128i *)srcPtr);           /* load [R00|G00|B00|R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05] - Need RGB 00-03 */
    px[1] = _mm_loadu_si128((__m128i *)(srcPtr + 12));    /* load [R04|G04|B04|R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09] - Need RGB 04-07 */
    p[0] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMaskR));    /* Contains R00-03 */
    p[1] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMaskR));    /* Contains R04-07 */
    p[2] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMaskG));    /* Contains G00-03 */
    p[3] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMaskG));    /* Contains G04-07 */
    p[4] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMaskB));    /* Contains B00-03 */
    p[5] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMaskB));    /* Contains B04-07 */
}

inline void rpp_load16_u8_to_f32_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px;
    px = _mm_loadu_si128((__m128i *)srcPtr);
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask00To03), _mm_shuffle_epi8(px, xmm_pxMask04To07)));    /* Contains pixels 01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask08To11), _mm_shuffle_epi8(px, xmm_pxMask12To15)));    /* Contains pixels 09-16 */
}

inline void rpp_load24_u8_to_f32_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px1, px2;
    px1 = _mm_loadu_si128((__m128i *)(srcPtr));
    px2 = _mm_loadl_epi64((__m128i *)(srcPtr + 16));

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask00To03), _mm_shuffle_epi8(px1, xmm_pxMask04To07)));  /* Contains pixels 01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask08To11), _mm_shuffle_epi8(px1, xmm_pxMask12To15)));  /* Contains pixels 09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask00To03), _mm_shuffle_epi8(px2, xmm_pxMask04To07)));  /* Contains pixels 17-24 */
}

inline void rpp_load32_u8_to_f32_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m256i px = _mm256_loadu_si256((__m256i *)srcPtr);
    __m128i px1 = _mm256_castps256_ps128(px);
    __m128i px2 = _mm256_extractf128_si256(px, 1);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask00To03), _mm_shuffle_epi8(px1, xmm_pxMask04To07))); // Contains pixels 01-08
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask08To11), _mm_shuffle_epi8(px1, xmm_pxMask12To15))); // Contains pixels 09-16
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask00To03), _mm_shuffle_epi8(px2, xmm_pxMask04To07))); // Contains pixels 17-24
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask08To11), _mm_shuffle_epi8(px2, xmm_pxMask12To15))); // Contains pixels 25-32
}

inline void rpp_load40_u8_to_f32_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m256i px1 = _mm256_loadu_si256((__m256i *)srcPtr);     // Load the first 32 bytes
    __m128i px2 = _mm_loadu_si128((__m128i *)(srcPtr + 32)); // Load the remaining 8 bytes
    __m128i px1Low  = _mm256_castsi256_si128(px1);
    __m128i px1High = _mm256_extractf128_si256(px1, 1);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1Low, xmm_pxMask00To03), _mm_shuffle_epi8(px1Low, xmm_pxMask04To07))); // Pixels 01-08
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1Low, xmm_pxMask08To11), _mm_shuffle_epi8(px1Low, xmm_pxMask12To15))); // Pixels 09-16
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1High, xmm_pxMask00To03), _mm_shuffle_epi8(px1High, xmm_pxMask04To07))); // Pixels 17-24
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1High, xmm_pxMask08To11), _mm_shuffle_epi8(px1High, xmm_pxMask12To15))); // Pixels 25-32
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask00To03), _mm_shuffle_epi8(px2, xmm_pxMask04To07)));        // Pixels 33-40
}

inline void rpp_load8_u8_to_f32_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px;
    px = _mm_loadu_si128((__m128i *)srcPtr);
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask00To03), _mm_shuffle_epi8(px, xmm_pxMask04To07)));    /* Contains pixels 01-08 */
}

inline void rpp_load16_u8_to_f32_mirror_avx(Rpp8u *srcPtr, __m256 *p)
{
    __m128i px;
    px = _mm_loadu_si128((__m128i *)srcPtr);    /* load pixels 0-15 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask15To12), _mm_shuffle_epi8(px, xmm_pxMask11To08)));    /* Contains pixels 15-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask07To04), _mm_shuffle_epi8(px, xmm_pxMask03To00)));    /* Contains pixels 7-0 */
}

inline void rpp_store16_f32_to_u8_avx(Rpp8u *dstPtr, __m256 *p)
{
    __m256i pxCvt;
    __m128i px[3];
    pxCvt = _mm256_cvtps_epi32(p[0]);
    px[1] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for R */
    pxCvt = _mm256_cvtps_epi32(p[1]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[1], px[2]);    /* pack pixels 0-15 */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);
}

inline void rpp_load8_u8_to_f64_avx(Rpp8u *srcPtr, __m256d *p)
{
    __m128i px;
    px = _mm_loadu_si128((__m128i *)srcPtr);
    p[0] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px, xmm_pxMask00To03));    /* Contains pixels 01-04 */
    p[1] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px, xmm_pxMask04To07));    /* Contains pixels 05-08 */
}

inline void rpp_load8_i8_to_f64_avx(Rpp8s *srcPtr, __m256d *p)
{
    __m128i px;
    px = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));
    p[0] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px, xmm_pxMask00To03));    /* Contains pixels 01-04 */
    p[1] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px, xmm_pxMask04To07));    /* Contains pixels 05-08 */
}

inline void rpp_load8_i8_to_f32_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m128i px;
    px = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask00To03), _mm_shuffle_epi8(px, xmm_pxMask04To07)));    /* Contains pixels 01-08 */
}

inline void rpp_load16_u8_to_u32_avx(Rpp8u *srcPtr, __m256i *p)
{
    __m128i px;
    px = _mm_loadu_si128((__m128i *)srcPtr);
    p[0] = _mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask00To03), _mm_shuffle_epi8(px, xmm_pxMask04To07));    /* Contains pixels 01-08 */
    p[1] = _mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask08To11), _mm_shuffle_epi8(px, xmm_pxMask12To15));    /* Contains pixels 09-16 */
}

inline void rpp_load96_u8_avx(Rpp8u *srcPtrR, Rpp8u *srcPtrG, Rpp8u *srcPtrB, __m256i *p)
{
    p[0] = _mm256_loadu_si256((__m256i *)srcPtrR);
    p[1] = _mm256_loadu_si256((__m256i *)srcPtrG);
    p[2] = _mm256_loadu_si256((__m256i *)srcPtrB);
}

inline void rpp_load96_i8_avx(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m256i *p)
{
    p[0] = _mm256_load_si256((__m256i *)srcPtrR);
    p[1] = _mm256_load_si256((__m256i *)srcPtrG);
    p[2] = _mm256_load_si256((__m256i *)srcPtrB);
}

inline void rpp_load24_f32pkd3_to_f32pln3_avx(Rpp32f *srcPtr, __m256 *p)
{
    __m128 p128[8];
    p128[0] = _mm_loadu_ps(srcPtr);
    p128[1] = _mm_loadu_ps(srcPtr + 3);
    p128[2] = _mm_loadu_ps(srcPtr + 6);
    p128[3] = _mm_loadu_ps(srcPtr + 9);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    p128[4] = _mm_loadu_ps(srcPtr + 12);
    p128[5] = _mm_loadu_ps(srcPtr + 15);
    p128[6] = _mm_loadu_ps(srcPtr + 18);
    p128[7] = _mm_loadu_ps(srcPtr + 21);
    _MM_TRANSPOSE4_PS(p128[4], p128[5], p128[6], p128[7]);
    p[0] = _mm256_setr_m128(p128[0], p128[4]);
    p[1] = _mm256_setr_m128(p128[1], p128[5]);
    p[2] = _mm256_setr_m128(p128[2], p128[6]);
}

inline void rpp_load24_f32pkd3_to_f32pln3_mirror_avx(Rpp32f *srcPtr, __m256 *p)
{
    __m128 p128[8];
    __m256i pxMask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    p128[0] = _mm_loadu_ps(srcPtr); /* loads R01|G01|B01|R02 */
    p128[1] = _mm_loadu_ps(srcPtr + 3); /* loads R02|G02|B02|R03 */
    p128[2] = _mm_loadu_ps(srcPtr + 6); /* loads R03|G03|B03|R04 */
    p128[3] = _mm_loadu_ps(srcPtr + 9); /* loads R04|G04|B04|R05 */
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]); /* Transpose the 4x4 matrix and forms [[R01 R02 R03 R04][B01 B02 B03 B04][G01 G02 G03 G03][R02 R03 R04 R05]] */
    p128[4] = _mm_loadu_ps(srcPtr + 12); /* loads R05|G05|B05|R06 */
    p128[5] = _mm_loadu_ps(srcPtr + 15); /* loads R06|G06|B06|R07 */
    p128[6] = _mm_loadu_ps(srcPtr + 18); /* loads R07|G07|B07|R08 */
    p128[7] = _mm_loadu_ps(srcPtr + 21); /* loads R08|G08|B08|R09 */
    _MM_TRANSPOSE4_PS(p128[4], p128[5], p128[6], p128[7]); /* Transpose the 4x4 matrix and forms [[R05 R06 R07 R08][B05 B06 B07 B08][G05 G06 G07 G08][R06 R07 R08 R09]] */
    p[0] = _mm256_setr_m128(p128[0], p128[4]); /* packs as R01-R08 */
    p[1] = _mm256_setr_m128(p128[1], p128[5]); /* packs as G01-R08 */
    p[2] = _mm256_setr_m128(p128[2], p128[6]); /* packs as B01-R08 */

    p[0] = _mm256_permutevar8x32_ps(p[0], pxMask); /* shuffle as R08-R01 */
    p[1] = _mm256_permutevar8x32_ps(p[1], pxMask); /* shuffle as G08-G01 */
    p[2] = _mm256_permutevar8x32_ps(p[2], pxMask); /* shuffle as B08-B01 */
}

inline void rpp_store24_f32pln3_to_f32pln3_avx(Rpp32f *dstPtrR, Rpp32f *dstPtrG, Rpp32f *dstPtrB, __m256 *p)
{
    _mm256_storeu_ps(dstPtrR, p[0]);
    _mm256_storeu_ps(dstPtrG, p[1]);
    _mm256_storeu_ps(dstPtrB, p[2]);
}

inline void rpp_load24_f32pkd3_to_f64pln3_avx(Rpp32f *srcPtr, __m256d *p)
{
    __m128 p128[8];
    p128[0] = _mm_loadu_ps(srcPtr);
    p128[1] = _mm_loadu_ps(srcPtr + 3);
    p128[2] = _mm_loadu_ps(srcPtr + 6);
    p128[3] = _mm_loadu_ps(srcPtr + 9);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    p128[4] = _mm_loadu_ps(srcPtr + 12);
    p128[5] = _mm_loadu_ps(srcPtr + 15);
    p128[6] = _mm_loadu_ps(srcPtr + 18);
    p128[7] = _mm_loadu_ps(srcPtr + 21);
    _MM_TRANSPOSE4_PS(p128[4], p128[5], p128[6], p128[7]);
    p[0] = _mm256_cvtps_pd(p128[0]);
    p[1] = _mm256_cvtps_pd(p128[4]);
    p[2] = _mm256_cvtps_pd(p128[1]);
    p[3] = _mm256_cvtps_pd(p128[5]);
    p[4] = _mm256_cvtps_pd(p128[2]);
    p[5] = _mm256_cvtps_pd(p128[6]);
}

inline void rpp_load24_f32pln3_to_f32pln3_avx(Rpp32f *srcPtrR, Rpp32f *srcPtrG, Rpp32f *srcPtrB, __m256 *p)
{
    p[0] = _mm256_loadu_ps(srcPtrR);
    p[1] = _mm256_loadu_ps(srcPtrG);
    p[2] = _mm256_loadu_ps(srcPtrB);
}

inline void rpp_load24_f32pln3_to_f32pln3_mirror_avx(Rpp32f *srcPtrR, Rpp32f *srcPtrG, Rpp32f *srcPtrB, __m256 *p)
{
    __m256i pxMask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    p[0] = _mm256_loadu_ps(srcPtrR); /* loads pixels R01-R08 */
    p[1] = _mm256_loadu_ps(srcPtrG); /* loads pixels G01-G08 */
    p[2] = _mm256_loadu_ps(srcPtrB); /* loads pixels G01-B08 */

    p[0] = _mm256_permutevar8x32_ps(p[0], pxMask); /* shuffle as R08-R01 */
    p[1] = _mm256_permutevar8x32_ps(p[1], pxMask); /* shuffle as G08-G01 */
    p[2] = _mm256_permutevar8x32_ps(p[2], pxMask); /* shuffle as B08-B01 */
}

inline void rpp_store24_f32pln3_to_f32pkd3_avx(Rpp32f *dstPtr, __m256 *p)
{
    __m256 pTemp[4], pRow[4];
    pTemp[0] = _mm256_shuffle_ps(p[0], p[1], 0x44); /* shuffle to get R01|R02|G01|G02|R05|R06|G05|G06 */
    pTemp[2] = _mm256_shuffle_ps(p[0], p[1], 0xEE); /* shuffle to get R03|R04|G03|G04|R07|R08|G07|G08 */
    pTemp[1] = _mm256_shuffle_ps(p[2], avx_p0, 0x44); /* shuffle to get B01|B02|00|00|B05|B06|00|00 */
    pTemp[3] = _mm256_shuffle_ps(p[2], avx_p0, 0xEE); /* shuffle to get B03|B04|00|00|B07|B08|00|00 */
    pRow[0] = _mm256_shuffle_ps(pTemp[0], pTemp[1], 0x88); /* shuffle to get R01|G01|B01|00|R05|G05|B05|00 */
    pRow[1] = _mm256_shuffle_ps(pTemp[0], pTemp[1], 0xDD); /* shuffle to get R02|G02|B02|00|R06|G06|B05|00 */
    pRow[2] = _mm256_shuffle_ps(pTemp[2], pTemp[3], 0x88); /* shuffle to get R03|G03|B03|00|R07|G07|B05|00 */
    pRow[3] = _mm256_shuffle_ps(pTemp[2], pTemp[3], 0xDD); /* shuffle to get R04|G04|B04|00|R08|G08|B05|00 */

    __m128 p128[4];
    p128[1] = _mm256_castps256_ps128(pRow[1]); /* get R01|G01|B01|00 */
    p128[2] = _mm256_castps256_ps128(pRow[2]); /* get R02|G02|B02|00 */
    p128[3] = _mm256_castps256_ps128(pRow[3]); /* get R03|G03|B03|00 */
    p128[0] = _mm256_castps256_ps128(pRow[0]); /* get R04|G04|B04|00 */
    _mm_storeu_ps(dstPtr, p128[0]);
    _mm_storeu_ps(dstPtr + 3, p128[1]);
    _mm_storeu_ps(dstPtr + 6, p128[2]);
    _mm_storeu_ps(dstPtr + 9, p128[3]);

    p128[0] = _mm256_extractf128_ps(pRow[0], 1); /* get R05|G05|B05|00 */
    p128[1] = _mm256_extractf128_ps(pRow[1], 1); /* get R06|G06|B06|00 */
    p128[2] = _mm256_extractf128_ps(pRow[2], 1); /* get R07|G07|B07|00 */
    p128[3] = _mm256_extractf128_ps(pRow[3], 1); /* get R08|G08|B08|00 */
    _mm_storeu_ps(dstPtr + 12, p128[0]);
    _mm_storeu_ps(dstPtr + 15, p128[1]);
    _mm_storeu_ps(dstPtr + 18, p128[2]);
    _mm_storeu_ps(dstPtr + 21, p128[3]);
}

inline void rpp_load24_f32pln3_to_f64pln3_avx(Rpp32f *srcPtrR, Rpp32f *srcPtrG, Rpp32f *srcPtrB, __m256d *p)
{
    __m128 px128[6];
    __m256 px[3];
    px[0] = _mm256_loadu_ps(srcPtrR);
    px[1] = _mm256_loadu_ps(srcPtrG);
    px[2] = _mm256_loadu_ps(srcPtrB);
    px128[0] = _mm256_castps256_ps128(px[0]);
    px128[1] = _mm256_extractf128_ps(px[0], 1);
    px128[2] = _mm256_castps256_ps128(px[1]);
    px128[3] = _mm256_extractf128_ps(px[1], 1);
    px128[4] = _mm256_castps256_ps128(px[2]);
    px128[5] = _mm256_extractf128_ps(px[2], 1);
    p[0] = _mm256_cvtps_pd(px128[0]);
    p[1] = _mm256_cvtps_pd(px128[1]);
    p[2] = _mm256_cvtps_pd(px128[2]);
    p[3] = _mm256_cvtps_pd(px128[3]);
    p[4] = _mm256_cvtps_pd(px128[4]);
    p[5] = _mm256_cvtps_pd(px128[5]);
}

inline void rpp_load16_f32_to_f32_avx(Rpp32f *srcPtr, __m256 *p)
{
    p[0] = _mm256_loadu_ps(srcPtr);
    p[1] = _mm256_loadu_ps(srcPtr + 8);
}

inline void rpp_load24_f32_to_f32_avx(Rpp32f *srcPtr, __m256 *p)
{
    p[0] = _mm256_loadu_ps(srcPtr);
    p[1] = _mm256_loadu_ps(srcPtr + 8);
    p[2] = _mm256_loadu_ps(srcPtr + 16);
}

inline void rpp_load32_f32_to_f32_avx(Rpp32f *srcPtr, __m256 *p)
{
    p[0] = _mm256_loadu_ps(srcPtr);
    p[1] = _mm256_loadu_ps(srcPtr + 8);
    p[2] = _mm256_loadu_ps(srcPtr + 16);
    p[3] = _mm256_loadu_ps(srcPtr + 24);
}

inline void rpp_load40_f32_to_f32_avx(Rpp32f *srcPtr, __m256 *p)
{
    p[0] = _mm256_loadu_ps(srcPtr);
    p[1] = _mm256_loadu_ps(srcPtr + 8);
    p[2] = _mm256_loadu_ps(srcPtr + 16);
    p[3] = _mm256_loadu_ps(srcPtr + 24);
    p[4] = _mm256_loadu_ps(srcPtr + 32);
}

inline void rpp_store16_f32_to_f32_avx(Rpp32f *dstPtr, __m256 *p)
{
    _mm256_storeu_ps(dstPtr, p[0]);
    _mm256_storeu_ps(dstPtr + 8, p[1]);
}

inline void rpp_store16_f32_to_f16_avx(Rpp16f *dstPtr, __m256 *p)
{
    __m128i px128[2];
    px128[0] = _mm256_cvtps_ph(p[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm256_cvtps_ph(p[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 8), px128[1]);
}

inline void rpp_load8_f32_to_f32_avx(Rpp32f *srcPtr, __m256 *p)
{
    p[0] = _mm256_loadu_ps(srcPtr);
}

inline void rpp_load8_f32_to_f32_mirror_avx(Rpp32f *srcPtr, __m256 *p)
{
    __m256i pxMask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    p[0] = _mm256_loadu_ps(srcPtr);
    p[0] = _mm256_permutevar8x32_ps(p[0], pxMask); /* shuffle as R08-R01 */
}

inline void rpp_store8_f32_to_f32_avx(Rpp32f *dstPtr, __m256 *p)
{
    _mm256_storeu_ps(dstPtr, p[0]);
}

inline void rpp_load8_f32_to_f64_avx(Rpp32f *srcPtr, __m256d *p)
{
    __m128 px128[2];
    __m256 px;
    px = _mm256_loadu_ps(srcPtr);
    px128[0] = _mm256_castps256_ps128(px);
    px128[1] = _mm256_extractf128_ps(px, 1);
    p[0] = _mm256_cvtps_pd(px128[0]);
    p[1] = _mm256_cvtps_pd(px128[1]);
}

inline void rpp_store8_u32_to_u32_avx(Rpp32u *dstPtr, __m256i *p)
{
    _mm256_store_si256((__m256i *)dstPtr, p[0]);
}

inline void rpp_store8_i32_to_i32_avx(Rpp32s *dstPtr, __m256i *p)
{
    _mm256_store_si256((__m256i *)dstPtr, p[0]);
}

inline void rpp_store4_f64_to_f64_avx(Rpp64f *dstPtr, __m256d *p)
{
    _mm256_storeu_pd(dstPtr, p[0]);
}

inline void rpp_store16_u8_to_u8(Rpp8u *dstPtr, __m128i *p)
{
    _mm_storeu_si128((__m128i *)dstPtr, p[0]);
}

inline void rpp_store16_i8(Rpp8s *dstPtr, __m128i *p)
{
    _mm_store_si128((__m128i *)dstPtr, p[0]);
}

inline void rpp_store8_f32_to_f16_avx(Rpp16f *dstPtr, __m256 *p)
{
    __m128i px128 = _mm256_cvtps_ph(p[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128);
}

inline void rpp_load48_i8pkd3_to_f32pln3_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m128i px[4];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));           /* add I8 conversion param to load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 12)));    /* add I8 conversion param to load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 24)));    /* add I8 conversion param to load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 36)));    /* add I8 conversion param to load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskR), _mm_shuffle_epi8(px[1], xmm_pxMaskR)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskR), _mm_shuffle_epi8(px[3], xmm_pxMaskR)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskG), _mm_shuffle_epi8(px[1], xmm_pxMaskG)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskG), _mm_shuffle_epi8(px[3], xmm_pxMaskG)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMaskB), _mm_shuffle_epi8(px[1], xmm_pxMaskB)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMaskB), _mm_shuffle_epi8(px[3], xmm_pxMaskB)));    /* Contains B09-16 */
}

inline void rpp_load48_i8pkd3_to_f32pln3_mirror_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m128i px[4];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));           /* add I8 conversion param to load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 12)));    /* add I8 conversion param to load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 24)));    /* add I8 conversion param to load [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|R13|G13|B13|R14] - Need RGB 09-12 */
    px[3] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 36)));    /* add I8 conversion param to load [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|R17|G17|B17|R18] - Need RGB 13-16 */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMaskRMirror), _mm_shuffle_epi8(px[2], xmm_pxMaskRMirror)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMaskRMirror), _mm_shuffle_epi8(px[0], xmm_pxMaskRMirror)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMaskGMirror), _mm_shuffle_epi8(px[2], xmm_pxMaskGMirror)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMaskGMirror), _mm_shuffle_epi8(px[0], xmm_pxMaskGMirror)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[3], xmm_pxMaskBMirror), _mm_shuffle_epi8(px[2], xmm_pxMaskBMirror)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMaskBMirror), _mm_shuffle_epi8(px[0], xmm_pxMaskBMirror)));    /* Contains B09-16 */
}

inline void rpp_store48_f32pln3_to_i8pln3_avx(Rpp8s *dstPtrR, Rpp8s *dstPtrG, Rpp8s *dstPtrB, __m256 *p)
{
    __m256i pxCvt;
    __m128i px[4];

    pxCvt = _mm256_cvtps_epi32(p[0]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for R */
    pxCvt = _mm256_cvtps_epi32(p[1]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for R */
    pxCvt = _mm256_cvtps_epi32(p[2]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for G */
    pxCvt = _mm256_cvtps_epi32(p[3]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for G */
    px[1] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for G */
    pxCvt = _mm256_cvtps_epi32(p[4]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for B */
    pxCvt = _mm256_cvtps_epi32(p[5]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for B */
    px[2] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for B */
    px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    px[1] = _mm_sub_epi8(px[1], xmm_pxConvertI8);    /* convert back to i8 for px1 store */
    px[2] = _mm_sub_epi8(px[2], xmm_pxConvertI8);    /* convert back to i8 for px2 store */
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

inline void rpp_load48_i8pln3_to_f32pln3_avx(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m256 *p)
{
    __m128i px[3];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrR));    /* add I8 conversion param to load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrG));    /* add I8 conversion param to load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrB));    /* add I8 conversion param to load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask00To03), _mm_shuffle_epi8(px[0], xmm_pxMask04To07)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask08To11), _mm_shuffle_epi8(px[0], xmm_pxMask12To15)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask00To03), _mm_shuffle_epi8(px[1], xmm_pxMask04To07)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask08To11), _mm_shuffle_epi8(px[1], xmm_pxMask12To15)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[2], xmm_pxMask04To07)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[2], xmm_pxMask12To15)));    /* Contains B09-16 */
}

inline void rpp_load48_i8pln3_to_f32pln3_mirror_avx(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m256 *p)
{
    __m128i px[3];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrR));    /* add I8 conversion param to load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrG));    /* add I8 conversion param to load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrB));    /* add I8 conversion param to load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask15To12), _mm_shuffle_epi8(px[0], xmm_pxMask11To08)));    /* Contains R01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[0], xmm_pxMask07To04), _mm_shuffle_epi8(px[0], xmm_pxMask03To00)));    /* Contains R09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask15To12), _mm_shuffle_epi8(px[1], xmm_pxMask11To08)));    /* Contains G01-08 */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[1], xmm_pxMask07To04), _mm_shuffle_epi8(px[1], xmm_pxMask03To00)));    /* Contains G09-16 */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask15To12), _mm_shuffle_epi8(px[2], xmm_pxMask11To08)));    /* Contains B01-08 */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px[2], xmm_pxMask07To04), _mm_shuffle_epi8(px[2], xmm_pxMask03To00)));    /* Contains B09-16 */
}

inline void rpp_store48_f32pln3_to_i8pkd3_avx(Rpp8s *dstPtr, __m256 *p)
{
    __m256i pxCvt[3];
    __m128i px[5];
    __m128i pxMask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);

    pxCvt[0] = _mm256_cvtps_epi32(p[0]);    /* convert to int32 for R01-08 */
    pxCvt[1] = _mm256_cvtps_epi32(p[2]);    /* convert to int32 for G01-08 */
    pxCvt[2] = _mm256_cvtps_epi32(p[4]);    /* convert to int32 for B01-08 */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 0), _mm256_extracti128_si256(pxCvt[1], 0));    /* pack pixels 0-7 as R01-04|G01-04 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 0), xmm_px0);    /* pack pixels 8-15 as B01-04|X01-04 */
    px[0] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00] */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 1), _mm256_extracti128_si256(pxCvt[1], 1));    /* pack pixels 0-7 as R05-08|G05-08 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 1), xmm_px0);    /* pack pixels 8-15 as B05-08|X05-08 */
    px[1] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00] */
    pxCvt[0] = _mm256_cvtps_epi32(p[1]);    /* convert to int32 for R09-16 */
    pxCvt[1] = _mm256_cvtps_epi32(p[3]);    /* convert to int32 for G09-16 */
    pxCvt[2] = _mm256_cvtps_epi32(p[5]);    /* convert to int32 for B09-16 */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 0), _mm256_extracti128_si256(pxCvt[1], 0));    /* pack pixels 0-7 as R09-12|G09-12 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 0), xmm_px0);    /* pack pixels 8-15 as B09-12|X09-12 */
    px[2] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00] */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 1), _mm256_extracti128_si256(pxCvt[1], 1));    /* pack pixels 0-7 as R13-16|G13-16 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 1), xmm_px0);    /* pack pixels 8-15 as B13-16|X13-16 */
    px[3] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00] */
    px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    px[1] = _mm_sub_epi8(px[1], xmm_pxConvertI8);    /* convert back to i8 for px1 store */
    px[2] = _mm_sub_epi8(px[2], xmm_pxConvertI8);    /* convert back to i8 for px2 store */
    px[3] = _mm_sub_epi8(px[3], xmm_pxConvertI8);    /* convert back to i8 for px3 store */
    px[0] = _mm_shuffle_epi8(px[0], pxMask);    /* shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    px[1] = _mm_shuffle_epi8(px[1], pxMask);    /* shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    px[2] = _mm_shuffle_epi8(px[2], pxMask);    /* shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    px[3] = _mm_shuffle_epi8(px[3], pxMask);    /* shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px[1]);    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px[2]);    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px[3]);    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

inline void rpp_load24_i8pln3_to_f64pln3_avx(Rpp8s *srcPtrR, Rpp8s *srcPtrG, Rpp8s *srcPtrB, __m256d *p)
{
    __m128i px[3];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrR));    /* add I8 conversion param to load [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrG));    /* add I8 conversion param to load [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    px[2] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtrB));    /* add I8 conversion param to load [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
    p[0] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMask00To03));    /* Contains R01-04 */
    p[1] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMask04To07));    /* Contains R05-08 */
    p[2] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMask00To03));    /* Contains G01-04 */
    p[3] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMask04To07));    /* Contains G05-08 */
    p[4] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[2], xmm_pxMask00To03));    /* Contains B01-04 */
    p[5] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[2], xmm_pxMask04To07));    /* Contains B05-08 */
}

inline void rpp_load24_i8pkd3_to_f64pln3_avx(Rpp8s *srcPtr, __m256d *p)
{
    __m128i px[2];

    px[0] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));           /* add I8 conversion param to load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01-04 */
    px[1] = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 12)));    /* add I8 conversion param to load [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|R09|G09|B09|R10] - Need RGB 05-08 */
    p[0] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMaskR));    /* Contains R01-04 */
    p[1] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMaskR));    /* Contains R05-08 */
    p[2] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMaskG));    /* Contains G01-04 */
    p[3] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMaskG));    /* Contains G05-08 */
    p[4] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[0], xmm_pxMaskB));    /* Contains B01-04 */
    p[5] = _mm256_cvtepi32_pd(_mm_shuffle_epi8(px[1], xmm_pxMaskB));    /* Contains B05-08 */
}

inline void rpp_load16_i8_to_f32_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m128i px;
    px = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));    /* add I8 conversion param to load */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask00To03), _mm_shuffle_epi8(px, xmm_pxMask04To07)));    /* Contains pixels 01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask08To11), _mm_shuffle_epi8(px, xmm_pxMask12To15)));    /* Contains pixels 09-16 */
}

inline void rpp_load16_abs_i16_to_f32_avx(Rpp16s *srcPtr, __m256 *p)
{
    __m256i px =  _mm256_loadu_si256((__m256i *)srcPtr);  

    //Extracting 16 bits from the px and converting from 16 bit int to 32 bit int
    __m256i px0 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(px, 0)); 
    __m256i px1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(px, 1)); 

    //Taking absolute values for i32 
    __m256i abs_px0 = _mm256_abs_epi32(px0);  
    __m256i abs_px1 = _mm256_abs_epi32(px1); 
    
    // Convert 32 bit int to 32 bit floats
    p[0] = _mm256_cvtepi32_ps(abs_px0); 
    p[1] = _mm256_cvtepi32_ps(abs_px1); 
}

inline void rpp_load24_i8_to_f32_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m128i px1, px2;
    px1 = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr)));
    px2 = _mm_add_epi8(xmm_pxConvertI8, _mm_loadl_epi64((__m128i *)(srcPtr + 16)));

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask00To03), _mm_shuffle_epi8(px1, xmm_pxMask04To07)));  /* Contains pixels 01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask08To11), _mm_shuffle_epi8(px1, xmm_pxMask12To15)));  /* Contains pixels 09-16 */
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask00To03), _mm_shuffle_epi8(px2, xmm_pxMask04To07)));  /* Contains pixels 17-24 */
}

inline void rpp_load32_i8_to_f32_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m256i px = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtr));
    __m128i px1 = _mm256_castps256_ps128(px);
    __m128i px2 = _mm256_extractf128_si256(px, 1);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask00To03), _mm_shuffle_epi8(px1, xmm_pxMask04To07))); // Contains pixels 01-08
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1, xmm_pxMask08To11), _mm_shuffle_epi8(px1, xmm_pxMask12To15))); // Contains pixels 09-16
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask00To03), _mm_shuffle_epi8(px2, xmm_pxMask04To07))); // Contains pixels 17-24
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask08To11), _mm_shuffle_epi8(px2, xmm_pxMask12To15))); // Contains pixels 25-32
}

inline void rpp_load40_i8_to_f32_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m256i px1 = _mm256_add_epi8(avx_pxConvertI8, _mm256_loadu_si256((__m256i *)srcPtr));     // Load the first 32 bytes
    __m128i px2 = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)(srcPtr + 32))); // Load the remaining 8 bytes
    __m128i px1Low  = _mm256_castsi256_si128(px1);
    __m128i px1High = _mm256_extractf128_si256(px1, 1);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1Low, xmm_pxMask00To03), _mm_shuffle_epi8(px1Low, xmm_pxMask04To07))); // Pixels 01-08
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1Low, xmm_pxMask08To11), _mm_shuffle_epi8(px1Low, xmm_pxMask12To15))); // Pixels 09-16
    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1High, xmm_pxMask00To03), _mm_shuffle_epi8(px1High, xmm_pxMask04To07))); // Pixels 17-24
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px1High, xmm_pxMask08To11), _mm_shuffle_epi8(px1High, xmm_pxMask12To15))); // Pixels 25-32
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px2, xmm_pxMask00To03), _mm_shuffle_epi8(px2, xmm_pxMask04To07)));        // Pixels 33-40
}

inline void rpp_load16_i8_to_f32_mirror_avx(Rpp8s *srcPtr, __m256 *p)
{
    __m128i px;
    px = _mm_add_epi8(xmm_pxConvertI8, _mm_loadu_si128((__m128i *)srcPtr));    /* add I8 conversion param to load */
    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask15To12), _mm_shuffle_epi8(px, xmm_pxMask11To08)));    /* Contains pixels 01-08 */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128i(_mm_shuffle_epi8(px, xmm_pxMask07To04), _mm_shuffle_epi8(px, xmm_pxMask03To00)));    /* Contains pixels 09-16 */
}

inline void rpp_store16_f32_to_i8_avx(Rpp8s *dstPtr, __m256 *p)
{
    __m256i pxCvt;
    __m128i px[3];
    pxCvt = _mm256_cvtps_epi32(p[0]);
    px[1] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for R */
    pxCvt = _mm256_cvtps_epi32(p[1]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[1], px[2]);    /* pack pixels 0-15 */
    px[0] = _mm_sub_epi8(px[0], xmm_pxConvertI8);    /* convert back to i8 for px0 store */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);
}

inline void rpp_load16_i8_to_i32_avx(Rpp8s *srcPtr, __m256i *p)
{
    __m128i px;
    px = _mm_loadu_si128((__m128i *)srcPtr);
    p[0] = _mm256_cvtepi8_epi32(px);    /* Contains pixels 01-08 */
    p[1] = _mm256_cvtepi8_epi32(_mm_shuffle_epi8(px, xmm_pxMask08To15));    /* Contains pixels 09-16 */
}

inline void rpp_normalize48_avx(__m256 *p)
{
    p[0] = _mm256_mul_ps(p[0], avx_p1op255);
    p[1] = _mm256_mul_ps(p[1], avx_p1op255);
    p[2] = _mm256_mul_ps(p[2], avx_p1op255);
    p[3] = _mm256_mul_ps(p[3], avx_p1op255);
    p[4] = _mm256_mul_ps(p[4], avx_p1op255);
    p[5] = _mm256_mul_ps(p[5], avx_p1op255);
}

inline void rpp_multiply48_constant(__m256 *p, __m256 pMultiplier)
{
    p[0] = _mm256_mul_ps(p[0], pMultiplier);
    p[1] = _mm256_mul_ps(p[1], pMultiplier);
    p[2] = _mm256_mul_ps(p[2], pMultiplier);
    p[3] = _mm256_mul_ps(p[3], pMultiplier);
    p[4] = _mm256_mul_ps(p[4], pMultiplier);
    p[5] = _mm256_mul_ps(p[5], pMultiplier);
}

inline void rpp_multiply48_constant(__m128 *p, __m128 pMultiplier)
{
    p[0] = _mm_mul_ps(p[0], pMultiplier);
    p[1] = _mm_mul_ps(p[1], pMultiplier);
    p[2] = _mm_mul_ps(p[2], pMultiplier);
    p[3] = _mm_mul_ps(p[3], pMultiplier);
    p[4] = _mm_mul_ps(p[4], pMultiplier);
    p[5] = _mm_mul_ps(p[5], pMultiplier);
    p[6] = _mm_mul_ps(p[6], pMultiplier);
    p[7] = _mm_mul_ps(p[7], pMultiplier);
    p[8] = _mm_mul_ps(p[8], pMultiplier);
    p[9] = _mm_mul_ps(p[9], pMultiplier);
    p[10] = _mm_mul_ps(p[10], pMultiplier);
    p[11] = _mm_mul_ps(p[11], pMultiplier);
}

inline void rpp_multiply16_constant(__m256 *p, __m256 pMultiplier)
{
    p[0] = _mm256_mul_ps(p[0], pMultiplier);
    p[1] = _mm256_mul_ps(p[1], pMultiplier);
}

inline void rpp_multiply16_constant(__m128 *p, __m128 pMultiplier)
{
    p[0] = _mm_mul_ps(p[0], pMultiplier);
    p[1] = _mm_mul_ps(p[1], pMultiplier);
    p[2] = _mm_mul_ps(p[2], pMultiplier);
    p[3] = _mm_mul_ps(p[3], pMultiplier);
}

template <typename FuncType, typename... ArgTypes>
inline void rpp_simd_load(FuncType &&rpp_simd_load_routine, ArgTypes&&... args)
{
    std::forward<FuncType>(rpp_simd_load_routine)(std::forward<ArgTypes>(args)...);
}

template <typename FuncType, typename... ArgTypes>
inline void rpp_simd_store(FuncType &&rpp_simd_store_routine, ArgTypes&&... args)
{
    std::forward<FuncType>(rpp_simd_store_routine)(std::forward<ArgTypes>(args)...);
}

// Shuffle floats in `src` by using SSE2 `pshufd` instead of `shufps`, if possible.
#define SIMD_SHUFFLE_PS(src, imm) \
    _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(src), imm))

#define CHECK_SIMD  0
#define FP_BITS     16
#define FP_MUL      (1<<FP_BITS)

const __m128 xmm_full = _mm_set1_ps((float)0xFFFFFFFF);
const __m128 xmm_sn = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
const __m128 xmm_m6 = _mm_set1_ps((float)-6.0);
const __m128 xmm_eps = _mm_set1_ps((float)1e-9f);
const __m128 xmm_1o3 = _mm_set1_ps((float)1.0/3.0);
const __m128 xmm_m4o6 = _mm_set1_ps((float) -4.0/6.0);
const __m128 xmm_abs = _mm_set1_ps((float)0x80000000);

const __m128 xmm_4o6_2o6_3o6_0  = _mm_set_ps(4.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 0.0f);
const __m128 m6_m6_p6_p0        = _mm_set_ps(-6.0f ,-6.0f , 6.0f , 0.0f);
const __m128 p1_p1_m2_p0        = _mm_set_ps(1.0f , 1.0f ,-2.0f , 0.0f);
const __m128 m1_m1_m1_p1        = _mm_set_ps(-1.0f ,-1.0f ,-1.0f , 1.0f);

SIMD_CONST_PI(full       , 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
SIMD_CONST_PI(sn         , 0x80000000, 0x80000000, 0x80000000, 0x80000000);
SIMD_CONST_PS(m6_m6_m6_m6,-6.0f ,-6.0f ,-6.0f ,-6.0f);
SIMD_CONST_PS(m4o6_m4o6_m4o6_m4o6,-4.0f / 6.0f,-4.0f / 6.0f,-4.0f / 6.0f,-4.0f / 6.0f);
SIMD_CONST_PS(eps        , 1e-9f, 1e-9f, 1e-9f, 1e-9f);
SIMD_CONST_PS(p1         , 1.0f , 1.0f , 1.0f , 1.0f);

SIMD_CONST_PS(p4o6_p2o6_p3o6_p0  , 4.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 0.0f);
SIMD_CONST_PI(abs        , 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
SIMD_CONST_PS(m6_m6_p6_p0,-6.0f ,-6.0f , 6.0f , 0.0f);
SIMD_CONST_PS(p1_p1_m2_p0, 1.0f , 1.0f ,-2.0f , 0.0f);
SIMD_CONST_PS(m1_m1_m1_p1,-1.0f ,-1.0f ,-1.0f , 1.0f);
SIMD_CONST_PS(p0         , 0.0f , 0.0f , 0.0f , 0.0f);

inline __m128i _mm_mullo_epi8(__m128i a, __m128i b)
{
    __m128i zero = _mm_setzero_si128();
    __m128i Alo = _mm_cvtepu8_epi16(a);
    __m128i Ahi = _mm_unpackhi_epi8(a, zero);
    __m128i Blo = _mm_cvtepu8_epi16(b);
    __m128i Bhi = _mm_unpackhi_epi8(b, zero);
    __m128i Clo = _mm_mullo_epi16(Alo, Blo);
    __m128i Chi = _mm_mullo_epi16(Ahi, Bhi);
    __m128i maskLo = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i maskHi = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    __m128i C = _mm_or_si128(_mm_shuffle_epi8(Clo, maskLo), _mm_shuffle_epi8(Chi, maskHi));

     return C;
}

static inline Rpp32u HorMin(__m128i pmin)
{
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm_min_epu8(pmin, _mm_shuffle_epi32(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_shufflelo_epi16(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin = _mm_min_epu8(pmin, _mm_srli_epi16(pmin, 8));
    return (_mm_cvtsi128_si32(pmin) & 0x000000FF);
}

static inline Rpp32u HorMax(__m128i pmax)
{
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm_min_epu8(pmax, _mm_shuffle_epi32(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_shufflelo_epi16(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax = _mm_min_epu8(pmax, _mm_srli_epi16(pmax, 8));
    return (_mm_cvtsi128_si32(pmax) & 0x000000FF);
}

#if __AVX__
static inline Rpp32u HorMin256(__m256i pmin)
{
    __m128i pmin_128;
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(3, 2, 3, 2)));
    pmin = _mm256_min_epu8(pmin, _mm256_permute4x64_epi64(pmin, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = M256I(pmin).m256i_i128[0];
    pmin_128 = _mm_min_epu8(pmin_128, _mm_shufflelo_epi16(pmin_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmin_128 = _mm_min_epu8(pmin_128, _mm_srli_epi16(pmin_128, 8));
    return (_mm_cvtsi128_si32(pmin_128) & 0x000000FF);
}

static inline Rpp32u HorMax256(__m256i pmax)
{
    __m128i pmax_128;
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(3, 2, 3, 2)));
    pmax = _mm256_max_epu8(pmax, _mm256_permute4x64_epi64(pmax, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = M256I(pmax).m256i_i128[0];
    pmax_128 = _mm_max_epi8(pmax_128, _mm_shufflelo_epi16(pmax_128, _MM_SHUFFLE(1, 1, 1, 1)));
    pmax_128 = _mm_max_epi8(pmax_128, _mm_srli_epi16(pmax_128, 8));
    return (_mm_cvtsi128_si32(pmax_128) & 0x000000FF);
}
#endif

static  inline __m128 fast_exp_sse (__m128 x)
{
    __m128 t, f, e, p, r;
    __m128i i, j;
    __m128 l2e = _mm_set1_ps (1.442695041f);  /* log2(e) */
    __m128 c0  = _mm_set1_ps (0.3371894346f);
    __m128 c1  = _mm_set1_ps (0.657636276f);
    __m128 c2  = _mm_set1_ps (1.00172476f);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    t = _mm_mul_ps (x, l2e);             /* t = log2(e) * x */
#ifdef __SSE4_1__
    e = _mm_floor_ps (t);                /* floor(t) */
    i = _mm_cvtps_epi32 (e);             /* (int)floor(t) */
#else /* __SSE4_1__*/
    i = _mm_cvttps_epi32 (t);            /* i = (int)t */
    j = _mm_srli_epi32 (_mm_castps_si128 (x), 31); /* signbit(t) */
    i = _mm_sub_epi32 (i, j);            /* (int)t - signbit(t) */
    e = _mm_cvtepi32_ps (i);             /* floor(t) ~= (int)t - signbit(t) */
#endif /* __SSE4_1__*/
    f = _mm_sub_ps (t, e);               /* f = t - floor(t) */
    p = c0;                              /* c0 */
    p = _mm_mul_ps (p, f);               /* c0 * f */
    p = _mm_add_ps (p, c1);              /* c0 * f + c1 */
    p = _mm_mul_ps (p, f);               /* (c0 * f + c1) * f */
    p = _mm_add_ps (p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    j = _mm_slli_epi32 (i, 23);          /* i << 23 */
    r = _mm_castsi128_ps (_mm_add_epi32 (j, _mm_castps_si128 (p))); /* r = p * 2^i*/
    return r;
}

#if __AVX2__
static inline __m256 fast_exp_avx(__m256 x)
{
    __m256 t, f, e, p, r;
    __m256i i, j;
    __m256 l2e = _mm256_set1_ps(1.442695041f);    /* log2(e) */
    __m256 c0  = _mm256_set1_ps(0.3371894346f);
    __m256 c1  = _mm256_set1_ps(0.657636276f);
    __m256 c2  = _mm256_set1_ps(1.00172476f);

    /* exp(x) = 2^i * 2^f; i = floor (log2(e) * x), 0 <= f <= 1 */
    t = _mm256_mul_ps(x, l2e);             /* t = log2(e) * x */
    e = _mm256_floor_ps(t);                /* floor(t) */
    i = _mm256_cvtps_epi32(e);             /* (int)floor(t) */
    f = _mm256_sub_ps(t, e);               /* f = t - floor(t) */
    p = c0;                                /* c0 */
    p = _mm256_mul_ps(p, f);               /* c0 * f */
    p = _mm256_add_ps(p, c1);              /* c0 * f + c1 */
    p = _mm256_mul_ps(p, f);               /* (c0 * f + c1) * f */
    p = _mm256_add_ps(p, c2);              /* p = (c0 * f + c1) * f + c2 ~= 2^f */
    j = _mm256_slli_epi32(i, 23);          /* i << 23 */
    r = _mm256_castsi256_ps(_mm256_add_epi32(j, _mm256_castps_si256(p)));    /* r = p * 2^i*/
    return r;
}
#endif

#define set1_ps_hex(x) _mm_castsi128_ps(_mm_set1_epi32(x))
#define set1_ps_hex_avx(x) _mm256_castsi256_ps(_mm256_set1_epi32(x))

static const __m128 _ps_0 = _mm_set1_ps(0.f);
static const __m128 _ps_1 = _mm_set1_ps(1.f);
static const __m128 _ps_0p5 = _mm_set1_ps(0.5f);
static const __m128 _ps_n0p5 = _mm_set1_ps(-0.5f);
static const __m128 _ps_1p5 = _mm_set1_ps(1.5f);
static const __m128 _ps_min_norm_pos = set1_ps_hex(0x00800000);
static const __m128 _ps_mant_mask = set1_ps_hex(0x7f800000);
static const __m128 _ps_inv_mant_mask = set1_ps_hex(~0x7f800000);
static const __m128 _ps_sign_mask = set1_ps_hex(0x80000000);
static const __m128 _ps_inv_sign_mask = set1_ps_hex(~0x80000000);

static const __m128i _pi32_1 = _mm_set1_epi32(1);
static const __m128i _pi32_inv1 = _mm_set1_epi32(~1);
static const __m128i _pi32_2 = _mm_set1_epi32(2);
static const __m128i _pi32_4 = _mm_set1_epi32(4);
static const __m128i _pi32_0x7f = _mm_set1_epi32(0x7f);

static const __m128 _ps_minus_cephes_DP1 = _mm_set1_ps(-0.78515625f);
static const __m128 _ps_minus_cephes_DP2 = _mm_set1_ps(-2.4187564849853515625e-4f);
static const __m128 _ps_minus_cephes_DP3 = _mm_set1_ps(-3.77489497744594108e-8f);
static const __m128 _ps_sincof_p0 = _mm_set1_ps(-1.9515295891E-4f);
static const __m128 _ps_sincof_p1 = _mm_set1_ps( 8.3321608736E-3f);
static const __m128 _ps_sincof_p2 = _mm_set1_ps(-1.6666654611E-1f);
static const __m128 _ps_coscof_p0 = _mm_set1_ps( 2.443315711809948E-005f);
static const __m128 _ps_coscof_p1 = _mm_set1_ps(-1.388731625493765E-003f);
static const __m128 _ps_coscof_p2 = _mm_set1_ps( 4.166664568298827E-002f);
static const __m128 _ps_cephes_FOPI = _mm_set1_ps(1.27323954473516f); // 4 / M_PI

static const __m256 _ps_0p5_avx = _mm256_set1_ps(0.5f);
static const __m256 _ps_n0p5_avx = _mm256_set1_ps(-0.5f);
static const __m256 _ps_1p5_avx = _mm256_set1_ps(1.5f);
static const __m256 _ps_min_norm_pos_avx = set1_ps_hex_avx(0x00800000);
static const __m256 _ps_inv_mant_mask_avx = set1_ps_hex_avx(~0x7f800000);
static const __m256 _ps_sign_mask_avx = set1_ps_hex_avx(0x80000000);
static const __m256 _ps_inv_sign_mask_avx = set1_ps_hex_avx(~0x80000000);

static const __m256i _pi32_1_avx = _mm256_set1_epi32(1);
static const __m256i _pi32_inv1_avx = _mm256_set1_epi32(~1);
static const __m256i _pi32_2_avx = _mm256_set1_epi32(2);
static const __m256i _pi32_4_avx = _mm256_set1_epi32(4);
static const __m256i _pi32_0x7f_avx = _mm256_set1_epi32(0x7f);

static const __m256 _ps_minus_cephes_DP1_avx = _mm256_set1_ps(-0.78515625f);
static const __m256 _ps_minus_cephes_DP2_avx = _mm256_set1_ps(-2.4187564849853515625e-4f);
static const __m256 _ps_minus_cephes_DP3_avx = _mm256_set1_ps(-3.77489497744594108e-8f);
static const __m256 _ps_sincof_p0_avx = _mm256_set1_ps(-1.9515295891E-4f);
static const __m256 _ps_sincof_p1_avx = _mm256_set1_ps( 8.3321608736E-3f);
static const __m256 _ps_sincof_p2_avx = _mm256_set1_ps(-1.6666654611E-1f);
static const __m256 _ps_coscof_p0_avx = _mm256_set1_ps( 2.443315711809948E-005f);
static const __m256 _ps_coscof_p1_avx = _mm256_set1_ps(-1.388731625493765E-003f);
static const __m256 _ps_coscof_p2_avx = _mm256_set1_ps( 4.166664568298827E-002f);
static const __m256 _ps_cephes_FOPI_avx = _mm256_set1_ps(1.27323954473516f); // 4 / M_PI

static inline void sincos_ps(__m256 x, __m256 *s, __m256 *c)
{
    // Extract the sign bit (upper one)
    __m256 sign_bit_sin = _mm256_and_ps(x, _ps_sign_mask_avx);
    // take the absolute value
    x = _mm256_xor_ps(x, sign_bit_sin);

    // Scale by 4/Pi
    __m256 y = _mm256_mul_ps(x, _ps_cephes_FOPI_avx);

    // Store the integer part of y in emm2
    __m256i emm2 = _mm256_cvttps_epi32(y);

    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm256_add_epi32(emm2, _pi32_1_avx);
    emm2 = _mm256_and_si256(emm2, _pi32_inv1_avx);
    y = _mm256_cvtepi32_ps(emm2);

    __m256i emm4 = emm2;

    // Get the swap sign flag for the sine
    __m256i emm0 = _mm256_and_si256(emm2, _pi32_4_avx);
    emm0 = _mm256_slli_epi32(emm0, 29);
    __m256 swap_sign_bit_sin = _mm256_castsi256_ps(emm0);

    // Get the polynom selection mask for the sine
    emm2 = _mm256_and_si256(emm2, _pi32_2_avx);
    emm2 = _mm256_cmpeq_epi32(emm2, _mm256_setzero_si256());
    __m256 poly_mask = _mm256_castsi256_ps(emm2);
    // The magic pass: "Extended precision modular arithmetic - x = ((x - y * DP1) - y * DP2) - y * DP3;
    __m256 xmm1 = _mm256_mul_ps(y, _ps_minus_cephes_DP1_avx);
    __m256 xmm2 = _mm256_mul_ps(y, _ps_minus_cephes_DP2_avx);
    __m256 xmm3 = _mm256_mul_ps(y, _ps_minus_cephes_DP3_avx);
    x = _mm256_add_ps(_mm256_add_ps(x, xmm1), _mm256_add_ps(xmm2, xmm3));

    emm4 = _mm256_sub_epi32(emm4, _pi32_2_avx);
    emm4 = _mm256_andnot_si256(emm4, _pi32_4_avx);
    emm4 = _mm256_slli_epi32(emm4, 29);
    __m256 sign_bit_cos = _mm256_castsi256_ps(emm4);

    sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    // Evaluate the first polynom  (0 <= x <= Pi/4)
    __m256 z = _mm256_mul_ps(x,x);
    y = _ps_coscof_p0_avx;

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps_coscof_p1_avx);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _ps_coscof_p2_avx);
    y = _mm256_mul_ps(y, _mm256_mul_ps(z, z));
    __m256 tmp = _mm256_mul_ps(z, _ps_0p5_avx);
    y = _mm256_sub_ps(y, tmp);
    y = _mm256_add_ps(y, avx_p1);

    // Evaluate the second polynom  (Pi/4 <= x <= 0)

    __m256 y2 = _ps_sincof_p0_avx;
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps_sincof_p1_avx);
    y2 = _mm256_mul_ps(y2, z);
    y2 = _mm256_add_ps(y2, _ps_sincof_p2_avx);
    y2 = _mm256_mul_ps(y2, _mm256_mul_ps(z, x));
    y2 = _mm256_add_ps(y2, x);

    // Select the correct result from the two polynoms
    xmm3 = poly_mask;
    __m256 ysin2 = _mm256_and_ps(xmm3, y2);
    __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
    y2 = _mm256_sub_ps(y2,ysin2);
    y = _mm256_sub_ps(y, ysin1);

    xmm1 = _mm256_add_ps(ysin1,ysin2);
    xmm2 = _mm256_add_ps(y,y2);

    // Update the sign
    *s = _mm256_xor_ps(xmm1, sign_bit_sin);
    *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

static inline void sincos_ps(__m128 x, __m128 *s, __m128 *c)
{
    // Extract the sign bit (upper one)
    __m128 sign_bit_sin = _mm_and_ps(x, _ps_sign_mask);
    // take the absolute value
    x = _mm_xor_ps(x, sign_bit_sin);

    // Scale by 4/Pi
    __m128 y = _mm_mul_ps(x, _ps_cephes_FOPI);

    // Store the integer part of y in emm2
    __m128i emm2 = _mm_cvttps_epi32(y);

    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32(emm2, _pi32_1);
    emm2 = _mm_and_si128(emm2, _pi32_inv1);
    y = _mm_cvtepi32_ps(emm2);

    __m128i emm4 = emm2;

    // Get the swap sign flag for the sine
    __m128i emm0 = _mm_and_si128(emm2, _pi32_4);
    emm0 = _mm_slli_epi32(emm0, 29);
    __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

    // Get the polynom selection mask for the sine
    emm2 = _mm_and_si128(emm2, _pi32_2);
    emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
    __m128 poly_mask = _mm_castsi128_ps(emm2);
    // The magic pass: "Extended precision modular arithmetic - x = ((x - y * DP1) - y * DP2) - y * DP3;
    __m128 xmm1 = _mm_mul_ps(y, _ps_minus_cephes_DP1);
    __m128 xmm2 = _mm_mul_ps(y, _ps_minus_cephes_DP2);
    __m128 xmm3 = _mm_mul_ps(y, _ps_minus_cephes_DP3);
    x = _mm_add_ps(_mm_add_ps(x, xmm1), _mm_add_ps(xmm2, xmm3));

    emm4 = _mm_sub_epi32(emm4, _pi32_2);
    emm4 = _mm_andnot_si128(emm4, _pi32_4);
    emm4 = _mm_slli_epi32(emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps(emm4);

    sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

    // Evaluate the first polynom  (0 <= x <= Pi/4)
    __m128 z = _mm_mul_ps(x,x);
    y = _ps_coscof_p0;

    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, _ps_coscof_p1);
    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, _ps_coscof_p2);
    y = _mm_mul_ps(y, _mm_mul_ps(z, z));
    __m128 tmp = _mm_mul_ps(z, _ps_0p5);
    y = _mm_sub_ps(y, tmp);
    y = _mm_add_ps(y, _ps_1);

    // Evaluate the second polynom  (Pi/4 <= x <= 0)

    __m128 y2 = _ps_sincof_p0;
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_add_ps(y2, _ps_sincof_p1);
    y2 = _mm_mul_ps(y2, z);
    y2 = _mm_add_ps(y2, _ps_sincof_p2);
    y2 = _mm_mul_ps(y2, _mm_mul_ps(z, x));
    y2 = _mm_add_ps(y2, x);

    // Select the correct result from the two polynoms
    xmm3 = poly_mask;
    __m128 ysin2 = _mm_and_ps(xmm3, y2);
    __m128 ysin1 = _mm_andnot_ps(xmm3, y);
    y2 = _mm_sub_ps(y2,ysin2);
    y = _mm_sub_ps(y, ysin1);

    xmm1 = _mm_add_ps(ysin1,ysin2);
    xmm2 = _mm_add_ps(y,y2);

    // Update the sign
    *s = _mm_xor_ps(xmm1, sign_bit_sin);
    *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

static const __m128 _ps_atanrange_hi = _mm_set1_ps(2.414213562373095);
static const __m128 _ps_atanrange_lo = _mm_set1_ps(0.4142135623730950);
static const __m128 _ps_cephes_PIF = _mm_set1_ps(3.141592653589793238);
static const __m128 _ps_cephes_PIO2F = _mm_set1_ps(1.5707963267948966192);
static const __m128 _ps_cephes_PIO4F = _mm_set1_ps(0.7853981633974483096);

static const __m128 _ps_atancof_p0 = _mm_set1_ps(8.05374449538e-2);
static const __m128 _ps_atancof_p1 = _mm_set1_ps(1.38776856032e-1);
static const __m128 _ps_atancof_p2 = _mm_set1_ps(1.99777106478e-1);
static const __m128 _ps_atancof_p3 = _mm_set1_ps(3.33329491539e-1);

static inline __m128 atan_ps( __m128 x )
{
    __m128 sign_bit, y;

    sign_bit = x;
    // Take the absolute value
    x = _mm_and_ps( x, _ps_inv_sign_mask );
    // Extract the sign bit (upper one)
    sign_bit = _mm_and_ps( sign_bit, _ps_sign_mask );

    // Range reduction, init x and y depending on range

    // x > 2.414213562373095
    __m128 cmp0 = _mm_cmpgt_ps( x, _ps_atanrange_hi );
    // x > 0.4142135623730950
    __m128 cmp1 = _mm_cmpgt_ps( x, _ps_atanrange_lo );

    // x > 0.4142135623730950 && !( x > 2.414213562373095 )
    __m128 cmp2 = _mm_andnot_ps( cmp0, cmp1 );

    // -( 1.0/x )
    __m128 y0 = _mm_and_ps( cmp0, _ps_cephes_PIO2F );
    __m128 x0 = _mm_div_ps( _ps_1, x );
    x0 = _mm_xor_ps( x0, _ps_sign_mask );

    __m128 y1 = _mm_and_ps( cmp2, _ps_cephes_PIO4F );
    // (x-1.0)/(x+1.0)
    __m128 x1_o = _mm_sub_ps( x, _ps_1 );
    __m128 x1_u = _mm_add_ps( x, _ps_1 );
    __m128 x1 = _mm_div_ps( x1_o, x1_u );

    __m128 x2 = _mm_and_ps( cmp2, x1 );
    x0 = _mm_and_ps( cmp0, x0 );
    x2 = _mm_or_ps( x2, x0 );
    cmp1 = _mm_or_ps( cmp0, cmp2 );
    x2 = _mm_and_ps( cmp1, x2 );
    x = _mm_andnot_ps( cmp1, x );
    x = _mm_or_ps( x2, x );

    y = _mm_or_ps( y0, y1 );

    __m128 zz = _mm_mul_ps( x, x );
    __m128 acc = _ps_atancof_p0;
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_sub_ps( acc, _ps_atancof_p1 );
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_add_ps( acc, _ps_atancof_p2 );
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_sub_ps( acc, _ps_atancof_p3 );
    acc = _mm_mul_ps( acc, zz );
    acc = _mm_mul_ps( acc, x );
    acc = _mm_add_ps( acc, x );
    y = _mm_add_ps( y, acc );

    // Update the sign
    y = _mm_xor_ps( y, sign_bit );

    return y;
}

static inline __m128 atan2_ps( __m128 y, __m128 x )
{
    __m128 x_eq_0 = _mm_cmpeq_ps( x, _ps_0 );
    __m128 x_gt_0 = _mm_cmpgt_ps( x, _ps_0 );
    __m128 x_le_0 = _mm_cmple_ps( x, _ps_0 );
    __m128 y_eq_0 = _mm_cmpeq_ps( y, _ps_0 );
    __m128 x_lt_0 = _mm_cmplt_ps( x, _ps_0 );
    __m128 y_lt_0 = _mm_cmplt_ps( y, _ps_0 );

    __m128 zero_mask = _mm_and_ps( x_eq_0, y_eq_0 );
    __m128 zero_mask_other_case = _mm_and_ps( y_eq_0, x_gt_0 );
    zero_mask = _mm_or_ps( zero_mask, zero_mask_other_case );

    __m128 pio2_mask = _mm_andnot_ps( y_eq_0, x_eq_0 );
    __m128 pio2_mask_sign = _mm_and_ps( y_lt_0, _ps_sign_mask );
    __m128 pio2_result = _ps_cephes_PIO2F;
    pio2_result = _mm_xor_ps( pio2_result, pio2_mask_sign );
    pio2_result = _mm_and_ps( pio2_mask, pio2_result );

    __m128 pi_mask = _mm_and_ps( y_eq_0, x_le_0 );
    __m128 pi = _ps_cephes_PIF;
    __m128 pi_result = _mm_and_ps( pi_mask, pi );

    __m128 swap_sign_mask_offset = _mm_and_ps( x_lt_0, y_lt_0 );
    swap_sign_mask_offset = _mm_and_ps( swap_sign_mask_offset, _ps_sign_mask );

    __m128 offset0 = _mm_setzero_ps();
    __m128 offset1 = _ps_cephes_PIF;
    offset1 = _mm_xor_ps( offset1, swap_sign_mask_offset );

    __m128 offset = _mm_andnot_ps( x_lt_0, offset0 );
    offset = _mm_and_ps( x_lt_0, offset1 );

    __m128 arg = _mm_div_ps( y, x );
    __m128 atan_result = atan_ps( arg );
    atan_result = _mm_add_ps( atan_result, offset );

    // Select between zero_result, pio2_result and atan_result

    __m128 result = _mm_andnot_ps( zero_mask, pio2_result );
    atan_result = _mm_andnot_ps( pio2_mask, atan_result );
    atan_result = _mm_andnot_ps( pio2_mask, atan_result);
    result = _mm_or_ps( result, atan_result );
    result = _mm_or_ps( result, pi_result );

    return result;
}

static const __m256 _ps_atanrange_hi_avx = _mm256_set1_ps(2.414213562373095);
static const __m256 _ps_atanrange_lo_avx = _mm256_set1_ps(0.4142135623730950);
static const __m256 _ps_cephes_PIF_avx = _mm256_set1_ps(3.141592653589793238);
static const __m256 _ps_cephes_PIO2F_avx = _mm256_set1_ps(1.5707963267948966192);
static const __m256 _ps_cephes_PIO4F_avx = _mm256_set1_ps(0.7853981633974483096);

static const __m256 _ps_atancof_p0_avx = _mm256_set1_ps(8.05374449538e-2);
static const __m256 _ps_atancof_p1_avx = _mm256_set1_ps(1.38776856032e-1);
static const __m256 _ps_atancof_p2_avx = _mm256_set1_ps(1.99777106478e-1);
static const __m256 _ps_atancof_p3_avx = _mm256_set1_ps(3.33329491539e-1);

// AVX2 version of the atan_ps() SSE version
static inline __m256 atan_ps(__m256 x)
{
    __m256 sign_bit, y;

    sign_bit = x;
    // Take the absolute value
    x = _mm256_and_ps(x, _ps_inv_sign_mask_avx);
    // Extract the sign bit (upper one)
    sign_bit = _mm256_and_ps(sign_bit, _ps_sign_mask_avx);

    // Range reduction, init x and y depending on range
    // x > 2.414213562373095
    __m256 cmp0 = _mm256_cmp_ps(x, _ps_atanrange_hi_avx, _CMP_GT_OS);
    // x > 0.4142135623730950
    __m256 cmp1 = _mm256_cmp_ps(x, _ps_atanrange_lo_avx, _CMP_GT_OS);

    // x > 0.4142135623730950 && !(x > 2.414213562373095)
    __m256 cmp2 = _mm256_andnot_ps(cmp0, cmp1);

    // -(1.0/x);
    __m256 y0 = _mm256_and_ps(cmp0, _ps_cephes_PIO2F_avx);
    __m256 x0 = _mm256_div_ps(avx_p1, x);
    x0 = _mm256_xor_ps(x0, _ps_sign_mask_avx);

    __m256 y1 = _mm256_and_ps(cmp2, _ps_cephes_PIO4F_avx);
    // (x-1.0)/(x+1.0)
    __m256 x1_o = _mm256_sub_ps(x, avx_p1);
    __m256 x1_u = _mm256_add_ps(x, avx_p1);
    __m256 x1 = _mm256_div_ps(x1_o, x1_u);

    __m256 x2 = _mm256_and_ps(cmp2, x1);
    x0 = _mm256_and_ps(cmp0, x0);
    x2 = _mm256_or_ps(x2, x0);
    cmp1 = _mm256_or_ps(cmp0, cmp2);
    x2 = _mm256_and_ps(cmp1, x2);
    x = _mm256_andnot_ps(cmp1, x);
    x = _mm256_or_ps(x2, x);

    y = _mm256_or_ps(y0, y1);

    __m256 zz = _mm256_mul_ps(x, x);
    __m256 acc = _ps_atancof_p0_avx;
    acc = _mm256_fmsub_ps(acc, zz, _ps_atancof_p1_avx);
    acc = _mm256_fmadd_ps(acc, zz, _ps_atancof_p2_avx);
    acc = _mm256_fmsub_ps(acc, zz, _ps_atancof_p3_avx);
    acc = _mm256_mul_ps(acc, zz);
    acc = _mm256_fmadd_ps(acc, x, x);
    y = _mm256_add_ps(y, acc);

    // Update the sign
    y = _mm256_xor_ps(y, sign_bit);

    return y;
}

// AVX2 version of the atan2_ps() SSE version
static inline __m256 atan2_ps(__m256 y, __m256 x)
{
    __m256 x_eq_0 = _mm256_cmp_ps(x, avx_p0, _CMP_EQ_OQ);
    __m256 x_gt_0 = _mm256_cmp_ps(x, avx_p0, _CMP_GT_OS);
    __m256 x_le_0 = _mm256_cmp_ps(x, avx_p0, _CMP_LE_OS);
    __m256 y_eq_0 = _mm256_cmp_ps(y, avx_p0, _CMP_EQ_OQ);
    __m256 x_lt_0 = _mm256_cmp_ps(x, avx_p0, _CMP_LT_OS);
    __m256 y_lt_0 = _mm256_cmp_ps(y, avx_p0, _CMP_LT_OS);

    // Computes a zero mask, set if either both x=y=0 or y=0&x>0
    __m256 zero_mask = _mm256_and_ps(x_eq_0, y_eq_0);
    __m256 zero_mask_other_case = _mm256_and_ps(y_eq_0, x_gt_0);
    zero_mask = _mm256_or_ps(zero_mask, zero_mask_other_case);

    // Computes pio2 intermediate result, set if (y!0 and x=0) & (pi/2 XOR (upper bit y<0))
    __m256 pio2_mask = _mm256_andnot_ps(y_eq_0, x_eq_0);
    __m256 pio2_mask_sign = _mm256_and_ps(y_lt_0, _ps_sign_mask_avx);
    __m256 pio2_result = _ps_cephes_PIO2F_avx;
    pio2_result = _mm256_xor_ps(pio2_result, pio2_mask_sign);
    pio2_result = _mm256_and_ps(pio2_mask, pio2_result);

    // Computes pi intermediate result, set if y=0&x<0 and pi
    __m256 pi_mask = _mm256_and_ps(y_eq_0, x_le_0);
    __m256 pi_result = _mm256_and_ps(pi_mask, _ps_cephes_PIF_avx);

    // Computes swap_sign_mask_offset, set if x<0 & y<0 of sign bit(uppermost bit)
    __m256 swap_sign_mask_offset = _mm256_and_ps(x_lt_0, y_lt_0);
    swap_sign_mask_offset = _mm256_and_ps(swap_sign_mask_offset, _ps_sign_mask_avx);

     // Computes offset, set based on pi, swap_sign_mask_offset and x<0
    __m256 offset0 = _mm256_xor_ps(_ps_cephes_PIF_avx, swap_sign_mask_offset);
    __m256 offset = _mm256_andnot_ps(x_lt_0, avx_p0);
    offset = _mm256_and_ps(x_lt_0, offset0);

    // Computes division of x,y
    __m256 arg = _mm256_div_ps(y, x);
    __m256 atan_result = atan_ps(arg);
    atan_result = _mm256_add_ps(atan_result, offset);

    // Select between zero_result, pio2_result and atan_result
    __m256 result = _mm256_andnot_ps(zero_mask, pio2_result);
    atan_result = _mm256_andnot_ps(pio2_mask, atan_result);
    atan_result = _mm256_andnot_ps(pio2_mask, atan_result);
    result = _mm256_or_ps(result, atan_result);
    result = _mm256_or_ps(result, pi_result);

    return result;
}

// Modified AVX2 version of the original SSE version at https://github.com/RJVB/sse_mathfun/blob/master/sse_mathfun.h
static inline __m256 log_ps(__m256 x)
{
    __m256 e;
    __m256i emm0;
    __m256 one = *(__m256 *)&avx_p1;
    __m256 invalid_mask = _mm256_cmp_ps(x, avx_p0, _CMP_LE_OQ);

    // cut off denormalized stuff
    x = _mm256_max_ps(x, *(__m256 *)&_ps_min_norm_pos_avx);

    // part 1: x = frexpf(x, &e);
    emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    // keep only the fractional part
    x = _mm256_and_ps(x, *(__m256 *)&_ps_inv_mant_mask_avx);
    x = _mm256_or_ps(x, *(__m256 *)&_ps_0p5_avx);

    emm0 = _mm256_sub_epi32(emm0, *(__m256i *)&_pi32_0x7f_avx);
    e = _mm256_cvtepi32_ps(emm0);

    e = _mm256_add_ps(e, one);

    // part 2: if( x < SQRTHF ) { e -= 1; x = x + x - 1.0; } else { x = x - 1.0; }
    __m256 z, y;
    __m256 mask = _mm256_cmp_ps(x, *(__m256 *)&avx_cephesSQRTHF, _CMP_LT_OQ);
    __m256 tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);
    z = _mm256_mul_ps(x,x);
    y = *(__m256 *)&avx_cephesLogP0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, *(__m256 *)&avx_cephesLogP8);
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    tmp = _mm256_mul_ps(e, *(__m256 *)&avx_cephesLogQ1);
    y = _mm256_add_ps(y, tmp);
    tmp = _mm256_mul_ps(z, *(__m256 *)&_ps_0p5_avx);
    y = _mm256_sub_ps(y, tmp);
    tmp = _mm256_mul_ps(e, *(__m256 *)&avx_cephesLogQ2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN

    return x;
}

// Modified version of the original SSE version at https://github.com/RJVB/sse_mathfun/blob/master/sse_mathfun.h
static inline __m128 log_ps(__m128 x)
{
    __m128 e;
    __m128i emm0;
    __m128 one = *(__m128 *)&_ps_1;
    __m128 invalid_mask = _mm_cmple_ps(x, xmm_p0);

    // cut off denormalized stuff
    x = _mm_max_ps(x, *(__m128 *)&_ps_min_norm_pos);

    // part 1: x = frexpf(x, &e);
    emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

    // keep only the fractional part
    x = _mm_and_ps(x, *(__m128 *)&_ps_inv_mant_mask);
    x = _mm_or_ps(x, *(__m128 *)&_ps_0p5);

    emm0 = _mm_sub_epi32(emm0, *(__m128i *)&_pi32_0x7f);
    e = _mm_cvtepi32_ps(emm0);

    e = _mm_add_ps(e, one);

    // part 2: if( x < SQRTHF ) { e -= 1; x = x + x - 1.0; } else { x = x - 1.0; }
    __m128 z, y;
    __m128 mask = _mm_cmplt_ps(x, *(__m128 *)&xmm_cephesSQRTHF);
    __m128 tmp = _mm_and_ps(x, mask);
    x = _mm_sub_ps(x, one);
    e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    x = _mm_add_ps(x, tmp);
    z = _mm_mul_ps(x,x);
    y = *(__m128 *)&xmm_cephesLogP0;
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP1);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP2);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP3);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP4);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP5);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP6);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP7);
    y = _mm_mul_ps(y, x);
    y = _mm_add_ps(y, *(__m128 *)&xmm_cephesLogP8);
    y = _mm_mul_ps(y, x);
    y = _mm_mul_ps(y, z);
    tmp = _mm_mul_ps(e, *(__m128 *)&xmm_cephesLogQ1);
    y = _mm_add_ps(y, tmp);
    tmp = _mm_mul_ps(z, *(__m128 *)&_ps_0p5);
    y = _mm_sub_ps(y, tmp);
    tmp = _mm_mul_ps(e, *(__m128 *)&xmm_cephesLogQ2);
    x = _mm_add_ps(x, y);
    x = _mm_add_ps(x, tmp);
    x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN

    return x;
}

inline Rpp32f rpp_hsum_ps(__m128 x)
{
    __m128 shuf = _mm_movehdup_ps(x);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);        // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline Rpp32f rpp_hsum_ps(__m256 x)
{
    __m128 p0 = _mm256_extractf128_ps(x, 1); // Contains x7, x6, x5, x4
    __m128 p1 = _mm256_castps256_ps128(x);   // Contains x3, x2, x1, x0
    __m128 sum = _mm_add_ps(p0, p1);         // Contains x3 + x7, x2 + x6, x1 + x5, x0 + x4
    p0 = sum;                                // Contains -, -, x1 + x5, x0 + x4
    p1 = _mm_movehl_ps(sum, sum);            // Contains -, -, x3 + x7, x2 + x6
    sum = _mm_add_ps(p0, p1);                // Contains -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6
    p0 = sum;                                // Contains -, -, -, x0 + x2 + x4 + x6
    p1 = _mm_shuffle_ps(sum, sum, 0x1);      // Contains -, -, -, x1 + x3 + x5 + x7
    sum = _mm_add_ss(p0, p1);                // Contains -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    return _mm_cvtss_f32(sum);
}

/* Computes inverse square root */
inline Rpp32f rpp_rsqrt_ps(Rpp32f x)
{
    __m128 X = _mm_set_ss(x);
    __m128 tmp = _mm_rsqrt_ss(X);
    Rpp32f y = _mm_cvtss_f32(tmp);
    return y * (1.5f - x * 0.5f * y * y);
}

/* Compute inverse square root */
/* SSE matches to 6 decimal places with raw C version due to newton rhapson approximation*/
inline void rpp_rsqrt_sse(Rpp32f *input, Rpp64s numElements, Rpp32f eps, Rpp32f rdiv, Rpp32f mul)
{
    Rpp64s i = 0;
    __m128 rdivx4 = _mm_set1_ps(rdiv);
    __m128 mulx4 = _mm_set1_ps(mul * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m128 epsx4 = _mm_set1_ps(eps);
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            x = _mm_add_ps(x, epsx4);
            __m128 y = _mm_rsqrt_ps(x);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            __m128 mask = _mm_cmpneq_ps(x, xmm_p0);
            __m128 y = _mm_rsqrt_ps(x);
            y = _mm_and_ps(y, mask);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt_ps(input[i] * rdiv + eps) * mul;
    }
    else
    {
        for (; i < numElements; i++)
        {
            Rpp32f x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt_ps(x) * mul : 0;
        }
    }
}

/* Compute inverse square root */
/* AVX2 matches to 6 decimal places with raw C version due to newton rhapson approximation*/
inline void rpp_rsqrt_avx(Rpp32f *input, Rpp32s numElements, Rpp32f eps, Rpp32f rdiv, Rpp32f scale)
{
    Rpp32s i = 0;
    __m256 rdivx8 = _mm256_set1_ps(rdiv);
    __m256 mulx8 = _mm256_set1_ps(scale * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m256 epsx8 = _mm256_set1_ps(eps);
        for (; i + 8 <= numElements; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&input[i]);
            x = _mm256_mul_ps(x, rdivx8);
            x = _mm256_add_ps(x, epsx8);
            __m256 y = _mm256_rsqrt_ps(x);
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 xy2 = _mm256_mul_ps(x, y2);
            __m256 three_minus_xy2 = _mm256_sub_ps(avx_p3, xy2);
            y = _mm256_mul_ps(y, three_minus_xy2);
            y = _mm256_mul_ps(y, mulx8);
            _mm256_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 8 <= numElements; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&input[i]);
            x = _mm256_mul_ps(x, rdivx8);
            __m256 mask = _mm256_cmp_ps(x, avx_p0, _CMP_NEQ_OQ);
            __m256 y = _mm256_rsqrt_ps(x);
            y = _mm256_and_ps(y, mask);
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 xy2 = _mm256_mul_ps(x, y2);
            __m256 three_minus_xy2 = _mm256_sub_ps(avx_p3, xy2);
            y = _mm256_mul_ps(y, three_minus_xy2);
            y = _mm256_mul_ps(y, mulx8);
            _mm256_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt_ps(input[i] * rdiv + eps) * scale;
    }
    else
    {
        for (; i < numElements; i++)
        {
            Rpp32f x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt_ps(x) * scale : 0;
        }
    }
}

static inline void fast_matmul4x4_sse(float *A, float *B, float *C)
{
    __m128 row1 = _mm_load_ps(&B[0]);                   // Row 0 of B
    __m128 row2 = _mm_load_ps(&B[4]);                   // Row 1 of B
    __m128 row3 = _mm_load_ps(&B[8]);                   // Row 2 of B
    __m128 row4 = _mm_load_ps(&B[12]);                  // Row 3 of B

    for(int i = 0; i < 4; i++)
    {
        __m128 brod1 = _mm_set1_ps(A[4 * i + 0]);       // Example for row 0 computation -> A[0][0] is broadcasted
        __m128 brod2 = _mm_set1_ps(A[4 * i + 1]);       // Example for row 0 computation -> A[0][1] is broadcasted
        __m128 brod3 = _mm_set1_ps(A[4 * i + 2]);       // Example for row 0 computation -> A[0][2] is broadcasted
        __m128 brod4 = _mm_set1_ps(A[4 * i + 3]);       // Example for row 0 computation -> A[0][3] is broadcasted

        __m128 row = _mm_add_ps(                        // Example for row 0 computation -> P + Q
                        _mm_add_ps(                     // Example for row 0 computation -> P = A[0][0] * B[0][0] + A[0][1] * B[1][0]
                            _mm_mul_ps(brod1, row1),    // Example for row 0 computation -> (A[0][0] * B[0][0], A[0][0] * B[0][1], A[0][0] * B[0][2], A[0][0] * B[0][3])
                            _mm_mul_ps(brod2, row2)),   // Example for row 0 computation -> (A[0][1] * B[1][0], A[0][1] * B[1][1], A[0][1] * B[1][2], A[0][1] * B[1][3])
                        _mm_add_ps(                     // Example for row 0 computation -> Q = A[0][2] * B[2][0] + A[0][3] * B[3][0]
                            _mm_mul_ps(brod3, row3),    // Example for row 0 computation -> (A[0][2] * B[2][0], A[0][2] * B[2][1], A[0][2] * B[2][2], A[0][2] * B[2][3])
                            _mm_mul_ps(brod4, row4)));  // Example for row 0 computation -> (A[0][3] * B[3][0], A[0][3] * B[3][1], A[0][3] * B[3][2], A[0][3] * B[3][3])

        _mm_store_ps(&C[4*i], row);                     // Example for row 0 computation -> Storing whole computed row 0
    }
}

/* Generic interpolation loads  */

inline void rpp_generic_nn_load_u8pkd3(Rpp8u *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128i &p)
{
    __m128i px[4];
    px[0] = invalidLoad[0] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[0]));  // LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01
    px[1] = invalidLoad[1] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[1]));  // LOC1 load [R11|G11|B11|R12|G12|B12|R13|G13|B13|R14|G14|B14|R15|G15|B15|R16] - Need RGB 11
    px[2] = invalidLoad[2] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[2]));  // LOC2 load [R21|G21|B21|R22|G22|B22|R23|G23|B23|R24|G24|B24|R25|G25|B25|R26] - Need RGB 21
    px[3] = invalidLoad[3] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[3]));  // LOC3 load [R31|G31|B31|R32|G32|B32|R33|G33|B33|R34|G34|B34|R35|G35|B35|R36] - Need RGB 31
    px[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R01|G01|B01|R02|R11|G11|B11|R12|R21|G21|B21|R22|R31|G31|B31|R32]
    p = _mm_shuffle_epi8(px[0], xmm_pkd_mask);    // Shuffle to obtain 4 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|00|00|00|00]
}

inline void rpp_generic_nn_load_u8pkd3_avx(Rpp8u *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256i &p)
{
    __m128i px[7];
    px[0] = invalidLoad[0] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[0]));  // LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01
    px[1] = invalidLoad[1] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[1]));  // LOC1 load [R11|G11|B11|R12|G12|B12|R13|G13|B13|R14|G14|B14|R15|G15|B15|R16] - Need RGB 11
    px[2] = invalidLoad[2] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[2]));  // LOC2 load [R21|G21|B21|R22|G22|B22|R23|G23|B23|R24|G24|B24|R25|G25|B25|R26] - Need RGB 21
    px[3] = invalidLoad[3] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[3]));  // LOC3 load [R31|G31|B31|R32|G32|B32|R33|G33|B33|R34|G34|B34|R35|G35|B35|R36] - Need RGB 31
    px[4] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R01|G01|B01|R02|R11|G11|B11|R12|R21|G21|B21|R22|R31|G31|B31|R32]
    px[4] = _mm_shuffle_epi8(px[4], xmm_pkd_mask); // shuffle to obtain 4 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|00|00|00|00]

    px[0] = invalidLoad[4] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[4]));  // LOC4 load [R41|G41|B41|R42|G42|B42|R43|G43|B43|R44|G44|B44|R45|G45|B45|R46] - Need RGB 41
    px[1] = invalidLoad[5] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[5]));  // LOC5 load [R51|G51|B51|R52|G52|B52|R53|G53|B53|R54|G54|B54|R55|G55|B55|R56] - Need RGB 51
    px[2] = invalidLoad[6] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[6]));  // LOC6 load [R61|G61|B61|R62|G62|B62|R63|G63|B63|R64|G64|B64|R65|G65|B65|R66] - Need RGB 61
    px[3] = invalidLoad[7] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[7]));  // LOC7 load [R71|G71|B71|R72|G72|B72|R73|G73|B73|R74|G74|B74|R75|G75|B75|R76] - Need RGB 71
    px[5] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R41|G41|B41|R42|R51|G51|B51|R52|R61|G61|B61|R62|R71|G71|B71|R72]
    px[5] = _mm_shuffle_epi8(px[5], xmm_pkd_mask); // shuffle to obtain 4 RGB [R41|G41|B41|R51|G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00]

    px[6] = _mm_shuffle_epi8(px[5], xmm_pxMask00); // shuffle to move 0-3 of px[5] to 12-15
    px[4] = _mm_add_epi8(px[4], px[6]); // add px[4] and px[5]
    px[5] = _mm_shuffle_epi8(px[5], xmm_pxMask04To11); // shuffle to move values at 4-11 of px[5] to 0-7
    p = _mm256_setr_m128i(px[4], px[5]); // Merge to obtain 8 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|R41|G41|B41|R51|G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00|00|00|00|00]
}

inline void rpp_generic_nn_load_u8pln1(Rpp8u *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128i &p)
{
    __m128i px[4];
    px[0] = invalidLoad[0] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[0]));  // LOC0 load [R01|R02|R03|R04|R05|R06...] - Need R01
    px[1] = invalidLoad[1] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[1]));  // LOC1 load [R11|R12|R13|R14|R15|R16...] - Need R11
    px[2] = invalidLoad[2] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[2]));  // LOC2 load [R21|R22|R23|R24|R25|R26...] - Need R21
    px[3] = invalidLoad[3] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[3]));  // LOC3 load [R31|R32|R33|R34|R35|R36...] - Need R31
    px[0] = _mm_unpacklo_epi8(px[0], px[2]);    // unpack 8 lo-pixels of px[0] and px[2]
    px[1] = _mm_unpacklo_epi8(px[1], px[3]);    // unpack 8 lo-pixels of px[1] and px[3]
    p = _mm_unpacklo_epi8(px[0], px[1]);    // unpack to obtain [R01|R11|R21|R31|00|00|00|00|00|00|00|00|00|00|00|00]
}

inline void rpp_generic_nn_load_u8pln1_avx(Rpp8u *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256i &p)
{
    Rpp8u buffer[16] = {0};
    for(int i = 0; i < 8; i++)
    {
        if(!invalidLoad[i])
            buffer[i] = *(srcPtrChannel + srcLoc[i]);
    }
    __m128i px = _mm_loadu_si128((__m128i *)buffer);
    p = _mm256_setr_m128i(px, xmm_px0);
}

inline void rpp_generic_nn_load_f32pkd3_to_f32pln3(Rpp32f *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128 *p)
{
    p[0] = invalidLoad[0] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[0]);  // LOC0 load [R01|G01|B01|R02] - Need RGB 01
    p[1] = invalidLoad[1] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[1]);  // LOC1 load [R11|G11|B11|R12] - Need RGB 11
    p[2] = invalidLoad[2] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[2]);  // LOC2 load [R21|G21|B21|R22] - Need RGB 21
    __m128 pTemp = invalidLoad[3] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[3]);  // LOC2 load [R31|G31|B31|R32] - Need RGB 31
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], pTemp); // Transpose to obtain RGB in each vector
}

inline void rpp_generic_nn_load_f32pkd3_to_f32pln3_avx(Rpp32f *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256 *p)
{
    p[0] = _mm256_setr_ps((!invalidLoad[0]) ? srcPtrChannel[srcLoc[0]]: 0, (!invalidLoad[1]) ? srcPtrChannel[srcLoc[1]]: 0,           // Get R01-R08. load the values from input using srcLoc buffer if invalidLoad is 0, else set the values to 0
                          (!invalidLoad[2]) ? srcPtrChannel[srcLoc[2]]: 0, (!invalidLoad[3]) ? srcPtrChannel[srcLoc[3]]: 0,
                          (!invalidLoad[4]) ? srcPtrChannel[srcLoc[4]]: 0, (!invalidLoad[5]) ? srcPtrChannel[srcLoc[5]]: 0,
                          (!invalidLoad[6]) ? srcPtrChannel[srcLoc[6]]: 0, (!invalidLoad[7]) ? srcPtrChannel[srcLoc[7]]: 0);
    p[1] = _mm256_setr_ps((!invalidLoad[0]) ? srcPtrChannel[srcLoc[0] + 1]: 0, (!invalidLoad[1]) ? srcPtrChannel[srcLoc[1] + 1]: 0,   // Get G01-R08. load the values from input using srcLoc buffer if invalidLoad is 0, else set the values to 0
                          (!invalidLoad[2]) ? srcPtrChannel[srcLoc[2] + 1]: 0, (!invalidLoad[3]) ? srcPtrChannel[srcLoc[3] + 1]: 0,
                          (!invalidLoad[4]) ? srcPtrChannel[srcLoc[4] + 1]: 0, (!invalidLoad[5]) ? srcPtrChannel[srcLoc[5] + 1]: 0,
                          (!invalidLoad[6]) ? srcPtrChannel[srcLoc[6] + 1]: 0, (!invalidLoad[7]) ? srcPtrChannel[srcLoc[7] + 1]: 0);
    p[2] = _mm256_setr_ps((!invalidLoad[0]) ? srcPtrChannel[srcLoc[0] + 2]: 0, (!invalidLoad[1]) ? srcPtrChannel[srcLoc[1] + 2]: 0,   // Get B01-R08. load the values from input using srcLoc buffer if invalidLoad is 0, else set the values to 0
                          (!invalidLoad[2]) ? srcPtrChannel[srcLoc[2] + 2]: 0, (!invalidLoad[3]) ? srcPtrChannel[srcLoc[3] + 2]: 0,
                          (!invalidLoad[4]) ? srcPtrChannel[srcLoc[4] + 2]: 0, (!invalidLoad[5]) ? srcPtrChannel[srcLoc[5] + 2]: 0,
                          (!invalidLoad[6]) ? srcPtrChannel[srcLoc[6] + 2]: 0, (!invalidLoad[7]) ? srcPtrChannel[srcLoc[7] + 2]: 0);
}

inline void rpp_generic_nn_load_f32pkd3_to_f32pkd3(Rpp32f *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128 *p)
{
    p[0] = invalidLoad[0] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[0]);  // LOC0 load [R01|G01|B01|R02] - Need RGB 01
    p[1] = invalidLoad[1] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[1]);  // LOC1 load [R11|G11|B11|R12] - Need RGB 11
    p[2] = invalidLoad[2] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[2]);  // LOC2 load [R21|G21|B21|R22] - Need RGB 21
    p[3] = invalidLoad[3] ? xmm_p0 : _mm_loadu_ps(srcPtrChannel + srcLoc[3]);  // LOC2 load [R31|G31|B31|R32] - Need RGB 31
}

inline void rpp_generic_nn_load_f32pkd3_to_f32pkd3_avx(Rpp32f *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256 *p)
{
    p[0] = _mm256_setr_ps((!invalidLoad[0]) ? srcPtrChannel[srcLoc[0]]: 0, (!invalidLoad[0]) ? srcPtrChannel[srcLoc[0] + 1]: 0,        // Get R01|G01|B01|R02|B02|G02|R03|G03
                          (!invalidLoad[0]) ? srcPtrChannel[srcLoc[0] + 2]: 0, (!invalidLoad[1]) ? srcPtrChannel[srcLoc[1]]: 0,        // load the values from input using srcLoc buffer if invalidLoad is 0, else set the values to 0
                          (!invalidLoad[1]) ? srcPtrChannel[srcLoc[1] + 1]: 0, (!invalidLoad[1]) ? srcPtrChannel[srcLoc[1] + 2]: 0,
                          (!invalidLoad[2]) ? srcPtrChannel[srcLoc[2]]: 0, (!invalidLoad[2]) ? srcPtrChannel[srcLoc[2] + 1]: 0);
    p[1] = _mm256_setr_ps((!invalidLoad[2]) ? srcPtrChannel[srcLoc[2] + 2]: 0, (!invalidLoad[3]) ? srcPtrChannel[srcLoc[3]]: 0,        // Get B03|R04|G04|B04|R05|G05|B05|R06
                          (!invalidLoad[3]) ? srcPtrChannel[srcLoc[3] + 1]: 0, (!invalidLoad[3]) ? srcPtrChannel[srcLoc[3] + 2]: 0,    // load the values from input using srcLoc buffer if invalidLoad is 0, else set the values to 0
                          (!invalidLoad[4]) ? srcPtrChannel[srcLoc[4]]: 0, (!invalidLoad[4]) ? srcPtrChannel[srcLoc[4] + 1]: 0,
                          (!invalidLoad[4]) ? srcPtrChannel[srcLoc[4] + 2]: 0, (!invalidLoad[5]) ? srcPtrChannel[srcLoc[5]]: 0);
    p[2] = _mm256_setr_ps((!invalidLoad[5]) ? srcPtrChannel[srcLoc[5] + 1]: 0, (!invalidLoad[5]) ? srcPtrChannel[srcLoc[5] + 2]: 0,    // Get G06|B06|R07|G07|B07|R08|G08|B08
                          (!invalidLoad[6]) ? srcPtrChannel[srcLoc[6]]: 0, (!invalidLoad[6]) ? srcPtrChannel[srcLoc[6] + 1]: 0,        // load the values from input using srcLoc buffer if invalidLoad is 0, else set the values to 0
                          (!invalidLoad[6]) ? srcPtrChannel[srcLoc[6] + 2]: 0, (!invalidLoad[7]) ? srcPtrChannel[srcLoc[7]]: 0,
                          (!invalidLoad[7]) ? srcPtrChannel[srcLoc[7] + 1]: 0, (!invalidLoad[7]) ? srcPtrChannel[srcLoc[7] + 2]: 0);
}

inline void rpp_generic_nn_load_f32pln1(Rpp32f *srcPtrChanel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128 &p)
{
    __m128 pTemp[4];
    pTemp[0] = invalidLoad[0] ? xmm_p0 : _mm_loadu_ps(srcPtrChanel + srcLoc[0]);  // LOC0 load [R01|R02|R03|R04] - Need R01
    pTemp[1] = invalidLoad[1] ? xmm_p0 : _mm_loadu_ps(srcPtrChanel + srcLoc[1]);  // LOC1 load [R11|R12|R13|R14] - Need R11
    pTemp[2] = invalidLoad[2] ? xmm_p0 : _mm_loadu_ps(srcPtrChanel + srcLoc[2]);  // LOC2 load [R21|R22|R23|R24] - Need R21
    pTemp[3] = invalidLoad[3] ? xmm_p0 : _mm_loadu_ps(srcPtrChanel + srcLoc[3]);  // LOC3 load [R31|R32|R33|R34] - Need R31
    pTemp[0] = _mm_unpacklo_ps(pTemp[0], pTemp[2]);
    pTemp[1] = _mm_unpacklo_ps(pTemp[1], pTemp[3]);
    p = _mm_unpacklo_ps(pTemp[0], pTemp[1]);    // Unpack to obtain [R01|R11|R21|R31]
}

inline void rpp_generic_nn_load_f32pln1_avx(Rpp32f *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256 &p)
{
    __m256i pxLoadMask = _mm256_setr_epi32((!invalidLoad[0]) ? 0x80000000 : 0, (!invalidLoad[1]) ? 0x80000000 : 0,  // Set MSB of 32 bit value to 1 if invalidLoad value is 0
                                          (!invalidLoad[2]) ? 0x80000000 : 0, (!invalidLoad[3]) ? 0x80000000 : 0,
                                          (!invalidLoad[4]) ? 0x80000000 : 0, (!invalidLoad[5]) ? 0x80000000 : 0,
                                          (!invalidLoad[6]) ? 0x80000000 : 0, (!invalidLoad[7]) ? 0x80000000 : 0);
    __m256i pxSrcLoc = _mm256_loadu_si256((__m256i *)srcLoc);   // Load the source location values passed
    __m256 pSrcLoc = _mm256_castsi256_ps(pxSrcLoc);   // cast to float
    p = _mm256_mask_i32gather_ps(avx_p0, srcPtrChannel, pSrcLoc, pxLoadMask, 4);   // if the MSB of 32 bit value is set, then load from corresponding location value in pSrcLoc. Otherwise set the 32 bit value to 0
}

inline void rpp_generic_nn_load_i8pkd3(Rpp8s *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128i &p)
{
    __m128i px[4];
    px[0] = invalidLoad[0] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[0]));  // LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01
    px[1] = invalidLoad[1] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[1]));  // LOC1 load [R11|G11|B11|R12|G12|B12|R13|G13|B13|R14|G14|B14|R15|G15|B15|R16] - Need RGB 11
    px[2] = invalidLoad[2] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[2]));  // LOC2 load [R21|G21|B21|R22|G22|B22|R23|G23|B23|R24|G24|B24|R25|G25|B25|R26] - Need RGB 21
    px[3] = invalidLoad[3] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[3]));  // LOC3 load [R31|G31|B31|R32|G32|B32|R33|G33|B33|R34|G34|B34|R35|G35|B35|R36] - Need RGB 31
    px[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R01|G01|B01|R02|R11|G11|B11|R12|R21|G21|B21|R22|R31|G31|B31|R32]
    p = _mm_shuffle_epi8(px[0], xmm_pkd_mask);    // Shuffle to obtain 4 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|00|00|00|00]
}

inline void rpp_generic_nn_load_i8pkd3_avx(Rpp8s *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256i &p)
{
    __m128i px[7];
    px[0] = invalidLoad[0] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[0]));  // LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01
    px[1] = invalidLoad[1] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[1]));  // LOC1 load [R11|G11|B11|R12|G12|B12|R13|G13|B13|R14|G14|B14|R15|G15|B15|R16] - Need RGB 11
    px[2] = invalidLoad[2] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[2]));  // LOC2 load [R21|G21|B21|R22|G22|B22|R23|G23|B23|R24|G24|B24|R25|G25|B25|R26] - Need RGB 21
    px[3] = invalidLoad[3] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[3]));  // LOC3 load [R31|G31|B31|R32|G32|B32|R33|G33|B33|R34|G34|B34|R35|G35|B35|R36] - Need RGB 31
    px[4] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R01|G01|B01|R02|R11|G11|B11|R12|R21|G21|B21|R22|R31|G31|B31|R32]
    px[4] = _mm_shuffle_epi8(px[4], xmm_pkd_mask); // shuffle to obtain 4 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|00|00|00|00]

    px[0] = invalidLoad[4] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[4]));  // LOC4 load [R41|G41|B41|R42|G42|B42|R43|G43|B43|R44|G44|B44|R45|G45|B45|R46] - Need RGB 41
    px[1] = invalidLoad[5] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[5]));  // LOC5 load [R51|G51|B51|R52|G52|B52|R53|G53|B53|R54|G54|B54|R55|G55|B55|R56] - Need RGB 51
    px[2] = invalidLoad[6] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[6]));  // LOC6 load [R61|G61|B61|R62|G62|B62|R63|G63|B63|R64|G64|B64|R65|G65|B65|R66] - Need RGB 61
    px[3] = invalidLoad[7] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[7]));  // LOC7 load [R71|G71|B71|R72|G72|B72|R73|G73|B73|R74|G74|B74|R75|G75|B75|R76] - Need RGB 71
    px[5] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R41|G41|B41|R42|R51|G51|B51|R52|R61|G61|B61|R62|R71|G71|B71|R72]
    px[5] = _mm_shuffle_epi8(px[5], xmm_pkd_mask); // shuffle to obtain 4 RGB [R41|G41|B41|R51|G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00]

    px[6] = _mm_shuffle_epi8(px[5], xmm_pxMask00); // shuffle to move 0-3 of px[5] to 12-15
    px[4] = _mm_add_epi8(px[4], px[6]); // add px[4] and px[5]
    px[5] = _mm_shuffle_epi8(px[5], xmm_pxMask04To11); // shuffle to move values at 4-11 of px[5] to 0-7
    p = _mm256_setr_m128i(px[4], px[5]); // Merge to obtain 8 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|R41|G41|B41|R51|G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00|00|00|00|00]
}

inline void rpp_generic_nn_load_i8pln1(Rpp8s *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m128i &p)
{
    __m128i px[4];
    px[0] = invalidLoad[0] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[0]));  // LOC0 load [R01|R02|R03|R04|R05|R06...] - Need R01
    px[1] = invalidLoad[1] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[1]));  // LOC1 load [R11|R12|R13|R14|R15|R16...] - Need R11
    px[2] = invalidLoad[2] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[2]));  // LOC2 load [R21|R22|R23|R24|R25|R26...] - Need R21
    px[3] = invalidLoad[3] ? xmm_px0 : _mm_loadu_si128((__m128i *)(srcPtrChannel + srcLoc[3]));  // LOC3 load [R31|R32|R33|R34|R35|R36...] - Need R31
    px[0] = _mm_unpacklo_epi8(px[0], px[2]);    // unpack 8 lo-pixels of px[0] and px[2]
    px[1] = _mm_unpacklo_epi8(px[1], px[3]);    // unpack 8 lo-pixels of px[1] and px[3]
    p = _mm_unpacklo_epi8(px[0], px[1]);    // unpack to obtain [R01|R11|R21|R31|00|00|00|00|00|00|00|00|00|00|00|00]
}

inline void rpp_generic_nn_load_i8pln1_avx(Rpp8s *srcPtrChannel, Rpp32s *srcLoc, Rpp32s *invalidLoad, __m256i &p)
{
    Rpp8s buffer[16] = {0};
    for(int i = 0; i < 8; i++)
    {
        if(!invalidLoad[i])
            buffer[i] = *(srcPtrChannel + srcLoc[i]);
    }
    __m128i px = _mm_loadu_si128((__m128i *)buffer);
    p = _mm256_setr_m128i(px, xmm_px0);
}

inline void rpp_generic_bilinear_load_mask_avx(__m256 &pSrcY, __m256 &pSrcX, __m256 *pRoiLTRB, Rpp32s *invalidLoadMask)
{
    _mm256_storeu_si256((__m256i*) invalidLoadMask, _mm256_cvtps_epi32(_mm256_or_ps(                                    // Vectorized ROI boundary check for 8 locations
        _mm256_or_ps(_mm256_cmp_ps(pSrcX, pRoiLTRB[0], _CMP_LT_OQ), _mm256_cmp_ps(pSrcY, pRoiLTRB[1], _CMP_LT_OQ)),
        _mm256_or_ps(_mm256_cmp_ps(_mm256_floor_ps(pSrcX), pRoiLTRB[2], _CMP_GT_OQ), _mm256_cmp_ps(_mm256_floor_ps(pSrcY), pRoiLTRB[3], _CMP_GT_OQ))
    )));
}

template <typename T>
inline void rpp_generic_bilinear_load_1c_avx(T *srcPtrChannel, RpptDescPtr srcDescPtr, RpptBilinearNbhoodLocsVecLen8 &srcLocs, __m256 &pSrcY, __m256 &pSrcX, __m256 *pRoiLTRB, __m256 *pSrc)
{
    Rpp32s invalidLoadMask[8];
    RpptBilinearNbhoodValsVecLen8 srcVals;
    memset(&srcVals, 0, sizeof(RpptBilinearNbhoodValsVecLen8));
    rpp_generic_bilinear_load_mask_avx(pSrcY, pSrcX, pRoiLTRB, invalidLoadMask);
    for (int j = 0; j < 8; j++)
    {
        if (invalidLoadMask[j] == 0) // Loading specific pixels where invalidLoadMask is set to 0
        {
            srcVals.srcValsTL.data[j] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsTL.data[j]];
            srcVals.srcValsTR.data[j] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsTR.data[j]];
            srcVals.srcValsBL.data[j] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsBL.data[j]];
            srcVals.srcValsBR.data[j] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsBR.data[j]];
        }
    }
    pSrc[0] = _mm256_loadu_ps(&srcVals.srcValsTL.data[0]);      // R channel Top-Left
    pSrc[1] = _mm256_loadu_ps(&srcVals.srcValsTR.data[0]);      // R channel Top-Right
    pSrc[2] = _mm256_loadu_ps(&srcVals.srcValsBL.data[0]);      // R channel Bottom-Left
    pSrc[3] = _mm256_loadu_ps(&srcVals.srcValsBR.data[0]);      // R channel Bottom-Right
}

template <typename T>
inline void rpp_generic_bilinear_load_3c_avx(T *srcPtrChannel, RpptDescPtr srcDescPtr, RpptBilinearNbhoodLocsVecLen8 &srcLocs, __m256 &pSrcY, __m256 &pSrcX, __m256 *pRoiLTRB, __m256 *pSrc)
{
    Rpp32s invalidLoadMask[8];
    RpptBilinearNbhoodValsVecLen8 srcVals;
    memset(&srcVals, 0, sizeof(RpptBilinearNbhoodValsVecLen8));
    rpp_generic_bilinear_load_mask_avx(pSrcY, pSrcX, pRoiLTRB, invalidLoadMask);
    for (int j = 0; j < 8; j++)
    {
        if (invalidLoadMask[j] == 0) // Loading specific pixels where invalidLoadMask is set to 0
        {
            for (int c = 0; c < srcDescPtr->c * 8; c += 8)
            {
                Rpp32s pos = c + j;
                srcVals.srcValsTL.data[pos] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsTL.data[pos]];
                srcVals.srcValsTR.data[pos] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsTR.data[pos]];
                srcVals.srcValsBL.data[pos] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsBL.data[pos]];
                srcVals.srcValsBR.data[pos] = (Rpp32f) srcPtrChannel[srcLocs.srcLocsBR.data[pos]];
            }
        }
    }
    pSrc[0] = _mm256_loadu_ps(&srcVals.srcValsTL.data[0]);      // R channel Top-Left
    pSrc[1] = _mm256_loadu_ps(&srcVals.srcValsTR.data[0]);      // R channel Top-Right
    pSrc[2] = _mm256_loadu_ps(&srcVals.srcValsBL.data[0]);      // R channel Bottom-Left
    pSrc[3] = _mm256_loadu_ps(&srcVals.srcValsBR.data[0]);      // R channel Bottom-Right
    pSrc[4] = _mm256_loadu_ps(&srcVals.srcValsTL.data[8]);      // G channel Top-Left
    pSrc[5] = _mm256_loadu_ps(&srcVals.srcValsTR.data[8]);      // G channel Top-Right
    pSrc[6] = _mm256_loadu_ps(&srcVals.srcValsBL.data[8]);      // G channel Bottom-Left
    pSrc[7] = _mm256_loadu_ps(&srcVals.srcValsBR.data[8]);      // G channel Bottom-Right
    pSrc[8] = _mm256_loadu_ps(&srcVals.srcValsTL.data[16]);     // B channel Top-Left
    pSrc[9] = _mm256_loadu_ps(&srcVals.srcValsTR.data[16]);     // B channel Top-Right
    pSrc[10] = _mm256_loadu_ps(&srcVals.srcValsBL.data[16]);    // B channel Bottom-Left
    pSrc[11] = _mm256_loadu_ps(&srcVals.srcValsBR.data[16]);    // B channel Bottom-Right
}

/* Resize loads and stores */
inline void rpp_bilinear_load_u8pkd3_to_f32pln3_avx(Rpp8u **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    __m128i px[8];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[0]));  /* Top Row LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[1]));  /* Top Row LOC1 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[0] = _mm_unpacklo_epi8(px[0], px[1]);                                /* unpack 8 lo-pixels of px[0] and px[1] */
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[2]));  /* Top Row LOC2 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[3]));  /* Top Row LOC3 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[1] = _mm_unpacklo_epi8(px[2], px[3]);    /* unpack 8 lo-pixels of px[2] and px[3] */
    px[2] = _mm_unpacklo_epi16(px[0], px[1]);   /* unpack to obtain [R01|R01|R01|R01|G01|G01|G01|G01|B01|B01|B01|B01|R02|R02|R02|R02] */
    px[3] = _mm_unpackhi_epi16(px[0], px[1]);   /* unpack to obtain [G02|G02|G02|G02|B02|B02|B02|B02|R03|R03|R03|R03|G03|G03|G03|G03] */

    /* Repeat the above steps for next 4 dst locations*/
    px[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[4]));
    px[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[5]));
    px[4] = _mm_unpacklo_epi8(px[4], px[5]);
    px[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[6]));
    px[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[7]));
    px[5] = _mm_unpacklo_epi8(px[6], px[7]);
    px[6] = _mm_unpacklo_epi16(px[4], px[5]);
    px[7] = _mm_unpackhi_epi16(px[4], px[5]);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[6], xmm_pxMask00To03)));  /* Contains TopRow R01 for all the dst locations */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask12To15), _mm_shuffle_epi8(px[6], xmm_pxMask12To15)));  /* Contains TopRow R02 for all the dst locations */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask04To07), _mm_shuffle_epi8(px[6], xmm_pxMask04To07)));  /* Contains TopRow G01 for all the dst locations */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask00To03), _mm_shuffle_epi8(px[7], xmm_pxMask00To03)));  /* Contains TopRow G02 for all the dst locations */
    p[8] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[6], xmm_pxMask08To11)));  /* Contains TopRow B01 for all the dst locations */
    p[9] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask04To07), _mm_shuffle_epi8(px[7], xmm_pxMask04To07)));  /* Contains TopRow B02 for all the dst locations */

    /* Repeat above steps to obtain pixels from the BottomRow*/
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[0]));
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[1]));
    px[0] = _mm_unpacklo_epi8(px[0], px[1]);
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[2]));
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[3]));
    px[1] = _mm_unpacklo_epi8(px[2], px[3]);
    px[2] = _mm_unpacklo_epi16(px[0], px[1]);
    px[3] = _mm_unpackhi_epi16(px[0], px[1]);

    px[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[4]));
    px[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[5]));
    px[4] = _mm_unpacklo_epi8(px[4], px[5]);
    px[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[6]));
    px[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[7]));
    px[5] = _mm_unpacklo_epi8(px[6], px[7]);
    px[6] = _mm_unpacklo_epi16(px[4], px[5]);
    px[7] = _mm_unpackhi_epi16(px[4], px[5]);

    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[6], xmm_pxMask00To03)));  /* Contains BottomRow R01 for all the dst locations */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask12To15), _mm_shuffle_epi8(px[6], xmm_pxMask12To15)));  /* Contains BottomRow R02 for all the dst locations */
    p[6] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask04To07), _mm_shuffle_epi8(px[6], xmm_pxMask04To07)));  /* Contains BottomRow G01 for all the dst locations */
    p[7] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask00To03), _mm_shuffle_epi8(px[7], xmm_pxMask00To03)));  /* Contains BottomRow G02 for all the dst locations */
    p[10] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[6], xmm_pxMask08To11))); /* Contains BottomRow B01 for all the dst locations */
    p[11] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask04To07), _mm_shuffle_epi8(px[7], xmm_pxMask04To07))); /* Contains BottomRow B02 for all the dst locations */

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
        p[4] = _mm256_blendv_ps(p[4], p[5], pLowerBoundMask);
        p[6] = _mm256_blendv_ps(p[6], p[7], pLowerBoundMask);
        p[8] = _mm256_blendv_ps(p[8], p[9], pLowerBoundMask);
        p[10] = _mm256_blendv_ps(p[10], p[11], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
        p[5] = _mm256_blendv_ps(p[5], p[4], pUpperBoundMask);
        p[7] = _mm256_blendv_ps(p[7], p[6], pUpperBoundMask);
        p[9] = _mm256_blendv_ps(p[9], p[8], pUpperBoundMask);
        p[11] = _mm256_blendv_ps(p[11], p[10], pUpperBoundMask);
    }
}

inline void rpp_bilinear_load_u8pln1_to_f32pln1_avx(Rpp8u **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    __m128i pxTemp[8];
    pxTemp[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[0]));  /* Top Row load LOC0 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[1]));  /* Top Row load LOC1 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[2]));  /* Top Row load LOC2 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[3]));  /* Top Row load LOC3 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[0] = _mm_unpacklo_epi8(pxTemp[0], pxTemp[1]);    /* unpack 8 lo-pixels of px[0] and px[1] */
    pxTemp[1] = _mm_unpacklo_epi8(pxTemp[2], pxTemp[3]);    /* unpack 8 lo-pixels of px[2] and px[3] */
    pxTemp[0] = _mm_unpacklo_epi16(pxTemp[0], pxTemp[1]);   /* unpack 8 lo-pixels to obtain 1st and 2nd pixels of TopRow*/

    /* Repeat the above steps for next 4 dst locations*/
    pxTemp[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[4]));
    pxTemp[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[5]));
    pxTemp[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[6]));
    pxTemp[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[7]));
    pxTemp[4] = _mm_unpacklo_epi8(pxTemp[4], pxTemp[5]);
    pxTemp[5] = _mm_unpacklo_epi8(pxTemp[6], pxTemp[7]);
    pxTemp[4] = _mm_unpacklo_epi16(pxTemp[4], pxTemp[5]);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask00To03), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask00To03)));    /* Contains 1st pixels of 8 locations from Top row */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask04To07), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask04To07)));    /* Contains 2nd pixels of 8 locations from Top row */

    /* Repeat above steps to obtain pixels from the BottomRow*/
    pxTemp[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[0]));
    pxTemp[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[1]));
    pxTemp[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[2]));
    pxTemp[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[3]));
    pxTemp[0] = _mm_unpacklo_epi8(pxTemp[0], pxTemp[1]);
    pxTemp[1] = _mm_unpacklo_epi8(pxTemp[2], pxTemp[3]);
    pxTemp[0] = _mm_unpacklo_epi16(pxTemp[0], pxTemp[1]);

    pxTemp[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[4]));
    pxTemp[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[5]));
    pxTemp[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[6]));
    pxTemp[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[7]));
    pxTemp[4] = _mm_unpacklo_epi8(pxTemp[4], pxTemp[5]);
    pxTemp[5] = _mm_unpacklo_epi8(pxTemp[6], pxTemp[7]);
    pxTemp[4] = _mm_unpacklo_epi16(pxTemp[4], pxTemp[5]);

    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask00To03), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask00To03)));    /* Contains 1st pixels of 8 locations from Bottom row */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask04To07), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask04To07)));    /* Contains 2nd pixels of 8 locations from Bottom row */

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
    }
}

inline void rpp_store24_f32pln3_to_u8pkd3_avx(Rpp8u* dstPtr, __m256* p)
{
    __m256i px1 = _mm256_packus_epi32(_mm256_cvtps_epi32(p[0]), _mm256_cvtps_epi32(p[1])); /* Pack p[0] and p[1] (R and G channels) */
    __m256i px2 = _mm256_packus_epi32(_mm256_cvtps_epi32(p[2]), avx_px0);                  /* Pack p[2] and zeros (B channel)*/
    px1 = _mm256_packus_epi16(px1, px2);                    /* Pack to obtain |R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|...|R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|... */
    px1 = _mm256_shuffle_epi8(px1, avx_pxShufflePkd);       /* Shuffle to obtain in RGB packed format */
    px1 = _mm256_permutevar8x32_epi32(px1, avx_pxPermPkd);  /* Permute to eliminate the zeros in between */
    _mm256_storeu_si256((__m256i *)(dstPtr), px1);          /* store the 24 U8 pixels in dst */
}

inline void rpp_store8_f32pln1_to_u8pln1_avx(Rpp8u* dstPtr, __m256 &p)
{
    __m256i px1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_cvtps_epi32(p), avx_px0), _MM_SHUFFLE(3,1,2,0));
    px1 = _mm256_packus_epi16(px1, avx_px0);
    rpp_storeu_si64((__m128i *)(dstPtr), _mm256_castsi256_si128(px1));
}

inline void rpp_store24_f32pln3_to_u8pln3_avx(Rpp8u* dstRPtr, Rpp8u* dstGPtr, Rpp8u* dstBPtr, __m256* p)
{
    rpp_store8_f32pln1_to_u8pln1_avx(dstRPtr, p[0]);
    rpp_store8_f32pln1_to_u8pln1_avx(dstGPtr, p[1]);
    rpp_store8_f32pln1_to_u8pln1_avx(dstBPtr, p[2]);
}

inline void rpp_bilinear_load_i8pkd3_to_f32pln3_avx(Rpp8s **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    __m128i px[8];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[0]));  /* Top Row LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[1]));  /* Top Row LOC1 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[0] = _mm_unpacklo_epi8(px[0], px[1]);                                /* unpack 8 lo-pixels of px[0] and px[1] */
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[2]));  /* Top Row LOC2 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[3]));  /* Top Row LOC3 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|...] - Need RGB 01-02 */
    px[1] = _mm_unpacklo_epi8(px[2], px[3]);                                /* unpack 8 lo-pixels of px[2] and px[3] */
    px[2] = _mm_add_epi8(_mm_unpacklo_epi16(px[0], px[1]), xmm_pxConvertI8);    /* unpack to obtain [R01|R01|R01|R01|G01|G01|G01|G01|B01|B01|B01|B01|R02|R02|R02|R02] */
    px[3] = _mm_add_epi8(_mm_unpackhi_epi16(px[0], px[1]), xmm_pxConvertI8);    /* unpack to obtain [G02|G02|G02|G02|B02|B02|B02|B02|R03|R03|R03|R03|G03|G03|G03|G03] */

    /* Repeat the above steps for next 4 dst locations*/
    px[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[4]));
    px[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[5]));
    px[4] = _mm_unpacklo_epi8(px[4], px[5]);
    px[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[6]));
    px[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[7]));
    px[5] = _mm_unpacklo_epi8(px[6], px[7]);
    px[6] = _mm_add_epi8(_mm_unpacklo_epi16(px[4], px[5]), xmm_pxConvertI8);
    px[7] = _mm_add_epi8(_mm_unpackhi_epi16(px[4], px[5]), xmm_pxConvertI8);

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[6], xmm_pxMask00To03)));  /* Contains TopRow 1st pixels R channel for all the dst locations */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask12To15), _mm_shuffle_epi8(px[6], xmm_pxMask12To15)));  /* Contains TopRow 2nd pixels R channel for all the dst locations */
    p[4] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask04To07), _mm_shuffle_epi8(px[6], xmm_pxMask04To07)));  /* Contains TopRow 1st pixels R channel for all the dst locations */
    p[5] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask00To03), _mm_shuffle_epi8(px[7], xmm_pxMask00To03)));  /* Contains TopRow 2nd pixels R channel for all the dst locations */
    p[8] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[6], xmm_pxMask08To11)));  /* Contains TopRow 1st pixels R channel for all the dst locations */
    p[9] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask04To07), _mm_shuffle_epi8(px[7], xmm_pxMask04To07)));  /* Contains TopRow 2nd pixels R channel for all the dst locations */

    /* Repeat above steps to obtain pixels from the BottomRow*/
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[0]));
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[1]));
    px[0] = _mm_unpacklo_epi8(px[0], px[1]);
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[2]));
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[3]));
    px[1] = _mm_unpacklo_epi8(px[2], px[3]);
    px[2] = _mm_add_epi8(_mm_unpacklo_epi16(px[0], px[1]), xmm_pxConvertI8);
    px[3] = _mm_add_epi8(_mm_unpackhi_epi16(px[0], px[1]), xmm_pxConvertI8);

    px[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[4]));
    px[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[5]));
    px[4] = _mm_unpacklo_epi8(px[4], px[5]);
    px[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[6]));
    px[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[7]));
    px[5] = _mm_unpacklo_epi8(px[6], px[7]);
    px[6] = _mm_add_epi8(_mm_unpacklo_epi16(px[4], px[5]), xmm_pxConvertI8);
    px[7] = _mm_add_epi8(_mm_unpackhi_epi16(px[4], px[5]), xmm_pxConvertI8);

    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask00To03), _mm_shuffle_epi8(px[6], xmm_pxMask00To03)));  /* Contains BottomRow 1st pixels R channel for all the dst locations */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask12To15), _mm_shuffle_epi8(px[6], xmm_pxMask12To15)));  /* Contains BottomRow 2nd pixels R channel for all the dst locations */
    p[6] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask04To07), _mm_shuffle_epi8(px[6], xmm_pxMask04To07)));  /* Contains BottomRow 1st pixels G channel for all the dst locations */
    p[7] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask00To03), _mm_shuffle_epi8(px[7], xmm_pxMask00To03)));  /* Contains BottomRow 2nd pixels G channel for all the dst locations */
    p[10] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[2], xmm_pxMask08To11), _mm_shuffle_epi8(px[6], xmm_pxMask08To11))); /* Contains BottomRow 1st pixels B channel for all the dst locations */
    p[11] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(px[3], xmm_pxMask04To07), _mm_shuffle_epi8(px[7], xmm_pxMask04To07))); /* Contains BottomRow 2nd pixels B channel for all the dst locations */

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
        p[4] = _mm256_blendv_ps(p[4], p[5], pLowerBoundMask);
        p[6] = _mm256_blendv_ps(p[6], p[7], pLowerBoundMask);
        p[8] = _mm256_blendv_ps(p[8], p[9], pLowerBoundMask);
        p[10] = _mm256_blendv_ps(p[10], p[11], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
        p[5] = _mm256_blendv_ps(p[5], p[4], pUpperBoundMask);
        p[7] = _mm256_blendv_ps(p[7], p[6], pUpperBoundMask);
        p[9] = _mm256_blendv_ps(p[9], p[8], pUpperBoundMask);
        p[11] = _mm256_blendv_ps(p[11], p[10], pUpperBoundMask);
    }
}

inline void rpp_bilinear_load_i8pln1_to_f32pln1_avx(Rpp8s **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    __m128i pxTemp[8];
    pxTemp[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[0]));  /* Top Row load LOC0 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[1]));  /* Top Row load LOC1 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[2]));  /* Top Row load LOC2 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[3]));  /* Top Row load LOC3 [R01|R02|R03|R04|R05|R06|R07|...|R16] Need R01-02 */
    pxTemp[0] = _mm_unpacklo_epi8(pxTemp[0], pxTemp[1]);    /* unpack 8 lo-pixels of px[0] and px[1] */
    pxTemp[1] = _mm_unpacklo_epi8(pxTemp[2], pxTemp[3]);    /* unpack 8 lo-pixels of px[2] and px[3] */
    pxTemp[0] = _mm_unpacklo_epi16(pxTemp[0], pxTemp[1]);   /* unpack 8 lo-pixels to obtain 1st and 2nd pixels of TopRow*/

    /* Repeat the above steps for next 4 dst locations*/
    pxTemp[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[4]));
    pxTemp[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[5]));
    pxTemp[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[6]));
    pxTemp[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[0] + loc[7]));
    pxTemp[4] = _mm_unpacklo_epi8(pxTemp[4], pxTemp[5]);
    pxTemp[5] = _mm_unpacklo_epi8(pxTemp[6], pxTemp[7]);
    pxTemp[4] = _mm_unpacklo_epi16(pxTemp[4], pxTemp[5]);

    pxTemp[0] = _mm_add_epi8(pxTemp[0], xmm_pxConvertI8);   /* add I8 conversion param */
    pxTemp[4] = _mm_add_epi8(pxTemp[4], xmm_pxConvertI8);   /* add I8 conversion param */

    p[0] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask00To03), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask00To03)));    /* Contains 1st pixels of 8 locations from Top row */
    p[1] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask04To07), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask04To07)));    /* Contains 2nd pixels of 8 locations from Top row */

    /* Repeat above steps to obtain pixels from the BottomRow*/
    pxTemp[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[0]));
    pxTemp[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[1]));
    pxTemp[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[2]));
    pxTemp[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[3]));
    pxTemp[0] = _mm_unpacklo_epi8(pxTemp[0], pxTemp[1]);
    pxTemp[1] = _mm_unpacklo_epi8(pxTemp[2], pxTemp[3]);
    pxTemp[0] = _mm_unpacklo_epi16(pxTemp[0], pxTemp[1]);

    pxTemp[4] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[4]));
    pxTemp[5] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[5]));
    pxTemp[6] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[6]));
    pxTemp[7] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp[1] + loc[7]));
    pxTemp[4] = _mm_unpacklo_epi8(pxTemp[4], pxTemp[5]);
    pxTemp[5] = _mm_unpacklo_epi8(pxTemp[6], pxTemp[7]);
    pxTemp[4] = _mm_unpacklo_epi16(pxTemp[4], pxTemp[5]);

    pxTemp[0] = _mm_add_epi8(pxTemp[0], xmm_pxConvertI8);
    pxTemp[4] = _mm_add_epi8(pxTemp[4], xmm_pxConvertI8);

    p[2] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask00To03), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask00To03)));  /* Contains 1st pixels of 8 locations from Bottom row */
    p[3] = _mm256_cvtepi32_ps(_mm256_setr_m128(_mm_shuffle_epi8(pxTemp[0], xmm_pxMask04To07), _mm_shuffle_epi8(pxTemp[4], xmm_pxMask04To07)));  /* Contains 2nd pixels of 8 locations from Bottom row */

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
    }
}

inline void rpp_store24_f32pln3_to_i8pkd3_avx(Rpp8s* dstPtr, __m256* p)
{
    __m256i px1 = _mm256_packus_epi32(_mm256_cvtps_epi32(p[0]), _mm256_cvtps_epi32(p[1]));  /* Pack the R and G channels to single vector*/
    __m256i px2 = _mm256_packus_epi32(_mm256_cvtps_epi32(p[2]), avx_px0);                   /* Pack the B channel with zeros to single vector */
    px1 = _mm256_packus_epi16(px1, px2);
    px1 = _mm256_shuffle_epi8(px1, avx_pxShufflePkd);       /* Shuffle the pixels to obtain RGB in packed format */
    px1 = _mm256_permutevar8x32_epi32(px1, avx_pxPermPkd);  /* Permute to get continuous RGB pixels */
    px1 = _mm256_sub_epi8(px1, avx_pxConvertI8);            /* add I8 conversion param */
    _mm256_storeu_si256((__m256i *)(dstPtr), px1);          /* store the 12 U8 pixels in dst */
}

inline void rpp_store8_f32pln1_to_i8pln1_avx(Rpp8s* dstPtr, __m256 &p)
{
    __m256i px1 = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_cvtps_epi32(p), avx_px0), _MM_SHUFFLE(3,1,2,0));
    px1 = _mm256_sub_epi8(_mm256_packus_epi16(px1, avx_px0), avx_pxConvertI8);  /* Pack and add I8 conversion param */
    rpp_storeu_si64((__m128i *)(dstPtr), _mm256_castsi256_si128(px1));          /* store the 4 pixels in dst */
}

inline void rpp_store24_f32pln3_to_i8pln3_avx(Rpp8s* dstRPtr, Rpp8s* dstGPtr, Rpp8s* dstBPtr, __m256* p)
{
    rpp_store8_f32pln1_to_i8pln1_avx(dstRPtr, p[0]);
    rpp_store8_f32pln1_to_i8pln1_avx(dstGPtr, p[1]);
    rpp_store8_f32pln1_to_i8pln1_avx(dstBPtr, p[2]);
}

inline void rpp_bilinear_load_f32pkd3_to_f32pln3_avx(Rpp32f **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    __m256 pTemp[10];
    pTemp[0] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[0]);   /* Top Row load LOC0 [R01|G01|B01|R02|G02|B02|XX|XX] Need RGB 01-02 */
    pTemp[1] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[1]);   /* Top Row load LOC1 [R01|G01|B01|R02|G02|B02|XX|XX] Need RGB 01-02 */
    pTemp[2] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[2]);   /* Top Row load LOC2 [R01|G01|B01|R02|G02|B02|XX|XX] Need RGB 01-02 */
    pTemp[3] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[3]);   /* Top Row load LOC3 [R01|G01|B01|R02|G02|B02|XX|XX] Need RGB 01-02 */

    pTemp[4] = _mm256_unpacklo_ps(pTemp[0], pTemp[1]);  /* Unpack to obtain [R01|R01|G01|G01|G02|G02|B02|B02] of LOC0 & LOC1 */
    pTemp[5] = _mm256_unpackhi_ps(pTemp[0], pTemp[1]);  /* Unpack to obtain [B01|B01|R02|R02|XX|XX|XX|XX] of LOC0 & LOC1*/
    pTemp[6] = _mm256_unpacklo_ps(pTemp[2], pTemp[3]);  /* Unpack to obtain [R01|R01|G01|G01|G02|G02|B02|B02] of LOC2 & LOC3 */
    pTemp[7] = _mm256_unpackhi_ps(pTemp[2], pTemp[3]);  /* Unpack to obtain [B01|B01|R02|R02|XX|XX|XX|XX] of LOC2 & LOC3 */

    pTemp[8] = _mm256_unpacklo_pd(pTemp[4], pTemp[6]);  /* Unpack to obtain [R01|R01|R01|R01|G02|G02|G02|G02] */
    pTemp[9] = _mm256_unpackhi_pd(pTemp[4], pTemp[6]);  /* Unpack to obtain [G01|G01|G01|G01|B02|B02|B02|B02] */
    p[8] = _mm256_unpacklo_pd(pTemp[5], pTemp[7]);      /* Unpack to obtain [B01|B01|B01|B01|XX|XX|XX|XX] */
    p[1] = _mm256_unpackhi_pd(pTemp[5], pTemp[7]);      /* Unpack to obtain [R02|R02|R02|R02|XX|XX|XX|XX] */

    /* Repeat the above steps for next 4 dst locations*/
    pTemp[0] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[4]);
    pTemp[1] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[5]);
    pTemp[2] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[6]);
    pTemp[3] = _mm256_loadu_ps(srcRowPtrsForInterp[0] + loc[7]);

    pTemp[4] = _mm256_unpacklo_ps(pTemp[0], pTemp[1]);
    pTemp[5] = _mm256_unpackhi_ps(pTemp[0], pTemp[1]);
    pTemp[6] = _mm256_unpacklo_ps(pTemp[2], pTemp[3]);
    pTemp[7] = _mm256_unpackhi_ps(pTemp[2], pTemp[3]);

    pTemp[0] = _mm256_unpacklo_pd(pTemp[4], pTemp[6]);
    pTemp[1] = _mm256_unpackhi_pd(pTemp[4], pTemp[6]);
    pTemp[2] = _mm256_unpacklo_pd(pTemp[5], pTemp[7]);
    pTemp[3] = _mm256_unpackhi_pd(pTemp[5], pTemp[7]);

    p[0] = _mm256_permute2f128_ps(pTemp[8], pTemp[0], 32);  /* Permute to obtain R01 of 8 dst pixels in Top Row*/
    p[4] = _mm256_permute2f128_ps(pTemp[9], pTemp[1], 32);  /* Permute to obtain G01 of 8 dst pixels in Top Row*/
    p[8] = _mm256_permute2f128_ps(p[8], pTemp[2], 32);      /* Permute to obtain B01 of 8 dst pixels in Top Row*/

    p[1] = _mm256_permute2f128_ps(p[1], pTemp[3], 32);      /* Permute to obtain R02 of 8 dst pixels in Top Row*/
    p[5] = _mm256_permute2f128_ps(pTemp[8], pTemp[0], 49);  /* Permute to obtain G02 of 8 dst pixels in Top Row*/
    p[9] = _mm256_permute2f128_ps(pTemp[9], pTemp[1], 49);  /* Permute to obtain B02 of 8 dst pixels in Top Row*/

    /* Repeat above steps to obtain pixels from the BottomRow*/
    pTemp[0] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[0]);
    pTemp[1] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[1]);
    pTemp[2] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[2]);
    pTemp[3] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[3]);

    pTemp[4] = _mm256_unpacklo_ps(pTemp[0], pTemp[1]);
    pTemp[5] = _mm256_unpackhi_ps(pTemp[0], pTemp[1]);
    pTemp[6] = _mm256_unpacklo_ps(pTemp[2], pTemp[3]);
    pTemp[7] = _mm256_unpackhi_ps(pTemp[2], pTemp[3]);

    pTemp[8] = _mm256_unpacklo_pd(pTemp[4], pTemp[6]);
    pTemp[9] = _mm256_unpackhi_pd(pTemp[4], pTemp[6]);
    p[10] = _mm256_unpacklo_pd(pTemp[5], pTemp[7]);
    p[3] = _mm256_unpackhi_pd(pTemp[5], pTemp[7]);

    pTemp[0] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[4]);
    pTemp[1] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[5]);
    pTemp[2] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[6]);
    pTemp[3] = _mm256_loadu_ps(srcRowPtrsForInterp[1] + loc[7]);

    pTemp[4] = _mm256_unpacklo_ps(pTemp[0], pTemp[1]);
    pTemp[5] = _mm256_unpackhi_ps(pTemp[0], pTemp[1]);
    pTemp[6] = _mm256_unpacklo_ps(pTemp[2], pTemp[3]);
    pTemp[7] = _mm256_unpackhi_ps(pTemp[2], pTemp[3]);

    pTemp[0] = _mm256_unpacklo_pd(pTemp[4], pTemp[6]);
    pTemp[1] = _mm256_unpackhi_pd(pTemp[4], pTemp[6]);
    pTemp[2] = _mm256_unpacklo_pd(pTemp[5], pTemp[7]);
    pTemp[3] = _mm256_unpackhi_pd(pTemp[5], pTemp[7]);

    p[2] = _mm256_permute2f128_ps(pTemp[8], pTemp[0], 32);  /* Permute to obtain R01 of 8 dst pixels in Bottom Row*/
    p[6] = _mm256_permute2f128_ps(pTemp[9], pTemp[1], 32);  /* Permute to obtain G01 of 8 dst pixels in Bottom Row*/
    p[10] = _mm256_permute2f128_ps(p[10], pTemp[2], 32);    /* Permute to obtain B01 of 8 dst pixels in Bottom Row*/

    p[3] = _mm256_permute2f128_ps(p[3], pTemp[3], 32);      /* Permute to obtain R02 of 8 dst pixels in Botttom Row*/
    p[7] = _mm256_permute2f128_ps(pTemp[8], pTemp[0], 49);  /* Permute to obtain G02 of 8 dst pixels in Bottom Row*/
    p[11] = _mm256_permute2f128_ps(pTemp[9], pTemp[1], 49); /* Permute to obtain B02 of 8 dst pixels in Bottom Row*/

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
        p[4] = _mm256_blendv_ps(p[4], p[5], pLowerBoundMask);
        p[6] = _mm256_blendv_ps(p[6], p[7], pLowerBoundMask);
        p[8] = _mm256_blendv_ps(p[8], p[9], pLowerBoundMask);
        p[10] = _mm256_blendv_ps(p[10], p[11], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
        p[5] = _mm256_blendv_ps(p[5], p[4], pUpperBoundMask);
        p[7] = _mm256_blendv_ps(p[7], p[6], pUpperBoundMask);
        p[9] = _mm256_blendv_ps(p[9], p[8], pUpperBoundMask);
        p[11] = _mm256_blendv_ps(p[11], p[10], pUpperBoundMask);
    }
}

inline void rpp_bilinear_load_f32pln1_to_f32pln1_avx(Rpp32f **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    __m128 pTemp[6];
    pTemp[0] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[0]);   /* Top Row load LOC0 [R01|R02|R03|R04] Need R01-02 */
    pTemp[1] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[1]);   /* Top Row load LOC1 [R01|R02|R03|R04] Need R01-02 */
    pTemp[2] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[2]);   /* Top Row load LOC2 [R01|R02|R03|R04] Need R01-02 */
    pTemp[3] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[3]);   /* Top Row load LOC3 [R01|R02|R03|R04] Need R01-02 */
    _MM_TRANSPOSE4_PS(pTemp[0], pTemp[1], pTemp[2], pTemp[3]);  /* Transpose to obtain the R01 and R02 pixels of 4 locations in same vector*/

    /* Repeat the above steps for next 4 dst locations*/
    pTemp[2] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[4]);
    pTemp[3] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[5]);
    pTemp[4] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[6]);
    pTemp[5] = _mm_loadu_ps(srcRowPtrsForInterp[0] + loc[7]);
    _MM_TRANSPOSE4_PS(pTemp[2], pTemp[3], pTemp[4], pTemp[5]);
    p[0] = _mm256_setr_m128(pTemp[0], pTemp[2]);    /* Set to obtain the 1st pixels of 8 dst locations in single vector */
    p[1] = _mm256_setr_m128(pTemp[1], pTemp[3]);    /* Set to obtain the 2nd pixels of 8 dst locations in single vector */

    /* Repeat above steps to obtain pixels from the BottomRow*/
    pTemp[0] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[0]);
    pTemp[1] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[1]);
    pTemp[2] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[2]);
    pTemp[3] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[3]);
    _MM_TRANSPOSE4_PS(pTemp[0], pTemp[1], pTemp[2], pTemp[3]);

    pTemp[2] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[4]);
    pTemp[3] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[5]);
    pTemp[4] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[6]);
    pTemp[5] = _mm_loadu_ps(srcRowPtrsForInterp[1] + loc[7]);
    _MM_TRANSPOSE4_PS(pTemp[2], pTemp[3], pTemp[4], pTemp[5]);
    p[2] = _mm256_setr_m128(pTemp[0], pTemp[2]);
    p[3] = _mm256_setr_m128(pTemp[1], pTemp[3]);

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
    }
}

inline void rpp_store8_f32pln1_to_f32pln1_avx(Rpp32f* dstPtr, __m256 p)
{
    _mm256_storeu_ps(dstPtr, p);   /* store the 8 pixels in dst*/
}

inline void rpp_bilinear_load_f16pkd3_to_f32pln3_avx(Rpp16f **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    Rpp32f topRow0[3][8], topRow1[3][8], bottomRow0[3][8], bottomRow1[3][8];
    for(int cnt = 0; cnt < 8; cnt++)
    {
        *(topRow0[0] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt]);
        *(topRow0[1] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt] + 1);
        *(topRow0[2] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt] + 2);
        *(topRow1[0] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt] + 3);
        *(topRow1[1] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt] + 4);
        *(topRow1[2] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt] + 5);

        *(bottomRow0[0] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt]);
        *(bottomRow0[1] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt] + 1);
        *(bottomRow0[2] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt] + 2);
        *(bottomRow1[0] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt] + 3);
        *(bottomRow1[1] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt] + 4);
        *(bottomRow1[2] + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt] + 5);
    }

    p[0] = _mm256_loadu_ps(topRow0[0]);
    p[4] = _mm256_loadu_ps(topRow0[1]);
    p[8] = _mm256_loadu_ps(topRow0[2]);

    p[1] = _mm256_loadu_ps(topRow1[0]);
    p[5] = _mm256_loadu_ps(topRow1[1]);
    p[9] = _mm256_loadu_ps(topRow1[2]);

    p[2] = _mm256_loadu_ps(bottomRow0[0]);
    p[6] = _mm256_loadu_ps(bottomRow0[1]);
    p[10] = _mm256_loadu_ps(bottomRow0[2]);

    p[3] = _mm256_loadu_ps(bottomRow1[0]);
    p[7] = _mm256_loadu_ps(bottomRow1[1]);
    p[11] = _mm256_loadu_ps(bottomRow1[2]);

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
        p[4] = _mm256_blendv_ps(p[4], p[5], pLowerBoundMask);
        p[6] = _mm256_blendv_ps(p[6], p[7], pLowerBoundMask);
        p[8] = _mm256_blendv_ps(p[8], p[9], pLowerBoundMask);
        p[10] = _mm256_blendv_ps(p[10], p[11], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
        p[5] = _mm256_blendv_ps(p[5], p[4], pUpperBoundMask);
        p[7] = _mm256_blendv_ps(p[7], p[6], pUpperBoundMask);
        p[9] = _mm256_blendv_ps(p[9], p[8], pUpperBoundMask);
        p[11] = _mm256_blendv_ps(p[11], p[10], pUpperBoundMask);
    }
}

inline void rpp_bilinear_load_f16pln1_to_f32pln1_avx(Rpp16f **srcRowPtrsForInterp, Rpp32s *loc, __m256* p, __m256i &pxSrcLoc, __m256i &pxMaxSrcLoc, Rpp32s maxSrcLoc)
{
    Rpp32f topRow0[8], topRow1[8], bottomRow0[8], bottomRow1[8];
    for(int cnt = 0; cnt < 8; cnt++)
    {
        *(topRow0 + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt]);
        *(topRow1 + cnt) = (Rpp32f) *(srcRowPtrsForInterp[0] + loc[cnt] + 1);
        *(bottomRow0 + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt]);
        *(bottomRow1 + cnt) = (Rpp32f) *(srcRowPtrsForInterp[1] + loc[cnt] + 1);
    }
    p[0] = _mm256_loadu_ps(topRow0);
    p[1] = _mm256_loadu_ps(topRow1);
    p[2] = _mm256_loadu_ps(bottomRow0);
    p[3] = _mm256_loadu_ps(bottomRow1);

    if(loc[0] < 0 || loc[7] < 0) // If any src location below min src location is encountered replace the source pixel loaded with first pixel of the row
    {
       __m256 pLowerBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(avx_px0, pxSrcLoc)); // Mask set to true if the location is below min src location
        p[0] = _mm256_blendv_ps(p[0], p[1], pLowerBoundMask);
        p[2] = _mm256_blendv_ps(p[2], p[3], pLowerBoundMask);
    }
    else if(loc[7] > maxSrcLoc || loc[0] > maxSrcLoc) // If any src location beyond max src location -1 is encountered replace the source pixel loaded with first pixel of the row
    {
        __m256 pUpperBoundMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(pxSrcLoc, pxMaxSrcLoc)); // Mask set to true if the location is beyond max src location - 1
        p[1] = _mm256_blendv_ps(p[1], p[0], pUpperBoundMask);
        p[3] = _mm256_blendv_ps(p[3], p[2], pUpperBoundMask);
    }
}

inline void rpp_store24_f32pln3_to_f16pkd3_avx(Rpp16f* dstPtr, __m256* p)
{
    __m128 p128[4];
    p128[0] = _mm256_extractf128_ps(p[0], 0);
    p128[1] = _mm256_extractf128_ps(p[1], 0);
    p128[2] = _mm256_extractf128_ps(p[2], 0);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);

    __m128i px128[4];
    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 3), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 6), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 9), px128[3]);

    p128[0] = _mm256_extractf128_ps(p[0], 1);
    p128[1] = _mm256_extractf128_ps(p[1], 1);
    p128[2] = _mm256_extractf128_ps(p[2], 1);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);

    px128[0] = _mm_cvtps_ph(p128[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm_cvtps_ph(p128[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm_cvtps_ph(p128[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[3] = _mm_cvtps_ph(p128[3], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px128[0]);
    _mm_storeu_si128((__m128i *)(dstPtr + 15), px128[1]);
    _mm_storeu_si128((__m128i *)(dstPtr + 18), px128[2]);
    _mm_storeu_si128((__m128i *)(dstPtr + 21), px128[3]);
}

inline void rpp_store8_f32pln1_to_f16pln1_avx(Rpp16f* dstPtr, __m256 p)
{
    __m128i px128 = _mm256_cvtps_ph(p, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstPtr, px128);
}

inline void rpp_store24_f32pln3_to_f16pln3_avx(Rpp16f* dstRPtr, Rpp16f* dstGPtr, Rpp16f* dstBPtr, __m256* p)
{
    __m128i px128[3];
    px128[0] = _mm256_cvtps_ph(p[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[1] = _mm256_cvtps_ph(p[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    px128[2] = _mm256_cvtps_ph(p[2], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    _mm_storeu_si128((__m128i *)dstRPtr, px128[0]);
    _mm_storeu_si128((__m128i *)dstGPtr, px128[1]);
    _mm_storeu_si128((__m128i *)dstBPtr, px128[2]);
}

inline void rpp_resize_load(Rpp8u *srcPtr, __m128 *p)
{
    rpp_load16_u8_to_f32(srcPtr, p);
}

inline void rpp_resize_load(Rpp32f *srcPtr, __m128 *p)
{
    rpp_load4_f32_to_f32(srcPtr, p);
}

inline void rpp_resize_load(Rpp8s *srcPtr, __m128 *p)
{
    rpp_load16_i8_to_f32(srcPtr, p);
}

inline void rpp_resize_load(Rpp16f *srcPtr, __m128 *p)
{
    Rpp32f srcPtrTemp_ps[8];
    for(int cnt = 0; cnt < 8; cnt ++)
        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtr + cnt);

    rpp_load4_f32_to_f32(srcPtrTemp_ps, p);
    rpp_load4_f32_to_f32(srcPtrTemp_ps + 4, p + 1);
}

inline void rpp_resize_store(Rpp8u *dstPtr, __m128 *p)
{
    rpp_store16_f32_to_u8(dstPtr, p);
}

inline void rpp_resize_store(Rpp32f *dstPtr, __m128 *p)
{
    rpp_store4_f32_to_f32(dstPtr, p);
}

inline void rpp_resize_store(Rpp8s *dstPtr, __m128 *p)
{
    rpp_store16_f32_to_i8(dstPtr, p);
}

inline void rpp_resize_store(Rpp16f *dstPtr, __m128 *p)
{
    Rpp32f dstPtrTemp_ps[8];
    rpp_store4_f32_to_f32(dstPtrTemp_ps, p);
    rpp_store4_f32_to_f32(dstPtrTemp_ps + 4, p + 1);
    for(int cnt = 0; cnt < 8; cnt ++)
        *(dstPtr + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
}

inline void rpp_resize_store_pln3(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m128 *p)
{
    rpp_store48_f32pln3_to_u8pln3(dstPtrR, dstPtrG, dstPtrB, p);
}

inline void rpp_resize_store_pln3(Rpp32f *dstPtrR, Rpp32f *dstPtrG, Rpp32f *dstPtrB, __m128 *p)
{
    rpp_store12_f32pln3_to_f32pln3(dstPtrR, dstPtrG, dstPtrB, p);
}

inline void rpp_resize_store_pln3(Rpp16f *dstPtrR, Rpp16f *dstPtrG, Rpp16f *dstPtrB, __m128 *p)
{
    __m128 temp[3];
    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
    temp[0] = p[0]; /* R channel */
    temp[1] = p[2]; /* G channel */
    temp[2] = p[4]; /* B channel */
    rpp_store12_f32pln3_to_f32pln3(dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, temp);
    temp[0] = p[1]; /* R channel */
    temp[1] = p[3]; /* R channel */
    temp[2] = p[5]; /* R channel */
    rpp_store12_f32pln3_to_f32pln3(dstPtrTempR_ps + 4, dstPtrTempG_ps + 4, dstPtrTempB_ps + 4, temp);

    for(int cnt = 0; cnt < 8; cnt++)
    {
        *(dstPtrR + cnt) = (Rpp16f) *(dstPtrTempR_ps + cnt);
        *(dstPtrG + cnt) = (Rpp16f) *(dstPtrTempG_ps + cnt);
        *(dstPtrB + cnt) = (Rpp16f) *(dstPtrTempB_ps + cnt);
    }
}

inline void rpp_resize_store_pln3(Rpp8s *dstPtrR, Rpp8s *dstPtrG, Rpp8s *dstPtrB, __m128 *p)
{
    rpp_store48_f32pln3_to_i8pln3(dstPtrR, dstPtrG, dstPtrB, p);
}

inline void rpp_resize_store_pkd3(Rpp8u *dstPtr, __m128 *p)
{
    rpp_store48_f32pln3_to_u8pkd3(dstPtr, p);
}

inline void rpp_resize_store_pkd3(Rpp32f *dstPtr, __m128 *p)
{
    rpp_store12_f32pln3_to_f32pkd3(dstPtr, p);
}

inline void rpp_resize_store_pkd3(Rpp16f *dstPtr, __m128 *p)
{
    __m128 temp[3];
    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
    temp[0] = p[0]; /* R channel */
    temp[1] = p[2]; /* G channel */
    temp[2] = p[4]; /* B channel */
    rpp_store12_f32pln3_to_f32pln3(dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, temp);
    temp[0] = p[1]; /* R channel */
    temp[1] = p[3]; /* G channel */
    temp[2] = p[5]; /* B channel */
    rpp_store12_f32pln3_to_f32pln3(dstPtrTempR_ps + 4, dstPtrTempG_ps + 4, dstPtrTempB_ps + 4, temp);

    for(int cnt = 0; cnt < 8; cnt++)
    {
        *dstPtr++ = (Rpp16f) *(dstPtrTempR_ps + cnt);
        *dstPtr++ = (Rpp16f) *(dstPtrTempG_ps + cnt);
        *dstPtr++ = (Rpp16f) *(dstPtrTempB_ps + cnt);
    }
}

inline void rpp_resize_store_pkd3(Rpp8s *dstPtr, __m128 *p)
{
    rpp_store48_f32pln3_to_i8pkd3(dstPtr, p);
}

inline void rpp_resize_nn_load_u8pkd3(Rpp8u *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));  // LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));  // LOC1 load [R11|G11|B11|R12|G12|B12|R13|G13|B13|R14|G14|B14|R15|G15|B15|R16] - Need RGB 11
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));  // LOC2 load [R21|G21|B21|R22|G22|B22|R23|G23|B23|R24|G24|B24|R25|G25|B25|R26] - Need RGB 21
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));  // LOC3 load [R31|G31|B31|R32|G32|B32|R33|G33|B33|R34|G34|B34|R35|G35|B35|R36] - Need RGB 31
    px[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R01|G01|B01|R02|R11|G11|B11|R12|R21|G21|B21|R22|R31|G31|B31|R32]
    p = _mm_shuffle_epi8(px[0], xmm_pkd_mask);    // Shuffle to obtain 4 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|00|00|00|00]
}

template<typename T>
inline void rpp_resize_nn_extract_pkd3_avx(T *srcRowPtrsForInterp, Rpp32s *loc, __m256i &p)
{
    p = _mm256_setr_epi8(*(srcRowPtrsForInterp + loc[0]), *(srcRowPtrsForInterp + loc[0] + 1), *(srcRowPtrsForInterp + loc[0] + 2), 
                         *(srcRowPtrsForInterp + loc[1]), *(srcRowPtrsForInterp + loc[1] + 1), *(srcRowPtrsForInterp + loc[1] + 2), 
                         *(srcRowPtrsForInterp + loc[2]), *(srcRowPtrsForInterp + loc[2] + 1), *(srcRowPtrsForInterp + loc[2] + 2),
                         *(srcRowPtrsForInterp + loc[3]), *(srcRowPtrsForInterp + loc[3] + 1), *(srcRowPtrsForInterp + loc[3] + 2),
                         *(srcRowPtrsForInterp + loc[4]), *(srcRowPtrsForInterp + loc[4] + 1), *(srcRowPtrsForInterp + loc[4] + 2),
                         *(srcRowPtrsForInterp + loc[5]), *(srcRowPtrsForInterp + loc[5] + 1), *(srcRowPtrsForInterp + loc[5] + 2),
                         *(srcRowPtrsForInterp + loc[6]), *(srcRowPtrsForInterp + loc[6] + 1), *(srcRowPtrsForInterp + loc[6] + 2),
                         *(srcRowPtrsForInterp + loc[7]), *(srcRowPtrsForInterp + loc[7] + 1), *(srcRowPtrsForInterp + loc[7] + 2),
                         0, 0, 0, 0, 0, 0, 0, 0);
}

inline void rpp_resize_nn_load_u8pln1(Rpp8u *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));  // LOC0 load [R01|R02|R03|R04|R05|R06...] - Need R01
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));  // LOC1 load [R11|R12|R13|R14|R15|R16...] - Need R11
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));  // LOC2 load [R21|R22|R23|R24|R25|R26...] - Need R21
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));  // LOC3 load [R31|R32|R33|R34|R35|R36...] - Need R31
    px[0] = _mm_unpacklo_epi8(px[0], px[2]);    // unpack 8 lo-pixels of px[0] and px[2]
    px[1] = _mm_unpacklo_epi8(px[1], px[3]);    // unpack 8 lo-pixels of px[1] and px[3]
    p = _mm_unpacklo_epi8(px[0], px[1]);    // unpack to obtain [R01|R11|R21|R31|00|00|00|00|00|00|00|00|00|00|00|00]
}

template<typename T>
inline void rpp_resize_nn_extract_pln1_avx(T *srcRowPtrsForInterp, Rpp32s *loc, __m256i &p)
{
    p = _mm256_setr_epi8(*(srcRowPtrsForInterp + loc[0]), *(srcRowPtrsForInterp + loc[1]), 
                         *(srcRowPtrsForInterp + loc[2]), *(srcRowPtrsForInterp + loc[3]),
                         *(srcRowPtrsForInterp + loc[4]), *(srcRowPtrsForInterp + loc[5]),
                         *(srcRowPtrsForInterp + loc[6]), *(srcRowPtrsForInterp + loc[7]),
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

inline void rpp_resize_nn_load_f32pkd3_to_f32pln3(Rpp32f *srcRowPtrsForInterp, Rpp32s *loc, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcRowPtrsForInterp + loc[0]);  // LOC0 load [R01|G01|B01|R02] - Need RGB 01
    p[1] = _mm_loadu_ps(srcRowPtrsForInterp + loc[1]);  // LOC1 load [R11|G11|B11|R12] - Need RGB 11
    p[2] = _mm_loadu_ps(srcRowPtrsForInterp + loc[2]);  // LOC2 load [R21|G21|B21|R22] - Need RGB 21
    __m128 pTemp = _mm_loadu_ps(srcRowPtrsForInterp + loc[3]);  // LOC2 load [R31|G31|B31|R32]  - Need RGB 31
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], pTemp); // Transpose to obtain RGB in each vector
}

inline void rpp_resize_nn_load_f32pkd3_to_f32pln3_avx(Rpp32f *srcRowPtrsForInterp, Rpp32s *loc, __m256 *p)
{
    __m128 p128[8];
    p128[0] = _mm_loadu_ps(srcRowPtrsForInterp + loc[0]);
    p128[1] = _mm_loadu_ps(srcRowPtrsForInterp + loc[1]);
    p128[2] = _mm_loadu_ps(srcRowPtrsForInterp + loc[2]);
    p128[3] = _mm_loadu_ps(srcRowPtrsForInterp + loc[3]);
    _MM_TRANSPOSE4_PS(p128[0], p128[1], p128[2], p128[3]);
    p128[4] = _mm_loadu_ps(srcRowPtrsForInterp + loc[4]);
    p128[5] = _mm_loadu_ps(srcRowPtrsForInterp + loc[5]);
    p128[6] = _mm_loadu_ps(srcRowPtrsForInterp + loc[6]);
    p128[7] = _mm_loadu_ps(srcRowPtrsForInterp + loc[7]);
    _MM_TRANSPOSE4_PS(p128[4], p128[5], p128[6], p128[7]);
    p[0] = _mm256_setr_m128(p128[0], p128[4]);
    p[1] = _mm256_setr_m128(p128[1], p128[5]);
    p[2] = _mm256_setr_m128(p128[2], p128[6]);
}

inline void rpp_resize_nn_load_f16pkd3_to_f32pln3_avx(Rpp16f *srcRowPtrsForInterp, Rpp32s *loc, __m256 *p)
{
    p[0] = _mm256_setr_ps((Rpp32f)*(srcRowPtrsForInterp + loc[0]), (Rpp32f)*(srcRowPtrsForInterp + loc[1]),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[2]), (Rpp32f)*(srcRowPtrsForInterp + loc[3]),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[4]), (Rpp32f)*(srcRowPtrsForInterp + loc[5]),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[6]), (Rpp32f)*(srcRowPtrsForInterp + loc[7]));

    p[1] = _mm256_setr_ps((Rpp32f)*(srcRowPtrsForInterp + loc[0] + 1), (Rpp32f)*(srcRowPtrsForInterp + loc[1] + 1),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[2] + 1), (Rpp32f)*(srcRowPtrsForInterp + loc[3] + 1),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[4] + 1), (Rpp32f)*(srcRowPtrsForInterp + loc[5] + 1),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[6] + 1), (Rpp32f)*(srcRowPtrsForInterp + loc[7] + 1));

    p[2] = _mm256_setr_ps((Rpp32f)*(srcRowPtrsForInterp + loc[0] + 2), (Rpp32f)*(srcRowPtrsForInterp + loc[1] + 2),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[2] + 2), (Rpp32f)*(srcRowPtrsForInterp + loc[3] + 2),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[4] + 2), (Rpp32f)*(srcRowPtrsForInterp + loc[5] + 2),
                          (Rpp32f)*(srcRowPtrsForInterp + loc[6] + 2), (Rpp32f)*(srcRowPtrsForInterp + loc[7] + 2));
}

inline void rpp_resize_nn_load_f32pln1(Rpp32f *srcRowPtrsForInterp, Rpp32s *loc, __m128 &p)
{
    __m128 pTemp[4];
    pTemp[0] = _mm_loadu_ps(srcRowPtrsForInterp + loc[0]);  // LOC0 load [R01|R02|R03|R04] - Need R01
    pTemp[1] = _mm_loadu_ps(srcRowPtrsForInterp + loc[1]);  // LOC1 load [R11|R12|R13|R14] - Need R11
    pTemp[2] = _mm_loadu_ps(srcRowPtrsForInterp + loc[2]);  // LOC2 load [R21|R22|R23|R24] - Need R21
    pTemp[3] = _mm_loadu_ps(srcRowPtrsForInterp + loc[3]);  // LOC3 load [R31|R32|R33|R34] - Need R31
    pTemp[0] = _mm_unpacklo_ps(pTemp[0], pTemp[2]);
    pTemp[1] = _mm_unpacklo_ps(pTemp[1], pTemp[3]);
    p = _mm_unpacklo_ps(pTemp[0], pTemp[1]);    // Unpack to obtain [R01|R11|R21|R31]
}

inline void rpp_resize_nn_load_f32pln1_avx(Rpp32f *srcRowPtrsForInterp, Rpp32s *loc, __m256 &p)
{
    p = _mm256_setr_ps(*(srcRowPtrsForInterp + loc[0]), *(srcRowPtrsForInterp + loc[1]),
                       *(srcRowPtrsForInterp + loc[2]), *(srcRowPtrsForInterp + loc[3]),
                       *(srcRowPtrsForInterp + loc[4]), *(srcRowPtrsForInterp + loc[5]),
                       *(srcRowPtrsForInterp + loc[6]), *(srcRowPtrsForInterp + loc[7]));
}

inline void rpp_resize_nn_load_f16pln1_avx(Rpp16f *srcRowPtrsForInterp, Rpp32s *loc, __m256 &p)
{
    p = _mm256_setr_ps((Rpp32f)*(srcRowPtrsForInterp + loc[0]), (Rpp32f)*(srcRowPtrsForInterp + loc[1]),
                       (Rpp32f)*(srcRowPtrsForInterp + loc[2]), (Rpp32f)*(srcRowPtrsForInterp + loc[3]),
                       (Rpp32f)*(srcRowPtrsForInterp + loc[4]), (Rpp32f)*(srcRowPtrsForInterp + loc[5]),
                       (Rpp32f)*(srcRowPtrsForInterp + loc[6]), (Rpp32f)*(srcRowPtrsForInterp + loc[7]));
}

inline void rpp_resize_nn_load_i8pkd3(Rpp8s *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));  // LOC0 load [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|R05|G05|B05|R06] - Need RGB 01
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));  // LOC1 load [R11|G11|B11|R12|G12|B12|R13|G13|B13|R14|G14|B14|R15|G15|B15|R16] - Need RGB 11
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));  // LOC2 load [R21|G21|B21|R22|G22|B22|R23|G23|B23|R24|G24|B24|R25|G25|B25|R26] - Need RGB 21
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));  // LOC3 load [R31|G31|B31|R32|G32|B32|R33|G33|B33|R34|G34|B34|R35|G35|B35|R36] - Need RGB 31
    px[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));    // Unpack to obtain [R01|G01|B01|R02|R11|G11|B11|R12|R21|G21|B21|R22|R31|G31|B31|R32]
    p = _mm_shuffle_epi8(px[0], xmm_pkd_mask);    // Shuffle to obtain 4 RGB [R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|00|00|00|00]
}

inline void rpp_resize_nn_load_i8pln1(Rpp8s *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));  // LOC0 load [R01|R02|R03|R04|R05|R06...] - Need R01
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));  // LOC1 load [R11|R12|R13|R14|R15|R16...] - Need R11
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));  // LOC2 load [R21|R22|R23|R24|R25|R26...] - Need R21
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));  // LOC3 load [R31|R32|R33|R34|R35|R36...] - Need R31
    px[0] = _mm_unpacklo_epi8(px[0], px[2]);    // unpack 8 lo-pixels of px[0] and px[2]
    px[1] = _mm_unpacklo_epi8(px[1], px[3]);    // unpack 8 lo-pixels of px[1] and px[3]
    p = _mm_unpacklo_epi8(px[0], px[1]);    // unpack to obtain [R01|R11|R21|R31|00|00|00|00|00|00|00|00|00|00|00|00]
}

inline void rpp_store12_u8pkd3_to_u8pln3(Rpp8u* dstPtrR, Rpp8u* dstPtrG, Rpp8u* dstPtrB, __m128i &p)
{
    rpp_storeu_si32((__m128i *)(dstPtrR), _mm_shuffle_epi8(p, xmm_char_maskR)); /* Shuffle and extract the R pixels*/
    rpp_storeu_si32((__m128i *)(dstPtrG), _mm_shuffle_epi8(p, xmm_char_maskG)); /* Shuffle and extract the G pixels*/
    rpp_storeu_si32((__m128i *)(dstPtrB), _mm_shuffle_epi8(p, xmm_char_maskB)); /* Shuffle and extract the B pixels*/
}

inline void rpp_store24_u8pkd3_to_u8pln3_avx(Rpp8u* dstPtrR, Rpp8u* dstPtrG, Rpp8u* dstPtrB, __m256i &p)
{
    __m128i p128[2];
    p128[0] = _mm256_castsi256_si128(p); /* R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|R41|G41|B41|R51 */
    rpp_storeu_si32((__m128i *)(dstPtrR), _mm_shuffle_epi8(p128[0], xmm_char_maskR)); /* shuffle to get R01-R04*/
    rpp_storeu_si32((__m128i *)(dstPtrG), _mm_shuffle_epi8(p128[0], xmm_char_maskG)); /* shuffle to get G01-G04*/
    rpp_storeu_si32((__m128i *)(dstPtrB), _mm_shuffle_epi8(p128[0], xmm_char_maskB)); /* shuffle to get B01-B04*/

    p128[1] = _mm256_extractf128_si256(p, 1); /* G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00|00|00|00|00 */
    const __m128i shuffleMask = _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0x80, 0x80, 0x80, 0x80);
    p128[0] = _mm_unpackhi_epi64(p128[0], xmm_p0); /* B21|R31|G31|B31|R41|G41|B41|R51|00|00|00|00|00|00|00|00 */
    p128[1] = _mm_unpacklo_epi64(p128[1], p128[0]); /* G51|B51|R61|G61|B61|R71|G71|B71|B21|R31|G31|B31|R41|G41|B41|R51 */
    p128[1] = _mm_shuffle_epi8(p128[1], shuffleMask); /* R41|G41|B41|R51|G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00 */
    rpp_storeu_si32((__m128i *)(dstPtrR + 4), _mm_shuffle_epi8(p128[1], xmm_char_maskR)); /* shuffle to get R05-R08*/
    rpp_storeu_si32((__m128i *)(dstPtrG + 4), _mm_shuffle_epi8(p128[1], xmm_char_maskG)); /* shuffle to get G05-G08*/
    rpp_storeu_si32((__m128i *)(dstPtrB + 4), _mm_shuffle_epi8(p128[1], xmm_char_maskB)); /* shuffle to get B05-B08*/
}

inline void rpp_store24_i8pkd3_to_i8pln3_avx(Rpp8s* dstPtrR, Rpp8s* dstPtrG, Rpp8s* dstPtrB, __m256i &p)
{
    __m128i p128[2];
    p128[0] = _mm256_castsi256_si128(p); /* R01|G01|B01|R11|G11|B11|R21|G21|B21|R31|G31|B31|R41|G41|B41|R51 */
    rpp_storeu_si32((__m128i *)(dstPtrR), _mm_shuffle_epi8(p128[0], xmm_char_maskR)); /* shuffle to get R01-R04*/
    rpp_storeu_si32((__m128i *)(dstPtrG), _mm_shuffle_epi8(p128[0], xmm_char_maskG)); /* shuffle to get G01-G04*/
    rpp_storeu_si32((__m128i *)(dstPtrB), _mm_shuffle_epi8(p128[0], xmm_char_maskB)); /* shuffle to get B01-B04*/

    p128[1] = _mm256_extractf128_si256(p, 1); /* G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00|00|00|00|00 */
    const __m128i shuffleMask = _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0x80, 0x80, 0x80, 0x80);
    p128[0] = _mm_unpackhi_epi64(p128[0], xmm_p0); /* B21|R31|G31|B31|R41|G41|B41|R51|00|00|00|00|00|00|00|00 */
    p128[1] = _mm_unpacklo_epi64(p128[1], p128[0]); /* G51|B51|R61|G61|B61|R71|G71|B71|B21|R31|G31|B31|R41|G41|B41|R51 */
    p128[1] = _mm_shuffle_epi8(p128[1], shuffleMask); /* R41|G41|B41|R51|G51|B51|R61|G61|B61|R71|G71|B71|00|00|00|00 */
    rpp_storeu_si32((__m128i *)(dstPtrR + 4), _mm_shuffle_epi8(p128[1], xmm_char_maskR)); /* shuffle to get R05-R08*/
    rpp_storeu_si32((__m128i *)(dstPtrG + 4), _mm_shuffle_epi8(p128[1], xmm_char_maskG)); /* shuffle to get G05-G08*/
    rpp_storeu_si32((__m128i *)(dstPtrB + 4), _mm_shuffle_epi8(p128[1], xmm_char_maskB)); /* shuffle to get B05-B08*/
}

inline void rpp_store12_u8_to_u8(Rpp8u* dstPtr, __m128i &p)
{
    _mm_storeu_si128((__m128i *)(dstPtr), p);
}

inline void rpp_store4_u8pln1_to_u8pln1(Rpp8u* dstPtr, __m128i &p)
{
    rpp_storeu_si32((__m128i *)(dstPtr), p);
}

inline void rpp_store4_i8pln1_to_i8pln1(Rpp8s* dstPtr, __m128i &p)
{
    rpp_storeu_si32((__m128i *)(dstPtr), p);
}

inline void rpp_store24_u8_to_u8_avx(Rpp8u* dstPtr, __m256i &p)
{
    _mm256_storeu_si256((__m256i *)(dstPtr), p);
}

inline void rpp_store24_i8_to_i8_avx(Rpp8s* dstPtr, __m256i &p)
{
    _mm256_storeu_si256((__m256i *)(dstPtr), p);
}

inline void rpp_store12_u8pln3_to_u8pkd3(Rpp8u* dstPtr, __m128i *p)
{
    __m128i px[4];
    px[0] = _mm_unpacklo_epi8(p[0], p[1]);
    px[1] = _mm_unpacklo_epi64(px[0], p[2]);
    _mm_storeu_si128((__m128i *)(dstPtr), _mm_shuffle_epi8(px[1], xmm_store4_pkd_pixels));
}

inline void rpp_store24_u8pln3_to_u8pkd3_avx(Rpp8u* dstPtr, __m256i *p)
{
    __m128i px[5];
    px[0] = _mm256_castsi256_si128(p[0]);            /* R01|R11|R21|R31|R41|R51|R61|R71|00|00|00|00|00|00|00|00] */
    px[1] = _mm256_castsi256_si128(p[1]);            /* G01|G11|G21|G31|G41|G51|G61|G71|00|00|00|00|00|00|00|00] */
    px[2] = _mm256_castsi256_si128(p[2]);            /* B01|B11|B21|B31|B41|B51|B61|B71|00|00|00|00|00|00|00|00] */

    px[3] = _mm_unpacklo_epi8(px[0], px[1]);         /* unpack as R01|G01|R11|G11|R21|G21|R31|G31|R41|G41|R51|G51|R61|G61|R71|G71 */
    px[4] = _mm_unpacklo_epi64(px[3], px[2]);        /* unpack as R01|G01|R11|G11|R21|G21|R31|G31|B01|B11|B21|B31|B41|B51|B61|B71 */
    _mm_storeu_si128((__m128i *)(dstPtr), _mm_shuffle_epi8(px[4], xmm_store4_pkd_pixels)); /* shuffle to get RGB 00-03 */

    const __m128i xmm_shuffle_mask = _mm_setr_epi8(0, 1, 12, 2, 3, 13, 4, 5, 14, 6, 7, 15, 0x80, 0x80, 0x80, 0x80);
    px[4] = _mm_unpackhi_epi64(px[3], px[4]);        /* unpack as R41|G41|R51|G51|R61|G61|R71|G71|B01|B11|B21|B31|B41|B51|B61|B71] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(px[4], xmm_shuffle_mask)); /* shuffle to get RGB 04-07 */
}

inline void rpp_store12_i8pkd3_to_i8pln3(Rpp8s* dstPtrR, Rpp8s* dstPtrG, Rpp8s* dstPtrB, __m128i &p)
{
    rpp_storeu_si32((__m128i *)(dstPtrR), _mm_shuffle_epi8(p, xmm_char_maskR)); /* Shuffle and extract the R pixels*/
    rpp_storeu_si32((__m128i *)(dstPtrG), _mm_shuffle_epi8(p, xmm_char_maskG)); /* Shuffle and extract the G pixels*/
    rpp_storeu_si32((__m128i *)(dstPtrB), _mm_shuffle_epi8(p, xmm_char_maskB)); /* Shuffle and extract the B pixels*/
}



inline void rpp_store12_i8_to_i8(Rpp8s* dstPtr, __m128i &p)
{
    _mm_storeu_si128((__m128i *)(dstPtr), p);
}

inline void rpp_store12_i8pln3_to_i8pkd3(Rpp8s* dstPtr, __m128i *p)
{
    __m128i px[4];
    px[0] = _mm_unpacklo_epi8(p[0], p[1]);
    px[1] = _mm_unpacklo_epi64(px[0], p[2]);
    _mm_storeu_si128((__m128i *)(dstPtr), _mm_shuffle_epi8(px[1], xmm_store4_pkd_pixels));
}

inline void rpp_store24_i8pln3_to_i8pkd3_avx(Rpp8s* dstPtr, __m256i *p)
{
    __m128i px[5];
    px[0] = _mm256_castsi256_si128(p[0]);            // [R01|R11|R21|R31|R41|R51|R61|R71|00|00|00|00|00|00|00|00]
    px[1] = _mm256_castsi256_si128(p[1]);            // [G01|G11|G21|G31|G41|G51|G61|G71|00|00|00|00|00|00|00|00]
    px[2] = _mm256_castsi256_si128(p[2]);            // [B01|B11|B21|B31|B41|B51|B61|B71|00|00|00|00|00|00|00|00]

    px[3] = _mm_unpacklo_epi8(px[0], px[1]);         // [R01|G01|R11|G11|R21|G21|R31|G31|R41|G41|R51|G51|R61|G61|R71|G71]
    px[4] = _mm_unpacklo_epi64(px[3], px[2]);        // [R01|G01|R11|G11|R21|G21|R31|G31|B01|B11|B21|B31|B41|B51|B61|B71]
    _mm_storeu_si128((__m128i *)(dstPtr), _mm_shuffle_epi8(px[4], xmm_store4_pkd_pixels)); // shuffle to get RGB 00-03

    const __m128i xmm_shuffle_mask = _mm_setr_epi8(0, 1, 12, 2, 3, 13, 4, 5, 14, 6, 7, 15, 0x80, 0x80, 0x80, 0x80);
    px[4] = _mm_unpackhi_epi64(px[3], px[4]);        // [R41|G41|R51|G51|R61|G61|R71|G71|B01|B11|B21|B31|B41|B51|B61|B71]
    _mm_storeu_si128((__m128i *)(dstPtr + 12), _mm_shuffle_epi8(px[4], xmm_shuffle_mask)); // shuffle to get RGB 04-07
}

inline void rpp_store12_f32pkd3_to_f32pkd3(Rpp32f* dstPtr, __m128 *p)
{
    _mm_storeu_ps(dstPtr, p[0]); /* Store RGB set 1 */
    _mm_storeu_ps(dstPtr + 3, p[1]); /* Store RGB set 2 */
    _mm_storeu_ps(dstPtr + 6, p[2]); /* Store RGB set 3 */
    _mm_storeu_ps(dstPtr + 9, p[3]); /* Store RGB set 4 */
}

inline void rpp_store24_f32pkd3_to_f32pkd3_avx(Rpp32f* dstPtr, __m256 *p)
{
    _mm256_storeu_ps(dstPtr, p[0]); /* Store RGB set 1 */
    _mm256_storeu_ps(dstPtr + 8, p[1]); /* Store RGB set 2 */
    _mm256_storeu_ps(dstPtr + 16, p[2]); /* Store RGB set 3 */
}

inline void rpp_convert24_pkd3_to_pln3(__m128i &pxLower, __m128i &pxUpper, __m128i *pxDstChn)
{
    // pxLower = R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5 G5 B5 R6
    // pxUpper = G6 B6 R7 G7 B7 R8 G8 B8 0  0  0  0  0  0  0  0
    // shuffle1 - R1 R2 R3 R4 0 0 0 0 0 0 0 0 0 0 0 0
    // shuffle2 - G1 G2 G3 G4 0 0 0 0 0 0 0 0 0 0 0 0
    // shuffle3 - B1 B2 B3 B4 0 0 0 0 0 0 0 0 0 0 0 0
    // blend    - G6 B6 R7 G7 B7 R8 G8 B8 0 0 0 0 R5 G5 B5 R6
    // R5 R6 R7 R8 G5 G6 G7 G8 B5 B6 B7 B8 0 0 0 0
    // R1 R2 R3 R4 R5 R6 R7 R8 0  0  0  0  0  0  0  0
    // G1 G2 G3 G4 G5 G6 G7 G8 0  0  0  0  0  0  0  0
    // B1 B2 B3 B4 0  0  0  0  B5 B6 B7 B8 0  0  0  0
    // B1 B2 B3 B4 B5 B6 B7 B8 0  0  0  0  0  0  0  0

    __m128i pxTempUpper = _mm_blend_epi16(pxUpper, pxLower, 192);
    __m128i xmm_shuffle_mask = _mm_setr_epi8(12, 15, 2, 5, 13, 0, 3, 6, 14, 1, 4, 7, 0x80, 0x80, 0x80, 0x80);
    pxTempUpper = _mm_shuffle_epi8(pxTempUpper, xmm_shuffle_mask);

    pxDstChn[0] = _mm_unpacklo_epi32(_mm_shuffle_epi8(pxLower, xmm_char_maskR), pxTempUpper);
    pxDstChn[1] = _mm_blend_epi16(_mm_shuffle_epi8(pxLower, xmm_char_maskG), pxTempUpper, 12);

    xmm_shuffle_mask = _mm_setr_epi8(0, 1, 2, 3, 8, 9, 10, 11, 0x80, 0x80, 0x80, 0x80,0x80, 0x80, 0x80, 0x80);
    pxDstChn[2] = _mm_shuffle_epi8(_mm_blend_epi16(_mm_shuffle_epi8(pxLower, xmm_char_maskB), pxTempUpper, 48), xmm_shuffle_mask);
}

inline void rpp_convert72_pln3_to_pkd3(__m256i *pxSrc, __m128i *pxDst)
{
    const __m128i pxMask = _mm_setr_epi8(0, 1, 12, 2, 3, 13, 4, 5, 14, 6, 7, 15, 0x80, 0x80, 0x80, 0x80);

    __m256i px[2];
    px[0] = _mm256_unpacklo_epi8(pxSrc[0], pxSrc[1]);
    px[1] = _mm256_unpackhi_epi8(pxSrc[0], pxSrc[1]);

    __m128i pxTemp[4];
    // RGB 1-8
    pxTemp[0] = _mm256_castsi256_si128(px[0]);
    pxTemp[1] = _mm256_castsi256_si128(pxSrc[2]);

    // RGB 1-4, shuffle to get correct order
    // RGB 5-8, shuffle to get correct order
    pxTemp[2] = _mm_unpacklo_epi64(pxTemp[0], pxTemp[1]);
    pxTemp[3] = _mm_unpacklo_epi64(_mm_srli_si128(pxTemp[0], 8), pxTemp[1]);
    pxDst[0] = _mm_shuffle_epi8(pxTemp[2], xmm_store4_pkd_pixels);
    pxDst[1] = _mm_shuffle_epi8(pxTemp[3], pxMask);

    // RGB 9-16
    pxTemp[0] = _mm256_castsi256_si128(px[1]),
    pxTemp[1] = _mm256_castsi256_si128(pxSrc[2]);

    // RGB 9-12, shuffle to get correct order
    // RGB 13-15, shuffle to get correct order
    pxTemp[2] =  _mm_unpacklo_epi64(pxTemp[0], _mm_srli_si128(pxTemp[1], 8));
    pxTemp[3] = _mm_unpackhi_epi64(pxTemp[0], pxTemp[1]);
    pxDst[2] = _mm_shuffle_epi8(pxTemp[2], xmm_store4_pkd_pixels);
    pxDst[3] = _mm_shuffle_epi8(pxTemp[3], pxMask);

    // RGB 17-24
    pxTemp[0] = _mm256_extracti128_si256(px[0], 1),
    pxTemp[1] = _mm256_extracti128_si256(pxSrc[2], 1);

    // RGB 17-20, shuffle to get correct order
    // RGB 21-24, shuffle to get correct order
    pxTemp[2] = _mm_unpacklo_epi64(pxTemp[0], pxTemp[1]);
    pxTemp[3] = _mm_unpacklo_epi64(_mm_srli_si128(pxTemp[0], 8), pxTemp[1]);
    pxDst[4] = _mm_shuffle_epi8(pxTemp[2], xmm_store4_pkd_pixels);
    pxDst[5] = _mm_shuffle_epi8(pxTemp[3], pxMask);
}

inline void rpp_convert48_pln3_to_pkd3(__m128i *pxSrc, __m128i *pxDst)
{
    const __m128i pxMask = _mm_setr_epi8(0, 1, 12, 2, 3, 13, 4, 5, 14, 6, 7, 15, 0x80, 0x80, 0x80, 0x80);

    __m128i pxTemp[3];
    pxTemp[0] = _mm_unpacklo_epi8(pxSrc[0], pxSrc[1]);

    // RGB 1-4, shuffle to get correct order
    // RGB 5-8, shuffle to get correct order
    pxTemp[1] = _mm_unpacklo_epi64(pxTemp[0], pxSrc[2]);
    pxTemp[2] = _mm_unpacklo_epi64(_mm_srli_si128(pxTemp[0], 8), pxSrc[2]);
    pxDst[0] = _mm_shuffle_epi8(pxTemp[1], xmm_store4_pkd_pixels);
    pxDst[1] = _mm_shuffle_epi8(pxTemp[2], pxMask);

    pxTemp[0] = _mm_unpackhi_epi8(pxSrc[0], pxSrc[1]);

    // RGB 9-12, shuffle to get correct order
    // RGB 13-16, shuffle to get correct order
    pxTemp[1] = _mm_unpackhi_epi64(_mm_slli_si128(pxTemp[0], 8), pxSrc[2]);
    pxTemp[2] = _mm_unpackhi_epi64(pxTemp[0], pxSrc[2]);
    pxDst[2] = _mm_shuffle_epi8(pxTemp[1], xmm_store4_pkd_pixels);
    pxDst[3] = _mm_shuffle_epi8(pxTemp[2], pxMask);
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

inline void rpp_load16_f16_to_f32_avx(Rpp16f *srcPtr, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 8))));
}

inline void rpp_load24_f16_to_f32_avx(Rpp16f *srcPtr, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 8))));
    p[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 16))));
}

inline void rpp_load32_f16_to_f32_avx(Rpp16f *srcPtr, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 8))));
    p[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 16))));
    p[3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 24))));
}

inline void rpp_load40_f16_to_f32_avx(Rpp16f *srcPtr, __m256 *p)
{
    p[0] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr))));
    p[1] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 8))));
    p[2] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 16))));
    p[3] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 24))));
    p[4] = _mm256_cvtph_ps(_mm_castps_si128(_mm_loadu_ps(reinterpret_cast<Rpp32f *>(srcPtr + 32))));
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

#endif //AMD_RPP_RPP_CPU_SIMD_HPP
