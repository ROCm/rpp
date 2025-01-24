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

int BLOCK_SIZE = 8;

float baseChromaTable[8][8] = {
        {17, 18, 24, 47, 99, 99, 99, 99},
        {18, 21, 26, 66, 99, 99, 99, 99},
        {24, 26, 56, 99, 99, 99, 99, 99},
        {47, 66, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99}
};

float baseLumaTable[8][8] = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
};

constexpr float r1 = 1.3870398998f; // cos(1 / 16.0 * PI) * SQRT2
constexpr float r2 = 1.3065630198f; // cos(2 / 16.0 * PI) * SQRT2
constexpr float r3 = 1.1758755445f; // cos(3 / 16.0 * PI) * SQRT2
constexpr float r5 = 0.7856949568f; // cos(5 / 16.0 * PI) * SQRT2
constexpr float r6 = 0.5411961079f; // cos(6 / 16.0 * PI) * SQRT2
constexpr float r7 = 0.2758993804f; // cos(7 / 16.0 * PI) * SQRT2
constexpr float isqrt2 = 0.3535533905932737f; // 1.0f / SQRT2

void transpose_8x8_avx(__m256& a, __m256& b, __m256& c, __m256& d, __m256& e, __m256& f, __m256& g, __m256& h)
{
    __m256 ac0145 = _mm256_unpacklo_ps(a, c); // a0 c0 a1 c1 a4 c4 a5 c5
    __m256 ac2367 = _mm256_unpackhi_ps(a, c); // a2 c2 a3 c3 a6 c6 a7 c7
    __m256 bd0145 = _mm256_unpacklo_ps(b, d); // b0 d0 b1 d1 b4 d4 b5 d5
    __m256 bd2367 = _mm256_unpackhi_ps(b, d); // b2 d2 b3 d3 b6 d6 b7 d7
    __m256 eg0145 = _mm256_unpacklo_ps(e, g); // e0 g0 e1 g1 e4 g4 e5 g5
    __m256 eg2367 = _mm256_unpackhi_ps(e, g); // e2 g2 e3 g3 e6 g6 e7 g7
    __m256 fh0145 = _mm256_unpacklo_ps(f, h); // f0 h0 f1 h1 f4 h4 f5 h5
    __m256 fh2367 = _mm256_unpackhi_ps(f, h); // f2 h2 f3 h3 f6 h6 f7 h7

    __m256 abcd04 = _mm256_unpacklo_ps(ac0145, bd0145); // a0 b0 c0 d0 a4 b4 c4 d4
    __m256 abcd15 = _mm256_unpackhi_ps(ac0145, bd0145); // a1 b1 c1 d1 a5 b5 c5 d5
    __m256 abcd26 = _mm256_unpacklo_ps(ac2367, bd2367); // a2 b2 c2 d2 a6 b6 c6 d6
    __m256 abcd37 = _mm256_unpackhi_ps(ac2367, bd2367); // a3 b3 c3 d3 a7 b7 c7 d7
    __m256 efgh04 = _mm256_unpacklo_ps(eg0145, fh0145); // e0 f0 g0 h0 e4 f4 g4 h4
    __m256 efgh15 = _mm256_unpackhi_ps(eg0145, fh0145); // e1 f1 g1 h1 e5 f5 g5 h5
    __m256 efgh26 = _mm256_unpacklo_ps(eg2367, fh2367); // e2 f2 g2 h2 e6 f6 g6 h6
    __m256 efgh37 = _mm256_unpackhi_ps(eg2367, fh2367); // e3 f3 g3 h3 e7 f7 g7 h7

    a = _mm256_permute2f128_ps(abcd04, efgh04, (2 << 4) | 0); //a0 b0 c0 d0 e0 f0 g0 h0
    e = _mm256_permute2f128_ps(abcd04, efgh04, (3 << 4) | 1); //a4 b4 c4 d4 e4 f4 g4 h4
    b = _mm256_permute2f128_ps(abcd15, efgh15, (2 << 4) | 0); //a1 b1 c1 d1 e1 f1 g1 h1
    f = _mm256_permute2f128_ps(abcd15, efgh15, (3 << 4) | 1); //a5 b5 c5 d5 e5 f5 g5 h5
    c = _mm256_permute2f128_ps(abcd26, efgh26, (2 << 4) | 0); //a2 b2 c2 d2 e2 f2 g2 h2
    g = _mm256_permute2f128_ps(abcd26, efgh26, (3 << 4) | 1); //a6 b6 c6 d6 e6 f6 g6 h6
    d = _mm256_permute2f128_ps(abcd37, efgh37, (2 << 4) | 0); //a3 b3 c3 d3 e3 f3 g3 h3
    h = _mm256_permute2f128_ps(abcd37, efgh37, (3 << 4) | 1); //a7 b7 c7 d7 e7 f7 g7 h7
}

void dct_fwd_8x8_1d_avx2(
    __m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4, __m256& s5,
    __m256& s6, __m256& s7)
{
    const __m256 xr1 = _mm256_set1_ps(r1);
    const __m256 xr2 = _mm256_set1_ps(r2);
    const __m256 xr3 = _mm256_set1_ps(r3);
    const __m256 xr5 = _mm256_set1_ps(r5);
    const __m256 xr6 = _mm256_set1_ps(r6);
    const __m256 xr7 = _mm256_set1_ps(r7);
    const __m256 xisqrt2 = _mm256_set1_ps(isqrt2);

    __m256 t0 = _mm256_add_ps(s0, s7);
    __m256 t7 = _mm256_sub_ps(s0, s7);
    __m256 t1 = _mm256_add_ps(s1, s6);
    __m256 t6 = _mm256_sub_ps(s1, s6);
    __m256 t2 = _mm256_add_ps(s2, s5);
    __m256 t5 = _mm256_sub_ps(s2, s5);
    __m256 t3 = _mm256_add_ps(s3, s4);
    __m256 t4 = _mm256_sub_ps(s3, s4);

    __m256 c0 = _mm256_add_ps(t0, t3);
    __m256 c3 = _mm256_sub_ps(t0, t3);
    __m256 c1 = _mm256_add_ps(t1, t2);
    __m256 c2 = _mm256_sub_ps(t1, t2);

    s0 = _mm256_add_ps(c0, c1);
    s4 = _mm256_sub_ps(c0, c1);
    s2 = _mm256_fmadd_ps(c2, xr6, _mm256_mul_ps(c3, xr2));
    s6 = _mm256_fmsub_ps(c3, xr6, _mm256_mul_ps(c2, xr2));

    c3 = _mm256_fmadd_ps(t4, xr3, _mm256_mul_ps(t7, xr5));
    c0 = _mm256_fmsub_ps(t7, xr3, _mm256_mul_ps(t4, xr5));
    c2 = _mm256_fmadd_ps(t5, xr1, _mm256_mul_ps(t6, xr7));
    c1 = _mm256_fmsub_ps(t6, xr1, _mm256_mul_ps(t5, xr7));

    s3 = _mm256_sub_ps(c0, c2);
    s5 = _mm256_sub_ps(c3, c1);

    c0 = _mm256_mul_ps(_mm256_add_ps(c0, c2), xisqrt2);
    c3 = _mm256_mul_ps(_mm256_add_ps(c1, c3), xisqrt2);

    s1 = _mm256_add_ps(c0, c3);
    s7 = _mm256_sub_ps(c0, c3);
}

void idct_8x8_1d_avx2(__m256& s0, __m256& s1, __m256& s2, __m256& s3, __m256& s4, __m256& s5, __m256& s6, __m256& s7)
{
    __m256 z0 = _mm256_add_ps(s1, s7);
    __m256 z1 = _mm256_add_ps(s3, s5);
    __m256 z4 = _mm256_mul_ps(_mm256_add_ps(z0, z1), _mm256_set1_ps(r3));
    __m256 z2 = _mm256_fmadd_ps(_mm256_set1_ps(-r3 - r5), _mm256_add_ps(s3, s7), z4);
    __m256 z3 = _mm256_fmadd_ps(_mm256_set1_ps(-r3 + r5), _mm256_add_ps(s1, s5), z4);
    z0 = _mm256_mul_ps(z0, _mm256_set1_ps(-r3 + r7));
    z1 = _mm256_mul_ps(z1, _mm256_set1_ps(-r3 - r1));

    __m256 b3 = _mm256_fmadd_ps(s7, _mm256_set1_ps(-r1 + r3 + r5 - r7), _mm256_add_ps(z0, z2));
    __m256 b2 = _mm256_fmadd_ps(s5, _mm256_set1_ps(r1 + r3 - r5 + r7), _mm256_add_ps(z1, z3));
    __m256 b1 = _mm256_fmadd_ps(s3, _mm256_set1_ps(r1 + r3 + r5 - r7), _mm256_add_ps(z1, z2));
    __m256 b0 = _mm256_fmadd_ps(s1, _mm256_set1_ps(r1 + r3 - r5 - r7), _mm256_add_ps(z0, z3));

    z0 = _mm256_add_ps(s0, s4);
    z1 = _mm256_sub_ps(s0, s4);
    z4 = _mm256_mul_ps(_mm256_add_ps(s2, s6), _mm256_set1_ps(r6));

    z2 = _mm256_sub_ps(z4, _mm256_mul_ps(s6, _mm256_set1_ps(r2 + r6)));
    z3 = _mm256_fmadd_ps(s2, _mm256_set1_ps(r2 - r6), z4);

    __m256 a0 = _mm256_add_ps(z0, z3);
    __m256 a3 = _mm256_sub_ps(z0, z3);
    __m256 a1 = _mm256_add_ps(z1, z2);
    __m256 a2 = _mm256_sub_ps(z1, z2);

    s0 = _mm256_add_ps(a0, b0);
    s7 = _mm256_sub_ps(a0, b0);
    s1 = _mm256_add_ps(a1, b1);
    s6 = _mm256_sub_ps(a1, b1);
    s2 = _mm256_add_ps(a2, b2);
    s5 = _mm256_sub_ps(a2, b2);
    s3 = _mm256_add_ps(a3, b3);
    s4 = _mm256_sub_ps(a3, b3);

    // s0 = _mm256_max_ps(avx_p0, _mm256_min_ps(s0, avx_p255));
    // s1 = _mm256_max_ps(avx_p0, _mm256_min_ps(s1, avx_p255));
    // s2 = _mm256_max_ps(avx_p0, _mm256_min_ps(s2, avx_p255));
    // s3 = _mm256_max_ps(avx_p0, _mm256_min_ps(s3, avx_p255));
    // s4 = _mm256_max_ps(avx_p0, _mm256_min_ps(s4, avx_p255));
    // s5 = _mm256_max_ps(avx_p0, _mm256_min_ps(s5, avx_p255));
    // s6 = _mm256_max_ps(avx_p0, _mm256_min_ps(s7, avx_p255));
    // s7 = _mm256_max_ps(avx_p0, _mm256_min_ps(s0, avx_p255));
}

void quantizeBlockAVX2(__m256 *p, float quantTable[8][8])
{
    for (int i = 0; i < 8; i++)
    {
        __m256 quantRow = _mm256_loadu_ps(quantTable[i]);   // Load 8 floats
        p[i] = _mm256_div_ps(p[i], quantRow);   // Perform element-wise division for quantization
    }
}

inline void rgb_to_ycbcr_subsampled(__m256 *pRgb, __m256 *pY, __m256 *pCb, __m256 *pCr)
{
    // Coefficients for Y channel
     __m256 coeffY_R = _mm256_set1_ps(0.299f);
     __m256 coeffY_G = _mm256_set1_ps(0.587f);
     __m256 coeffY_B = _mm256_set1_ps(0.114f);

    // Coefficients for Cb channel
     __m256 coeffCb_R = _mm256_set1_ps(-0.168736f);
     __m256 coeffCb_G = _mm256_set1_ps(-0.331264f);
     __m256 coeffCb_B = _mm256_set1_ps(0.5f);

    // Coefficients for Cr channel
    __m256 coeffCr_R = _mm256_set1_ps(0.5f);
    __m256 coeffCr_G = _mm256_set1_ps(-0.418688f);
    __m256 coeffCr_B = _mm256_set1_ps(-0.081312f);
    __m256 offset = _mm256_set1_ps(128.0f);

    for(int i = 0; i < 16; i++)
    {
        int idx = i * 6;
        int idx1 = i * 2;
        // Compute Y
        pY[idx1] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffY_R), _mm256_mul_ps(pRgb[idx + 2], coeffY_G)),
            _mm256_mul_ps(pRgb[idx + 4], coeffY_B));
        pY[idx1 + 1] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffY_R), _mm256_mul_ps(pRgb[idx + 3], coeffY_G)),
            _mm256_mul_ps(pRgb[idx + 5], coeffY_B));

        pY[idx1] = _mm256_ceil_ps(pY[idx1]);
        pY[idx1 + 1] = _mm256_ceil_ps(pY[idx1 + 1]);

        pY[idx1] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[idx1], avx_p255));
        pY[idx1 + 1] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[idx1 + 1], avx_p255));
        pY[idx1] =_mm256_sub_ps(pY[idx1], avx_p128);
        pY[idx1 + 1] =_mm256_sub_ps(pY[idx1 + 1], avx_p128);

        // Compute Cb
        pCb[idx1] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffCb_R), _mm256_mul_ps(pRgb[idx + 2], coeffCb_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 4], coeffCb_B), offset));
        if(i ==0)
        {
            printf("\n cb in the function");
            rpp_mm256_print_ps(pCb[idx1]);
        }
        pCb[idx1 + 1] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffCb_R), _mm256_mul_ps(pRgb[idx + 3], coeffCb_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 5], coeffCb_B), offset));

        // Compute Cr
        pCr[idx1] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffCr_R), _mm256_mul_ps(pRgb[idx + 2], coeffCr_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 4], coeffCr_B), offset));
        pCr[idx1 + 1] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffCr_R), _mm256_mul_ps(pRgb[idx + 3], coeffCr_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 5], coeffCr_B), offset));

        // pCr[idx1] = _mm256_ceil_ps(pCr[idx1]);
        // pCr[idx1 + 1] = _mm256_ceil_ps(pCr[idx1 + 1]);

        // pCr[idx1] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[idx1], avx_p255));
        // pCr[idx1 + 1] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[idx1 + 1], avx_p255));

        __m256 cbTemp = _mm256_hadd_ps(pCb[idx1], pCb[idx1 + 1]);
        cbTemp = _mm256_permutevar8x32_ps(cbTemp, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        __m256 crTemp = _mm256_hadd_ps(pCr[idx1], pCr[idx1 + 1]);
        crTemp = _mm256_permutevar8x32_ps(crTemp, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        pCb[i] = _mm256_mul_ps(cbTemp, _mm256_set1_ps(0.5));
        pCr[i] = _mm256_mul_ps(crTemp, _mm256_set1_ps(0.5));
    }
    for(int i = 0; i < 8; i++)
    {
        int idx = i * 2;
        pCb[i] = _mm256_mul_ps((_mm256_add_ps(pCb[idx], pCb[idx + 1])), _mm256_set1_ps(0.5));
        pCr[i] = _mm256_mul_ps((_mm256_add_ps(pCb[idx], pCb[idx + 1])), _mm256_set1_ps(0.5));
        pCb[i] = _mm256_ceil_ps(pCb[i]);
        pCr[i] = _mm256_ceil_ps(pCr[i]);
        pCb[i] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCb[i], avx_p255));
        pCr[i] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[i], avx_p255));
        pCb[i] =_mm256_sub_ps(pCb[i], avx_p128);
        pCr[i] =_mm256_sub_ps(pCr[i], avx_p128);
    }
}

inline void ycbcr_to_rgb_subsampled(__m256* pY, __m256* pCb, __m256* pCr, __m256* pRgb) {
    // Coefficients for YCbCr to RGB conversion
    const __m256 coeffR_Y = _mm256_set1_ps(1.0f);
    const __m256 coeffR_Cr = _mm256_set1_ps(1.402f);

    const __m256 coeffG_Y = _mm256_set1_ps(1.0f);
    const __m256 coeffG_Cb = _mm256_set1_ps(-0.344136f);
    const __m256 coeffG_Cr = _mm256_set1_ps(-0.714136f);

    const __m256 coeffB_Y = _mm256_set1_ps(1.0f);
    const __m256 coeffB_Cb = _mm256_set1_ps(1.772f);

    const __m256 offset = _mm256_set1_ps(128.0f);

    for (int i = 0; i < 8; i++)
    {
        pCb[i] = _mm256_add_ps(pCb[i], avx_p128);
        pCr[i] = _mm256_add_ps(pCr[i], avx_p128);
        pCb[i] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCb[i], avx_p255));
        pCr[i] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[i], avx_p255));
        for(int j = 0; j < 4; j++)
        {
            pY[i * 4 + j] = _mm256_add_ps(pY[i*4 + i], avx_p128);
            pY[i * 4 + j] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[i * 4 + j], avx_p255));
        }
        // Upsample Cb and Cr (average neighboring values)
        __m256 cb_upsampled = _mm256_mul_ps(pCb[i], _mm256_set1_ps(2.0f));
        __m256 cr_upsampled = _mm256_mul_ps(pCr[i], _mm256_set1_ps(2.0f));

        // Add offsets back to Cb and Cr
        __m256 cb = _mm256_sub_ps(cb_upsampled, offset);
        __m256 cr = _mm256_sub_ps(cr_upsampled, offset);

        for (int j = 0; j < 4; j++)
        {
            int idx = i * 12 + (j / 2) * 6 + (j % 2);

            // Calculate R = Y + 1.402 * Cr
            __m256 r = _mm256_fmadd_ps(coeffR_Cr, cr, pY[i * 4 + j]);

            // Calculate G = Y - 0.344136 * Cb - 0.714136 * Cr
            __m256 g = _mm256_fmadd_ps(coeffG_Cr, cr, _mm256_fmadd_ps(coeffG_Cb, cb, pY[i * 4 + j]));

            printf("\n gval ");
            rpp_mm256_print_ps(g);

            // Calculate B = Y + 1.772 * Cb
            __m256 b = _mm256_fmadd_ps(coeffB_Cb, cb, pY[i * 4 + j]);

            r = _mm256_max_ps(avx_p0, _mm256_min_ps(r, avx_p255));
            g = _mm256_max_ps(avx_p0, _mm256_min_ps(g, avx_p255));
            b = _mm256_max_ps(avx_p0, _mm256_min_ps(b, avx_p255));

            // Store interleaved R, G, B
            pRgb[idx] = r;
            pRgb[idx + 2] = g;
            pRgb[idx + 4] = b;
        }
    }
}

RppStatus jpeg_compression_distortion_u8_u8_host_tensor(Rpp8u *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
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

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < 1; i += 16)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < 1; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    rgb_to_ycbcr_subsampled(pRgb, pY, pCb, pCr);
                    if(vectorLoopCount == 0)
                    {
                        rpp_mm256_print_ps(pRgb[0]);
                        rpp_mm256_print_ps(pRgb[1]);
                        rpp_mm256_print_ps(pRgb[2]);
                        rpp_mm256_print_ps(pRgb[3]);
                        rpp_mm256_print_ps(pRgb[4]);
                        rpp_mm256_print_ps(pRgb[5]);
                        rpp_mm256_print_ps(pY[0]);
                        rpp_mm256_print_ps(pY[1]);
                        for(int cb = 0; cb < 8; cb++)
                        {
                            rpp_mm256_print_ps(pCb[cb]);
                        }
                    }
                    for(int y = 0; y < 4; y++)
                    {
                        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        dct_fwd_8x8_1d_avx2(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        dct_fwd_8x8_1d_avx2(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        quantizeBlockAVX2(&pY[y * 8], baseLumaTable);
                        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        idct_8x8_1d_avx2(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                        idct_8x8_1d_avx2(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], pY[y * 8 + 4], pY[ y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);
                    }

                    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    dct_fwd_8x8_1d_avx2(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    if(vectorLoopCount == 0)
                    {
                        printf("\n after col DCT ");
                        rpp_mm256_print_ps(pCb[0]);
                    }
                    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    dct_fwd_8x8_1d_avx2(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    if(vectorLoopCount == 0)
                    {
                        printf("\n after row DCT ");
                        rpp_mm256_print_ps(pCb[0]);
                    }
                    quantizeBlockAVX2(pCb, baseChromaTable);
                    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    idct_8x8_1d_avx2(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);
                    idct_8x8_1d_avx2(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);

                    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    dct_fwd_8x8_1d_avx2(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    dct_fwd_8x8_1d_avx2(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    quantizeBlockAVX2(pCr, baseChromaTable);
                    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    idct_8x8_1d_avx2(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);
                    idct_8x8_1d_avx2(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);

                    ycbcr_to_rgb_subsampled(pY, pCb, pCr, pRgb);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    if(vectorLoopCount == 0)
                    {
                        std::cout<<"\n after inv DCT ";
                        rpp_mm256_print_ps(pCb[0]);
                        rpp_mm256_print_ps(pRgb[0]);
                        rpp_mm256_print_ps(pRgb[2]);
                        rpp_mm256_print_ps(pRgb[4]);
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}