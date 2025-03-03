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
#include <algorithm> // for std::clamp

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

inline float GetQualityFactorScale(int quality)
{
    quality = std::max(1, std::min(quality,100));
    float q_scale = 1.0f;
    if (quality < 50)
        q_scale = 50.0f / quality;
    else
        q_scale = 2.0f - (2 * quality / 100.0f);
    return q_scale;
}

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

float a = 1.387039845322148f;             // sqrt(2) * cos(    pi / 16);
float b = 1.306562964876377f;             // sqrt(2) * cos(    pi /  8);
float c = 1.175875602419359f;             // sqrt(2) * cos(3 * pi / 16);
float d = 0.785694958387102f;             // sqrt(2) * cos(5 * pi / 16);
float e = 0.541196100146197f;             // sqrt(2) * cos(3 * pi /  8);
float f = 0.275899379282943f;             // sqrt(2) * cos(7 * pi / 16);
float norm_factor = 0.3535533905932737f;  // 1 / sqrt(8)

void quantize_block(Rpp32f *block, Rpp32f quantTable[8][8], int stride)
{
    float scale = GetQualityFactorScale(50);
    for (int row = 0; row < 8; row++)
    {
        for(int col = 0; col < 8; col++)
        {
            int idx = row * stride + col;
            float Qcoeff = quantTable[row][col] * scale;
            Qcoeff = std::clamp(Qcoeff, 1.0f, 255.0f);
            block[idx] = Qcoeff * roundf(block[idx] / Qcoeff);
        }
    }
}

template <int stride>
void dct_fwd_8x8_1d(float* data)
{
  float x0 = data[0 * stride];
  float x1 = data[1 * stride];
  float x2 = data[2 * stride];
  float x3 = data[3 * stride];
  float x4 = data[4 * stride];
  float x5 = data[5 * stride];
  float x6 = data[6 * stride];
  float x7 = data[7 * stride];

  float tmp0 = x0 + x7;
  float tmp1 = x1 + x6;
  float tmp2 = x2 + x5;
  float tmp3 = x3 + x4;

  float tmp4 = x0 - x7;
  float tmp5 = x6 - x1;
  float tmp6 = x2 - x5;
  float tmp7 = x4 - x3;

  float tmp8 = tmp0 + tmp3;
  float tmp9 = tmp0 - tmp3;
  float tmp10 = tmp1 + tmp2;
  float tmp11 = tmp1 - tmp2;

  x0 = norm_factor * (tmp8 + tmp10);
  x2 = norm_factor * (b * tmp9 + e * tmp11);
  x4 = norm_factor * (tmp8 - tmp10);
  x6 = norm_factor * (e * tmp9 - b * tmp11);

  x1 = norm_factor * (a * tmp4 - c * tmp5 + d * tmp6 - f * tmp7);
  x3 = norm_factor * (c * tmp4 + f * tmp5 - a * tmp6 + d * tmp7);
  x5 = norm_factor * (d * tmp4 + a * tmp5 + f * tmp6 - c * tmp7);
  x7 = norm_factor * (f * tmp4 + d * tmp5 + c * tmp6 + a * tmp7);

  data[0 * stride] = x0;
  data[1 * stride] = x1;
  data[2 * stride] = x2;
  data[3 * stride] = x3;
  data[4 * stride] = x4;
  data[5 * stride] = x5;
  data[6 * stride] = x6;
  data[7 * stride] = x7;
}

template <int stride>
void dct_inv_8x8_1d(float *data) {
  float x0 = data[0 * stride];
  float x1 = data[1 * stride];
  float x2 = data[2 * stride];
  float x3 = data[3 * stride];
  float x4 = data[4 * stride];
  float x5 = data[5 * stride];
  float x6 = data[6 * stride];
  float x7 = data[7 * stride];

  float tmp0 = x0 + x4;
  float tmp1 = b * x2 + e * x6;

  float tmp2 = tmp0 + tmp1;
  float tmp3 = tmp0 - tmp1;
  float tmp4 = f * x7 + a * x1 + c * x3 + d * x5;
  float tmp5 = a * x7 - f * x1 + d * x3 - c * x5;

  float tmp6 = x0 - x4;
  float tmp7 = e * x2 - b * x6;

  float tmp8 = tmp6 + tmp7;
  float tmp9 = tmp6 - tmp7;
  float tmp10 = c * x1 - d * x7 - f * x3 - a * x5;
  float tmp11 = d * x1 + c * x7 - a * x3 + f * x5;

  x0 = norm_factor * (tmp2 + tmp4);
  x7 = norm_factor * (tmp2 - tmp4);
  x4 = norm_factor * (tmp3 + tmp5);
  x3 = norm_factor * (tmp3 - tmp5);

  x1 = norm_factor * (tmp8 + tmp10);
  x5 = norm_factor * (tmp9 - tmp11);
  x2 = norm_factor * (tmp9 + tmp11);
  x6 = norm_factor * (tmp8 - tmp10);

  data[0 * stride] = x0;
  data[1 * stride] = x1;
  data[2 * stride] = x2;
  data[3 * stride] = x3;
  data[4 * stride] = x4;
  data[5 * stride] = x5;
  data[6 * stride] = x6;
  data[7 * stride] = x7;
}

inline void dct_8x8_1d_avx2(__m256 *x) {
    // Extract each lane from x.
    Rpp32f val[8], temp[12];
    val[0] = _mm256_cvtss_f32(x[0]);
    val[1] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(1,1,1,1,1,1,1,1)));
    val[2] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(2,2,2,2,2,2,2,2)));
    val[3] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(3,3,3,3,3,3,3,3)));
    val[4] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(4,4,4,4,4,4,4,4)));
    val[5] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(5,5,5,5,5,5,5,5)));
    val[6] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(6,6,6,6,6,6,6,6)));
    val[7] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(7,7,7,7,7,7,7,7)));

    temp[0]  = val[0] + val[7];
    temp[1]  = val[1] + val[6];
    temp[2]  = val[2] + val[5];
    temp[3]  = val[3] + val[4];

    temp[4]  = val[0] - val[7];
    temp[5]  = val[6] - val[1];
    temp[6]  = val[2] - val[5];
    temp[7]  = val[4] - val[3];

    temp[8]  = temp[0] + temp[3];
    temp[9]  = temp[0] - temp[3];
    temp[10] = temp[1] + temp[2];
    temp[11] = temp[1] - temp[2];

    float y0 = norm_factor * (temp[8] + temp[10]);
    float y2 = norm_factor * (b * temp[9] + e * temp[11]);
    float y4 = norm_factor * (temp[8] - temp[10]);
    float y6 = norm_factor * (e * temp[9] - b * temp[11]);

    float y1 = norm_factor * (a * temp[4] - c * temp[5] + d * temp[6] - f * temp[7]);
    float y3 = norm_factor * (c * temp[4] + f * temp[5] - a * temp[6] + d * temp[7]);
    float y5 = norm_factor * (d * temp[4] + a * temp[5] + f * temp[6] - c * temp[7]);
    float y7 = norm_factor * (f * temp[4] + d * temp[5] + c * temp[6] + a * temp[7]);

    x[0] = _mm256_setr_ps(y0, y1, y2, y3, y4, y5, y6, y7);
}

inline void dct_inv_8x8_1d_avx2(__m256 *x)
{
    float x0 = _mm256_cvtss_f32(x[0]);
    float x1 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(1,1,1,1,1,1,1,1)));
    float x2 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(2,2,2,2,2,2,2,2)));
    float x3 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(3,3,3,3,3,3,3,3)));
    float x4 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(4,4,4,4,4,4,4,4)));
    float x5 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(5,5,5,5,5,5,5,5)));
    float x6 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(6,6,6,6,6,6,6,6)));
    float x7 = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(x[0], _mm256_setr_epi32(7,7,7,7,7,7,7,7)));

    float tmp0 = x0 + x4;
    float tmp1 = b * x2 + e * x6;
    float tmp2 = tmp0 + tmp1;
    float tmp3 = tmp0 - tmp1;
    float tmp4 = f * x7 + a * x1 + c * x3 + d * x5;
    float tmp5 = a * x7 - f * x1 + d * x3 - c * x5;
    float tmp6 = x0 - x4;
    float tmp7 = e * x2 - b * x6;
    float tmp8 = tmp6 + tmp7;
    float tmp9 = tmp6 - tmp7;
    float tmp10 = c * x1 - d * x7 - f * x3 - a * x5;
    float tmp11 = d * x1 + c * x7 - a * x3 + f * x5;

    float y0 = norm_factor * (tmp2 + tmp4);
    float y7 = norm_factor * (tmp2 - tmp4);
    float y4 = norm_factor * (tmp3 + tmp5);
    float y3 = norm_factor * (tmp3 - tmp5);
    float y1 = norm_factor * (tmp8 + tmp10);
    float y5 = norm_factor * (tmp9 - tmp11);
    float y2 = norm_factor * (tmp9 + tmp11);
    float y6 = norm_factor * (tmp8 - tmp10);

    x[0] =  _mm256_setr_ps(y0, y1, y2, y3, y4, y5, y6, y7);
}

void quantizeBlockAVX2(__m256 *p, float quantTable[8][8])
{
    float qscale = GetQualityFactorScale(50);
    __m256 scale = _mm256_set1_ps(qscale);
    for (int i = 0; i < 8; i++)
    {
        __m256 quantRow = _mm256_loadu_ps(quantTable[i]);   // Load 8 floats
        quantRow = _mm256_mul_ps(quantRow, scale);
        quantRow = _mm256_max_ps(avx_p1, _mm256_min_ps(quantRow, avx_p255));
        p[i] = _mm256_div_ps(p[i], quantRow);   // Perform element-wise division for quantization
        p[i] = _mm256_round_ps(p[i] , _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        p[i] = _mm256_mul_ps(p[i], quantRow);
    }
}

template<typename T>
inline void rgb_to_ycbcr_generic(T *srcPtr, int rowLimit, int colLimit, Rpp32f *y, Rpp32f *cb, Rpp32f *cr, RpptDescPtr srcDescPtr)
{
    T *srcPtrR = srcPtr;
    T *srcPtrG = srcPtrR + srcDescPtr->strides.cStride;
    T *srcPtrB = srcPtrG + srcDescPtr->strides.cStride;

    int wStride = srcDescPtr->strides.wStride;
    int hStride = srcDescPtr->strides.hStride;

    float r[256], g[256], b[256];
    // Process Y component and padding
    for (int row = 0; row < 16; row++)
    {
        for (int col = 0; col < 16; col++)
        {
            if (row < colLimit && col < rowLimit)
            {
                int idx = row * hStride + col * wStride;
                if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                {
                    r[row * 16 + col] = static_cast<float>(srcPtrR[idx]) * 255.0f;
                    g[row * 16 + col] = static_cast<float>(srcPtrG[idx]) * 255.0f;
                    b[row * 16 + col] = static_cast<float>(srcPtrB[idx]) * 255.0f;
                }
                else if constexpr (std::is_same<T, Rpp8s>::value)
                {
                    r[row * 16 + col] = static_cast<float>(srcPtrR[idx]) + 128.0f;
                    g[row * 16 + col] = static_cast<float>(srcPtrG[idx]) + 128.0f;
                    b[row * 16 + col] = static_cast<float>(srcPtrB[idx]) + 128.0f;
                }
                else
                {
                    r[row * 16 + col] = static_cast<float>(srcPtrR[idx]);
                    g[row * 16 + col] = static_cast<float>(srcPtrG[idx]);
                    b[row * 16 + col] = static_cast<float>(srcPtrB[idx]);
                }
                y[row * 16 + col] = std::clamp((0.299f * r[row * 16 + col] + 0.587f * g[row * 16 + col] + 0.114f * b[row * 16 + col]), 0.0f, 255.0f) - 128.0f;
            }
            else
            {
                r[row * 16 + col] = r[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
                g[row * 16 + col] = g[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
                b[row * 16 + col] = b[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
                y[row * 16 + col] = y[(std::min(row, colLimit - 1)) * 16 + std::min(col, rowLimit - 1)];
            }
        }
    }

    // Process Cb/Cr using 4:2:0 subsampling
    for (int row = 0; row < 16; row += 2)
    {
        for (int col = 0; col < 16; col += 2)
        {
            int id1 = row * 16 + col;
            int id2 = row * 16 + col + 1;
            int id3 = (row + 1) * 16 + col;
            int id4 = (row + 1) * 16 + col + 1;

            float avgR = (r[id1] + r[id2] + r[id3] + r[id4]) / 4.0f;
            float avgG = (g[id1] + g[id2] + g[id3] + g[id4]) / 4.0f;
            float avgB = (b[id1] + b[id2] + b[id3] + b[id4]) / 4.0f;

            // Convert to Cb/Cr
            int chromaIdx = (row / 2) * 8 + (col / 2);
            cb[chromaIdx] = std::clamp((-0.168736f * avgR - 0.331264f * avgG + 0.5f * avgB + 128.0f), 0.0f, 255.0f) - 128.0f;
            cr[chromaIdx] = std::clamp((0.5f * avgR - 0.418688f * avgG - 0.081312f * avgB + 128.0f), 0.0f, 255.0f) - 128.0f;
        }
    }
}

template <typename T>
inline void ycbcr_to_rgb_generic(T *dstPtr, int rowLimit, int colLimit, Rpp32f *y, Rpp32f *cb, Rpp32f *cr, RpptDescPtr dstDescPtr)
{
    T *dstPtrR = dstPtr;
    T *dstPtrG = dstPtrR + dstDescPtr->strides.cStride;
    T *dstPtrB = dstPtrG + dstDescPtr->strides.cStride;

    int hStride = dstDescPtr->strides.hStride;
    int wStride = dstDescPtr->strides.wStride;

    // Process 8x8 chroma blocks (mapping 4:2:0 chroma to 16x16 Y pixels)
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            // Get chroma values for this 2x2 block
            int cbcrIdx = row * 8 + col;
            float currCb = cb[cbcrIdx];
            float currCr = cr[cbcrIdx];

            // Process 2x2 Y pixels for each Cb/Cr pair
            for (int sub_row = 0; sub_row < 2; sub_row++)
            {
                for (int sub_col = 0; sub_col < 2; sub_col++)
                {
                    int y_idx = (row * 2 + sub_row) * 16 + (col * 2 + sub_col);
                    int dstIdx = (row * 2 + sub_row) * hStride + (col * 2 + sub_col) * wStride;

                    // Prevent out-of-bounds access
                    if ((row * 2 + sub_row) >= colLimit || (col * 2 + sub_col) >= rowLimit)
                        continue;

                    // Convert YCbCr to RGB
                    float yVal = y[y_idx] + 128.0f;
                    yVal = std::clamp(yVal, 0.0f, 255.0f);
                    float r = yVal + 1.402f * currCr;
                    float g = yVal - 0.344136f * currCb - 0.714136f * currCr;
                    float b = yVal + 1.772f * currCb;

                    // Corrected mapping to dstPtr using strides
                    if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                    {
                        dstPtrR[dstIdx] = static_cast<T>(std::clamp(r / 255.0f, 0.0f, 1.0f));
                        dstPtrG[dstIdx] = static_cast<T>(std::clamp(g / 255.0f, 0.0f, 1.0f));
                        dstPtrB[dstIdx] = static_cast<T>(std::clamp(b / 255.0f, 0.0f, 1.0f));
                    }
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                    {
                        dstPtrR[dstIdx] = static_cast<T>(std::clamp(r - 128.0f, -128.0f, 127.0f));
                        dstPtrG[dstIdx] = static_cast<T>(std::clamp(g - 128.0f, -128.0f, 127.0f));
                        dstPtrB[dstIdx] = static_cast<T>(std::clamp(b - 128.0f, -128.0f, 127.0f));
                    }
                    else
                    {
                        dstPtrR[dstIdx] = static_cast<T>(std::clamp(r, 0.0f, 255.0f));
                        dstPtrG[dstIdx] = static_cast<T>(std::clamp(g, 0.0f, 255.0f));
                        dstPtrB[dstIdx] = static_cast<T>(std::clamp(b, 0.0f, 255.0f));
                    }
                }
            }
        }
    }
}

template <typename T>
inline void jpeg_compression_distortion_generic(T *srcPtr, T *dstPtr, Rpp32f *scratchMem, Rpp32s rowLimit, Rpp32s colLimit, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    Rpp32f *y, *cb, *cr;
    y = scratchMem;
    cb = y + 256;
    cr = cb + 64;

    // Convert RGB to YCbCr
    rgb_to_ycbcr_generic(srcPtr, rowLimit, colLimit, y, cb, cr, srcDescPtr);

    for (int blockRow = 0; blockRow < 16; blockRow += 8)
    {
        for (int blockCol = 0; blockCol < 16; blockCol += 8)
        {
            Rpp32f *block = y + blockRow * 16 + blockCol;
            for(int row = 0; row < 8; row++)
                dct_fwd_8x8_1d<1>(block + row * 16);  // Row-wise DCT (stride = 1)
            for(int row = 0; row < 8; row++)
                dct_fwd_8x8_1d<16>(block + row); // Column-wise DCT (stride = 16)
            quantize_block(block, baseLumaTable, 16);
            for(int row = 0; row < 8; row++)
                dct_inv_8x8_1d<16>(block + row);
            for(int row = 0; row < 8; row++)
                dct_inv_8x8_1d<1>(block + row * 16);
        }
    }

    // Apply DCT for Cb and Cr channels (8x8 each)
    for(int row = 0; row < 8; row++)
        dct_fwd_8x8_1d<1>(cb + row * 8);  // Row-wise DCT (stride = 1)
    for(int row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(cb + row);  // Column-wise DCT (stride = 8)
    quantize_block(cb, baseChromaTable, 8);
    for(int row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(cb + row);
    for(int row = 0; row < 8; row++)
        dct_inv_8x8_1d<1>(cb + row * 8);

    for(int row = 0; row < 8; row++)
        dct_fwd_8x8_1d<1>(cr + row * 8);  // Row-wise DCT (stride = 1)
    for(int row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(cr + row);  // Column-wise DCT (stride = 8)
    quantize_block(cr, baseChromaTable, 8);
    for(int row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(cr + row);
    for(int row = 0; row < 8; row++)
        dct_inv_8x8_1d<1>(cr + row * 8);

    // Convert YCbCr back to RGB
    ycbcr_to_rgb_generic(dstPtr, rowLimit, colLimit, y, cb, cr, dstDescPtr);
}

template <typename T>
inline void jpeg_compression_distortion_pln_generic(T *srcPtr, T *dstPtr, Rpp32f *scratchMem, Rpp32s rowLimit, Rpp32s colLimit, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    float blockData[64];
    // Load 8x8 block with boundary handling and type conversion
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            if (row < colLimit && col < rowLimit)
            {
                int idx = row * srcDescPtr->strides.hStride + col;
                if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                    blockData[row * 8 + col] = (static_cast<float>(srcPtr[idx]) * 255.0f) - 128.0f;
                else if constexpr (std::is_same<T, Rpp8s>::value)
                    blockData[row * 8 + col] = static_cast<float>(srcPtr[idx]);
                else
                    blockData[row * 8 + col] = static_cast<float>(srcPtr[idx]) - 128.0f;
            }
            else
            {
                blockData[row * 8 + col] = blockData[(std::min(row, colLimit - 1)) * 8 + std::min(col, rowLimit - 1)];
            }
        }
    }

    // Apply DCT for Cb and Cr channels (8x8 each)
    for(int row = 0; row < 8; row++)
        dct_fwd_8x8_1d<1>(blockData + row * 8);  // Row-wise DCT (stride = 1)
    for(int row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(blockData + row);  // Column-wise DCT (stride = 8)
    quantize_block(blockData, baseLumaTable, 8);
    for(int row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(blockData + row);
    for(int row = 0; row < 8; row++)
        dct_inv_8x8_1d<1>(blockData + row * 8);

    // Store the processed block back to dstPtr with boundary checks and type conversion
    for (int row = 0; row < colLimit; row++)
    {
        for (int col = 0; col < rowLimit; col++)
        {
            int idx = row * dstDescPtr->strides.hStride + col;
            float value = std::clamp(blockData[row * 8 + col] + 128.0f, 0.0f, 255.0f);

            if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                dstPtr[idx] = static_cast<T>(value / 255.0f);
            else if constexpr (std::is_same<T, Rpp8s>::value)
                dstPtr[idx] = static_cast<T>(value - 128.0f);
            else
                dstPtr[idx] = static_cast<T>(value);
        }
    }
}

inline void rgb_to_ycbcr_subsampled(__m256 *pRgb, __m256 *pY, __m256 *pCb, __m256 *pCr)
{
    __m256 pRavg[16], pGavg[16], pBavg[16];
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
        // Compute Y
        pY[i] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffY_R), _mm256_mul_ps(pRgb[idx + 2], coeffY_G)),
            _mm256_mul_ps(pRgb[idx + 4], coeffY_B));
        pY[i + 16] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffY_R), _mm256_mul_ps(pRgb[idx + 3], coeffY_G)),
            _mm256_mul_ps(pRgb[idx + 5], coeffY_B));

        pY[i] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[i], avx_p255));
        pY[i + 16] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[i + 16], avx_p255));
        pY[i] =_mm256_sub_ps(pY[i], avx_p128);
        pY[i + 16] =_mm256_sub_ps(pY[i + 16], avx_p128);

        pRavg[i] = _mm256_hadd_ps(pRgb[idx], pRgb[idx + 1]);
        pGavg[i] = _mm256_hadd_ps(pRgb[idx + 2], pRgb[idx + 3]);
        pBavg[i] = _mm256_hadd_ps(pRgb[idx + 4], pRgb[idx + 5]);

        pRavg[i] = _mm256_permutevar8x32_ps(pRavg[i], _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        pGavg[i] = _mm256_permutevar8x32_ps(pGavg[i], _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
        pBavg[i] = _mm256_permutevar8x32_ps(pBavg[i], _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    }
    for (int i = 0; i < 16; i += 2)
    {
        int chromaIdx = i / 2;
        pRavg[i] = _mm256_mul_ps(_mm256_add_ps(pRavg[i], pRavg[i + 1]), _mm256_set1_ps(0.25f));
        pGavg[i] = _mm256_mul_ps(_mm256_add_ps(pGavg[i], pGavg[i + 1]), _mm256_set1_ps(0.25f));
        pBavg[i] = _mm256_mul_ps(_mm256_add_ps(pBavg[i], pBavg[i + 1]), _mm256_set1_ps(0.25f));
        pCb[chromaIdx] = _mm256_add_ps(
                 _mm256_add_ps(_mm256_mul_ps(pRavg[i], coeffCb_R), _mm256_mul_ps(pGavg[i], coeffCb_G)),
                 _mm256_add_ps(_mm256_mul_ps(pBavg[i], coeffCb_B), offset));
        pCr[chromaIdx] = _mm256_add_ps(
                 _mm256_add_ps(_mm256_mul_ps(pRavg[i], coeffCr_R), _mm256_mul_ps(pGavg[i], coeffCr_G)),
                 _mm256_add_ps(_mm256_mul_ps(pBavg[i], coeffCr_B), offset));
        pCb[chromaIdx] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCb[chromaIdx], avx_p255));
        pCr[chromaIdx] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[chromaIdx], avx_p255));
        pCb[chromaIdx] = _mm256_sub_ps(pCb[chromaIdx], avx_p128);
        pCr[chromaIdx] = _mm256_sub_ps(pCr[chromaIdx], avx_p128);
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
        __m256 cb = pCb[i];
        __m256 cr = pCr[i];
        for(int j = 0; j < 4; j++)
        {
            int yIdx = 2 * i + (j / 2) + ((j & 1) ? 16 : 0);
            int idx = i * 12 + (j / 2) * 6 + (j % 2);

            pY[yIdx] = _mm256_add_ps(pY[yIdx], avx_p128);
            pY[yIdx] = _mm256_min_ps(_mm256_max_ps(pY[yIdx], avx_p0), avx_p255);
            __m256 r = _mm256_add_ps(_mm256_mul_ps(coeffR_Cr, cr), pY[yIdx]);

            __m256 g = _mm256_add_ps(
                            _mm256_add_ps(pY[yIdx], _mm256_mul_ps(coeffG_Cb, cb)),
                            _mm256_mul_ps(coeffG_Cr, cr)
                        );

            __m256 b = _mm256_add_ps(_mm256_mul_ps(coeffB_Cb, cb), pY[yIdx]);

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

void process_jpeg_compression_distortion(__m256* pRgb, __m256* pY, __m256* pCb, __m256* pCr,
                                         float baseLumaTable[8][8], float baseChromaTable[8][8])
{
    rgb_to_ycbcr_subsampled(pRgb, pY, pCb, pCr);
    for (int y = 0; y < 4; y++) 
    {
        for (int row = 0; row < 8; row++)
            dct_8x8_1d_avx2(&pY[y * 8 + row]);

        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], 
                          pY[y * 8 + 4], pY[y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);

        for (int row = 0; row < 8; row++)
            dct_8x8_1d_avx2(&pY[y * 8 + row]);

        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], 
                          pY[y * 8 + 4], pY[y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);

        quantizeBlockAVX2(&pY[y * 8], baseLumaTable);

        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], 
                          pY[y * 8 + 4], pY[y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);

        for (int row = 0; row < 8; row++)
            dct_inv_8x8_1d_avx2(&pY[y * 8 + row]);

        transpose_8x8_avx(pY[y * 8], pY[y * 8 + 1], pY[y * 8 + 2], pY[y * 8 + 3], 
                          pY[y * 8 + 4], pY[y * 8 + 5], pY[y * 8 + 6], pY[y * 8 + 7]);

        for (int row = 0; row < 8; row++)
            dct_inv_8x8_1d_avx2(&pY[y * 8 + row]);
    }

    for (int row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCb[row]);

    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);

    for (int row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCb[row]);

    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);

    quantizeBlockAVX2(pCb, baseChromaTable);

    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);

    for (int row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCb[row]);

    transpose_8x8_avx(pCb[0], pCb[1], pCb[2], pCb[3], pCb[4], pCb[5], pCb[6], pCb[7]);

    for (int row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCb[row]);

    for (int row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCr[row]);

    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);

    for (int row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCr[row]);

    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);

    quantizeBlockAVX2(pCr, baseChromaTable);

    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);

    for (int row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCr[row]);

    transpose_8x8_avx(pCr[0], pCr[1], pCr[2], pCr[3], pCr[4], pCr[5], pCr[6], pCr[7]);

    for (int row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCr[row]);

    ycbcr_to_rgb_subsampled(pY, pCb, pCr, pRgb);
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
#pragma omp parallel for num_threads(1)
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
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * 16 * 16 * 3;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                alignedLength = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += 16 * srcDescPtr->strides.hStride;
                dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                alignedLength = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8u *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                int colLimit = ( i + 8) < roi.xywhROI.roiHeight ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(int row = 0; row < 8; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    quantizeBlockAVX2(p, baseLumaTable);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(int row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp8u *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    int rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += 8 * srcDescPtr->strides.hStride;
                dstPtrRow += 8 * dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                          RpptDescPtr srcDescPtr,
                                                          Rpp32f *dstPtr,
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

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = 0;
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = (i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_store48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            alignedLength = 0;
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                    srcPtrRow += 16 * srcDescPtr->strides.hStride;
                    dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                alignedLength = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp32f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                int colLimit = ( i + 8) < roi.xywhROI.roiHeight ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(int row = 0; row < 8; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    quantizeBlockAVX2(p, baseLumaTable);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(int row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp32f *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    int rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += 8 * srcDescPtr->strides.hStride;
                dstPtrRow += 8 * dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                          RpptDescPtr srcDescPtr,
                                                          Rpp16f *dstPtr,
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

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = 0;
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = (i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_store48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            alignedLength = 0;
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                    srcPtrRow += 16 * srcDescPtr->strides.hStride;
                    dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                alignedLength = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                int colLimit = ( i + 8) < roi.xywhROI.roiHeight ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(int row = 0; row < 8; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    quantizeBlockAVX2(p, baseLumaTable);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(int row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    int rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += 8 * srcDescPtr->strides.hStride;
                dstPtrRow += 8 * dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp8s *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(1)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + batchCount * 16 * 16 * 3;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = ( i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                alignedLength = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += 16 * srcDescPtr->strides.hStride;
                dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = (i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                alignedLength = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    int rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += 16 * srcDescPtr->strides.hStride;
                dstPtrRowR += 16 * dstDescPtr->strides.hStride;
                dstPtrRowG += 16 * dstDescPtr->strides.hStride;
                dstPtrRowB += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                int colLimit = (i + 16) < roi.xywhROI.roiHeight ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, baseLumaTable, baseChromaTable);
                    for(int row = 0; row < 16; row++)
                    {
                        Rpp8s *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    int rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += 16 * srcDescPtr->strides.hStride;
                srcPtrRowG += 16 * srcDescPtr->strides.hStride;
                srcPtrRowB += 16 * srcDescPtr->strides.hStride;
                dstPtrRow += 16 * dstDescPtr->strides.hStride;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                int colLimit = (i + 8) < roi.xywhROI.roiHeight ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(int row = 0; row < 8; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_i8_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    quantizeBlockAVX2(p, baseLumaTable);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
                    for(int row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(int row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp8s *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTempRow, p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    int rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : roi.xywhROI.roiWidth - vectorLoopCount;
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += 8 * srcDescPtr->strides.hStride;
                dstPtrRow += 8 * dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}