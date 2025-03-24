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

#include "host_tensor_executors.hpp"
#include "rpp_cpu_simd_math.hpp"
#include <algorithm> // for std::clamp

Rpp32s BLOCK_SIZE = 8;

const Rpp32f dctCoeff1 = 1.387039845322148f;  // sqrt(2) * cos(pi / 16)
const Rpp32f dctCoeff2 = 1.306562964876377f;  // sqrt(2) * cos(pi / 8)
const Rpp32f dctCoeff3 = 1.175875602419359f;  // sqrt(2) * cos(3 * pi / 16)
const Rpp32f dctCoeff4 = 0.785694958387102f;  // sqrt(2) * cos(5 * pi / 16)
const Rpp32f dctCoeff5 = 0.541196100146197f;  // sqrt(2) * cos(3 * pi / 8)
const Rpp32f dctCoeff6 = 0.275899379282943f;  // sqrt(2) * cos(7 * pi / 16)
const Rpp32f dctNormFactor = 0.3535533905932737f; // 1 / sqrt(8)

alignas(64) const Rpp32f chromaQuantTable[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

alignas(64) const Rpp32f lumaQuantTable[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
};

inline Rpp32f get_quality_factor(Rpp32s quality)
{
    quality = std::max(1, std::min(quality,100));
    Rpp32f qualityFactor = 1.0f;
    if (quality < 50)
        qualityFactor = 50.0f / quality;
    else
        qualityFactor = 2.0f - (2 * quality / 100.0f);
    return qualityFactor;
}

void upsample_avx2(__m256 input, __m256 *p)
{
    p[0] = _mm256_unpacklo_ps(input, input); // [0 0 1 1 2 2 3 3]
    p[1] = _mm256_unpackhi_ps(input, input); // [4 4 5 5 6 6 7 7]
}

void transpose_8x8_avx(__m256* p)
{
    __m256 temp[16];

    temp[0] = _mm256_unpacklo_ps(p[0], p[2]); 
    temp[1] = _mm256_unpackhi_ps(p[0], p[2]); 
    temp[2] = _mm256_unpacklo_ps(p[1], p[3]); 
    temp[3] = _mm256_unpackhi_ps(p[1], p[3]); 
    temp[4] = _mm256_unpacklo_ps(p[4], p[6]); 
    temp[5] = _mm256_unpackhi_ps(p[4], p[6]); 
    temp[6] = _mm256_unpacklo_ps(p[5], p[7]); 
    temp[7] = _mm256_unpackhi_ps(p[5], p[7]); 

    temp[8]  = _mm256_unpacklo_ps(temp[0], temp[2]); 
    temp[9]  = _mm256_unpackhi_ps(temp[0], temp[2]); 
    temp[10] = _mm256_unpacklo_ps(temp[1], temp[3]); 
    temp[11] = _mm256_unpackhi_ps(temp[1], temp[3]); 
    temp[12] = _mm256_unpacklo_ps(temp[4], temp[6]); 
    temp[13] = _mm256_unpackhi_ps(temp[4], temp[6]); 
    temp[14] = _mm256_unpacklo_ps(temp[5], temp[7]); 
    temp[15] = _mm256_unpackhi_ps(temp[5], temp[7]); 

    p[0] = _mm256_permute2f128_ps(temp[8], temp[12], (2 << 4) | 0); 
    p[4] = _mm256_permute2f128_ps(temp[8], temp[12], (3 << 4) | 1); 
    p[1] = _mm256_permute2f128_ps(temp[9], temp[13], (2 << 4) | 0); 
    p[5] = _mm256_permute2f128_ps(temp[9], temp[13], (3 << 4) | 1); 
    p[2] = _mm256_permute2f128_ps(temp[10], temp[14], (2 << 4) | 0); 
    p[6] = _mm256_permute2f128_ps(temp[10], temp[14], (3 << 4) | 1); 
    p[3] = _mm256_permute2f128_ps(temp[11], temp[15], (2 << 4) | 0); 
    p[7] = _mm256_permute2f128_ps(temp[11], temp[15], (3 << 4) | 1); 
}

void quantize_block(Rpp32f *block, const Rpp32f *quantTable, Rpp32s stride, Rpp32s qualityParam)
{
    Rpp32f qualityFactor = get_quality_factor(qualityParam);
    for (Rpp32s row = 0; row < 8; row++)
    {
        for (Rpp32s col = 0; col < 8; col++)
        {
            Rpp32s idx = row * stride + col;
            Rpp32f Qcoeff = quantTable[row * 8 + col] * qualityFactor; // Directly accessing 1D array
            Qcoeff = std::clamp(Qcoeff, 1.0f, 255.0f);
            block[idx] = Qcoeff * roundf(block[idx] / Qcoeff);
        }
    }
}

template <Rpp32s stride>
void dct_fwd_8x8_1d(Rpp32f* data)
{
  Rpp32f x0 = data[0 * stride];
  Rpp32f x1 = data[1 * stride];
  Rpp32f x2 = data[2 * stride];
  Rpp32f x3 = data[3 * stride];
  Rpp32f x4 = data[4 * stride];
  Rpp32f x5 = data[5 * stride];
  Rpp32f x6 = data[6 * stride];
  Rpp32f x7 = data[7 * stride];

  Rpp32f tmp0 = x0 + x7;
  Rpp32f tmp1 = x1 + x6;
  Rpp32f tmp2 = x2 + x5;
  Rpp32f tmp3 = x3 + x4;

  Rpp32f tmp4 = x0 - x7;
  Rpp32f tmp5 = x6 - x1;
  Rpp32f tmp6 = x2 - x5;
  Rpp32f tmp7 = x4 - x3;

  Rpp32f tmp8 = tmp0 + tmp3;
  Rpp32f tmp9 = tmp0 - tmp3;
  Rpp32f tmp10 = tmp1 + tmp2;
  Rpp32f tmp11 = tmp1 - tmp2;

  x0 = dctNormFactor * (tmp8 + tmp10);
  x2 = dctNormFactor * (dctCoeff2 * tmp9 + dctCoeff5 * tmp11);
  x4 = dctNormFactor * (tmp8 - tmp10);
  x6 = dctNormFactor * (dctCoeff5 * tmp9 - dctCoeff2 * tmp11);

  x1 = dctNormFactor * (dctCoeff1 * tmp4 - dctCoeff3 * tmp5 + dctCoeff4 * tmp6 - dctCoeff6 * tmp7);
  x3 = dctNormFactor * (dctCoeff3 * tmp4 + dctCoeff6 * tmp5 - dctCoeff1 * tmp6 + dctCoeff4 * tmp7);
  x5 = dctNormFactor * (dctCoeff4 * tmp4 + dctCoeff1 * tmp5 + dctCoeff6 * tmp6 - dctCoeff3 * tmp7);
  x7 = dctNormFactor * (dctCoeff6 * tmp4 + dctCoeff4 * tmp5 + dctCoeff3 * tmp6 + dctCoeff1 * tmp7);

  data[0 * stride] = x0;
  data[1 * stride] = x1;
  data[2 * stride] = x2;
  data[3 * stride] = x3;
  data[4 * stride] = x4;
  data[5 * stride] = x5;
  data[6 * stride] = x6;
  data[7 * stride] = x7;
}

template <Rpp32s stride>
void dct_inv_8x8_1d(Rpp32f *data) {
  Rpp32f x0 = data[0 * stride];
  Rpp32f x1 = data[1 * stride];
  Rpp32f x2 = data[2 * stride];
  Rpp32f x3 = data[3 * stride];
  Rpp32f x4 = data[4 * stride];
  Rpp32f x5 = data[5 * stride];
  Rpp32f x6 = data[6 * stride];
  Rpp32f x7 = data[7 * stride];

  Rpp32f tmp0 = x0 + x4;
  Rpp32f tmp1 = dctCoeff2 * x2 + dctCoeff5 * x6;

  Rpp32f tmp2 = tmp0 + tmp1;
  Rpp32f tmp3 = tmp0 - tmp1;
  Rpp32f tmp4 = dctCoeff6 * x7 + dctCoeff1 * x1 + dctCoeff3 * x3 + dctCoeff4 * x5;
  Rpp32f tmp5 = dctCoeff1 * x7 - dctCoeff6 * x1 + dctCoeff4 * x3 - dctCoeff3 * x5;

  Rpp32f tmp6 = x0 - x4;
  Rpp32f tmp7 = dctCoeff5 * x2 - dctCoeff2 * x6;

  Rpp32f tmp8 = tmp6 + tmp7;
  Rpp32f tmp9 = tmp6 - tmp7;
  Rpp32f tmp10 = dctCoeff3 * x1 - dctCoeff4 * x7 - dctCoeff6 * x3 - dctCoeff1 * x5;
  Rpp32f tmp11 = dctCoeff4 * x1 + dctCoeff3 * x7 - dctCoeff1 * x3 + dctCoeff6 * x5;

  x0 = dctNormFactor * (tmp2 + tmp4);
  x7 = dctNormFactor * (tmp2 - tmp4);
  x4 = dctNormFactor * (tmp3 + tmp5);
  x3 = dctNormFactor * (tmp3 - tmp5);

  x1 = dctNormFactor * (tmp8 + tmp10);
  x5 = dctNormFactor * (tmp9 - tmp11);
  x2 = dctNormFactor * (tmp9 + tmp11);
  x6 = dctNormFactor * (tmp8 - tmp10);

  data[0 * stride] = x0;
  data[1 * stride] = x1;
  data[2 * stride] = x2;
  data[3 * stride] = x3;
  data[4 * stride] = x4;
  data[5 * stride] = x5;
  data[6 * stride] = x6;
  data[7 * stride] = x7;
}

inline void dct_8x8_1d_avx2(__m256 *pVecDct) {
    // Extract each lane from x.
    Rpp32f x[8], temp[12];
    x[0] = _mm256_cvtss_f32(pVecDct[0]);
    x[1] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1)));
    x[2] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2)));
    x[3] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(3, 3, 3, 3, 3, 3, 3, 3)));
    x[4] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(4, 4, 4, 4, 4, 4, 4, 4)));
    x[5] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(5, 5, 5, 5, 5, 5, 5, 5)));
    x[6] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(6, 6, 6, 6, 6, 6, 6, 6)));
    x[7] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(7, 7, 7, 7, 7, 7 ,7 ,7)));

    temp[0]  = x[0] + x[7];
    temp[1]  = x[1] + x[6];
    temp[2]  = x[2] + x[5];
    temp[3]  = x[3] + x[4];

    temp[4]  = x[0] - x[7];
    temp[5]  = x[6] - x[1];
    temp[6]  = x[2] - x[5];
    temp[7]  = x[4] - x[3];

    temp[8]  = temp[0] + temp[3];
    temp[9]  = temp[0] - temp[3];
    temp[10] = temp[1] + temp[2];
    temp[11] = temp[1] - temp[2];

    x[0] = dctNormFactor * (temp[8] + temp[10]);
    x[2] = dctNormFactor * (dctCoeff2 * temp[9] + dctCoeff5 * temp[11]);
    x[4] = dctNormFactor * (temp[8] - temp[10]);
    x[6] = dctNormFactor * (dctCoeff5 * temp[9] - dctCoeff2 * temp[11]);

    x[1] = dctNormFactor * (dctCoeff1 * temp[4] - dctCoeff3 * temp[5] + dctCoeff4 * temp[6] - dctCoeff6 * temp[7]);
    x[3] = dctNormFactor * (dctCoeff3 * temp[4] + dctCoeff6 * temp[5] - dctCoeff1 * temp[6] + dctCoeff4 * temp[7]);
    x[5] = dctNormFactor * (dctCoeff4 * temp[4] + dctCoeff1 * temp[5] + dctCoeff6 * temp[6] - dctCoeff3 * temp[7]);
    x[7] = dctNormFactor * (dctCoeff6 * temp[4] + dctCoeff4 * temp[5] + dctCoeff3 * temp[6] + dctCoeff1 * temp[7]);

    pVecDct[0] = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
}

inline void dct_inv_8x8_1d_avx2(__m256 *pVecDct)
{
    Rpp32f x[8], temp[12];
    x[0] = _mm256_cvtss_f32(pVecDct[0]);
    x[1] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(1, 1, 1, 1, 1, 1, 1, 1)));
    x[2] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(2, 2, 2, 2, 2, 2, 2, 2)));
    x[3] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(3, 3, 3, 3, 3, 3, 3, 3)));
    x[4] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(4, 4, 4, 4, 4, 4, 4, 4)));
    x[5] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(5, 5, 5, 5, 5, 5, 5, 5)));
    x[6] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(6, 6, 6, 6, 6, 6, 6, 6)));
    x[7] = _mm256_cvtss_f32(_mm256_permutevar8x32_ps(pVecDct[0], _mm256_setr_epi32(7, 7, 7, 7, 7, 7, 7, 7)));

    temp[0] = x[0] + x[4];
    temp[1] = dctCoeff2 * x[2] + dctCoeff5 * x[6];
    temp[2] = temp[0] + temp[1];
    temp[3] = temp[0] - temp[1];
    temp[4] = dctCoeff6 * x[7] + dctCoeff1 * x[1] + dctCoeff3 * x[3] + dctCoeff4 * x[5];
    temp[5] = dctCoeff1 * x[7] - dctCoeff6 * x[1] + dctCoeff4 * x[3] - dctCoeff3 * x[5];
    temp[6] = x[0] - x[4];
    temp[7] = dctCoeff5 * x[2] - dctCoeff2 * x[6];
    temp[8] = temp[6] + temp[7];
    temp[9] = temp[6] - temp[7];
    temp[10] = dctCoeff3 * x[1] - dctCoeff4 * x[7] - dctCoeff6 * x[3] - dctCoeff1 * x[5];
    temp[11] = dctCoeff4 * x[1] + dctCoeff3 * x[7] - dctCoeff1 * x[3] + dctCoeff6 * x[5];

    x[0] = dctNormFactor * (temp[2] + temp[4]);
    x[7] = dctNormFactor * (temp[2] - temp[4]);
    x[4] = dctNormFactor * (temp[3] + temp[5]);
    x[3] = dctNormFactor * (temp[3] - temp[5]);
    x[1] = dctNormFactor * (temp[8] + temp[10]);
    x[5] = dctNormFactor * (temp[9] - temp[11]);
    x[2] = dctNormFactor * (temp[9] + temp[11]);
    x[6] = dctNormFactor * (temp[8] - temp[11]);

    pVecDct[0] =  _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
}

void quantize_block_avx2(__m256 *p, const Rpp32f *quantTable, Rpp32s qualityParam)
{
    Rpp32f qualityFactor = get_quality_factor(qualityParam);
    __m256 pQualityFactor = _mm256_set1_ps(qualityFactor);
    for (Rpp32s i = 0; i < 8; i++)
    {
        __m256 quantRow = _mm256_loadu_ps(&quantTable[i * 8]); // Load 8 values from the 1D array
        quantRow = _mm256_mul_ps(quantRow, pQualityFactor);
        quantRow = _mm256_max_ps(avx_p1, _mm256_min_ps(quantRow, avx_p255));

        p[i] = _mm256_div_ps(p[i], quantRow);   // Element-wise division
        p[i] = roundf_avx2(p[i]);               // Round the values
        p[i] = _mm256_mul_ps(p[i], quantRow);   // Multiply back with quantized coefficients
    }
}

template<typename T>
inline void rgb_to_ycbcr_generic(T *srcPtr, Rpp32s rowLimit, Rpp32s colLimit, Rpp32f *y, Rpp32f *cb, Rpp32f *cr, RpptDescPtr srcDescPtr)
{
    T *srcPtrR = srcPtr;
    T *srcPtrG = srcPtrR + srcDescPtr->strides.cStride;
    T *srcPtrB = srcPtrG + srcDescPtr->strides.cStride;

    Rpp32s wStride = srcDescPtr->strides.wStride;
    Rpp32s hStride = srcDescPtr->strides.hStride;

    Rpp32f r[256], g[256], b[256];
    // Process Y component and padding
    for (Rpp32s row = 0; row < 16; row++)
    {
        for (Rpp32s col = 0; col < 16; col++)
        {
            if (row < colLimit && col < rowLimit)
            {
                Rpp32s idx = row * hStride + col * wStride;
                if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                {
                    r[row * 16 + col] = static_cast<Rpp32f>(srcPtrR[idx]) * 255.0f;
                    g[row * 16 + col] = static_cast<Rpp32f>(srcPtrG[idx]) * 255.0f;
                    b[row * 16 + col] = static_cast<Rpp32f>(srcPtrB[idx]) * 255.0f;
                }
                else if constexpr (std::is_same<T, Rpp8s>::value)
                {
                    r[row * 16 + col] = static_cast<Rpp32f>(srcPtrR[idx]) + 128.0f;
                    g[row * 16 + col] = static_cast<Rpp32f>(srcPtrG[idx]) + 128.0f;
                    b[row * 16 + col] = static_cast<Rpp32f>(srcPtrB[idx]) + 128.0f;
                }
                else
                {
                    r[row * 16 + col] = static_cast<Rpp32f>(srcPtrR[idx]);
                    g[row * 16 + col] = static_cast<Rpp32f>(srcPtrG[idx]);
                    b[row * 16 + col] = static_cast<Rpp32f>(srcPtrB[idx]);
                }
                y[row * 16 + col] = std::clamp(((0.299f * r[row * 16 + col]) + (0.587f * g[row * 16 + col]) + (0.114f * b[row * 16 + col])), 0.0f, 255.0f) - 128.0f;
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
    for (Rpp32s row = 0; row < 16; row += 2)
    {
        for (Rpp32s col = 0; col < 16; col += 2)
        {
            Rpp32s id1 = row * 16 + col;
            Rpp32s id2 = row * 16 + col + 1;
            Rpp32s id3 = (row + 1) * 16 + col;
            Rpp32s id4 = (row + 1) * 16 + col + 1;

            Rpp32f avgR = (r[id1] + r[id2] + r[id3] + r[id4]) / 4.0f;
            Rpp32f avgG = (g[id1] + g[id2] + g[id3] + g[id4]) / 4.0f;
            Rpp32f avgB = (b[id1] + b[id2] + b[id3] + b[id4]) / 4.0f;

            // Convert to Cb/Cr
            Rpp32s chromaIdx = (row / 2) * 8 + (col / 2);
            cb[chromaIdx] = std::clamp(((-0.168736f * avgR) - (0.331264f * avgG) + (0.5f * avgB) + 128.0f), 0.0f, 255.0f) - 128.0f;
            cr[chromaIdx] = std::clamp(((0.5f * avgR) - (0.418688f * avgG) - (0.081312f * avgB) + 128.0f), 0.0f, 255.0f) - 128.0f;
        }
    }
}

template <typename T>
inline void ycbcr_to_rgb_generic(T *dstPtr, Rpp32s rowLimit, Rpp32s colLimit, Rpp32f *y, Rpp32f *cb, Rpp32f *cr, RpptDescPtr dstDescPtr)
{
    T *dstPtrR = dstPtr;
    T *dstPtrG = dstPtrR + dstDescPtr->strides.cStride;
    T *dstPtrB = dstPtrG + dstDescPtr->strides.cStride;

    Rpp32s hStride = dstDescPtr->strides.hStride;
    Rpp32s wStride = dstDescPtr->strides.wStride;

    // Process 8x8 chroma blocks (mapping 4:2:0 chroma to 16x16 Y pixels)
    for (Rpp32s row = 0; row < 8; row++)
    {
        for (Rpp32s col = 0; col < 8; col++)
        {
            // Get chroma values for this 2x2 block
            Rpp32s cbcrIdx = row * 8 + col;
            Rpp32f currCb = cb[cbcrIdx];
            Rpp32f currCr = cr[cbcrIdx];

            // Process 2x2 Y pixels for each Cb/Cr pair
            for (Rpp32s sub_row = 0; sub_row < 2; sub_row++)
            {
                for (Rpp32s sub_col = 0; sub_col < 2; sub_col++)
                {
                    Rpp32s y_idx = (row * 2 + sub_row) * 16 + (col * 2 + sub_col);
                    Rpp32s dstIdx = (row * 2 + sub_row) * hStride + (col * 2 + sub_col) * wStride;

                    // Prevent out-of-bounds access
                    if ((row * 2 + sub_row) >= colLimit || (col * 2 + sub_col) >= rowLimit)
                        continue;

                    // Convert YCbCr to RGB
                    Rpp32f yVal = y[y_idx] + 128.0f;
                    yVal = std::clamp(yVal, 0.0f, 255.0f);
                    Rpp32f r = yVal + 1.402f * currCr;
                    Rpp32f g = yVal - 0.344136f * currCb - 0.714136f * currCr;
                    Rpp32f b = yVal + 1.772f * currCb;

                    // Corrected mapping to dstPtr using strides
                    if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                    {
                        dstPtrR[dstIdx] = static_cast<T>(std::clamp((r / 255.0f), 0.0f, 1.0f));
                        dstPtrG[dstIdx] = static_cast<T>(std::clamp((g / 255.0f), 0.0f, 1.0f));
                        dstPtrB[dstIdx] = static_cast<T>(std::clamp((b / 255.0f), 0.0f, 1.0f));
                    }
                    else if constexpr (std::is_same<T, Rpp8s>::value)
                    {
                        dstPtrR[dstIdx] = static_cast<T>(std::clamp((r - 128.0f), -128.0f, 127.0f));
                        dstPtrG[dstIdx] = static_cast<T>(std::clamp((g - 128.0f), -128.0f, 127.0f));
                        dstPtrB[dstIdx] = static_cast<T>(std::clamp((b - 128.0f), -128.0f, 127.0f));
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
inline void jpeg_compression_distortion_generic(T *srcPtr, T *dstPtr, Rpp32f *scratchMem, Rpp32s rowLimit, Rpp32s colLimit, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, Rpp32s qualityParam)
{
    Rpp32f *y, *cb, *cr;
    y = scratchMem;
    cb = y + 256;
    cr = cb + 64;

    // Convert RGB to YCbCr
    rgb_to_ycbcr_generic(srcPtr, rowLimit, colLimit, y, cb, cr, srcDescPtr);

    for (Rpp32s blockRow = 0; blockRow < 16; blockRow += 8)
    {
        for (Rpp32s blockCol = 0; blockCol < 16; blockCol += 8)
        {
            Rpp32f *block = y + blockRow * 16 + blockCol;
            for(Rpp32s row = 0; row < 8; row++)
                dct_fwd_8x8_1d<1>(block + row * 16);  // Row-wise DCT (stride = 1)
            for(Rpp32s row = 0; row < 8; row++)
                dct_fwd_8x8_1d<16>(block + row); // Column-wise DCT (stride = 16)
            quantize_block(block, lumaQuantTable, 16, qualityParam);
            for(Rpp32s row = 0; row < 8; row++)
                dct_inv_8x8_1d<16>(block + row);
            for(Rpp32s row = 0; row < 8; row++)
                dct_inv_8x8_1d<1>(block + row * 16);
        }
    }

    // Apply DCT for Cb and Cr channels (8x8 each)
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<1>(cb + row * 8);  // Row-wise DCT (stride = 1)
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(cb + row);  // Column-wise DCT (stride = 8)
    quantize_block(cb, chromaQuantTable, 8, qualityParam);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(cb + row);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<1>(cb + row * 8);

    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<1>(cr + row * 8);  // Row-wise DCT (stride = 1)
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(cr + row);  // Column-wise DCT (stride = 8)
    quantize_block(cr, chromaQuantTable, 8, qualityParam);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(cr + row);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<1>(cr + row * 8);

    // Convert YCbCr back to RGB
    ycbcr_to_rgb_generic(dstPtr, rowLimit, colLimit, y, cb, cr, dstDescPtr);
}

template <typename T>
inline void jpeg_compression_distortion_pln_generic(T *srcPtr, T *dstPtr, Rpp32f *scratchMem, Rpp32s rowLimit, Rpp32s colLimit, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, Rpp32s qualityParam)
{
    Rpp32f blockData[64];
    // Load 8x8 block with boundary handling and type conversion
    for (Rpp32s row = 0; row < 8; row++)
    {
        for (Rpp32s col = 0; col < 8; col++)
        {
            if (row < colLimit && col < rowLimit)
            {
                Rpp32s idx = row * srcDescPtr->strides.hStride + col;
                if constexpr (std::is_same<T, Rpp32f>::value || std::is_same<T, Rpp16f>::value)
                    blockData[row * 8 + col] = (static_cast<Rpp32f>(srcPtr[idx]) * 255.0f) - 128.0f;
                else if constexpr (std::is_same<T, Rpp8s>::value)
                    blockData[row * 8 + col] = static_cast<Rpp32f>(srcPtr[idx]);
                else
                    blockData[row * 8 + col] = static_cast<Rpp32f>(srcPtr[idx]) - 128.0f;
            }
            else
            {
                blockData[row * 8 + col] = blockData[(std::min(row, colLimit - 1)) * 8 + std::min(col, rowLimit - 1)];
            }
        }
    }

    // Apply DCT for Cb and Cr channels (8x8 each)
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<1>(blockData + row * 8);  // Row-wise DCT (stride = 1)
    for(Rpp32s row = 0; row < 8; row++)
        dct_fwd_8x8_1d<8>(blockData + row);  // Column-wise DCT (stride = 8)
    quantize_block(blockData, lumaQuantTable, 8, qualityParam);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<8>(blockData + row);
    for(Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d<1>(blockData + row * 8);

    // Store the processed block back to dstPtr with boundary checks and type conversion
    for (Rpp32s row = 0; row < colLimit; row++)
    {
        for (Rpp32s col = 0; col < rowLimit; col++)
        {
            Rpp32s idx = row * dstDescPtr->strides.hStride + col;
            Rpp32f value = std::clamp((blockData[row * 8 + col] + 128.0f), 0.0f, 255.0f);

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
    __m256 pCoeffYR = _mm256_set1_ps(0.299f);
    __m256 pCoeffYG = _mm256_set1_ps(0.587f);
    __m256 pCoeffYB = _mm256_set1_ps(0.114f);

    // Coefficients for Cb channel
    __m256 pCoeffCbR = _mm256_set1_ps(-0.168736f);
    __m256 pCoeffCbG = _mm256_set1_ps(-0.331264f);
    __m256 pCoeffCbB = _mm256_set1_ps(0.5f);

    // Coefficients for Cr channel
    __m256 pCoeffCrR = _mm256_set1_ps(0.5f);
    __m256 pCoeffCrG = _mm256_set1_ps(-0.418688f);
    __m256 pCoeffCrB = _mm256_set1_ps(-0.081312f);
    __m256 offset = _mm256_set1_ps(128.0f);
    __m256i pxMask = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    __m256 pQuarterFactor = _mm256_set1_ps(0.25f);

    for(int idxY = 0, idxRGB = 0; idxY < 16; idxY++, idxRGB += 6)
    {
        // Compute Y
        pY[idxY] = _mm256_fmadd_ps(pRgb[idxRGB + 4], pCoeffYB, _mm256_fmadd_ps(pRgb[idxRGB + 2], pCoeffYG, _mm256_mul_ps(pRgb[idxRGB], pCoeffYR)));
        pY[idxY + 16] = _mm256_fmadd_ps(pRgb[idxRGB + 5], pCoeffYB, _mm256_fmadd_ps(pRgb[idxRGB + 3], pCoeffYG, _mm256_mul_ps(pRgb[idxRGB + 1], pCoeffYR)));

        pY[idxY] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[idxY], avx_p255));
        pY[idxY + 16] = _mm256_max_ps(avx_p0, _mm256_min_ps(pY[idxY + 16], avx_p255));
        pY[idxY] =_mm256_sub_ps(pY[idxY], avx_p128);
        pY[idxY + 16] =_mm256_sub_ps(pY[idxY + 16], avx_p128);

        pRavg[idxY] = _mm256_hadd_ps(pRgb[idxRGB], pRgb[idxRGB + 1]);
        pGavg[idxY] = _mm256_hadd_ps(pRgb[idxRGB + 2], pRgb[idxRGB + 3]);
        pBavg[idxY] = _mm256_hadd_ps(pRgb[idxRGB + 4], pRgb[idxRGB + 5]);

        pRavg[idxY] = _mm256_permutevar8x32_ps(pRavg[idxY], pxMask);
        pGavg[idxY] = _mm256_permutevar8x32_ps(pGavg[idxY], pxMask);
        pBavg[idxY] = _mm256_permutevar8x32_ps(pBavg[idxY], pxMask);
    }
    for (Rpp32s i = 0; i < 16; i += 2)
    {
        Rpp32s chromaIdx = i / 2;
        pRavg[i] = _mm256_mul_ps(_mm256_add_ps(pRavg[i], pRavg[i + 1]), pQuarterFactor);
        pGavg[i] = _mm256_mul_ps(_mm256_add_ps(pGavg[i], pGavg[i + 1]), pQuarterFactor);
        pBavg[i] = _mm256_mul_ps(_mm256_add_ps(pBavg[i], pBavg[i + 1]), pQuarterFactor);
        pCb[chromaIdx] = _mm256_fmadd_ps(pBavg[i], pCoeffCbB, _mm256_fmadd_ps(pGavg[i], pCoeffCbG, _mm256_fmadd_ps(pRavg[i], pCoeffCbR, offset)));
        pCr[chromaIdx] = _mm256_fmadd_ps(pBavg[i], pCoeffCrB, _mm256_fmadd_ps(pGavg[i], pCoeffCrG, _mm256_fmadd_ps(pRavg[i], pCoeffCrR, offset)));
        pCb[chromaIdx] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCb[chromaIdx], avx_p255));
        pCr[chromaIdx] = _mm256_max_ps(avx_p0, _mm256_min_ps(pCr[chromaIdx], avx_p255));
        pCb[chromaIdx] = _mm256_sub_ps(pCb[chromaIdx], avx_p128);
        pCr[chromaIdx] = _mm256_sub_ps(pCr[chromaIdx], avx_p128);
    }
}

inline void ycbcr_to_rgb_subsampled(__m256* pY, __m256* pCb, __m256* pCr, __m256* pRgb) {
    // Coefficients for YCbCr to RGB conversion
    const __m256 pCoeffRY= _mm256_set1_ps(1.0f);
    const __m256 pCoeffRCr = _mm256_set1_ps(1.402f);

    const __m256 pCoeffGY = _mm256_set1_ps(1.0f);
    const __m256 pCoeffGCb = _mm256_set1_ps(-0.344136f);
    const __m256 pCoeffGCr = _mm256_set1_ps(-0.714136f);

    const __m256 pCoeffBY = _mm256_set1_ps(1.0f);
    const __m256 pCoeffBCb = _mm256_set1_ps(1.772f);

    const __m256 offset = _mm256_set1_ps(128.0f);

    for (Rpp32s i = 0; i < 8; i++)
    {
        __m256 cb[2];
        upsample_avx2(pCb[i], &cb[0]);
        __m256 cr[2];
        upsample_avx2(pCr[i], &cr[0]);
        for(Rpp32s j = 0; j < 4; j++)
        {
            Rpp32s idxY = 2 * i + (j / 2) + ((j & 1) ? 16 : 0);
            Rpp32s idxRGB = i * 12 + (j / 2) * 6 + (j % 2);
            __m256 curCb = cb[j % 2];
            __m256 curCr = cr[j % 2];

            pY[idxY] = _mm256_add_ps(pY[idxY], avx_p128);
            pY[idxY] = _mm256_min_ps(_mm256_max_ps(pY[idxY], avx_p0), avx_p255);
            __m256 pR = _mm256_fmadd_ps(pCoeffRCr, curCr, pY[idxY]);

            __m256 pG = _mm256_fmadd_ps(pCoeffGCr, curCr, 
                            _mm256_fmadd_ps(pCoeffGCb, curCb, pY[idxY]));

            __m256 pB = _mm256_fmadd_ps(pCoeffBCb, curCb, pY[idxY]);

            pR = _mm256_max_ps(avx_p0, _mm256_min_ps(pR, avx_p255));
            pG = _mm256_max_ps(avx_p0, _mm256_min_ps(pG, avx_p255));
            pB = _mm256_max_ps(avx_p0, _mm256_min_ps(pB, avx_p255));

            // Store interleaved R, G, B
            pRgb[idxRGB] = pR;
            pRgb[idxRGB + 2] = pG;
            pRgb[idxRGB + 4] = pB;
        }
    }
}

void process_jpeg_compression_distortion(__m256* pRgb, __m256* pY, __m256* pCb, __m256* pCr,
                                         const Rpp32f *lumaQuantTable, const Rpp32f *chromaQuantTable, Rpp32s qualityParam)
{
    rgb_to_ycbcr_subsampled(pRgb, pY, pCb, pCr);
    for (Rpp32s idxY = 0; idxY < 32; idxY += 8) 
    {
        for (Rpp32s row = 0; row < 8; row++)
            dct_8x8_1d_avx2(&pY[idxY + row]);
        transpose_8x8_avx(&pY[idxY]);
        for (Rpp32s row = 0; row < 8; row++)
            dct_8x8_1d_avx2(&pY[idxY + row]);
        transpose_8x8_avx(&pY[idxY]);

        quantize_block_avx2(&pY[idxY], lumaQuantTable, qualityParam);

        transpose_8x8_avx(&pY[idxY]);
        for (Rpp32s row = 0; row < 8; row++)
            dct_inv_8x8_1d_avx2(&pY[idxY + row]);
        transpose_8x8_avx(&pY[idxY]);
        for (Rpp32s row = 0; row < 8; row++)
            dct_inv_8x8_1d_avx2(&pY[idxY + row]);
    }

    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCb[row]);
    transpose_8x8_avx(&pCb[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCb[row]);
    transpose_8x8_avx(&pCb[0]);

    quantize_block_avx2(pCb, chromaQuantTable, qualityParam);

    transpose_8x8_avx(&pCb[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCb[row]);
    transpose_8x8_avx(&pCb[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCb[row]);

    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCr[row]);
    transpose_8x8_avx(&pCr[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_8x8_1d_avx2(&pCr[row]);
    transpose_8x8_avx(&pCr[0]);

    quantize_block_avx2(pCr, chromaQuantTable, qualityParam);

    transpose_8x8_avx(&pCr[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCr[row]);
    transpose_8x8_avx(&pCr[0]);
    for (Rpp32s row = 0; row < 8; row++)
        dct_inv_8x8_1d_avx2(&pCr[row]);

    ycbcr_to_rgb_subsampled(pY, pCb, pCr, pRgb);
}

RppStatus jpeg_compression_distortion_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp8u *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32s *qualityTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(1)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
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
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels

        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

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
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
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

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
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
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8u *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp8u *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                        {
                            rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                            p[row] = _mm256_sub_ps(p[row], avx_p128);
                        }
                        else
                            p[row] = p[colLimit - 1];
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                          RpptDescPtr srcDescPtr,
                                                          Rpp32f *dstPtr,
                                                          RpptDescPtr dstDescPtr,
                                                          Rpp32s *qualityTensor,
                                                          RpptROIPtr roiTensorPtrSrc,
                                                          RpptRoiType roiType,
                                                          RppLayoutParams layoutParams,
                                                          rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
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
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels
        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_store48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                    srcPtrRow += srcIncrement;
                    dstPtrRow += dstIncrement;
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

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
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
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp32f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f32pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp32f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                          RpptDescPtr srcDescPtr,
                                                          Rpp16f *dstPtr,
                                                          RpptDescPtr dstDescPtr,
                                                          Rpp32s *qualityTensor,
                                                          RpptROIPtr roiTensorPtrSrc,
                                                          RpptRoiType roiType,
                                                          RppLayoutParams layoutParams,
                                                          rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
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
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels
        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f16pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f16pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                    srcPtrRow += srcIncrement;
                    dstPtrRow += dstIncrement;
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

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f16pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRowR, *dstPtrTempRowG, *dstPtrTempRowB;
                        dstPtrTempRowR = dstPtrTempR + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowG = dstPtrTempG + row * dstDescPtr->strides.hStride;
                        dstPtrTempRowB = dstPtrTempB + row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempRowR, dstPtrTempRowG, dstPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    srcPtrTemp += 48;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth * 3; vectorLoopCount += 48)
                {
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
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
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_f16pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow= dstPtrTemp+ row * dstDescPtr->strides.hStride;
                        rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    dstPtrTemp += 48;
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp16f *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        p[row] = _mm256_add_ps(p[row], avx_p128);
                        p[row] = _mm256_max_ps(avx_p0, _mm256_min_ps(p[row], avx_p255));
                        Rpp16f *dstPtrTempRow;
                        dstPtrTempRow = dstPtrTemp + row * dstDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTempRow, &p[row]);                                 // simd loads
                    }
#endif
                    dstPtrTemp += 8;
                    srcPtrTemp += 8;
                }
                for(; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount += 8)
                {
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus jpeg_compression_distortion_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp8s *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32s *qualityTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(1)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
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
        Rpp32s qualityParam = qualityTensor[batchCount];
        Rpp32f *scratchMem = handle.GetInitHandle()->mem.mcpu.scratchBufferHost + (batchCount * (16 * 16 * 3));  // (16 * 16) is the block size, and 3 represents the number of channels
        Rpp32s srcIncrement = 16 * srcDescPtr->strides.hStride;
        Rpp32s dstIncrement = 16 * dstDescPtr->strides.hStride;

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
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
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

            for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[32], pCr[32];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTempRow, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = (((vectorLoopCount / 3) + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth -  vectorLoopCount / 3);
                    jpeg_compression_distortion_generic(srcPtrTemp, dstPtrTempR, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTempR += vectorIncrementPerChannel;
                        srcPtrTemp += 48;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRowR += dstIncrement;
                dstPtrRowG += dstIncrement;
                dstPtrRowB += dstIncrement;
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
                Rpp32s colLimit = ((i + 16) < roi.xywhROI.roiHeight) ? 16 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pY[32], pCb[8], pCr[8];
                    for(Rpp32s row = 0; row < 16; row++)
                    {
                        Rpp8s *srcPtrTempRowR, *srcPtrTempRowG, *srcPtrTempRowB;
                        srcPtrTempRowR = srcPtrTempR + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowG = srcPtrTempG + row * srcDescPtr->strides.hStride;
                        srcPtrTempRowB = srcPtrTempB + row * srcDescPtr->strides.hStride;
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempRowR, srcPtrTempRowG, srcPtrTempRowB, &pRgb[row * 6]);                                 // simd loads
                    }
                    process_jpeg_compression_distortion(pRgb, pY, pCb, pCr, lumaQuantTable, chromaQuantTable, qualityParam);
                    for(Rpp32s row = 0; row < 16; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 16) < roi.xywhROI.roiWidth) ? 16 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_generic(srcPtrTempR, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 16)
                    {
                        dstPtrTemp += 48;
                        srcPtrTempR += vectorIncrementPerChannel;
                    }
                }
                srcPtrRowR += srcIncrement;
                srcPtrRowG += srcIncrement;
                srcPtrRowB += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
        else if((srcDescPtr->c == 1 ) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            srcIncrement = 8 * srcDescPtr->strides.hStride;
            dstIncrement = 8 * dstDescPtr->strides.hStride;
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 8)
            {
                Rpp32s colLimit = ((i + 8) < roi.xywhROI.roiHeight) ? 8 : (roi.xywhROI.roiHeight - i);
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32s vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
#if __AVX2__
                    __m256 p[8];
                    for(Rpp32s row = 0; row < 8; row++)
                    {
                        Rpp8s *srcPtrTempRow;
                        srcPtrTempRow = srcPtrTemp + row * srcDescPtr->strides.hStride;
                        if((row + i) < roi.xywhROI.roiHeight)
                            rpp_simd_load(rpp_load8_i8_to_f32_avx, srcPtrTempRow, &p[row]);                                 // simd loads
                        else
                            p[row] = p[colLimit - 1];
                        p[row] = _mm256_sub_ps(p[row], avx_p128);
                    }

                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    quantize_block_avx2(p, lumaQuantTable, qualityParam);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);
                    transpose_8x8_avx(&p[0]);
                    for(Rpp32s row = 0; row < 8; row++)
                        dct_inv_8x8_1d_avx2(&p[row]);

                    for(Rpp32s row = 0; row < 8; row++)
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
                    Rpp32s rowLimit = ((vectorLoopCount + 8) < roi.xywhROI.roiWidth) ? 8 : (roi.xywhROI.roiWidth - vectorLoopCount);
                    jpeg_compression_distortion_pln_generic(srcPtrTemp, dstPtrTemp, scratchMem, rowLimit, colLimit, srcDescPtr, dstDescPtr, qualityParam);
                    if(rowLimit == 8)
                    {
                        dstPtrTemp += 8;
                        srcPtrTemp += 8;
                    }
                }
                srcPtrRow += srcIncrement;
                dstPtrRow += dstIncrement;
            }
        }
    }
    return RPP_SUCCESS;
}