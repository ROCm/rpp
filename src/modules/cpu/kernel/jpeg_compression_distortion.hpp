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

static constexpr float a = 1.387039845322148f;
static constexpr float b = 1.306562964876377f;
static constexpr float c = 1.175875602419359f;
static constexpr float d = 0.785694958387102f;
static constexpr float e = 0.541196100146197f;
static constexpr float f = 0.275899379282943f;
__m256 norm_factor = _mm256_set1_ps(0.3535533905932737f);

inline float GetQualityScale(int quality)
{
  quality = std::max(1, std::min(quality, 100));
  float qScale = 1.0f;
  if (quality < 50)
    qScale = 50.0f / quality;
  else
    qScale = 2.0f - (2 * quality / 100.0f);
  return qScale;
}

// Compute quantization table using AVX
void ComputeQuantizationTableAVX(float* baseTable, float scale, float* outputTable)
{
    float scaledTable[BLOCK_SIZE * BLOCK_SIZE];

    // Scale each row
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
        __m256 row = load_float_avx(baseTable + i * BLOCK_SIZE);
        row = _mm256_mul_ps(row, _mm256_set1_ps(scale));
        store_uint8_avx(outputTable + i * BLOCK_SIZE, row);
    }
}

// Get Luma Quantization Table
void GetLumaQuantizationTable(int quality, float outputTable[BLOCK_SIZE][BLOCK_SIZE]) {
    float baseLumaTable[BLOCK_SIZE][BLOCK_SIZE] = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    float scale = GetQualityFactorScale(quality) / 100.0f;
    ComputeQuantizationTableAVX(&baseLumaTable[0][0], scale, &outputTable[0][0]);
}

// Get Chroma Quantization Table
void GetChromaQuantizationTable(int quality, uint8_t outputTable[BLOCK_SIZE][BLOCK_SIZE]) {
    float baseChromaTable[BLOCK_SIZE][BLOCK_SIZE] = {
        {17, 18, 24, 47, 99, 99, 99, 99},
        {18, 21, 26, 66, 99, 99, 99, 99},
        {24, 26, 56, 99, 99, 99, 99, 99},
        {47, 66, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99},
        {99, 99, 99, 99, 99, 99, 99, 99}
    };

    float scale = GetQualityFactorScale(quality) / 100.0f;
    ComputeQuantizationTableAVX(&baseChromaTable[0][0], scale, &outputTable[0][0]);
}

__m256 dct_row(__m256 x) {
    // Constants for DCT calculation
    __m256 c1 = _mm256_set1_ps(0.3535533906); // 1/sqrt(2)
    __m256 c2 = _mm256_set1_ps(0.9238795325); // cos(pi/8)
    __m256 c3 = _mm256_set1_ps(0.7071067812); // cos(2*pi/8)
    __m256 c4 = _mm256_set1_ps(0.3826834324); // cos(3*pi/8)

    // Perform DCT calculations
    __m256 t0 = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 0x21)); // x[0] + x[4]
    __m256 t1 = _mm256_sub_ps(x, _mm256_permute2f128_ps(x, x, 0x21)); // x[0] - x[4]
    __m256 t2 = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 0x31)); // x[2] + x[6]
    __m256 t3 = _mm256_sub_ps(x, _mm256_permute2f128_ps(x, x, 0x31)); // x[2] - x[6]
    __m256 t4 = _mm256_add_ps(_mm256_permute2f128_ps(t0, t0, 0x21), t2); // (x[0]+x[4]) + (x[2]+x[6])
    __m256 t5 = _mm256_sub_ps(_mm256_permute2f128_ps(t0, t0, 0x21), t2); // (x[0]+x[4]) - (x[2]+x[6])
    __m256 t6 = _mm256_add_ps(t1, _mm256_permute2f128_ps(t3, t3, 0x21)); // (x[0]-x[4]) + (x[2]-x[6])
    __m256 t7 = _mm256_sub_ps(t1, _mm256_permute2f128_ps(t3, t3, 0x21)); // (x[0]-x[4]) - (x[2]-x[6])

    __m256 x0 = _mm256_mul_ps(t4, c1); // (x[0]+x[4]+x[2]+x[6]) * c1
    __m256 x1 = _mm256_mul_ps(t5, c2); // (x[0]+x[4]-x[2]-x[6]) * c2
    __m256 x2 = _mm256_mul_ps(t6, c3); // (x[0]-x[4]+x[2]-x[6]) * c3
    __m256 x3 = _mm256_mul_ps(t7, c4); // (x[0]-x[4]-x[2]+x[6]) * c4

    return _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
}

__m256 dct_col(__m256 x) {
    // Transpose the vector for column-wise DCT
    __m256 t0 = _mm256_unpacklo_ps(x, _mm256_setzero_ps());
    __m256 t1 = _mm256_unpackhi_ps(x, _mm256_setzero_ps());
    __m256 t2 = _mm256_unpacklo_ps(t1, t0); 

    // Perform DCT on the transposed vector
    return dct_row(t2);
}

// Function to perform 8x8 2D IDCT using AVX2
__m256 idct_row(__m256 x) {
    // Constants for IDCT calculation
    __m256 c1 = _mm256_set1_ps(0.3535533906); // 1/sqrt(2)
    __m256 c2 = _mm256_set1_ps(0.9238795325); // cos(pi/8)
    __m256 c3 = _mm256_set1_ps(0.7071067812); // cos(2*pi/8)
    __m256 c4 = _mm256_set1_ps(0.3826834324); // cos(3*pi/8)

    // Perform IDCT calculations
    __m256 t0 = _mm256_mul_ps(x, c1);
    __m256 t1 = _mm256_mul_ps(x, c2);
    __m256 t2 = _mm256_mul_ps(x, c3);
    __m256 t3 = _mm256_mul_ps(x, c4);

    __m256 x0 = _mm256_add_ps(_mm256_add_ps(t0, t1), _mm256_add_ps(t2, t3));
    __m256 x1 = _mm256_sub_ps(_mm256_add_ps(t0, t1), _mm256_add_ps(t2, t3));
    __m256 x2 = _mm256_sub_ps(_mm256_add_ps(t0, t2), _mm256_add_ps(t1, t3));
    __m256 x3 = _mm256_sub_ps(_mm256_add_ps(t0, t3), _mm256_add_ps(t1, t2));

    __m256 t4 = _mm256_add_ps(x0, x1);
    __m256 t5 = _mm256_sub_ps(x0, x1);
    __m256 t6 = _mm256_add_ps(x2, x3);
    __m256 t7 = _mm256_sub_ps(x2, x3);

    __m256 r0 = _mm256_add_ps(t4, t6);
    __m256 r1 = _mm256_sub_ps(t5, _mm256_permute2f128_ps(t7, t7, 0x21));
    __m256 r2 = _mm256_sub_ps(t5, _mm256_permute2f128_ps(t7, t7, 0x31));
    __m256 r3 = _mm256_add_ps(t4, t7);

    return _mm256_blend_ps(r0, r1, 0x55);
}

void dct_fwd_8x8_1d_avx2(__m256 *x)
{
    // Temporary values (sum and difference)
    __m256 tmp0 = _mm256_add_ps(x[0], x[7]);
    __m256 tmp1 = _mm256_add_ps(x[1], x[6]);
    __m256 tmp2 = _mm256_add_ps(x[2], x[5]);
    __m256 tmp3 = _mm256_add_ps(x[3], x[4]);

    __m256 tmp4 = _mm256_sub_ps(x[0], x[7]);
    __m256 tmp5 = _mm256_sub_ps(x[6], x[1]);
    __m256 tmp6 = _mm256_sub_ps(x[2], x[5]);
    __m256 tmp7 = _mm256_sub_ps(x[4], x[3]);

    // Additional sums and differences
    __m256 tmp8 = _mm256_add_ps(tmp0, tmp3);
    __m256 tmp9 = _mm256_sub_ps(tmp0, tmp3);
    __m256 tmp10 = _mm256_add_ps(tmp1, tmp2);
    __m256 tmp11 = _mm256_sub_ps(tmp1, tmp2);

    // Apply DCT formula (with constants a, b, c, d, e, f)
    x[0] = _mm256_mul_ps(norm_factor, _mm256_add_ps(tmp8, tmp10));
    x[2] = _mm256_mul_ps(norm_factor, _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(b), tmp9),
        _mm256_mul_ps(_mm256_set1_ps(e), tmp11)
    ));
    x[4] = _mm256_mul_ps(norm_factor, _mm256_sub_ps(tmp8, tmp10));
    x[6] = _mm256_mul_ps(norm_factor, _mm256_sub_ps(
        _mm256_mul_ps(_mm256_set1_ps(e), tmp9),
        _mm256_mul_ps(_mm256_set1_ps(b), tmp11)
    ));

    x[1] = _mm256_mul_ps(norm_factor, _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(a), tmp4),
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(-c), tmp5),
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(d), tmp6),
                _mm256_mul_ps(_mm256_set1_ps(-f), tmp7)
            )
        )
    ));

    x[3] = _mm256_mul_ps(norm_factor, _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(c), tmp4),
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(f), tmp5),
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(-a), tmp6),
                _mm256_mul_ps(_mm256_set1_ps(d), tmp7)
            )
        )
    ));

    x[5] = _mm256_mul_ps(norm_factor, _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(d), tmp4),
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(a), tmp5),
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(f), tmp6),
                _mm256_mul_ps(_mm256_set1_ps(-c), tmp7)
            )
        )
    ));

    x[7] = _mm256_mul_ps(norm_factor, _mm256_add_ps(
        _mm256_mul_ps(_mm256_set1_ps(f), tmp4),
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_set1_ps(d), tmp5),
            _mm256_add_ps(
                _mm256_mul_ps(_mm256_set1_ps(c), tmp6),
                _mm256_mul_ps(_mm256_set1_ps(a), tmp7)
            )
        )
    ));
}

inline void rgb_to_ycbcr(__m256 *pRgb, __m256 *pYCbCr)
{
    // Coefficients for Y channel
    const __m256 coeffY_R = _mm256_set1_ps(0.299f);
    const __m256 coeffY_G = _mm256_set1_ps(0.587f);
    const __m256 coeffY_B = _mm256_set1_ps(0.114f);

    // Coefficients for Cb channel
    const __m256 coeffCb_R = _mm256_set1_ps(-0.168736f);
    const __m256 coeffCb_G = _mm256_set1_ps(-0.331264f);
    const __m256 coeffCb_B = _mm256_set1_ps(0.5f);

    // Coefficients for Cr channel
    const __m256 coeffCr_R = _mm256_set1_ps(0.5f);
    const __m256 coeffCr_G = _mm256_set1_ps(-0.418688f);
    const __m256 coeffCr_B = _mm256_set1_ps(-0.081312f);

    const __m256 offset = _mm256_set1_ps(128.0f);

    for(int i = 0; i < 16; i++)
    {
        int idx = i * 6;
        // Compute Y
        __m256 pY[i] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffY_R), _mm256_mul_ps(pRgb[idx + 2], coeffY_G)),
            _mm256_mul_ps(pRgb[idx + 4], coeffY_B));
        __m256 pY[i + 16] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffY_R), _mm256_mul_ps(pRgb[idx + 3], coeffY_G)),
            _mm256_mul_ps(pRgb[5], coeffY_B));

        // Compute Cb
        __m256 pCb[i] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffCb_R), _mm256_mul_ps(pRgb[idx + 2], coeffCb_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 4], coeffCb_B), offset));
        __m256 pCb[i + 16] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffCb_R), _mm256_mul_ps(pRgb[idx + 3], coeffCb_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 5], coeffCb_B), offset));

        // Compute Cr
        __m256 pCr[i] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx], coeffCr_R), _mm256_mul_ps(pRgb[idx + 2], coeffCr_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 4], coeffCr_B), offset));
        __m256 pCr[i + 16] = _mm256_add_ps(
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 1], coeffCr_R), _mm256_mul_ps(pRgb[idx + 3], coeffCr_G)),
            _mm256_add_ps(_mm256_mul_ps(pRgb[idx + 5], coeffCr_B), offset));
    }

}

RppStatus jpeg_compression_u8_u8_host_tensor(Rpp8u *srcPtr,
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
        Rpp32u alignedLength = (bufferLength / 48) * 48;
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

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i += 16)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 pRgb[96];
                    __m256 pYCbCr[96];
                    for(int row = 0; row < 16; row++)
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTemp, pRgb[row * 6]);                                 // simd loads
                    rgb_to_ycbcr(pRgb, y, cb, cr);
                    for(int row = 0; row < 16; row++)
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTemp, pRgb[row * 6]);
                    dct_fwd_8x8_1d_avx2(pY[0], p)
                }
#endif
            }
        }
    }
}