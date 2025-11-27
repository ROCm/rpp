/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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
constexpr int HISTOGRAM_BINS = 256;

// ---- BT.601 full-range coefficients ----
const float cYR = 0.299000f;
const float cYG = 0.587000f;
const float cYB = 0.114000f;
const float cUR = -0.168736f;
const float cUG = -0.331264f;
const float cUB =  0.500000f;
const float cVR =  0.500000f;
const float cVG = -0.418688f;
const float cVB = -0.081312f;
const float cR_V =  1.402000f;
const float cG_U =  0.344136f;
const float cG_V =  0.714136f;
const float cB_U =  1.772000f;

// YCbCr conversion coefficients
const __m256 pCYR = _mm256_set1_ps(cYR);
const __m256 pCYG = _mm256_set1_ps(cYG);
const __m256 pCYB = _mm256_set1_ps(cYB);
const __m256 pCUR = _mm256_set1_ps(cUR);
const __m256 pCUG = _mm256_set1_ps(cUG);
const __m256 pCUB = _mm256_set1_ps(cUB);
const __m256 pCVR = _mm256_set1_ps(cVR);
const __m256 pCVG = _mm256_set1_ps(cVG);
const __m256 pCVB = _mm256_set1_ps(cVB);

// YCbCr to RGB conversion coefficients
const __m256 pCRV = _mm256_set1_ps(cR_V);
const __m256 pCGU = _mm256_set1_ps(cG_U);
const __m256 pCGV = _mm256_set1_ps(cG_V);
const __m256 pCBU = _mm256_set1_ps(cB_U);

inline void rgb_to_ycbcr_compute(Rpp8u *srcR, Rpp8u *srcG, Rpp8u *srcB,
                                 Rpp8u *dstY, Rpp8u *dstCb, Rpp8u *dstCr)
{
    Rpp32f r = (Rpp32f)(*srcR);
    Rpp32f g = (Rpp32f)(*srcG);
    Rpp32f b = (Rpp32f)(*srcB);

    Rpp32f yF = fmaf(r, cYR, fmaf(g, cYG, b * cYB));                 // Y  = 0.299*R + 0.587*G + 0.114*B
    Rpp32f uF = fmaf(r, cUR, fmaf(g, cUG, fmaf(b, cUB, 128.0f)));    // Cb = -0.168736*R - 0.331264*G + 0.500*B + 128
    Rpp32f vF = fmaf(r, cVR, fmaf(g, cVG, fmaf(b, cVB, 128.0f)));    // Cr = 0.500*R - 0.418688*G - 0.081312*B + 128

    *dstY = (yF  < 0 ? 0 : (yF > 255 ? 255 : std::nearbyintf(yF)));
    *dstCb = (uF < 0 ? 0 : (uF > 255 ? 255 : std::nearbyintf(uF)));
    *dstCr = (vF < 0 ? 0 : (vF > 255 ? 255 : std::nearbyintf(vF)));
}

inline void ycbcr_to_rgb_compute(Rpp8u *srcY, Rpp8u *srcCb, Rpp8u *srcCr,
                                 Rpp8u *dstR, Rpp8u *dstG, Rpp8u *dstB)
{
    Rpp32f Y = (Rpp32f)(*srcY);
    Rpp32f U = (Rpp32f)(*srcCb) - 128.0f;
    Rpp32f V = (Rpp32f)(*srcCr) - 128.0f;

    Rpp32f R = Y + cR_V * V;
    Rpp32f G = Y - cG_U * U - cG_V * V;
    Rpp32f B = Y + cB_U * U;

    *dstR = (Rpp8u)(R < 0 ? 0 : (R > 255 ? 255 : std::nearbyintf(R)));
    *dstG = (Rpp8u)(G < 0 ? 0 : (G > 255 ? 255 : std::nearbyintf(G)));
    *dstB = (Rpp8u)(B < 0 ? 0 : (B > 255 ? 255 : std::nearbyintf(B)));
}

#if __AVX2__

inline void rgb_to_ycbcr_avx(__m256 *p)
{
    // --- Block 0 (first 8 pixels) ---
    __m256 y0 = _mm256_fmadd_ps(p[0], pCYR, _mm256_fmadd_ps(p[2], pCYG, _mm256_mul_ps(p[4], pCYB)));                // Y = R*cYR + G*cYG + B*cYB
    __m256 u0 = _mm256_fmadd_ps(p[0], pCUR, _mm256_fmadd_ps(p[2], pCUG, _mm256_fmadd_ps(p[4], pCUB, avx_p128)));    // U = R*cUR + G*cUG + B*cUB + 128
    __m256 v0 = _mm256_fmadd_ps(p[0], pCVR, _mm256_fmadd_ps(p[2], pCVG, _mm256_fmadd_ps(p[4], pCVB, avx_p128)));    // V = R*cVR + G*cVG + B*cVB + 128

    p[0] = _mm256_min_ps(_mm256_max_ps(y0, avx_p0), avx_p255);
    p[2] = _mm256_min_ps(_mm256_max_ps(u0, avx_p0), avx_p255);
    p[4] = _mm256_min_ps(_mm256_max_ps(v0, avx_p0), avx_p255);

    // --- Block 1 (next 8 pixels) ---
    __m256 y1 = _mm256_fmadd_ps(p[1], pCYR, _mm256_fmadd_ps(p[3], pCYG, _mm256_mul_ps(p[5], pCYB)));                // Y = R*cYR + G*cYG + B*cYB
    __m256 u1 = _mm256_fmadd_ps(p[1], pCUR, _mm256_fmadd_ps(p[3], pCUG, _mm256_fmadd_ps(p[5], pCUB, avx_p128)));    // U = R*cUR + G*cUG + B*cUB + 128
    __m256 v1 = _mm256_fmadd_ps(p[1], pCVR, _mm256_fmadd_ps(p[3], pCVG, _mm256_fmadd_ps(p[5], pCVB, avx_p128)));    // V = R*cVR + G*cVG + B*cVB + 128

    p[1] = _mm256_min_ps(_mm256_max_ps(y1, avx_p0), avx_p255);
    p[3] = _mm256_min_ps(_mm256_max_ps(u1, avx_p0), avx_p255);
    p[5] = _mm256_min_ps(_mm256_max_ps(v1, avx_p0), avx_p255);
}

inline void ycbcr_to_rgb_avx(__m256 *p)
{
    // --- Block 0 (first 8 pixels) ---
    __m256 U0 = _mm256_sub_ps(p[2], avx_p128);  // U = Cb - 128
    __m256 V0 = _mm256_sub_ps(p[4], avx_p128);  // V = Cr - 128

    __m256 r0 = _mm256_fmadd_ps(V0, pCRV, p[0]);                                // R = Y + 1.402 * V
    __m256 g0 = _mm256_fnmadd_ps(V0, pCGV, _mm256_fnmadd_ps(U0, pCGU, p[0]));   // G = Y - 0.714136 * V - 0.344136*U
    __m256 b0 = _mm256_fmadd_ps(U0, pCBU, p[0]);                                // B = Y + 1.772 * U

    p[0] = _mm256_min_ps(_mm256_max_ps(r0, avx_p0), avx_p255);
    p[2] = _mm256_min_ps(_mm256_max_ps(g0, avx_p0), avx_p255);
    p[4] = _mm256_min_ps(_mm256_max_ps(b0, avx_p0), avx_p255);

    // --- Block 1 (next 8 pixels) ---
    __m256 U1 = _mm256_sub_ps(p[3], avx_p128);  // U = Cb - 128
    __m256 V1 = _mm256_sub_ps(p[5], avx_p128);  // V = Cr - 128

    __m256 r1 = _mm256_fmadd_ps(V1, pCRV, p[1]);                                // R = Y + 1.402 * V
    __m256 g1 = _mm256_fnmadd_ps(V1, pCGV, _mm256_fnmadd_ps(U1, pCGU, p[1]));   // G = Y - 0.714136 * V - 0.344136*U
    __m256 b1 = _mm256_fmadd_ps(U1, pCBU, p[1]);                                // B = Y + 1.772 * U

    p[1] = _mm256_min_ps(_mm256_max_ps(r1, avx_p0), avx_p255);
    p[3] = _mm256_min_ps(_mm256_max_ps(g1, avx_p0), avx_p255);
    p[5] = _mm256_min_ps(_mm256_max_ps(b1, avx_p0), avx_p255);
}

#endif

inline void collect_hist_pln_tensor_host(Rpp8u *srcPtr,
                                         Rpp32u *hist,
                                         Rpp32u roiWidth,
                                         Rpp32u roiHeight,
                                         Rpp32u rowStride)
{
    for (int y = 0; y < roiHeight; y++)
    {
        Rpp8u *srcRow = srcPtr + y * rowStride;
        for (int x = 0; x < roiWidth; x++)
            hist[srcRow[x]]++;
    }
}

inline void collect_hist_y_buffer(const Rpp8u *yBuf,
                                  Rpp32u *hist,
                                  Rpp32u pixels)
{
    for (Rpp32u i = 0; i < pixels; i++)
        hist[yBuf[i]]++;
}

inline void build_lut_from_hist_host(const Rpp32u *hist,
                                     Rpp8u *lut,
                                     Rpp32u img_size)
{
    Rpp32u cdf[HISTOGRAM_BINS];
    Rpp32u cdf_accum = 0;

    Rpp32u min_cdf = 0;

    for (int i = 0; i < HISTOGRAM_BINS; i++)
    {
        cdf_accum += hist[i];
        cdf[i] = cdf_accum;

        if (min_cdf == 0 && cdf[i] != 0)
            min_cdf = cdf[i];
    }

    // denominator = pixels - mincdf
    float denominator = std::max((float)(img_size - min_cdf), 1.0f);
    bool is_uniform = (min_cdf == img_size);

    if (is_uniform) {
        for (int i = 0; i < HISTOGRAM_BINS; ++i)
            lut[i] = i;
        return;
    }
    const float mult_scalar = 255.0f / denominator;
    int vectorLoopCount = 0;

#if __AVX2__
    __m256 v_min_cdf = _mm256_set1_ps((float)min_cdf);
    __m256 v_mult    = _mm256_set1_ps(mult_scalar);
    for (; vectorLoopCount <= HISTOGRAM_BINS; vectorLoopCount += 16)
    {
        // Load CDF[0..7] → convert to float
        __m256i ci0 = _mm256_loadu_si256((__m256i const*)(cdf + vectorLoopCount));
        __m256  cf0 = _mm256_cvtepi32_ps(ci0);
        __m256  r0  = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(cf0, v_min_cdf), v_mult), avx_p255);
        __m256i ri0 = _mm256_cvtps_epi32(r0);

        // Load CDF[8..15] → convert to float
        __m256i ci1 = _mm256_loadu_si256((__m256i const*)(cdf + vectorLoopCount + 8));
        __m256  cf1 = _mm256_cvtepi32_ps(ci1);
        __m256  r1  = _mm256_min_ps(_mm256_mul_ps(_mm256_sub_ps(cf1, v_min_cdf), v_mult), avx_p255);
        __m256i ri1 = _mm256_cvtps_epi32(r1);

        // Convert 32-bit → 16-bit
         __m128i pack16_0 = _mm_packs_epi32(_mm256_castsi256_si128(ri0), _mm256_extracti128_si256(ri0, 1));
        __m128i pack16_1 = _mm_packs_epi32(_mm256_castsi256_si128(ri1), _mm256_extracti128_si256(ri1, 1));

        // Convert 16-bit → 8-bit
        __m128i pack8 = _mm_packus_epi16(pack16_0, pack16_1);

        // Store 16 LUT entries
        _mm_storeu_si128((__m128i*)(lut + vectorLoopCount), pack8);
    }
#else
    for (; vectorLoopCount < HISTOGRAM_BINS; vectorLoopCount++)
    {
        Rpp32f eq = ((Rpp32f)cdf[vectorLoopCount] - (Rpp32f)min_cdf) * mult_scalar;
        if (eq > 255.0f) eq = 255.0f;
        if (eq < 0.0f)   eq = 0.0f;
        lut[vectorLoopCount] = (Rpp8u)round(eq);
    }
#endif
}

inline void apply_lut_tensor(const Rpp8u *src,
                             Rpp8u *dst,
                             Rpp32u roiWidth,
                             Rpp32u roiHeight,
                             const Rpp8u *lut,
                             Rpp32u srcRowStride,
                             Rpp32u dstRowStride)
{
    for (Rpp32u y = 0; y < roiHeight; y++)
    {
        const Rpp8u* srcRow = src + y * srcRowStride;
        Rpp8u* dstRow = dst + y * dstRowStride;

        for (Rpp32u x = 0; x < roiWidth; x++)
            dstRow[x] = lut[srcRow[x]]; // Apply LUT only to the Y channel value
    }
}

inline void histogram_equalize_host_compute(const Rpp8u *srcY, Rpp8u *dstY, Rpp32u roiWidth, Rpp32u roiHeight, Rpp32u pixels)
{
    Rpp32u hist[HISTOGRAM_BINS] = {0};
    Rpp8u lut[HISTOGRAM_BINS];

    collect_hist_y_buffer(srcY, hist, pixels); // collect historgram values
    build_lut_from_hist_host(hist, lut, pixels); // build LUT from historgarm 
    apply_lut_tensor(srcY, dstY, roiWidth, roiHeight, lut, roiWidth, roiWidth); // apply LUT
}

RppStatus histogram_equalize_u8_u8_host_tensor(Rpp8u *srcPtr,
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

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u roiWidth = roi.xywhROI.roiWidth;
        Rpp32u roiHeight = roi.xywhROI.roiHeight;

        Rpp32u pixels = roiWidth * roiHeight;

        Rpp8u *scratchBase = reinterpret_cast<Rpp8u *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);

        Rpp8u *yBuf = scratchBase + batchCount * (pixels * 3);
        Rpp8u *cbBuf = yBuf + pixels;
        Rpp8u *crBuf = cbBuf + pixels;
        Rpp8u *dstYBuf = crBuf + pixels;

#if __AVX2__
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        Rpp32u alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
#endif

        // Histogram Equalise without fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB, *yPtr, *cbPtr, *crPtr;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    rgb_to_ycbcr_avx(p);
                    rpp_store48_f32pln3_to_u8pln3_avx(yPtr, cbPtr, crPtr, p);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    rgb_to_ycbcr_compute(srcPtrTempR++, srcPtrTempG++, srcPtrTempB++, yPtr++, cbPtr++, crPtr++);

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(yPtr, cbPtr, crPtr, p);
                    ycbcr_to_rgb_avx(p);
                    rpp_store48_f32pln3_to_u8pln3_avx(dstPtrTempR, dstPtrTempG, dstPtrTempB, p);

                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Histogram Equalise without fused output-layout toggle (NCHW -> NHCW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow, *yPtr, *cbPtr, *crPtr;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    rgb_to_ycbcr_avx(p);
                    rpp_store48_f32pln3_to_u8pln3_avx(yPtr, cbPtr, crPtr, p);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    rgb_to_ycbcr_compute(srcPtrTempR++, srcPtrTempG++, srcPtrTempB++, yPtr++, cbPtr++, crPtr++);

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
            
            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_load48_u8pln3_to_f32pln3_avx(yPtr, cbPtr, crPtr, p);
                    ycbcr_to_rgb_avx(p);
                    rpp_store48_f32pln3_to_u8pkd3_avx(dstPtrTemp, p);
                    dstPtrTemp += vectorIncrement;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                {
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTemp, dstPtrTemp + 1, dstPtrTemp + 2);
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Histogram Equalise without fused output-layout toggle (NHCW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB, *yPtr, *cbPtr, *crPtr;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;
#if __AVX2__
            alignedLength = ((roi.xywhROI.roiWidth / vectorIncrement) - 1) * vectorIncrement;
#endif
            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);     // simd loads
                    rgb_to_ycbcr_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, yPtr, cbPtr, crPtr, p);   // simd stores

                    srcPtrTemp += vectorIncrement;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                {
                    rgb_to_ycbcr_compute(srcPtrTemp, srcPtrTemp + 1, srcPtrTemp + 2, yPtr++, cbPtr++, crPtr++);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, yPtr, cbPtr, crPtr, p);     // simd loads
                    ycbcr_to_rgb_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores

                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++);

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Histogram Equalise without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow, *yPtr, *cbPtr, *crPtr;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            yPtr = yBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;
#if __AVX2__
            alignedLength = ((roi.xywhROI.roiWidth / vectorIncrement) - 1) * vectorIncrement;
#endif
            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6], pY, pU, pV;
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);     // simd loads
                    rgb_to_ycbcr_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, yPtr, cbPtr, crPtr, p);   // simd stores

                    srcPtrTemp += vectorIncrement;
                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                {
                    rgb_to_ycbcr_compute(srcPtrTemp, srcPtrTemp + 1, srcPtrTemp + 2, yPtr++, cbPtr++, crPtr++);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
            }

            histogram_equalize_host_compute(yBuf, dstYBuf, roiWidth, roiHeight, pixels);
            yPtr = dstYBuf;
            cbPtr = cbBuf;
            crPtr = crBuf;

            for(int i = 0; i < roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6], pY, pU, pV;
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, yPtr, cbPtr, crPtr, p);     // simd loads
                    ycbcr_to_rgb_avx(p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);   // simd stores

                    yPtr += vectorIncrementPerChannel;
                    cbPtr += vectorIncrementPerChannel;
                    crPtr += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < roiWidth; vectorLoopCount++)
                {
                    ycbcr_to_rgb_compute(yPtr++, cbPtr++, crPtr++, dstPtrTemp, dstPtrTemp + 1, dstPtrTemp + 2);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Histogram Equalise without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u hist[HISTOGRAM_BINS];
            memset(hist, 0, HISTOGRAM_BINS * sizeof(Rpp32u));
            Rpp8u lutBatch[HISTOGRAM_BINS];
            Rpp8u *srcPtr, *dstPtr;
            srcPtr = srcPtrChannel;
            dstPtr = dstPtrChannel;
            collect_hist_pln_tensor_host(srcPtr, hist, roiWidth, roiHeight, srcDescPtr->strides.hStride);
            build_lut_from_hist_host(hist, lutBatch, pixels);
            apply_lut_tensor(srcPtr, dstPtr, roiWidth, roiHeight, lutBatch, srcDescPtr->strides.hStride, dstDescPtr->strides.hStride);
        }
    }

    return RPP_SUCCESS;
}
