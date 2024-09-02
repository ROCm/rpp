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
#include <chrono>

// Constants to represent the rain intensity for different data types
#define RAIN_INTENSITY_8U = 200;   // Intensity value for Rpp8u
#define RAIN_INTENSITY_8S = 72;    // Intensity value for Rpp8s
#define RAIN_INTENSITY_FLOAT = 200 * ONE_OVER_255; // Intensity value for Rpp32f and Rpp16f

template<typename T>
inline void create_rain_layer(T *rainLayer, Rpp32f rainPercentage, RpptDescPtr srcDescPtr, Rpp32s slant, Rpp32u dropLength, Rpp32u bufferMultiplier)
{
    Rpp32f rainPercent = rainPercentage * 0.004f;
    Rpp32u numDrops = static_cast<Rpp32u>(rainPercent * srcDescPtr->h * srcDescPtr->w);
    std::srand(std::time(0));
    // Choose the rain intensity value based on the data type
    T rainValue = std::is_same<T, Rpp8u>::value ? static_cast<T>(RAIN_INTENSITY_8U) :
                  std::is_same<T, Rpp8s>::value ? static_cast<T>(RAIN_INTENSITY_8S) :
                  static_cast<T>(RAIN_INTENSITY_FLOAT);
    Rpp32f slantPerDropLength = static_cast<Rpp32f>(slant) / dropLength;
    for (Rpp32u i = 0; i < numDrops; i++)
    {
        Rpp32u xStart = rand() % (srcDescPtr->w - slant);
        Rpp32u yStart = rand() % (srcDescPtr->h - dropLength);
        for (Rpp32u j = 0; j < dropLength; j++)
        {
            Rpp32u x = xStart + j * slantPerDropLength;
            Rpp32u y = yStart + j;

            if ((x >= 0) && (x < srcDescPtr->w) && (y < srcDescPtr->h))
            {
                T *rainLayerTemp = rainLayer + y * srcDescPtr->w + x;
                *rainLayerTemp = rainValue;
            }
        }
    }
}

RppStatus rain_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32s slant,
                                 Rpp32f *alphaValues,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    Rpp8u *rainLayer = reinterpret_cast<Rpp8u *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    std::memset(rainLayer, 0, srcDescPtr->w * srcDescPtr->h * sizeof(Rpp8u));
    create_rain_layer(rainLayer, rainPercentage, srcDescPtr, slant, rainHeight, layoutParams.bufferMultiplier);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8u *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f alpha = alphaValues[batchCount];
        __m256 pMul = _mm256_set1_ps(alpha);

        // Rain with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[6], p2[2];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[5]), pMul, p1[5]);    // alpha-blending adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[0]) * alpha + srcPtr1Temp[0]))));
                    *dstPtrTempG++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[1]) * alpha + srcPtr1Temp[1]))));
                    *dstPtrTempB++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[2]) * alpha + srcPtr1Temp[2]))));
                    srcPtr1Temp += 3;
                    srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8u *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtrChannel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = rainLayer;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[2];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr2TempR, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[5]), pMul, p1[5]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempR) * alpha + *srcPtr1TempR++))));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempG) * alpha + *srcPtr1TempG++))));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempB) * alpha + *srcPtr1TempB++))));
                    dstPtrTemp += 3;
                    srcPtr2TempR++;
                }
                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[6], p2[2];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[5]), pMul, p1[5]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1Temp += 48;
                    srcPtr2Temp += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    *srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 32) * 32;
            for(Rpp32u c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtrChannel;
                srcPtr2Row = rainLayer;
                dstPtrRow = dstPtrChannel;
                for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;
                    Rpp32u vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                    {
                        __m256 p1[4], p2[4];
                        rpp_simd_load(rpp_load32_u8_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load32_u8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                        p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                        p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[2], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                        p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[3], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store32_f32_to_u8_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += 32;
                        srcPtr2Temp += 32;
                        dstPtrTemp += 32;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp++ - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus rain_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f rainPercentage,
                                   Rpp32u rainWidth,
                                   Rpp32u rainHeight,
                                   Rpp32s slant,
                                   Rpp32f *alphaValues,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
    {
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    Rpp32f *rainLayer = handle.GetInitHandle()->mem.mcpu.scratchBufferHost;
    std::memset(rainLayer, 0, srcDescPtr->strides.nStride * sizeof(Rpp32f));
    create_rain_layer(rainLayer, rainPercentage, srcDescPtr, slant, rainHeight, layoutParams.bufferMultiplier);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f alpha = alphaValues[batchCount];
        __m256 pMul = _mm256_set1_ps(alpha);

        // Rain with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[3], p2;
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp, &p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[2]), pMul, p1[2]);    // alpha-blending adjustment

                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores
                    srcPtr1Temp += 24;
                    srcPtr2Temp += 8;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = RPPPIXELCHECKF32((*srcPtr2Temp - srcPtr1Temp[0]) * alpha + srcPtr1Temp[0]);
                    *dstPtrTempG++ = RPPPIXELCHECKF32((*srcPtr2Temp - srcPtr1Temp[1]) * alpha + srcPtr1Temp[1]);
                    *dstPtrTempB++ = RPPPIXELCHECKF32((*srcPtr2Temp - srcPtr1Temp[2]) * alpha + srcPtr1Temp[2]);
                    srcPtr1Temp += 3;
                    srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtrChannel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = rainLayer;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[3], p2;
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2TempR, &p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 8;
                    srcPtr1TempG += 8;
                    srcPtr1TempB += 8;
                    srcPtr2TempR += 8;
                    srcPtr2TempG += 8;
                    srcPtr2TempB += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtr2TempR - *srcPtr1TempR) * alpha + *srcPtr1TempR++);
                    dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtr2TempR - *srcPtr1TempG) * alpha + *srcPtr1TempG++);
                    dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtr2TempR - *srcPtr1TempB) * alpha + *srcPtr1TempB++);
                    srcPtr2TempR++;
                    dstPtrTemp += 3;
                }
                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                {
                    __m256 p1[3], p2;
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp, &p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1Temp += 24;
                    srcPtr2Temp += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp++ = RPPPIXELCHECKF32((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++);
                    *dstPtrTemp++ = RPPPIXELCHECKF32((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++);
                    *dstPtrTemp++ = RPPPIXELCHECKF32((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++);
                    srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 8) * 8;
            for(Rpp32u c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtrChannel;
                srcPtr2Row = rainLayer;
                dstPtrRow = dstPtrChannel;

                for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;
                    Rpp32u vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 p1, p2;
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr1Temp, &p1);    // simd loads
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtr2Temp, &p2);    // simd loads
                        p1 = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1), pMul, p1);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p1);    // simd stores

                        srcPtr1Temp += 8;
                        srcPtr2Temp += 8;
                        dstPtrTemp += 8;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        *dstPtrTemp++ = RPPPIXELCHECKF32((*srcPtr2Temp++ - *srcPtr1Temp) * alpha + *srcPtr1Temp++);
                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                rainLayer += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus rain_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f rainPercentage,
                                   Rpp32u rainWidth,
                                   Rpp32u rainHeight,
                                   Rpp32s slant,
                                   Rpp32f *alphaValues,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    Rpp16f *rainLayer = reinterpret_cast<Rpp16f *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    std::memset(rainLayer, 0, srcDescPtr->strides.nStride * sizeof(Rpp16f));
    create_rain_layer(rainLayer, rainPercentage, srcDescPtr, slant, rainHeight, layoutParams.bufferMultiplier);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f alpha = alphaValues[batchCount];
        __m256 pMul = _mm256_set1_ps(alpha);

        // Rain with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                {
                    __m256 p1[3], p2;
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtr2Temp, &p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[2]), pMul, p1[2]);    // alpha-blending adjustment

                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores
                    srcPtr1Temp += 24;
                    srcPtr2Temp += 8;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[0]) * alpha + srcPtr1Temp[0])));
                    *dstPtrTempG++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[0]) * alpha + srcPtr1Temp[0])));
                    *dstPtrTempB++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[0]) * alpha + srcPtr1Temp[0])));
                    srcPtr1Temp += 3;
                    srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp16f *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtrChannel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = rainLayer;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p1[3], p2;
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtr2TempR, &p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 8;
                    srcPtr1TempG += 8;
                    srcPtr1TempB += 8;
                    srcPtr2TempR += 8;
                    srcPtr2TempG += 8;
                    srcPtr2TempB += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempR) * alpha + *srcPtr1TempR++)));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempG) * alpha + *srcPtr1TempG++)));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempB) * alpha + *srcPtr1TempB++)));
                    srcPtr2TempR++;
                    dstPtrTemp += 3;
                }
                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                {
                    __m256 p1[3], p2;
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtr2Temp, &p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1Temp += 24;
                    srcPtr2Temp += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++)));
                    *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++)));
                    *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++)));
                    srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 8) * 8;
            for(Rpp32u c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtrChannel;
                srcPtr2Row = rainLayer;
                dstPtrRow = dstPtrChannel;

                for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;
                    Rpp32u vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 p1, p2;
                        rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtr1Temp, &p1);    // simd loads
                        rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtr2Temp, &p2);    // simd loads
                        p1 = _mm256_fmadd_ps(_mm256_sub_ps(p2, p1), pMul, p1);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, &p1);    // simd stores

                        srcPtr1Temp += 8;
                        srcPtr2Temp += 8;
                        dstPtrTemp += 8;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(static_cast<Rpp32f>((*srcPtr2Temp++ - *srcPtr1Temp) * alpha + *srcPtr1Temp++)));
                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                rainLayer += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus rain_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32s slant,
                                 Rpp32f *alphaValues,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    Rpp8s *rainLayer = reinterpret_cast<Rpp8s *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    std::memset(rainLayer, 0x81, srcDescPtr->strides.nStride * sizeof(Rpp8s));
    create_rain_layer(rainLayer, rainPercentage, srcDescPtr, slant, rainHeight, layoutParams.bufferMultiplier);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp8s *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32f alpha = alphaValues[batchCount];
        __m256 pMul = _mm256_set1_ps(alpha);

        // Rain with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[6], p2[2];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[5]), pMul, p1[5]);    // alpha-blending adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[0]) * alpha + srcPtr1Temp[0]))));
                    *dstPtrTempG++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[1]) * alpha + srcPtr1Temp[1]))));
                    *dstPtrTempB++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - srcPtr1Temp[2]) * alpha + srcPtr1Temp[2]))));
                    srcPtr1Temp += 3;
                    srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8s *srcPtr1RowR, *srcPtr1RowG, *srcPtr1RowB, *srcPtr2RowR, *srcPtr2RowG, *srcPtr2RowB, *dstPtrRow;
            srcPtr1RowR = srcPtrChannel;
            srcPtr1RowG = srcPtr1RowR + srcDescPtr->strides.cStride;
            srcPtr1RowB = srcPtr1RowG + srcDescPtr->strides.cStride;
            srcPtr2RowR = rainLayer;
            srcPtr2RowG = srcPtr2RowR + srcDescPtr->strides.cStride;
            srcPtr2RowB = srcPtr2RowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1TempR, *srcPtr1TempG, *srcPtr1TempB, *srcPtr2TempR, *srcPtr2TempG, *srcPtr2TempB, *dstPtrTemp;
                srcPtr1TempR = srcPtr1RowR;
                srcPtr1TempG = srcPtr1RowG;
                srcPtr1TempB = srcPtr1RowB;
                srcPtr2TempR = srcPtr2RowR;
                srcPtr2TempG = srcPtr2RowG;
                srcPtr2TempB = srcPtr2RowB;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[2];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr2TempR, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[5]), pMul, p1[5]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempR) * alpha + *srcPtr1TempR++))));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempG) * alpha + *srcPtr1TempG++))));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2TempR - *srcPtr1TempB) * alpha + *srcPtr1TempB++))));
                    srcPtr2TempR++;
                    dstPtrTemp += 3;
                }
                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
            srcPtr1Row = srcPtrChannel;
            srcPtr2Row = rainLayer;
            dstPtrRow = dstPtrChannel;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                srcPtr1Temp = srcPtr1Row;
                srcPtr2Temp = srcPtr2Row;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[6], p2[2];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[5]), pMul, p1[5]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1Temp += 48;
                    srcPtr2Temp += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    *srcPtr2Temp++;
                }
                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->w;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Rain without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 32) * 32;
            for(Rpp32u c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtr1Row, *srcPtr2Row, *dstPtrRow;
                srcPtr1Row = srcPtrChannel;
                srcPtr2Row = rainLayer;
                dstPtrRow = dstPtrChannel;
                for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtr1Temp, *srcPtr2Temp, *dstPtrTemp;
                    srcPtr1Temp = srcPtr1Row;
                    srcPtr2Temp = srcPtr2Row;
                    dstPtrTemp = dstPtrRow;
                    Rpp32u vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                    {
                        __m256 p1[4], p2[4];
                        rpp_simd_load(rpp_load32_i8_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load32_i8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                        p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                        p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[2], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                        p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[3], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store32_f32_to_i8_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp += 32;
                        srcPtr2Temp += 32;
                        dstPtrTemp += 32;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>((*srcPtr2Temp++ - *srcPtr1Temp) * alpha + *srcPtr1Temp++))));
                    srcPtr1Row += srcDescPtr->strides.hStride;
                    srcPtr2Row += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                rainLayer += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}
