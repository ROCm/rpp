#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include <random>
#include <chrono>

RppStatus rain_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32s slant,
                                 Rpp32f *alpha,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    Rpp32f rainPercent = rainPercentage * 0.004f;
    Rpp32u numDrops = static_cast<Rpp32u>(rainPercent * srcDescPtr->h * srcDescPtr->w);
    std::srand(std::time(0));
    Rpp8u *rainLayer = reinterpret_cast<Rpp8u *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    std::memset(rainLayer, 0, srcDescPtr->strides.nStride * sizeof(Rpp8u));

    for (Rpp32u i = 0; i < numDrops; i++)
    {
        Rpp32u xStart = rand() % (srcDescPtr->w - slant);
        Rpp32u yStart = rand() % (srcDescPtr->h - rainHeight);
        for (Rpp32u j = 0; j < rainHeight; j++)// height -  drop length
        {
            Rpp32u x = xStart + j * slant / rainHeight;
            Rpp32u y = yStart + j;

            if (x >= 0 && x < srcDescPtr->w && y < srcDescPtr->h)
            {
                Rpp32s loc = y * srcDescPtr->strides.hStride + x * layoutParams.bufferMultiplier;
                Rpp8u *rainLayerTemp = rainLayer + loc;

                for (Rpp32u k = 0; k < rainWidth; k++) // 
                {
                    Rpp32u idx = k * layoutParams.bufferMultiplier;
                    *(rainLayerTemp + idx) = 200;
                    if (layoutParams.bufferMultiplier > 1)
                    {
                        *(rainLayerTemp + idx + srcDescPtr->strides.cStride) = 200;
                        *(rainLayerTemp + idx + 2 * srcDescPtr->strides.cStride) = 200;
                    }
                }
            }
        }
    }

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

        __m256 pMul = _mm256_set1_ps(alpha[batchCount]);

        // Blend with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[2], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[3], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[4], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[5], p1[5]), pMul, p1[5]);    // alpha-blending adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(srcPtr1Temp[0] * (1 - alpha[batchCount]) + srcPtr2Temp[0] * alpha[batchCount]))));
                    *dstPtrTempG = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(srcPtr1Temp[1] * (1 - alpha[batchCount]) + srcPtr2Temp[1] * alpha[batchCount]))));
                    *dstPtrTempB = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(srcPtr1Temp[2] * (1 - alpha[batchCount]) + srcPtr2Temp[2] * alpha[batchCount]))));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[2], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[3], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[4], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[5], p1[5]), pMul, p1[5]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1TempR * (1 - alpha[batchCount]) + *srcPtr2TempR * alpha[batchCount]))));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1TempG * (1 - alpha[batchCount]) + *srcPtr2TempG * alpha[batchCount]))));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1TempB * (1 - alpha[batchCount]) + *srcPtr2TempB * alpha[batchCount]))));

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m256 p1[2], p2[2];

                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                        p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1Temp * (1 - alpha[batchCount]) + *srcPtr2Temp * alpha[batchCount]))));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }

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
                                 Rpp32f *alpha,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    Rpp32f rainPercent = rainPercentage * 0.004f;
    Rpp32u numDrops = static_cast<Rpp32u>(rainPercent * srcDescPtr->h * srcDescPtr->w);
    std::srand(std::time(0));
    Rpp8s *rainLayer = reinterpret_cast<Rpp8s *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    std::memset(rainLayer, 0x81, srcDescPtr->strides.nStride * sizeof(Rpp8s));

    for (Rpp32u i = 0; i < numDrops; i++)
    {
        Rpp32u xStart = rand() % (srcDescPtr->w - slant);
        Rpp32u yStart = rand() % (srcDescPtr->h - rainHeight);
        for (Rpp32u j = 0; j < rainHeight; j++)// height -  drop length
        {
            Rpp32u x = xStart + j * slant / rainHeight;
            Rpp32u y = yStart + j;

            if (x >= 0 && x < srcDescPtr->w && y < srcDescPtr->h)
            {
                Rpp32s loc = y * srcDescPtr->strides.hStride + x * layoutParams.bufferMultiplier;
                Rpp8s *rainLayerTemp = rainLayer + loc;

                for (Rpp32u k = 0; k < rainWidth; k++)
                {
                    Rpp32u idx = k * layoutParams.bufferMultiplier;
                    *(rainLayerTemp + idx) = 72;
                    if (layoutParams.bufferMultiplier > 1)
                    {
                        *(rainLayerTemp + idx + srcDescPtr->strides.cStride) = 72;
                        *(rainLayerTemp + idx + 2 * srcDescPtr->strides.cStride) = 72;
                    }
                }
            }
        }
    }

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

        __m256 pMul = _mm256_set1_ps(alpha[batchCount]);

        // Blend with fused output-layout toggle (NHWC -> NCHW)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr1Temp, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtr2Temp, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[2], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[3], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[4], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[5], p1[5]), pMul, p1[5]);    // alpha-blending adjustment

                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p1);    // simd stores
                    srcPtr1Temp += 48;
                    srcPtr2Temp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(srcPtr1Temp[0] * (1 - alpha[batchCount]) + srcPtr2Temp[0] * alpha[batchCount]))));
                    *dstPtrTempG = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(srcPtr1Temp[1] * (1 - alpha[batchCount]) + srcPtr2Temp[1] * alpha[batchCount]))));
                    *dstPtrTempB = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(srcPtr1Temp[2] * (1 - alpha[batchCount]) + srcPtr2Temp[2] * alpha[batchCount]))));

                    srcPtr1Temp += 3;
                    srcPtr2Temp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtr1Row += srcDescPtr->strides.hStride;
                srcPtr2Row += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Blend with fused output-layout toggle (NCHW -> NHWC)
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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m256 p1[6], p2[6];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr1TempR, srcPtr1TempG, srcPtr1TempB, p1);    // simd loads
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtr2TempR, srcPtr2TempG, srcPtr2TempB, p2);    // simd loads
                    p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                    p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                    p1[2] = _mm256_fmadd_ps(_mm256_sub_ps(p2[2], p1[2]), pMul, p1[2]);    // alpha-blending adjustment
                    p1[3] = _mm256_fmadd_ps(_mm256_sub_ps(p2[3], p1[3]), pMul, p1[3]);    // alpha-blending adjustment
                    p1[4] = _mm256_fmadd_ps(_mm256_sub_ps(p2[4], p1[4]), pMul, p1[4]);    // alpha-blending adjustment
                    p1[5] = _mm256_fmadd_ps(_mm256_sub_ps(p2[5], p1[5]), pMul, p1[5]);    // alpha-blending adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p1);    // simd stores

                    srcPtr1TempR += 16;
                    srcPtr1TempG += 16;
                    srcPtr1TempB += 16;
                    srcPtr2TempR += 16;
                    srcPtr2TempG += 16;
                    srcPtr2TempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1TempR * (1 - alpha[batchCount]) + *srcPtr2TempR * alpha[batchCount]))));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1TempG * (1 - alpha[batchCount]) + *srcPtr2TempG * alpha[batchCount]))));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1TempB * (1 - alpha[batchCount]) + *srcPtr2TempB * alpha[batchCount]))));

                    srcPtr1TempR++;
                    srcPtr2TempR++;
                    srcPtr1TempG++;
                    srcPtr2TempG++;
                    srcPtr1TempB++;
                    srcPtr2TempB++;
                    dstPtrTemp += 3;
                }

                srcPtr1RowR += srcDescPtr->strides.hStride;
                srcPtr1RowG += srcDescPtr->strides.hStride;
                srcPtr1RowB += srcDescPtr->strides.hStride;
                srcPtr2RowR += srcDescPtr->strides.hStride;
                srcPtr2RowG += srcDescPtr->strides.hStride;
                srcPtr2RowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Blend without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

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
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m256 p1[2], p2[2];

                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr1Temp, p1);    // simd loads
                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtr2Temp, p2);    // simd loads
                        p1[0] = _mm256_fmadd_ps(_mm256_sub_ps(p2[0], p1[0]), pMul, p1[0]);    // alpha-blending adjustment
                        p1[1] = _mm256_fmadd_ps(_mm256_sub_ps(p2[1], p1[1]), pMul, p1[1]);    // alpha-blending adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p1);    // simd stores

                        srcPtr1Temp +=16;
                        srcPtr2Temp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = static_cast<Rpp8s>(RPPPIXELCHECKI8(std::nearbyintf(static_cast<Rpp32f>(*srcPtr1Temp * (1 - alpha[batchCount]) + *srcPtr2Temp * alpha[batchCount]))));

                        srcPtr1Temp++;
                        srcPtr2Temp++;
                        dstPtrTemp++;
                    }

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