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

RppStatus tensor_min_u8_u8_host(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *minArr,
                                Rpp32u minArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        // Tensor min 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp8u min = 255;
            Rpp8u resultAvx[16];

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pMin = _mm256_set1_epi8((char)255);
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_loadu_si256((__m256i *)srcPtrTemp);
                    pMin = _mm256_min_epu8(p1, pMin);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    min = std::min(*srcPtrTemp++, min);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m128i result;
            reduce_min_32_host(&pMin, &result);
            rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

            min = std::min(std::min(resultAvx[0], resultAvx[1]), min);
#endif
            minArr[batchCount] = min;
        }

        // Tensor min 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp8u minC = 255, minR = 255, minG = 255, minB = 255;
            Rpp8u resultAvx[16];

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256i pMinR = _mm256_set1_epi8((char)255);
            __m256i pMinG = pMinR;
            __m256i pMinB = pMinR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i p[3];
                    rpp_simd_load(rpp_load96_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_min_96_host(p, &pMinR, &pMinG, &pMinB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    minR = std::min(*srcPtrTempR++, minR);
                    minG = std::min(*srcPtrTempG++, minG);
                    minB = std::min(*srcPtrTempB++, minB);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m128i result;
            reduce_min_96_host(&pMinR, &pMinG, &pMinB, &result);
            rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

            minR = std::min(resultAvx[0], minR);
            minG = std::min(resultAvx[1], minG);
            minB = std::min(resultAvx[2], minB);
#endif
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = minR;
			minArr[minArrIndex + 1] = minG;
			minArr[minArrIndex + 2] = minB;
			minArr[minArrIndex + 3] = minC;
        }

        // Tensor min 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp8u minC = 255, minR = 255, minG = 255, minB = 255;
            Rpp8u resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;

                __m128i pMinR = _mm_set1_epi8((char)255);
                __m128i pMinG = pMinR;
                __m128i pMinB = pMinR;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;

                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i p[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp, p);
                        compute_min_48_host(p, &pMinR, &pMinG, &pMinB);

                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        minR = std::min(srcPtrTemp[0], minR);
                        minG = std::min(srcPtrTemp[1], minG);
                        minB = std::min(srcPtrTemp[2], minB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }

                __m128i result;
                reduce_min_48_host(&pMinR, &pMinG, &pMinB, &result);
                rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                minR = std::min(resultAvx[0], minR);
                minG = std::min(resultAvx[1], minG);
                minB = std::min(resultAvx[2], minB);
            }
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = minR;
			minArr[minArrIndex + 1] = minG;
			minArr[minArrIndex + 2] = minB;
			minArr[minArrIndex + 3] = minC;
        }
    }
    return RPP_SUCCESS;
}

RppStatus tensor_min_f32_f32_host(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *minArr,
                                  Rpp32u minArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Tensor min 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp32f min = 255.0;
            Rpp32f resultAvx[4];

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256 pMin = _mm256_set1_ps(255.0);
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p1;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p1);
                    compute_min_float8_host(&p1, &pMin);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    min = std::min(*srcPtrTemp++, min);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }

#if __AVX2__
            __m128 result;
            reduce_min_float8_host(&pMin, &result);
            rpp_simd_store(rpp_store4_f32_to_f32, resultAvx, &result);
            min = std::min(std::min(resultAvx[0], resultAvx[1]), min);
#endif
            minArr[batchCount] = min;
        }

        // Tensor min 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp32f minC = 255.0, minR = 255.0, minG = 255.0, minB = 255.0;
            Rpp32f resultAvx[8];

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256 pMinR = _mm256_set1_ps(255.0);
            __m256 pMinG = pMinR;
            __m256 pMinB = pMinR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_min_float24_host(p, &pMinR, &pMinG, &pMinB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    minR = std::min(*srcPtrTempR++, minR);
                    minG = std::min(*srcPtrTempG++, minG);
                    minB = std::min(*srcPtrTempB++, minB);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m256 result;
            reduce_min_float24_host(&pMinR, &pMinG, &pMinB, &result);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

            minR = std::min(std::min(resultAvx[0], resultAvx[1]), minR);
            minG = std::min(std::min(resultAvx[2], resultAvx[3]), minG);
            minB = std::min(std::min(resultAvx[4], resultAvx[5]), minB);
#endif
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = minR;
			minArr[minArrIndex + 1] = minG;
			minArr[minArrIndex + 2] = minB;
			minArr[minArrIndex + 3] = minC;
        }

        // Tensor min 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32u vectorIncrement = 24;
            Rpp32f minC = 255.0, minR = 255.0, minG = 255.0, minB = 255.0;
            Rpp32f resultAvx[8];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow;
                srcPtrRow = srcPtrChannel;

#if __AVX2__
                __m256 pMinR = _mm256_set1_ps(255.0);
                __m256 pMinG = pMinR;
                __m256 pMinB = pMinR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);
                        compute_min_float24_host(p, &pMinR, &pMinG, &pMinB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        minR = std::min(srcPtrTemp[0], minR);
                        minG = std::min(srcPtrTemp[1], minG);
                        minB = std::min(srcPtrTemp[2], minB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }

#if __AVX2__
                __m256 result;
                reduce_min_float24_host(&pMinR, &pMinG, &pMinB, &result);
                rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

                minR = std::min(std::min(resultAvx[0], resultAvx[1]), minR);
                minG = std::min(std::min(resultAvx[2], resultAvx[3]), minG);
                minB = std::min(std::min(resultAvx[4], resultAvx[5]), minB);
#endif
            }
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = minR;
			minArr[minArrIndex + 1] = minG;
			minArr[minArrIndex + 2] = minB;
			minArr[minArrIndex + 3] = minC;
        }
    }
    return RPP_SUCCESS;
}

RppStatus tensor_min_f16_f16_host(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *minArr,
                                  Rpp32u minArrLength,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Tensor min 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp32f min = 255.0;
            Rpp32f resultAvx[4];

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256 pMin = _mm256_set1_ps(255.0);
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                    {
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
                    }
                    __m256 p1;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, &p1);
                    compute_min_float8_host(&p1, &pMin);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    min = std::min((Rpp32f)*srcPtrTemp++, min);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }

#if __AVX2__
            __m128 result;
            reduce_min_float8_host(&pMin, &result);
            rpp_simd_store(rpp_store4_f32_to_f32, resultAvx, &result);
            min = std::min(std::min(resultAvx[0], resultAvx[1]), min);
#endif
            minArr[batchCount] = (Rpp16f) min;
        }

        // Tensor min 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp32f minC = 255.0, minR = 255.0, minG = 255.0, minB = 255.0;
            Rpp32f resultAvx[8];

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256 pMinR = _mm256_set1_ps(255.0);
            __m256 pMinG = pMinR;
            __m256 pMinB = pMinR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);
                    compute_min_float24_host(p, &pMinR, &pMinG, &pMinB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    minR = std::min((Rpp32f)*srcPtrTempR++, minR);
                    minG = std::min((Rpp32f)*srcPtrTempG++, minG);
                    minB = std::min((Rpp32f)*srcPtrTempB++, minB);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m256 result;
            reduce_min_float24_host(&pMinR, &pMinG, &pMinB, &result);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

            minR = std::min(std::min(resultAvx[0], resultAvx[1]), minR);
            minG = std::min(std::min(resultAvx[2], resultAvx[3]), minG);
            minB = std::min(std::min(resultAvx[4], resultAvx[5]), minB);
#endif
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = (Rpp16f) minR;
			minArr[minArrIndex + 1] = (Rpp16f) minG;
			minArr[minArrIndex + 2] = (Rpp16f) minB;
			minArr[minArrIndex + 3] = (Rpp16f) minC;
        }

        // Tensor min 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 24) * 24;
            Rpp32u vectorIncrement = 24;
            Rpp32f minC = 255.0, minR = 255.0, minG = 255.0, minB = 255.0;
            Rpp32f resultAvx[8];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow;
                srcPtrRow = srcPtrChannel;

#if __AVX2__
                __m256 pMinR = _mm256_set1_ps(255.0);
                __m256 pMinG = pMinR;
                __m256 pMinB = pMinR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        Rpp32f srcPtrTemp_ps[24];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        {
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
                        }
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);
                        compute_min_float24_host(p, &pMinR, &pMinG, &pMinB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        minR = std::min((Rpp32f)srcPtrTemp[0], minR);
                        minG = std::min((Rpp32f)srcPtrTemp[1], minG);
                        minB = std::min((Rpp32f)srcPtrTemp[2], minB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }

#if __AVX2__
                __m256 result;
                reduce_min_float24_host(&pMinR, &pMinG, &pMinB, &result);
                rpp_simd_store(rpp_store8_f32_to_f32_avx, resultAvx, &result);

                minR = std::min(std::min(resultAvx[0], resultAvx[1]), minR);
                minG = std::min(std::min(resultAvx[2], resultAvx[3]), minG);
                minB = std::min(std::min(resultAvx[4], resultAvx[5]), minB);
#endif
            }
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = (Rpp16f) minR;
			minArr[minArrIndex + 1] = (Rpp16f) minG;
			minArr[minArrIndex + 2] = (Rpp16f) minB;
			minArr[minArrIndex + 3] = (Rpp16f) minC;
        }
    }
    return RPP_SUCCESS;
}

RppStatus tensor_min_i8_i8_host(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *minArr,
                                Rpp32u minArrLength,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        // Tensor min 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp8s min = 127;
            Rpp8s resultAvx[16];

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pMin = _mm256_set1_epi8((char)127);
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1 = _mm256_load_si256((__m256i *)srcPtrTemp);
                    pMin = _mm256_min_epi8(p1, pMin); //compare and store min of 32 values into global min

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    min = std::min((*srcPtrTemp++), min);
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }

#if __AVX2__
            __m128i result;
            reduce_min_i32_host(&pMin, &result);
            rpp_simd_store(rpp_store16_i8, resultAvx, &result);

            min = std::min(std::min(resultAvx[0], resultAvx[1]), min);
#endif
            minArr[batchCount] = min;
        }

        // Tensor min 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp8s minC = 127, minR = 127, minG = 127, minB = 127;
            Rpp8s resultAvx[16];

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256i pMinR = _mm256_set1_epi8((char)127);
            __m256i pMinG = pMinR;
            __m256i pMinB = pMinR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i p[3];
                    rpp_simd_load(rpp_load96_i8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_min_i96_host(p, &pMinR, &pMinG, &pMinB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    minR = std::min(*srcPtrTempR++, minR);
                    minG = std::min(*srcPtrTempG++, minG);
                    minB = std::min(*srcPtrTempB++, minB);
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            __m128i result;
            reduce_min_i96_host(&pMinR, &pMinG, &pMinB, &result);
            rpp_simd_store(rpp_store16_i8, resultAvx, &result);

            minR = std::min(resultAvx[0], minR);
            minG = std::min(resultAvx[1], minG);
            minB = std::min(resultAvx[2], minB);
#endif
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = minR;
			minArr[minArrIndex + 1] = minG;
			minArr[minArrIndex + 2] = minB;
			minArr[minArrIndex + 3] = minC;
        }

        // Tensor min 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u minArrIndex = batchCount * 4;
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp8s minC = 127, minR = 127, minG = 127, minB = 127;
            Rpp8s resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow;
                srcPtrRow = srcPtrChannel;

                __m128i pMinR = _mm_set1_epi8((char)127);
                __m128i pMinG = pMinR;
                __m128i pMinB = pMinR;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX2__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i p[3];
                        rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp, p);
                        compute_min_i48_host(p, &pMinR, &pMinG, &pMinB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        minR = std::min(srcPtrTemp[0], minR);
                        minG = std::min(srcPtrTemp[1], minG);
                        minB = std::min(srcPtrTemp[2], minB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
#if __AVX2__
                __m128i result;
                reduce_min_i48_host(&pMinR, &pMinG, &pMinB, &result);
                rpp_simd_store(rpp_store16_i8, resultAvx, &result);

                minR = std::min(resultAvx[0], minR);
                minG = std::min(resultAvx[1], minG);
                minB = std::min(resultAvx[2], minB);
#endif
            }
			minC = std::min(std::min(minR, minG), minB);
            minArr[minArrIndex] = minR;
			minArr[minArrIndex + 1] = minG;
			minArr[minArrIndex + 2] = minB;
			minArr[minArrIndex + 3] = minC;
        }
    }
    return RPP_SUCCESS;
}
