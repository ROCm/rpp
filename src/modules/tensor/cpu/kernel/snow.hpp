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

RppStatus snow_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f *brightnessCoefficientTensor,
                                 Rpp32f *snowThresholdTensor,
                                 Rpp32s *darkModeTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = ((snowThresholdTensor[batchCount] * 127.5f) + 85.0f) * ONE_OVER_255;
        Rpp32s darkMode = darkModeTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<float>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f; 
                    pixel.G *= 255.0f; 
                    pixel.B *= 255.0f; 
                    *dstPtrTempR = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    *dstPtrTempG = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    *dstPtrTempB = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f; 
                    pixel.G *= 255.0f; 
                    pixel.B *= 255.0f; 
                    dstPtrTemp[0] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    dstPtrTemp[1] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    dstPtrTemp[2] = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    pixel.G *= 255.0f;
                    pixel.B *= 255.0f;
                    *dstPtrTempR = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    *dstPtrTempG = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.G)));
                    *dstPtrTempB = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.B)));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__ 
            alignedLength = (bufferLength & ~15);
#endif
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    rpp_simd_load(rpp_normalize16_avx, p);                                                          // simd normalize
                    compute_snow_8_host(&p[0], pSnowParams);                                                        // snow adjustment
                    compute_snow_8_host(&p[1], pSnowParams);                                                        // snow adjustment
                    rpp_multiply16_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);                                    // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTemp) * ONE_OVER_255;
                    pixel.G = static_cast<Rpp32f>(*srcPtrTemp) * ONE_OVER_255;
                    pixel.B = static_cast<Rpp32f>(*srcPtrTemp) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f; 
                    *dstPtrTemp++ = static_cast<Rpp8u>(RPPPIXELCHECK(std::nearbyintf(pixel.R)));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

    }

    return RPP_SUCCESS;
}

RppStatus snow_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *brightnessCoefficientTensor,
                                   Rpp32f *snowThresholdTensor,
                                   Rpp32s *darkModeTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = ((snowThresholdTensor[batchCount] * (127.5)) + 85) * ONE_OVER_255;
        Rpp32s darkMode = darkModeTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<float>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = srcPtrTemp[0];
                    pixel.G = srcPtrTemp[1];
                    pixel.B = srcPtrTemp[2];
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR = RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG = RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB = RPPPIXELCHECKF32(pixel.B);

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTempR;
                    pixel.G = *srcPtrTempG;
                    pixel.B = *srcPtrTempB;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(pixel.B);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = srcPtrTemp[0];
                    pixel.G = srcPtrTemp[1];
                    pixel.B = srcPtrTemp[2];
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = RPPPIXELCHECKF32(pixel.R);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(pixel.G);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(pixel.B);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTempR;
                    pixel.G = *srcPtrTempG;
                    pixel.B = *srcPtrTempB;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR = RPPPIXELCHECKF32(pixel.R);
                    *dstPtrTempG = RPPPIXELCHECKF32(pixel.G);
                    *dstPtrTempB = RPPPIXELCHECKF32(pixel.B);

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = (bufferLength & ~15);
#endif
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);                                 // simd loads
                    compute_snow_8_host(&p, pSnowParams);                                                       // snow adjustment
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, &p);                              // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = *srcPtrTemp;
                    pixel.G = *srcPtrTemp;
                    pixel.B = *srcPtrTemp;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTemp++ = RPPPIXELCHECK(pixel.R);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus snow_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *brightnessCoefficientTensor,
                                   Rpp32f *snowThresholdTensor,
                                   Rpp32s *darkModeTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = ((snowThresholdTensor[batchCount] * (127.5)) + 85) * ONE_OVER_255;
        Rpp32s darkMode = darkModeTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<float>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    *dstPtrTempG = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    *dstPtrTempB = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);                              // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR);
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG);
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);                              // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixel.G = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixel.B = static_cast<Rpp32f>(srcPtrTemp[2]);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    dstPtrTemp[1] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    dstPtrTemp[2] = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_snow_24_host(p[0], p[1], p[2], pSnowParams);                                            // snow adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrTempR);
                    pixel.G = static_cast<Rpp32f>(*srcPtrTempG);
                    pixel.B = static_cast<Rpp32f>(*srcPtrTempB);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    *dstPtrTempG = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.G));
                    *dstPtrTempB = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.B));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = (bufferLength & ~15);
#endif 
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p;
                    rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, &p);                                 // simd loads 
                    compute_snow_8_host(&p, pSnowParams);                                                    // snow adjustment
                    rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, &p);                               // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>(*srcPtrChannel);
                    pixel.G = static_cast<Rpp32f>(*srcPtrChannel);
                    pixel.B = static_cast<Rpp32f>(*srcPtrChannel);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTemp++ = static_cast<Rpp16f>(RPPPIXELCHECKF32(pixel.R));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus snow_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f *brightnessCoefficientTensor,
                                 Rpp32f *snowThresholdTensor,
                                 Rpp32s *darkModeTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32u batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f brightnessCoefficient = brightnessCoefficientTensor[batchCount];
        Rpp32f snowThreshold = ((snowThresholdTensor[batchCount] * (127.5)) + 85) * ONE_OVER_255;
        Rpp32s darkMode = darkModeTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        __m256 pSnowParams[3];
        pSnowParams[0] = _mm256_set1_ps(brightnessCoefficient);
        pSnowParams[1] = _mm256_set1_ps(snowThreshold);
        pSnowParams[2] = _mm256_set1_ps(static_cast<float>(darkMode));
#endif

        // Snow with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    *dstPtrTempG = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    *dstPtrTempB = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Snow with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                               // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(*srcPtrTempR) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(*srcPtrTempG) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(*srcPtrTempB) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                               // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(srcPtrTemp[0]) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(srcPtrTemp[1]) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(srcPtrTemp[2]) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    dstPtrTemp[0] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    dstPtrTemp[1] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    dstPtrTemp[2] = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_simd_load(rpp_normalize48_avx, p);                                                          // simd normalize
                    compute_snow_24_host(p[0], p[2], p[4], pSnowParams);                                            // snow adjustment
                    compute_snow_24_host(p[1], p[3], p[5], pSnowParams);                                            // snow adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = (static_cast<Rpp32f>(*srcPtrTempR) + 128.0f) * ONE_OVER_255;
                    pixel.G = (static_cast<Rpp32f>(*srcPtrTempG) + 128.0f) * ONE_OVER_255;
                    pixel.B = (static_cast<Rpp32f>(*srcPtrTempB) + 128.0f) * ONE_OVER_255;
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    *dstPtrTempR = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    *dstPtrTempG = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.G - 128.0f));
                    *dstPtrTempB = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.B - 128.0f));

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Snow without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = (bufferLength & ~15);
#endif
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for (Rpp32u i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32u vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    rpp_simd_load(rpp_normalize16_avx, p);                                                          // simd normalize
                    compute_snow_8_host(&p[0], pSnowParams);                                                        // snow adjustment
                    compute_snow_8_host(&p[1], pSnowParams);                                                        // snow adjustment
                    rpp_multiply16_constant(p, avx_p255);                                                           // simd denormalize
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);                                       // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    RpptFloatRGB pixel;
                    pixel.R = static_cast<Rpp32f>((*srcPtrTemp + 128.0f) * ONE_OVER_255);
                    pixel.G = static_cast<Rpp32f>((*srcPtrTemp + 128.0f) * ONE_OVER_255);
                    pixel.B = static_cast<Rpp32f>((*srcPtrTemp + 128.0f) * ONE_OVER_255);
                    compute_snow_host(&pixel, brightnessCoefficient, snowThreshold, darkMode);
                    pixel.R *= 255.0f;
                    *dstPtrTemp++ = static_cast<Rpp8s>(RPPPIXELCHECKI8(pixel.R - 128.0f));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
