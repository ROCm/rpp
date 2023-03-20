/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_gaussian_noise_params_initialize_4_host_sse(Rpp32f &mean, Rpp32f &stdDev, __m128 *pGaussianNoiseParams)
{
    pGaussianNoiseParams[0] = _mm_set1_ps(mean);
    pGaussianNoiseParams[1] = _mm_set1_ps(stdDev);
}

inline void compute_gaussian_noise_params_initialize_8_host_avx(Rpp32f &mean, Rpp32f &stdDev, __m256 *pGaussianNoiseParams)
{
    pGaussianNoiseParams[0] = _mm256_set1_ps(mean);
    pGaussianNoiseParams[1] = _mm256_set1_ps(stdDev);
}

RppStatus gaussian_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempG = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempB = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);

                    srcPtrTemp+=3;
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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[1] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[2] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[1] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    dstPtrTemp[2] = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
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
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempG++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                    *dstPtrTempB++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~15;

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    rpp_multiply16_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);                                       // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);                                             // simd loads
                    rpp_multiply16_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);                                           // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (Rpp8u)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTemp++ * ONE_OVER_255, &xorwowState, mean, stdDev) * 255.0f);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);                                    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);       // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = compute_gaussian_noise_1_host(srcPtrTemp[0], &xorwowState, mean, stdDev);
                    *dstPtrTempG = compute_gaussian_noise_1_host(srcPtrTemp[1], &xorwowState, mean, stdDev);
                    *dstPtrTempB = compute_gaussian_noise_1_host(srcPtrTemp[2], &xorwowState, mean, stdDev);

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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);         // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                                  // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = compute_gaussian_noise_1_host(*srcPtrTempR, &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = compute_gaussian_noise_1_host(*srcPtrTempG, &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = compute_gaussian_noise_1_host(*srcPtrTempB, &xorwowState, mean, stdDev);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);                                    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                                  // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = compute_gaussian_noise_1_host(srcPtrTemp[0], &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = compute_gaussian_noise_1_host(srcPtrTemp[1], &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = compute_gaussian_noise_1_host(srcPtrTemp[2], &xorwowState, mean, stdDev);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
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
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);         // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);       // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = compute_gaussian_noise_1_host(*srcPtrTempR++, &xorwowState, mean, stdDev);
                    *dstPtrTempG++ = compute_gaussian_noise_1_host(*srcPtrTempG++, &xorwowState, mean, stdDev);
                    *dstPtrTempB++ = compute_gaussian_noise_1_host(*srcPtrTempB++, &xorwowState, mean, stdDev);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            Rpp32u vectorIncrementPerChannelDouble = 2 * vectorIncrementPerChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannelDouble)
                {
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);                                        // simd loads
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);                                      // simd stores
#else
                    __m128 p[2];
                    rpp_simd_load(rpp_load8_f32_to_f32, srcPtrTemp, p);                                             // simd loads
                    compute_gaussian_noise_8_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);  // gaussian_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32, dstPtrTemp, p);                                           // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannelDouble;
                    dstPtrTemp += vectorIncrementPerChannelDouble;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = compute_gaussian_noise_1_host(*srcPtrTemp++, &xorwowState, mean, stdDev);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);                                     // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);         // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);                                     // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);     // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0], &xorwowState, mean, stdDev);
                    *dstPtrTempG = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1], &xorwowState, mean, stdDev);
                    *dstPtrTempB = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2], &xorwowState, mean, stdDev);

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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);         // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);                                   // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);     // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);                                   // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR, &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG, &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB, &xorwowState, mean, stdDev);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);                             // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);                           // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);                                 // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);                               // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0], &xorwowState, mean, stdDev);
                    dstPtrTemp[1] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1], &xorwowState, mean, stdDev);
                    dstPtrTemp[2] = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2], &xorwowState, mean, stdDev);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
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
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);         // gaussian_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);     // gaussian_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);  // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR++, &xorwowState, mean, stdDev);
                    *dstPtrTempG++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG++, &xorwowState, mean, stdDev);
                    *dstPtrTempB++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB++, &xorwowState, mean, stdDev);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            Rpp32u vectorIncrementPerChannelDouble = 2 * vectorIncrementPerChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannelDouble)
                {
                    Rpp32f srcPtrTemp_ps[16], dstPtrTemp_ps[16];
                    for(int cnt = 0; cnt < vectorIncrementPerChannelDouble; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp_ps, p);                                     // simd loads
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp_ps, p);                                   // simd stores
#else
                    __m128 p[2];
                    rpp_simd_load(rpp_load8_f32_to_f32, srcPtrTemp_ps, p);                                          // simd loads
                    compute_gaussian_noise_8_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);  // gaussian_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32, dstPtrTemp_ps, p);                                        // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannelDouble; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrementPerChannelDouble;
                    dstPtrTemp += vectorIncrementPerChannelDouble;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (Rpp16f)compute_gaussian_noise_1_host((Rpp32f)*srcPtrTemp++, &xorwowState, mean, stdDev);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus gaussian_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        __m256 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_8_host_avx(mean, stdDev, pGaussianNoiseParams);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        __m128 pGaussianNoiseParams[2];
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        compute_gaussian_noise_params_initialize_4_host_sse(mean, stdDev, pGaussianNoiseParams);
#endif

        // Gaussian Noise with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempG = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempB = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);

                    srcPtrTemp+=3;
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

        // Gaussian Noise with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[1] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[2] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);

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

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);                                 // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);                               // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);                                     // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);                                   // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[0] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[1] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[1] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    dstPtrTemp[2] = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)srcPtrTemp[2] * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
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
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);      // simd loads
                    rpp_multiply48_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);          // simd loads
                    rpp_multiply48_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply48_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);        // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempR++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempG++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempG++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                    *dstPtrTempB++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTempB++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~15;

            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);                                         // simd loads
                    rpp_multiply16_constant(p, avx_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, avx_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);                                       // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);                                             // simd loads
                    rpp_multiply16_constant(p, xmm_p1op255);                                                        // u8 normalization to range[0,1]
                    compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                    rpp_multiply16_constant(p, xmm_p255);                                                           // u8 un-normalization
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);                                           // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (Rpp8s)(compute_gaussian_noise_1_host((Rpp32f)*srcPtrTemp++ * ONE_OVER_255 + RPP_128_OVER_255, &xorwowState, mean, stdDev) * 255.0f - 128.0f);
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
