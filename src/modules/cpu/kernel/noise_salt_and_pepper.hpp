#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus salt_and_pepper_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *noiseProbabilityTensor,
                                                  Rpp32f *saltProbabilityTensor,
                                                  Rpp32f *saltValueTensor,
                                                  Rpp32f *pepperValueTensor,
                                                  RpptXorwowState *xorwowInitialStatePtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp8u salt = (Rpp8u) (saltValueTensor[batchCount] * 255.0f);
        Rpp8u pepper = (Rpp8u) (pepperValueTensor[batchCount] * 255.0f);
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

        RpptXorwowState xorwowState;
        xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr[0].x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr[0].x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr[0].x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr[0].x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr[0].counter + offset;

#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset, xorwowInitialStatePtr[4].x[0] + offset, xorwowInitialStatePtr[5].x[0] + offset, xorwowInitialStatePtr[6].x[0] + offset, xorwowInitialStatePtr[7].x[0] + offset);
        pxXorwowStateX[1] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset, xorwowInitialStatePtr[4].x[1] + offset, xorwowInitialStatePtr[5].x[1] + offset, xorwowInitialStatePtr[6].x[1] + offset, xorwowInitialStatePtr[7].x[1] + offset);
        pxXorwowStateX[2] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset, xorwowInitialStatePtr[4].x[2] + offset, xorwowInitialStatePtr[5].x[2] + offset, xorwowInitialStatePtr[6].x[2] + offset, xorwowInitialStatePtr[7].x[2] + offset);
        pxXorwowStateX[3] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset, xorwowInitialStatePtr[4].x[3] + offset, xorwowInitialStatePtr[5].x[3] + offset, xorwowInitialStatePtr[6].x[3] + offset, xorwowInitialStatePtr[7].x[3] + offset);
        pxXorwowStateX[4] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset, xorwowInitialStatePtr[4].x[4] + offset, xorwowInitialStatePtr[5].x[4] + offset, xorwowInitialStatePtr[6].x[4] + offset, xorwowInitialStatePtr[7].x[4] + offset);
        pxXorwowStateCounter = _mm256_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset, xorwowInitialStatePtr[4].counter + offset, xorwowInitialStatePtr[5].counter + offset, xorwowInitialStatePtr[6].counter + offset, xorwowInitialStatePtr[7].counter + offset);

        __m256 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm256_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm256_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm256_set1_ps(salt);
        pSaltAndPepperNoiseParams[3] = _mm256_set1_ps(pepper);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset);
        pxXorwowStateX[1] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset);
        pxXorwowStateX[2] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset);
        pxXorwowStateX[3] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset);
        pxXorwowStateX[4] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset);
        pxXorwowStateCounter = _mm_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset);

        __m128 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm_set1_ps(salt);
        pSaltAndPepperNoiseParams[3] = _mm_set1_ps(pepper);
#endif

        // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp8u dst[3];
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dst, srcPtrTemp, 3);
                    else
                        if(randomNumberFloat <= saltProbability)
                            memset(dst, salt, 3);
                        else
                            memset(dst, pepper, 3);
                    *dstPtrTempR = dst[0];
                    *dstPtrTempG = dst[1];
                    *dstPtrTempB = dst[2];

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

        // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp8u src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, src, 3);
                    else
                        if(randomNumberFloat <= saltProbability)
                            memset(dstPtrTemp, salt, 3);
                        else
                            memset(dstPtrTemp, pepper, 3);

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

        // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, srcPtrTemp, 3);
                    else
                        if(randomNumberFloat <= saltProbability)
                            memset(dstPtrTemp, salt, 3);
                        else
                            memset(dstPtrTemp, pepper, 3);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
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
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                    {
                        *dstPtrTempR = *srcPtrTempR;
                        *dstPtrTempG = *srcPtrTempG;
                        *dstPtrTempB = *srcPtrTempB;
                    }
                    else
                    {
                        if(randomNumberFloat <= saltProbability)
                        {
                            *dstPtrTempR = salt;
                            *dstPtrTempG = salt;
                            *dstPtrTempB = salt;
                        }
                        else
                        {
                            *dstPtrTempR = pepper;
                            *dstPtrTempG = pepper;
                            *dstPtrTempB = pepper;
                        }
                    }

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

        // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        *dstPtrTemp = *srcPtrTemp;
                    else
                        if(randomNumberFloat <= saltProbability)
                            *dstPtrTemp = salt;
                        else
                            *dstPtrTemp = pepper;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus salt_and_pepper_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *noiseProbabilityTensor,
                                                    Rpp32f *saltProbabilityTensor,
                                                    Rpp32f *saltValueTensor,
                                                    Rpp32f *pepperValueTensor,
                                                    RpptXorwowState *xorwowInitialStatePtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp32f salt = saltValueTensor[batchCount];
        Rpp32f pepper = pepperValueTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowState xorwowState;
        xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr[0].x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr[0].x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr[0].x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr[0].x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr[0].counter + offset;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset, xorwowInitialStatePtr[4].x[0] + offset, xorwowInitialStatePtr[5].x[0] + offset, xorwowInitialStatePtr[6].x[0] + offset, xorwowInitialStatePtr[7].x[0] + offset);
        pxXorwowStateX[1] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset, xorwowInitialStatePtr[4].x[1] + offset, xorwowInitialStatePtr[5].x[1] + offset, xorwowInitialStatePtr[6].x[1] + offset, xorwowInitialStatePtr[7].x[1] + offset);
        pxXorwowStateX[2] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset, xorwowInitialStatePtr[4].x[2] + offset, xorwowInitialStatePtr[5].x[2] + offset, xorwowInitialStatePtr[6].x[2] + offset, xorwowInitialStatePtr[7].x[2] + offset);
        pxXorwowStateX[3] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset, xorwowInitialStatePtr[4].x[3] + offset, xorwowInitialStatePtr[5].x[3] + offset, xorwowInitialStatePtr[6].x[3] + offset, xorwowInitialStatePtr[7].x[3] + offset);
        pxXorwowStateX[4] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset, xorwowInitialStatePtr[4].x[4] + offset, xorwowInitialStatePtr[5].x[4] + offset, xorwowInitialStatePtr[6].x[4] + offset, xorwowInitialStatePtr[7].x[4] + offset);
        pxXorwowStateCounter = _mm256_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset, xorwowInitialStatePtr[4].counter + offset, xorwowInitialStatePtr[5].counter + offset, xorwowInitialStatePtr[6].counter + offset, xorwowInitialStatePtr[7].counter + offset);

        __m256 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm256_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm256_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm256_set1_ps(salt);
        pSaltAndPepperNoiseParams[3] = _mm256_set1_ps(pepper);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset);
        pxXorwowStateX[1] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset);
        pxXorwowStateX[2] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset);
        pxXorwowStateX[3] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset);
        pxXorwowStateX[4] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset);
        pxXorwowStateCounter = _mm_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset);

        __m128 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm_set1_ps(salt);
        pSaltAndPepperNoiseParams[3] = _mm_set1_ps(pepper);
#endif

        // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[8];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f dst[3];
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dst, srcPtrTemp, 12);
                    else
                        if(randomNumberFloat <= saltProbability)
                            std::fill_n(dst, 3, salt);
                        else
                            std::fill_n(dst, 3, pepper);
                    *dstPtrTempR = dst[0];
                    *dstPtrTempG = dst[1];
                    *dstPtrTempB = dst[2];

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

        // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, src, 12);
                    else
                        if(randomNumberFloat <= saltProbability)
                            std::fill_n(dstPtrTemp, 3, salt);
                        else
                            std::fill_n(dstPtrTemp, 3, pepper);

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

        // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, srcPtrTemp, 12);
                    else
                        if(randomNumberFloat <= saltProbability)
                            std::fill_n(dstPtrTemp, 3, salt);
                        else
                            std::fill_n(dstPtrTemp, 3, pepper);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
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
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                    {
                        *dstPtrTempR = *srcPtrTempR;
                        *dstPtrTempG = *srcPtrTempG;
                        *dstPtrTempB = *srcPtrTempB;
                    }
                    else
                    {
                        if(randomNumberFloat <= saltProbability)
                        {
                            *dstPtrTempR = salt;
                            *dstPtrTempG = salt;
                            *dstPtrTempB = salt;
                        }
                        else
                        {
                            *dstPtrTempR = pepper;
                            *dstPtrTempG = pepper;
                            *dstPtrTempB = pepper;
                        }
                    }

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
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);    // simd loads
                    compute_salt_and_pepper_noise_8_host(&p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);    // simd stores
#else
                    __m128 p;
                    rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, &p);    // simd loads
                    compute_salt_and_pepper_noise_4_host(&p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        *dstPtrTemp = *srcPtrTemp;
                    else
                        if(randomNumberFloat <= saltProbability)
                            *dstPtrTemp = salt;
                        else
                            *dstPtrTemp = pepper;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus salt_and_pepper_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *noiseProbabilityTensor,
                                                    Rpp32f *saltProbabilityTensor,
                                                    Rpp32f *saltValueTensor,
                                                    Rpp32f *pepperValueTensor,
                                                    RpptXorwowState *xorwowInitialStatePtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp16f salt = (Rpp16f) saltValueTensor[batchCount];
        Rpp16f pepper = (Rpp16f) pepperValueTensor[batchCount];
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        RpptXorwowState xorwowState;
        xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr[0].x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr[0].x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr[0].x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr[0].x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr[0].counter + offset;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset, xorwowInitialStatePtr[4].x[0] + offset, xorwowInitialStatePtr[5].x[0] + offset, xorwowInitialStatePtr[6].x[0] + offset, xorwowInitialStatePtr[7].x[0] + offset);
        pxXorwowStateX[1] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset, xorwowInitialStatePtr[4].x[1] + offset, xorwowInitialStatePtr[5].x[1] + offset, xorwowInitialStatePtr[6].x[1] + offset, xorwowInitialStatePtr[7].x[1] + offset);
        pxXorwowStateX[2] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset, xorwowInitialStatePtr[4].x[2] + offset, xorwowInitialStatePtr[5].x[2] + offset, xorwowInitialStatePtr[6].x[2] + offset, xorwowInitialStatePtr[7].x[2] + offset);
        pxXorwowStateX[3] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset, xorwowInitialStatePtr[4].x[3] + offset, xorwowInitialStatePtr[5].x[3] + offset, xorwowInitialStatePtr[6].x[3] + offset, xorwowInitialStatePtr[7].x[3] + offset);
        pxXorwowStateX[4] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset, xorwowInitialStatePtr[4].x[4] + offset, xorwowInitialStatePtr[5].x[4] + offset, xorwowInitialStatePtr[6].x[4] + offset, xorwowInitialStatePtr[7].x[4] + offset);
        pxXorwowStateCounter = _mm256_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset, xorwowInitialStatePtr[4].counter + offset, xorwowInitialStatePtr[5].counter + offset, xorwowInitialStatePtr[6].counter + offset, xorwowInitialStatePtr[7].counter + offset);

        __m256 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm256_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm256_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm256_set1_ps(salt);
        pSaltAndPepperNoiseParams[3] = _mm256_set1_ps(pepper);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset);
        pxXorwowStateX[1] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset);
        pxXorwowStateX[2] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset);
        pxXorwowStateX[3] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset);
        pxXorwowStateX[4] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset);
        pxXorwowStateCounter = _mm_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset);

        __m128 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm_set1_ps(salt);
        pSaltAndPepperNoiseParams[3] = _mm_set1_ps(pepper);
#endif

        // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[8];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
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
                    Rpp16f dst[3];
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dst, srcPtrTemp, 6);
                    else
                        if(randomNumberFloat <= saltProbability)
                            std::fill_n(dst, 3, salt);
                        else
                            std::fill_n(dst, 3, pepper);
                    *dstPtrTempR = dst[0];
                    *dstPtrTempG = dst[1];
                    *dstPtrTempB = dst[2];

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

        // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
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
                    Rpp16f src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, src, 6);
                    else
                        if(randomNumberFloat <= saltProbability)
                            std::fill_n(dstPtrTemp, 3, salt);
                        else
                            std::fill_n(dstPtrTemp, 3, pepper);

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

        // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, srcPtrTemp, 6);
                    else
                        if(randomNumberFloat <= saltProbability)
                            std::fill_n(dstPtrTemp, 3, salt);
                        else
                            std::fill_n(dstPtrTemp, 3, pepper);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    compute_salt_and_pepper_noise_24_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    compute_salt_and_pepper_noise_12_host(p, p + 1, p + 2, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
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
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                    {
                        *dstPtrTempR = *srcPtrTempR;
                        *dstPtrTempG = *srcPtrTempG;
                        *dstPtrTempB = *srcPtrTempB;
                    }
                    else
                    {
                        if(randomNumberFloat <= saltProbability)
                        {
                            *dstPtrTempR = salt;
                            *dstPtrTempG = salt;
                            *dstPtrTempB = salt;
                        }
                        else
                        {
                            *dstPtrTempR = pepper;
                            *dstPtrTempG = pepper;
                            *dstPtrTempB = pepper;
                        }
                    }

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
                dstPtrRowR += srcDescPtr->strides.hStride;
                dstPtrRowG += srcDescPtr->strides.hStride;
                dstPtrRowB += srcDescPtr->strides.hStride;
            }
        }

        // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p;
                    rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, &p);    // simd loads
                    compute_salt_and_pepper_noise_8_host(&p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, &p);    // simd stores
#else
                    __m128 p;
                    rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, &p);    // simd loads
                    compute_salt_and_pepper_noise_4_host(&p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, &p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        *dstPtrTemp = *srcPtrTemp;
                    else
                        if(randomNumberFloat <= saltProbability)
                            *dstPtrTemp = salt;
                        else
                            *dstPtrTemp = pepper;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus salt_and_pepper_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *noiseProbabilityTensor,
                                                  Rpp32f *saltProbabilityTensor,
                                                  Rpp32f *saltValueTensor,
                                                  Rpp32f *pepperValueTensor,
                                                  RpptXorwowState *xorwowInitialStatePtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f noiseProbability = noiseProbabilityTensor[batchCount];
        Rpp32f saltProbability = saltProbabilityTensor[batchCount] * noiseProbability;
        Rpp8s salt = (Rpp8s) ((saltValueTensor[batchCount] * 255.0f) - 128.0f);
        Rpp8s pepper = (Rpp8s) ((pepperValueTensor[batchCount] * 255.0f) - 128.0f);
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

        RpptXorwowState xorwowState;
        xorwowState.x[0] = xorwowInitialStatePtr[0].x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr[0].x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr[0].x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr[0].x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr[0].x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr[0].counter + offset;

#if __AVX2__
        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset, xorwowInitialStatePtr[4].x[0] + offset, xorwowInitialStatePtr[5].x[0] + offset, xorwowInitialStatePtr[6].x[0] + offset, xorwowInitialStatePtr[7].x[0] + offset);
        pxXorwowStateX[1] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset, xorwowInitialStatePtr[4].x[1] + offset, xorwowInitialStatePtr[5].x[1] + offset, xorwowInitialStatePtr[6].x[1] + offset, xorwowInitialStatePtr[7].x[1] + offset);
        pxXorwowStateX[2] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset, xorwowInitialStatePtr[4].x[2] + offset, xorwowInitialStatePtr[5].x[2] + offset, xorwowInitialStatePtr[6].x[2] + offset, xorwowInitialStatePtr[7].x[2] + offset);
        pxXorwowStateX[3] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset, xorwowInitialStatePtr[4].x[3] + offset, xorwowInitialStatePtr[5].x[3] + offset, xorwowInitialStatePtr[6].x[3] + offset, xorwowInitialStatePtr[7].x[3] + offset);
        pxXorwowStateX[4] = _mm256_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset, xorwowInitialStatePtr[4].x[4] + offset, xorwowInitialStatePtr[5].x[4] + offset, xorwowInitialStatePtr[6].x[4] + offset, xorwowInitialStatePtr[7].x[4] + offset);
        pxXorwowStateCounter = _mm256_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset, xorwowInitialStatePtr[4].counter + offset, xorwowInitialStatePtr[5].counter + offset, xorwowInitialStatePtr[6].counter + offset, xorwowInitialStatePtr[7].counter + offset);

        __m256 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm256_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm256_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm256_set1_ps((Rpp32s)salt + 128);
        pSaltAndPepperNoiseParams[3] = _mm256_set1_ps((Rpp32s)pepper + 128);
#else
        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        pxXorwowStateX[0] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[0] + offset, xorwowInitialStatePtr[1].x[0] + offset, xorwowInitialStatePtr[2].x[0] + offset, xorwowInitialStatePtr[3].x[0] + offset);
        pxXorwowStateX[1] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[1] + offset, xorwowInitialStatePtr[1].x[1] + offset, xorwowInitialStatePtr[2].x[1] + offset, xorwowInitialStatePtr[3].x[1] + offset);
        pxXorwowStateX[2] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[2] + offset, xorwowInitialStatePtr[1].x[2] + offset, xorwowInitialStatePtr[2].x[2] + offset, xorwowInitialStatePtr[3].x[2] + offset);
        pxXorwowStateX[3] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[3] + offset, xorwowInitialStatePtr[1].x[3] + offset, xorwowInitialStatePtr[2].x[3] + offset, xorwowInitialStatePtr[3].x[3] + offset);
        pxXorwowStateX[4] = _mm_setr_epi32(xorwowInitialStatePtr[0].x[4] + offset, xorwowInitialStatePtr[1].x[4] + offset, xorwowInitialStatePtr[2].x[4] + offset, xorwowInitialStatePtr[3].x[4] + offset);
        pxXorwowStateCounter = _mm_setr_epi32(xorwowInitialStatePtr[0].counter + offset, xorwowInitialStatePtr[1].counter + offset, xorwowInitialStatePtr[2].counter + offset, xorwowInitialStatePtr[3].counter + offset);

        __m128 pSaltAndPepperNoiseParams[4];
        pSaltAndPepperNoiseParams[0] = _mm_set1_ps(noiseProbability);
        pSaltAndPepperNoiseParams[1] = _mm_set1_ps(saltProbability);
        pSaltAndPepperNoiseParams[2] = _mm_set1_ps((Rpp32s)salt + 128);
        pSaltAndPepperNoiseParams[3] = _mm_set1_ps((Rpp32s)pepper + 128);
#endif

        // Salt and Pepper Noise with fused output-layout toggle (NHWC -> NCHW)
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
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp8s dst[3];
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dst, srcPtrTemp, 3);
                    else
                        if(randomNumberFloat <= saltProbability)
                            memset(dst, salt, 3);
                        else
                            memset(dst, pepper, 3);
                    *dstPtrTempR = dst[0];
                    *dstPtrTempG = dst[1];
                    *dstPtrTempB = dst[2];

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

        // Salt and Pepper Noise with fused output-layout toggle (NCHW -> NHWC)
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
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp8s src[3] = {*srcPtrTempR, *srcPtrTempG, *srcPtrTempB};
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, src, 3);
                    else
                        if(randomNumberFloat <= saltProbability)
                            memset(dstPtrTemp, salt, 3);
                        else
                            memset(dstPtrTemp, pepper, 3);

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

        // Salt and Pepper Noise without fused output-layout toggle (NHWC -> NHWC)
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
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        memcpy(dstPtrTemp, srcPtrTemp, 3);
                    else
                        if(randomNumberFloat <= saltProbability)
                            memset(dstPtrTemp, salt, 3);
                        else
                            memset(dstPtrTemp, pepper, 3);

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Salt and Pepper Noise without fused output-layout toggle (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_salt_and_pepper_noise_48_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
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
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                    {
                        *dstPtrTempR = *srcPtrTempR;
                        *dstPtrTempG = *srcPtrTempG;
                        *dstPtrTempB = *srcPtrTempB;
                    }
                    else
                    {
                        if(randomNumberFloat <= saltProbability)
                        {
                            *dstPtrTempR = salt;
                            *dstPtrTempG = salt;
                            *dstPtrTempB = salt;
                        }
                        else
                        {
                            *dstPtrTempR = pepper;
                            *dstPtrTempG = pepper;
                            *dstPtrTempB = pepper;
                        }
                    }

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

        // Salt and Pepper Noise without fused output-layout toggle single channel (NCHW -> NCHW)
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
                    rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                    compute_salt_and_pepper_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pSaltAndPepperNoiseParams);    // salt_and_pepper_noise adjustment
                    rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    float randomNumberFloat = rpp_host_rng_xorwow_f32(&xorwowState);
                    if (randomNumberFloat > noiseProbability)
                        *dstPtrTemp = *srcPtrTemp;
                    else
                        if(randomNumberFloat <= saltProbability)
                            *dstPtrTemp = salt;
                        else
                            *dstPtrTemp = pepper;

                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
