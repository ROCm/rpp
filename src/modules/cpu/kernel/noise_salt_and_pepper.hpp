#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

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
        RpptXorwowState xorwowState;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr->counter + offset;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

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
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
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
        RpptXorwowState xorwowState;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr->counter + offset;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

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
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
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
        RpptXorwowState xorwowState;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr->counter + offset;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

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
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
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
        RpptXorwowState xorwowState;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;
        xorwowState.x[0] = xorwowInitialStatePtr->x[0] + offset;
        xorwowState.x[1] = xorwowInitialStatePtr->x[1] + offset;
        xorwowState.x[2] = xorwowInitialStatePtr->x[2] + offset;
        xorwowState.x[3] = xorwowInitialStatePtr->x[3] + offset;
        xorwowState.x[4] = xorwowInitialStatePtr->x[4] + offset;
        xorwowState.counter = xorwowInitialStatePtr->counter + offset;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

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
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
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
