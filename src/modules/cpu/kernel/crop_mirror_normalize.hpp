#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus crop_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *meanTensor,
                                                  Rpp32f *stdDevTensor,
                                                  Rpp32u *mirrorTensor,
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

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pCMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pCMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pCMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);     //simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[0] - mean[0]) * invStdDev[0])));
                        *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[1] - mean[1]) * invStdDev[1])));
                        *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[2] - mean[2]) * invStdDev[2])));

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
            else
            {
                Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      //simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[0] - mean[0]) * invStdDev[0])));
                        *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[1] - mean[1]) * invStdDev[1])));
                        *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[2] - mean[2]) * invStdDev[2])));

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
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempR)) - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempG)) - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempB)) - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempR)) - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempG)) - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempB)) - mean[2]) * invStdDev[2]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);     //simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[0] - mean[0]) * invStdDev[0])));
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[1] - mean[1]) * invStdDev[1])));
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[2] - mean[2]) * invStdDev[2])));
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[0] - mean[0]) * invStdDev[0])));
                        dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[1] - mean[1]) * invStdDev[1])));
                        dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[2] - mean[2]) * invStdDev[2])));
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];

                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) - mean[c]) * invStdDev[c]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];

                            rpp_simd_load(rpp_load16_u8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) - mean[c]) * invStdDev[c]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
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

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pCMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c] * ONE_OVER_255;
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pCMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pCMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = RPPPIXELCHECKF32((srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = RPPPIXELCHECKF32((srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = RPPPIXELCHECKF32((srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = RPPPIXELCHECKF32((srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = RPPPIXELCHECKF32((srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = RPPPIXELCHECKF32((srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = RPPPIXELCHECKF32((*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32((*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32((*srcPtrTempB - mean[2]) * invStdDev[2]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        dstPtrTemp[0] = RPPPIXELCHECKF32((srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32((srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32((srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = RPPPIXELCHECKF32((srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32((srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32((srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[1];

                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp - mean[c]) * invStdDev[c]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[1];
                            rpp_simd_load(rpp_load8_f32_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp - mean[c]) * invStdDev[c]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
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

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pCMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c] * ONE_OVER_255;
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pCMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pCMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);     //simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

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
                        *dstPtrTempR = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp_ps, p);      // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                            dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                            dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                        }

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempB - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                        Rpp32f dstPtrTemp_ps[25];
                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                            srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                            srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                        }

                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_mirror_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempB - mean[2]) * invStdDev[2]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        Rpp32f srcPtrTemp_ps[24], dstPtrTemp_ps[25];
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];

                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        Rpp32f srcPtrTemp_ps[24], dstPtrTemp_ps[25];

                        srcPtrTemp -= vectorIncrement;
                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                        __m256 p[3];

                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_mirror_avx, srcPtrTemp_ps, p);    // simd loads
                        compute_cmn_24_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];

                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                            __m256 p[1];

                            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTemp - mean[c]) * invStdDev[c]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 8) * 8;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];

                            srcPtrTemp -= vectorIncrementPerChannel;
                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

                            __m256 p[1];

                            rpp_simd_load(rpp_load8_f32_to_f32_mirror_avx, srcPtrTemp_ps, p);    // simd loads
                            compute_cmn_8_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTemp - mean[c]) * invStdDev[c]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *meanTensor,
                                                  Rpp32f *stdDevTensor,
                                                  Rpp32u *mirrorTensor,
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

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pCMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / stdDevTensor[incrementPerImage + c];
            pCMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pCMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 - mean[0]) * invStdDev[0] - 128);
                        *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 - mean[1]) * invStdDev[1] - 128);
                        *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 - mean[2]) * invStdDev[2] - 128);

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
            else
            {
                Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 - mean[0]) * invStdDev[0] - 128);
                        *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 - mean[1]) * invStdDev[1] - 128);
                        *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 - mean[2]) * invStdDev[2] - 128);

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
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempR) + 128 - mean[0]) * invStdDev[0] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempG) + 128 - mean[1]) * invStdDev[1] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempB) + 128 - mean[2]) * invStdDev[2] - 128);

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
            else
            {
                Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd store

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempR) + 128 - mean[0]) * invStdDev[0] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempG) + 128 - mean[1]) * invStdDev[1] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTempB) + 128 - mean[2]) * invStdDev[2] - 128);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
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
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 - mean[0]) * invStdDev[0] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 - mean[1]) * invStdDev[1] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 - mean[2]) * invStdDev[2] - 128);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
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
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);      // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[0]) + 128 - mean[0]) * invStdDev[0] - 128);
                        dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[1]) + 128 - mean[1]) * invStdDev[1] - 128);
                        dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (srcPtrTemp[2]) + 128 - mean[2]) * invStdDev[2] - 128);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];

                            rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTemp) + 128 - mean[c]) * invStdDev[c] - 128);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
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
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;
                            __m256 p[2];

                            rpp_simd_load(rpp_load16_i8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8(((Rpp32f) (*srcPtrTemp) + 128 - mean[c]) * invStdDev[c] - 128);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *meanTensor,
                                                   Rpp32f *stdDevTensor,
                                                   Rpp32u *mirrorTensor,
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

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pCMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / (stdDevTensor[incrementPerImage + c] * 255);
            pCMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pCMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel;
        Rpp32f *dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        srcPtrTemp += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }

                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        *dstPtrTempR = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp32f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempB - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp32f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempB - mean[2]) * invStdDev[2]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];

                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                        srcPtrTemp += vectorIncrement;

                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        dstPtrTemp[0] = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];

                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp32f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp32f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];

                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTemp - mean[c]) * invStdDev[c]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp32f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp32f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];

                            rpp_simd_load(rpp_load16_u8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = RPPPIXELCHECKF32(((Rpp32f)*srcPtrTemp - mean[c]) * invStdDev[c]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus crop_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *meanTensor,
                                                   Rpp32f *stdDevTensor,
                                                   Rpp32u *mirrorTensor,
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

        std::vector<float> mean(srcDescPtr->c), invStdDev(srcDescPtr->c);
        Rpp32u incrementPerImage = srcDescPtr->c * batchCount;
        __m256 pCMNParams[2 * srcDescPtr->c];
        for(int c = 0; c < srcDescPtr->c; c++)
        {
            mean[c] = meanTensor[incrementPerImage + c];
            invStdDev[c] = 1.0 / (stdDevTensor[incrementPerImage + c] * 255);
            pCMNParams[2 * c] = _mm256_set1_ps(mean[c]);
            pCMNParams[2 * c + 1] = _mm256_set1_ps(invStdDev[c]);
        }
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel;
        Rpp16f *dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Crop Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];
                        Rpp32f dstPtrTempR_ps[16], dstPtrTempG_ps[16], dstPtrTempB_ps[16];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

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
                        *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRowR = dstPtrChannel;
                dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
                dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        Rpp32f dstPtrTempR_ps[16], dstPtrTempG_ps[16], dstPtrTempB_ps[16];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);     // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                        {
                            dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                            dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                            dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                        }

                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        srcPtrTemp -= 3;
                        *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);

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
        }

        // Crop Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp16f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[6];
                        Rpp32f dstPtrTemp_ps[49];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempB - mean[2]) * invStdDev[2]);

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
            else
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    Rpp16f *dstPtrTemp;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        srcPtrTempR -= vectorIncrementPerChannel;
                        srcPtrTempG -= vectorIncrementPerChannel;
                        srcPtrTempB -= vectorIncrementPerChannel;

                        __m256 p[6];
                        Rpp32f dstPtrTemp_ps[49];
                        rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_mirror_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        srcPtrTempR--;
                        srcPtrTempG--;
                        srcPtrTempB--;

                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempR - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempG - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTempB - mean[2]) * invStdDev[2]);

                        dstPtrTemp += 3;
                    }

                    srcPtrRowR += srcDescPtr->strides.hStride;
                    srcPtrRowG += srcDescPtr->strides.hStride;
                    srcPtrRowB += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            if(mirrorFlag == 0)
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m256 p[6];
                        Rpp32f dstPtrTemp_ps[49];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRow;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        srcPtrTemp -= vectorIncrement;

                        __m256 p[6];
                        Rpp32f dstPtrTemp_ps[49];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_mirror_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_48_host(p, pCMNParams);  // cmn adjustment
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < vectorIncrement; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                    {
                        srcPtrTemp -= 3;
                        dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[0] - mean[0]) * invStdDev[0]);
                        dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[1] - mean[1]) * invStdDev[1]);
                        dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)srcPtrTemp[2] - mean[2]) * invStdDev[2]);
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }

        // Crop Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if(mirrorFlag == 0)
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp16f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp16f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            __m256 p[2];
                            Rpp32f dstPtrTemp_ps[16];
                            rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                            srcPtrTemp += vectorIncrementPerChannel;
                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTemp - mean[c]) * invStdDev[c]);

                            srcPtrTemp++;
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else
            {
                Rpp32u alignedLength = (bufferLength / 16) * 16;
                srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x+roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
                for(int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *srcPtrRow;
                    Rpp16f *dstPtrRow;
                    srcPtrRow = srcPtrChannel;
                    dstPtrRow = dstPtrChannel;

                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp;
                        Rpp16f *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow;

                        int vectorLoopCount = 0;
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                        {
                            srcPtrTemp -= vectorIncrementPerChannel;

                            __m256 p[2];
                            Rpp32f dstPtrTemp_ps[16];
                            rpp_simd_load(rpp_load16_u8_to_f32_mirror_avx, srcPtrTemp, p);    // simd loads
                            compute_cmn_16_host(p, &pCMNParams[2 * c]);  // cmn adjustment
                            rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

                            for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                                dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

                            dstPtrTemp += vectorIncrementPerChannel;
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            srcPtrTemp--;

                            *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32(((Rpp32f)*srcPtrTemp - mean[c]) * invStdDev[c]);
                            dstPtrTemp++;
                        }
                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}