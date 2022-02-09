#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

RppStatus brightness_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROIPtr roiPtrDefault = &roiDefault;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr = &roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, roiPtr, roiPtrDefault, roiType);

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[0])) * alpha) + beta);
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[1])) * alpha) + beta);
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (srcPtrTemp[2])) * alpha) + beta);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempR)) * alpha) + beta);
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempG)) * alpha) + beta);
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTempB)) * alpha) + beta);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                        p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                        p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*srcPtrTemp)) * alpha) + beta);

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
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROIPtr roiPtrDefault = &roiDefault;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr = &roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, roiPtr, roiPtrDefault, roiType);

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount] * ONE_OVER_255;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = RPPPIXELCHECKF32(srcPtrTemp[0] * alpha + beta);
                    *dstPtrTempG = RPPPIXELCHECKF32(srcPtrTemp[1] * alpha + beta);
                    *dstPtrTempB = RPPPIXELCHECKF32(srcPtrTemp[2] * alpha + beta);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = RPPPIXELCHECKF32(*srcPtrTempR * alpha + beta);
                    dstPtrTemp[1] = RPPPIXELCHECKF32(*srcPtrTempG * alpha + beta);
                    dstPtrTemp[2] = RPPPIXELCHECKF32(*srcPtrTempB * alpha + beta);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                    {
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32(*srcPtrTemp * alpha + beta);

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
    }

    return RPP_SUCCESS;
}

RppStatus brightness_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *alphaTensor,
                                         Rpp32f *betaTensor,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROIPtr roiPtrDefault = &roiDefault;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr = &roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, roiPtr, roiPtrDefault, roiType);

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount] * ONE_OVER_255;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp += 12;
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0] * alpha + beta);
                    *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1] * alpha + beta);
                    *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2] * alpha + beta);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 12) * 12;

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB + cnt);
                    }

                    __m128 p[4];

                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Gs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                    {
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                    }

                    srcPtrTempR += 4;
                    srcPtrTempG += 4;
                    srcPtrTempB += 4;
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR * alpha + beta);
                    dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG * alpha + beta);
                    dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB * alpha + beta);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~3;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                    {
                        Rpp32f srcPtrTemp_ps[4], dstPtrTemp_ps[4];

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(srcPtrTemp_ps + cnt) = (Rpp16f) *(srcPtrTemp + cnt);
                        }

                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p);    // simd stores

                        for(int cnt = 0; cnt < 4; cnt++)
                        {
                            *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        }

                        srcPtrTemp += 4;
                        dstPtrTemp += 4;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp * alpha + beta);

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
    }

    return RPP_SUCCESS;
}

RppStatus brightness_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *alphaTensor,
                                       Rpp32f *betaTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROIPtr roiPtrDefault = &roiDefault;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr = &roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, roiPtr, roiPtrDefault, roiType);

        Rpp32f alpha = alphaTensor[batchCount];
        Rpp32f beta = betaTensor[batchCount];

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roiPtr->xywhROI.roiWidth * layoutParams.bufferMultiplier;

        __m128 pMul = _mm_set1_ps(alpha);
        __m128 pAdd = _mm_set1_ps(beta);

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += 48;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[0]) + 128) * alpha) + beta - 128);
                    *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[1]) + 128) * alpha) + beta - 128);
                    *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (srcPtrTemp[2]) + 128) * alpha) + beta - 128);

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

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128 p[12];

                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment Rs
                    p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment Rs
                    p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment Rs
                    p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment Rs
                    p[4] = _mm_fmadd_ps(p[4], pMul, pAdd);    // brightness adjustment Gs
                    p[5] = _mm_fmadd_ps(p[5], pMul, pAdd);    // brightness adjustment Gs
                    p[6] = _mm_fmadd_ps(p[6], pMul, pAdd);    // brightness adjustment Gs
                    p[7] = _mm_fmadd_ps(p[7], pMul, pAdd);    // brightness adjustment Gs
                    p[8] = _mm_fmadd_ps(p[8], pMul, pAdd);    // brightness adjustment Bs
                    p[9] = _mm_fmadd_ps(p[9], pMul, pAdd);    // brightness adjustment Bs
                    p[10] = _mm_fmadd_ps(p[10], pMul, pAdd);    // brightness adjustment Bs
                    p[11] = _mm_fmadd_ps(p[11], pMul, pAdd);    // brightness adjustment Bs
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempR) + 128) * alpha) + beta - 128);
                    dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempG) + 128) * alpha) + beta - 128);
                    dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTempB) + 128) * alpha) + beta - 128);

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

        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roiPtr->xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_fmadd_ps(p[0], pMul, pAdd);    // brightness adjustment
                        p[1] = _mm_fmadd_ps(p[1], pMul, pAdd);    // brightness adjustment
                        p[2] = _mm_fmadd_ps(p[2], pMul, pAdd);    // brightness adjustment
                        p[3] = _mm_fmadd_ps(p[3], pMul, pAdd);    // brightness adjustment
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((((Rpp32f) (*srcPtrTemp) + 128) * alpha) + beta - 128);

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
    }

    return RPP_SUCCESS;
}
