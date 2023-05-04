#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_water_src_loc_sse(__m128 &pDstY, __m128 &pDstX, __m128 &pSrcY, __m128 &pSrcX, __m128 *pWaterParams,
                                      __m128 &pSinFactor, __m128 &pCosFactor, __m128 &pRowLimit, __m128 &pColLimit,
                                      __m128 &pSrcStrideH, Rpp32s *srcLocArray, bool hasRGBChannels = false)
{
    pSrcY = _mm_fmadd_ps(pWaterParams[1], pCosFactor, pDstY);
    pSrcX = _mm_fmadd_ps(pWaterParams[0], pSinFactor, pDstX);
    pDstX = _mm_add_ps(pDstX, xmm_p4);
}

inline void compute_water_src_loc(Rpp32f dstY, Rpp32f dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f amplY, Rpp32f amplX,
                                  Rpp32f sinFactor, Rpp32f cosFactor, RpptROI *roiLTRB)
{
    srcY = dstY + amplY * cosFactor;
    srcX = dstX + amplX * sinFactor;
}

RppStatus water_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f amplX = amplitudeXTensor[batchCount];
        Rpp32f amplY = amplitudeYTensor[batchCount];
        Rpp32f freqX = frequencyXTensor[batchCount];
        Rpp32f freqY = frequencyYTensor[batchCount];
        Rpp32f phaseX = phaseXTensor[batchCount];
        Rpp32f phaseY = phaseYTensor[batchCount];

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLocArray[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};         // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m128 pWaterParams[6];
        pWaterParams[0] = _mm_set1_ps(amplX);
        pWaterParams[1] = _mm_set1_ps(amplY);
        pWaterParams[2] = _mm_set1_ps(freqX);
        pWaterParams[3] = _mm_set1_ps(freqY);
        pWaterParams[4] = _mm_set1_ps(phaseX);
        pWaterParams[5] = _mm_set1_ps(phaseY);

        // Water with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray, true);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow[3];
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelR, srcLocArray, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelG, srcLocArray, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelB, srcLocArray, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray, true);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad);
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrTempChn, srcLocArray, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_u8pln1_to_u8pln1, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus water_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *amplitudeXTensor,
                                    Rpp32f *amplitudeYTensor,
                                    Rpp32f *frequencyXTensor,
                                    Rpp32f *frequencyYTensor,
                                    Rpp32f *phaseXTensor,
                                    Rpp32f *phaseYTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f amplX = amplitudeXTensor[batchCount];
        Rpp32f amplY = amplitudeYTensor[batchCount];
        Rpp32f freqX = frequencyXTensor[batchCount];
        Rpp32f freqY = frequencyYTensor[batchCount];
        Rpp32f phaseX = phaseXTensor[batchCount];
        Rpp32f phaseY = phaseYTensor[batchCount];

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLocArray[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};         // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m128 pWaterParams[6];
        pWaterParams[0] = _mm_set1_ps(amplX);
        pWaterParams[1] = _mm_set1_ps(amplY);
        pWaterParams[2] = _mm_set1_ps(freqX);
        pWaterParams[3] = _mm_set1_ps(freqY);
        pWaterParams[4] = _mm_set1_ps(phaseX);
        pWaterParams[5] = _mm_set1_ps(phaseY);

        // Water with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128 pRow[3];
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray, true);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pln3, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128 pRow[4];
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelR, srcLocArray, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelG, srcLocArray, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelB, srcLocArray, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128 pRow[4];
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pkd3, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_f32pkd3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad);

                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrTempChn, srcLocArray, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus water_f16_f16_host_tensor(Rpp16f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *amplitudeXTensor,
                                    Rpp32f *amplitudeYTensor,
                                    Rpp32f *frequencyXTensor,
                                    Rpp32f *frequencyYTensor,
                                    Rpp32f *phaseXTensor,
                                    Rpp32f *phaseYTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f amplX = amplitudeXTensor[batchCount];
        Rpp32f amplY = amplitudeYTensor[batchCount];
        Rpp32f freqX = frequencyXTensor[batchCount];
        Rpp32f freqY = frequencyYTensor[batchCount];
        Rpp32f phaseX = phaseXTensor[batchCount];
        Rpp32f phaseY = phaseYTensor[batchCount];

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Water with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY, sinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp16f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus water_i8_i8_host_tensor(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f amplX = amplitudeXTensor[batchCount];
        Rpp32f amplY = amplitudeYTensor[batchCount];
        Rpp32f freqX = frequencyXTensor[batchCount];
        Rpp32f freqY = frequencyYTensor[batchCount];
        Rpp32f phaseX = phaseXTensor[batchCount];
        Rpp32f phaseY = phaseYTensor[batchCount];

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLocArray[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m128 pWaterParams[6];
        pWaterParams[0] = _mm_set1_ps(amplX);
        pWaterParams[1] = _mm_set1_ps(amplY);
        pWaterParams[2] = _mm_set1_ps(freqX);
        pWaterParams[3] = _mm_set1_ps(freqY);
        pWaterParams[4] = _mm_set1_ps(phaseX);
        pWaterParams[5] = _mm_set1_ps(phaseY);

        // Water with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray, true);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow[3];
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelR, srcLocArray, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelG, srcLocArray, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelB, srcLocArray, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray, true);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstX, dstY, sinFactor;
                __m128 pDstX, pDstY, pSinFactor;
                dstY = (Rpp32f)i;
                sinFactor= std::sin((freqX * dstY) + phaseX);
                pDstX = xmm_pDstLocInit;
                pDstY = _mm_set1_ps(dstY);
                pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    sincos_ps(_mm_fmadd_ps(pWaterParams[3], pDstX, pWaterParams[5]), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pWaterParams, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLocArray);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad);
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrTempChn, srcLocArray, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_i8pln1_to_i8pln1, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = vectorLoopCount;
                    cosFactor = std::cos((freqY * dstX) + phaseY);
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor, &roiLTRB);

                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for(int i = 0; i < srcDescPtr->c; i++)
                        compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}