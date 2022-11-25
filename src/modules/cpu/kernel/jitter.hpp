#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u *kernelSizeTensor,
                                       RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1)/2;
        Rpp32u widthLimit = roi.xywhROI.roiWidth;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = widthLimit & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0}; 

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m128 pDstLoc = xmm_pDstLocInit;
        __m128 pKernelSize = _mm_set1_ps(kernelSize);
        __m128 pChannel = _mm_set1_ps(layoutParams.bufferMultiplier);
        __m128 pHStride = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);
        __m128 pWidthLimit = _mm_set1_ps(roi.xywhROI.roiWidth-1);


        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                
                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_u8pkd3, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store4_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTempR++ = *(srcPtrChannel + rowcolLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + rowcolLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + rowcolLoc);
                }

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128i pxRow[3];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_u8pln1, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_nn_load_u8pln1, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_nn_load_u8pln1, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowR + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + rowcolLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_u8pkd3, srcPtrRow, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRow + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRow + 1 + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRow + 2 + rowcolLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pxRow;
                        rpp_simd_load(rpp_nn_load_u8pln1, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTempChn, pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for ( ;vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn = dstPtrTemp;
                    Rpp8u *srcPtrTempChn = srcPtrChannel;
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8u)*(srcPtrTempChn + rowcolLoc);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u *kernelSizeTensor,
                                       RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1)/2;
        Rpp32u widthLimit = roi.xywhROI.roiWidth;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = widthLimit & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0}; 

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m128 pDstLoc = xmm_pDstLocInit;
        __m128 pKernelSize = _mm_set1_ps(kernelSize);
        __m128 pChannel = _mm_set1_ps(layoutParams.bufferMultiplier);
        __m128 pHStride = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);
        __m128 pWidthLimit = _mm_set1_ps(roi.xywhROI.roiWidth-1);


        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                
                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128 pxRow[3];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_f32pkd3_to_f32pln3, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTempR++ = *(srcPtrChannel + rowcolLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + rowcolLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + rowcolLoc);
                }

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128 pxRow[4];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_f32pln1, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_nn_load_f32pln1, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_nn_load_f32pln1, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, layoutParams.bufferMultiplier, rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowR + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + rowcolLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    __m128 pRow;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, layoutParams.bufferMultiplier, rowcolLoc);
                    rpp_simd_load(rpp_load4_f32_to_f32, (srcPtrChannel + rowcolLoc), &pRow);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &pRow);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                pDstLoc = xmm_pDstLocInit;
                __m128 pRow = _mm_set1_ps(dstLocRow); 

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m128 pxRow;
                        rpp_simd_load(rpp_nn_load_f32pln1, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for ( ;vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn = dstPtrTemp;
                    Rpp32f *srcPtrTempChn = srcPtrChannel;
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp32f)*(srcPtrTempChn + rowcolLoc);
                        
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u *kernelSizeTensor,
                                       RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1)/2;
        Rpp32u widthLimit = roi.xywhROI.roiWidth;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = widthLimit & ~3;   // Align dst width to process 4 dst pixels per iteration
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0}; 

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);


        // Jitter with 3 channel inputs and outputs with Packed conversions
        if (srcDescPtr->c == 3 && dstDescPtr->c == 3)
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, layoutParams.bufferMultiplier, rowcolLoc);
                    *dstPtrTempR = (Rpp16f)*(srcPtrRowR + rowcolLoc);
                    *dstPtrTempG = (Rpp16f)*(srcPtrRowG + rowcolLoc);
                    *dstPtrTempB = (Rpp16f)*(srcPtrRowB + rowcolLoc);
                    dstPtrTempR = dstPtrTempR + dstDescPtr->strides.wStride;
                    dstPtrTempG = dstPtrTempG + dstDescPtr->strides.wStride;
                    dstPtrTempB = dstPtrTempB + dstDescPtr->strides.wStride;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with single channel inputs and outputs
        else
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTemp++ = (Rpp16f)*(srcPtrChannel + rowcolLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u *kernelSizeTensor,
                                       RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize-1)/2;
        Rpp32u widthLimit = roi.xywhROI.roiWidth;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = widthLimit & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0}; 

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m128 pDstLoc = xmm_pDstLocInit;
        __m128 pKernelSize = _mm_set1_ps(kernelSize);
        __m128 pChannel = _mm_set1_ps(layoutParams.bufferMultiplier);
        __m128 pHStride = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);
        __m128 pWidthLimit = _mm_set1_ps(roi.xywhROI.roiWidth-1);


        // Jitter with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                
                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_i8pkd3, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store4_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTempR++ = *(srcPtrChannel + rowcolLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + rowcolLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + rowcolLoc);
                }

                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Jitter with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128i pxRow[3];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_i8pln1, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_nn_load_i8pln1, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_nn_load_i8pln1, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowR + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + rowcolLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + rowcolLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Jitter without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_nn_load_i8pkd3, srcPtrRow, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                }

                for (; vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + rowcolLoc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + 1 + rowcolLoc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + 2 + rowcolLoc);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        // Jitter with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                __m128 pRow = _mm_set1_ps(dstLocRow); 
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    __m128 pCol = _mm_set1_ps(vectorLoopCount);
                    pCol = _mm_add_ps(pCol, pDstLoc);
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pxRow;
                        rpp_simd_load(rpp_nn_load_i8pln1, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTempChn, pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for ( ;vectorLoopCount < widthLimit; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn = dstPtrTemp;
                    Rpp8s *srcPtrTempChn = srcPtrChannel;
                    Rpp32s rowcolLoc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, srcDescPtr->c, rowcolLoc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8s)*(srcPtrTempChn + rowcolLoc);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
