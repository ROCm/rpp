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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0};

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m128 pKernelSize = _mm_set1_ps(kernelSize);
        __m128 pChannel = _mm_set1_ps(layoutParams.bufferMultiplier);
        __m128 pHStride = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);
        __m128 pWidthLimit = _mm_set1_ps(roi.xywhROI.roiWidth - 1);
        __m128 pBound = _mm_set1_ps(bound);

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
                __m128 pCol = xmm_pDstLocInit;

                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, bound, srcDescPtr->strides.hStride, srcDescPtr->c, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
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
                __m128 pCol = xmm_pDstLocInit;

                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow[3];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrRow, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTemp++ = *(srcPtrRow + loc);
                    *dstPtrTemp++ = *(srcPtrRow + 1 + loc);
                    *dstPtrTemp++ = *(srcPtrRow + 2 + loc);
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
                __m128 pCol = xmm_pDstLocInit;

                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pxRow;
                        rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTempChn, pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn = dstPtrTemp;
                    Rpp8u *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = *(srcPtrTempChn + loc);
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0}; 

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m128 pKernelSize = _mm_set1_ps(kernelSize);
        __m128 pChannel = _mm_set1_ps(layoutParams.bufferMultiplier);
        __m128 pHStride = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);
        __m128 pWidthLimit = _mm_set1_ps(roi.xywhROI.roiWidth-1);
        __m128 pBound = _mm_set1_ps(bound);


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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pxRow[3];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pxRow[4];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
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
#if __SSE4_1__
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    __m128 pRow;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    rpp_simd_load(rpp_load4_f32_to_f32, (srcPtrChannel + loc), &pRow);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &pRow);
                    dstPtrTemp += 3;
                }
#endif
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
                __m128 pRow = _mm_set1_ps(dstLocRow);
                __m128 pCol = xmm_pDstLocInit;

                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *srcPtrTempChn, *dstPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        __m128 pxRow;
                        rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn = dstPtrTemp;
                    Rpp32f *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp32f)*(srcPtrTempChn + loc);
                        
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
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
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, layoutParams.bufferMultiplier, loc);
                    *dstPtrTempR = (Rpp16f)*(srcPtrRowR + loc);
                    *dstPtrTempG = (Rpp16f)*(srcPtrRowG + loc);
                    *dstPtrTempB = (Rpp16f)*(srcPtrRowB + loc);
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
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTemp++ = (Rpp16f)*(srcPtrChannel + loc);
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

        Rpp32u kernelSize = kernelSizeTensor[batchCount];
        Rpp32u bound = (kernelSize - 1) / 2;
        Rpp32u heightLimit = roi.xywhROI.roiHeight - bound;
        Rpp32u offset = batchCount * srcDescPtr->strides.nStride;

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
        RpptXorwowStateBoxMuller xorwowState;
        Rpp32s srcLocArray[4] = {0};

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);
        __m128 pKernelSize = _mm_set1_ps(kernelSize);
        __m128 pChannel = _mm_set1_ps(layoutParams.bufferMultiplier);
        __m128 pHStride = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pHeightLimit = _mm_set1_ps(heightLimit);
        __m128 pWidthLimit = _mm_set1_ps(roi.xywhROI.roiWidth-1);
        __m128 pBound = _mm_set1_ps(bound);

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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrChannel, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pxRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTempR++ = *(srcPtrChannel + loc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + loc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + loc);
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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow[3];
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowR, srcLocArray, pxRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowG, srcLocArray, pxRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrRowB, srcLocArray, pxRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTemp++ = *(srcPtrRowR + loc);
                    *dstPtrTemp++ = *(srcPtrRowG + loc);
                    *dstPtrTemp++ = *(srcPtrRowB + loc);
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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pxRow;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrRow, srcLocArray, pxRow);
                    rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTemp, pxRow);
                    dstPtrTemp += vectorIncrement;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + loc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + 1 + loc);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrRow + 2 + loc);
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
                __m128 pCol = xmm_pDstLocInit;
                int vectorLoopCount = 0;
#if __SSE4_1__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_jitter_src_loc_sse(pxXorwowStateX, &pxXorwowStateCounter, pRow, pCol, pKernelSize, pBound, pHeightLimit, pWidthLimit, pHStride, pChannel, srcLocArray);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pxRow;
                        rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempChn, srcLocArray, pxRow);
                        rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTempChn, pxRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                    pCol = _mm_add_ps(xmm_p4, pCol);
                }
#endif
                for (;vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn = dstPtrTemp;
                    Rpp8s *srcPtrTempChn = srcPtrChannel;
                    Rpp32s loc;
                    compute_jitter_src_loc(&xorwowState, dstLocRow, vectorLoopCount, kernelSize, heightLimit, roi.xywhROI.roiWidth, srcDescPtr->strides.hStride, bound, srcDescPtr->c, loc);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8s)*(srcPtrTempChn + loc);
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