#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus warp_affine_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        roiLTRB.ltrbROI.lt.x = roi.xywhROI.xy.x;
        roiLTRB.ltrbROI.lt.y = roi.xywhROI.xy.y;
        roiLTRB.ltrbROI.rb.x = roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1;
        roiLTRB.ltrbROI.rb.y = roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1;
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = dstDescPtr->w & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pAffineMatrixTerm0 = _mm_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3);
        __m128 pAffineMatrixTerm3 = _mm_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3);
        __m128 pAffineMatrixTerm0Incr = _mm_set1_ps(affineMatrix_f6->data[0] * 4);
        __m128 pAffineMatrixTerm3Incr = _mm_set1_ps(affineMatrix_f6->data[3] * 4);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_generic_nn_load_u8pln1, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        roiLTRB.ltrbROI.lt.x = roi.xywhROI.xy.x;
        roiLTRB.ltrbROI.lt.y = roi.xywhROI.xy.y;
        roiLTRB.ltrbROI.rb.x = roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1;
        roiLTRB.ltrbROI.rb.y = roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1;
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = dstDescPtr->w & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pAffineMatrixTerm0 = _mm_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3);
        __m128 pAffineMatrixTerm3 = _mm_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3);
        __m128 pAffineMatrixTerm0Incr = _mm_set1_ps(affineMatrix_f6->data[0] * 4);
        __m128 pAffineMatrixTerm3Incr = _mm_set1_ps(affineMatrix_f6->data[3] * 4);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pln3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[4];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[4];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_f32pkd3_to_f32pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_f32pkd3_to_f32pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_generic_nn_load_f32pln1, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        roiLTRB.ltrbROI.lt.x = roi.xywhROI.xy.x;
        roiLTRB.ltrbROI.lt.y = roi.xywhROI.xy.y;
        roiLTRB.ltrbROI.rb.x = roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1;
        roiLTRB.ltrbROI.rb.y = roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1;
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = dstDescPtr->w & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pAffineMatrixTerm0 = _mm_setr_ps(0, affineMatrix_f6->data[0], affineMatrix_f6->data[0] * 2, affineMatrix_f6->data[0] * 3);
        __m128 pAffineMatrixTerm3 = _mm_setr_ps(0, affineMatrix_f6->data[3], affineMatrix_f6->data[3] * 2, affineMatrix_f6->data[3] * 3);
        __m128 pAffineMatrixTerm0Incr = _mm_set1_ps(affineMatrix_f6->data[0] * 4);
        __m128 pAffineMatrixTerm3Incr = _mm_set1_ps(affineMatrix_f6->data[3] * 4);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_i8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTemp, pRow);
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPkd;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                __m128 pSrcX, pSrcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                pSrcY = _mm_add_ps(_mm_set1_ps(srcY), pAffineMatrixTerm3);
                pSrcX = _mm_add_ps(_mm_set1_ps(srcX), pAffineMatrixTerm0);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_generic_nn_load_i8pln1, srcPtrTempChn, srcLoc, invalidLoad, pRow);
                        rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    compute_warp_affine_src_loc_next_term_sse(pSrcY, pSrcX, pAffineMatrixTerm3Incr, pAffineMatrixTerm0Incr);
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                srcY += (affineMatrix_f6->data[3] * vectorLoopCount);
                srcX += (affineMatrix_f6->data[0] * vectorLoopCount);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus warp_affine_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        roiLTRB.ltrbROI.lt.x = roi.xywhROI.xy.x;
        roiLTRB.ltrbROI.lt.y = roi.xywhROI.xy.y;
        roiLTRB.ltrbROI.rb.x = roi.xywhROI.xy.x + roi.xywhROI.roiWidth - 1;
        roiLTRB.ltrbROI.rb.y = roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1;
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f6 *affineMatrix_f6;
        affineMatrix_f6 = (Rpp32f6 *)affineTensor + batchCount;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        // Warp Affine with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp16f *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp Affine with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstDescPtr->h; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f srcX, srcY;
                compute_warp_affine_src_loc(i, vectorLoopCount, srcY, srcX, affineMatrix_f6, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_affine_src_loc_next_term(vectorLoopCount, srcY, srcX, affineMatrix_f6);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

// /************* BILINEAR INTERPOLATION *************/

// RppStatus resize_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
//                                             RpptDescPtr srcDescPtr,
//                                             Rpp8u *dstPtr,
//                                             RpptDescPtr dstDescPtr,
//                                             RpptImagePatchPtr dstImgSize,
//                                             RpptROIPtr roiTensorPtrSrc,
//                                             RpptRoiType roiType,
//                                             RppLayoutParams srcLayoutParams)
// {
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

// omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
//         Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
//         Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
//         Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
//         Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
//         Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
//         Rpp32s kernelSize = 2;
//         Rpp32f kernelRadius = 1.0f; // kernelSize / 2
//         Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
//         Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
//         Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
//         Rpp32s vectorIncrementPerChannel = 8;
//         Rpp32s vectorIncrementPkd = 24;

//         Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
//         srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
//         __m256 pWRatio = _mm256_set1_ps(wRatio);
//         __m256 pWOffset = _mm256_set1_ps(wOffset);
//         __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
//         __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
//         Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
//         Rpp32s srcLocArray[8] = {0};     // Since 8 dst pixels are processed per iteration
//         Rpp32s srcLocationRow, srcLocationColumn;

//         // Resize with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp8u *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8u *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
//                     rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp8u *dstPtrRow;
//             dstPtrRow = dstPtrChannel;
//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8u *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation

//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
//         else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp8u *dstPtrRow;
//             dstPtrRow = dstPtrChannel;
//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8u *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;
//                 Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     Rpp8u *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;
//                     __m256 pSrc[4], pDst;
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                         compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                         rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     Rpp8u *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp++;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }
//     }

//     return RPP_SUCCESS;
// }

// RppStatus resize_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
//                                               RpptDescPtr srcDescPtr,
//                                               Rpp32f *dstPtr,
//                                               RpptDescPtr dstDescPtr,
//                                               RpptImagePatchPtr dstImgSize,
//                                               RpptROIPtr roiTensorPtrSrc,
//                                               RpptRoiType roiType,
//                                               RppLayoutParams srcLayoutParams)
// {
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

// omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
//         Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
//         Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
//         Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
//         Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
//         Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
//         Rpp32s kernelSize = 2;
//         Rpp32f kernelRadius = 1.0f; // kernelSize / 2
//         Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
//         Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
//         Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
//         Rpp32s vectorIncrementPerChannel = 8;
//         Rpp32s vectorIncrementPkd = 24;

//         Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
//         srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
//         __m256 pWRatio = _mm256_set1_ps(wRatio);
//         __m256 pWOffset = _mm256_set1_ps(wOffset);
//         __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
//         __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
//         Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
//         Rpp32s srcLocArray[8] = {0};     // Since 8 dst pixels are processed per iteration
//         Rpp32s srcLocationRow, srcLocationColumn;

//         // Resize with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp32f *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp32f *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[4];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                     rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp32f *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp32f *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[4];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the col row location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
//         else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp32f *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp32f *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     Rpp32f *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;
//                     __m256 pSrc[4], pDst;
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients

//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);    // Load input pixels required for bilinear interpolation
//                         compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
//                         rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     Rpp32f *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp++;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }
//     }

//     return RPP_SUCCESS;
// }

// RppStatus resize_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
//                                               RpptDescPtr srcDescPtr,
//                                               Rpp16f *dstPtr,
//                                               RpptDescPtr dstDescPtr,
//                                               RpptImagePatchPtr dstImgSize,
//                                               RpptROIPtr roiTensorPtrSrc,
//                                               RpptRoiType roiType,
//                                               RppLayoutParams srcLayoutParams)
// {
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

// omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
//         Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
//         Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
//         Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
//         Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
//         Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
//         Rpp32s kernelSize = 2;
//         Rpp32f kernelRadius = 1.0f; // kernelSize / 2
//         Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
//         Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
//         Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
//         Rpp32s vectorIncrementPerChannel = 8;
//         Rpp32s vectorIncrementPkd = 24;

//         Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
//         srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
//         __m256 pWRatio = _mm256_set1_ps(wRatio);
//         __m256 pWOffset = _mm256_set1_ps(wOffset);
//         __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
//         __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
//         Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
//         Rpp32s srcLocArray[8] = {0};     // Since 8 dst pixels are processed per iteration
//         Rpp32s srcLocationRow, srcLocationColumn;

//         // Resize with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);    // Store dst pixels
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp16f *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp16f *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                     rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp16f *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp16f *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
//         else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp16f *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp16f *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     Rpp16f *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;

//                     __m256 pSrc[4], pDst;
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);    // Load input pixels required for bilinear interpolation
//                         compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                         rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     Rpp16f *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp++;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }
//     }

//     return RPP_SUCCESS;
// }

// RppStatus resize_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
//                                             RpptDescPtr srcDescPtr,
//                                             Rpp8s *dstPtr,
//                                             RpptDescPtr dstDescPtr,
//                                             RpptImagePatchPtr dstImgSize,
//                                             RpptROIPtr roiTensorPtrSrc,
//                                             RpptRoiType roiType,
//                                             RppLayoutParams srcLayoutParams)
// {
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

// omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
//         Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
//         Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
//         Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
//         Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
//         Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
//         Rpp32s kernelSize = 2;
//         Rpp32f kernelRadius = 1.0f; // kernelSize / 2
//         Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
//         Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
//         Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
//         Rpp32s vectorIncrementPerChannel = 8;
//         Rpp32s vectorIncrementPkd = 24;

//         Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
//         srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
//         __m256 pWRatio = _mm256_set1_ps(wRatio);
//         __m256 pWOffset = _mm256_set1_ps(wOffset);
//         __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
//         __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
//         Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
//         Rpp32s srcLocArray[8] = {0};     // Since 8 dst pixels are processed per iteration
//         Rpp32s srcLocationRow, srcLocationColumn;

//         // Resize with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp8s *dstPtrRow;
//             dstPtrRow = dstPtrChannel;

//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8s *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
//                     rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp8s *dstPtrRow;
//             dstPtrRow = dstPtrChannel;
//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8s *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;

//                 Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 pSrc[12], pDst[3];
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
//                     rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
//                     compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                     rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
//                     dstPtrTemp += vectorIncrementPkd;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
//                     dstPtrTemp += dstDescPtr->c;
//                 }

//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
//         else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp8s *dstPtrRow;
//             dstPtrRow = dstPtrChannel;
//             for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
//             {
//                 Rpp8s *dstPtrTemp;
//                 dstPtrTemp = dstPtrRow;
//                 Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
//                 compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
//                 compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
//                 pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
//                 pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
//                 pDstLoc = avx_pDstLocInit;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     Rpp8s *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;

//                     __m256 pSrc[4], pDst;
//                     __m256i pxSrcLoc;
//                     compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
//                         compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
//                         rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
//                 {
//                     Rpp8s *dstPtrTempChn;
//                     dstPtrTempChn = dstPtrTemp;
//                     compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
//                     compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
//                     for (int c = 0; c < dstDescPtr->c; c++)
//                     {
//                         compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
//                         dstPtrTempChn += dstDescPtr->strides.cStride;
//                     }
//                     dstPtrTemp++;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }
//     }

//     return RPP_SUCCESS;
// }
