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

/************* warp_perspective helpers *************/

inline void compute_warp_perspective_src_loc_next_term_avx(__m256 &pParCommon, __m256 &pParY, __m256 &pParX, __m256 &pSrcY, __m256 &pSrcX, __m256 &pPerspectiveMatrixTerm6Incr, __m256 &pPerspectiveMatrixTerm3Incr, __m256 &pPerspectiveMatrixTerm0Incr, __m256 roiHalfHeightVec, __m256 roiHalfWidthVec)
{
    pParCommon = _mm256_add_ps(pParCommon, pPerspectiveMatrixTerm6Incr);
    pParY = _mm256_add_ps(pParY, pPerspectiveMatrixTerm3Incr);
    pParX = _mm256_add_ps(pParX, pPerspectiveMatrixTerm0Incr);
    pSrcY = _mm256_add_ps(_mm256_div_ps(pParY, pParCommon), roiHalfHeightVec);
    pSrcX = _mm256_add_ps(_mm256_div_ps(pParX, pParCommon), roiHalfWidthVec);
}

inline void compute_warp_perspective_src_loc_params(Rpp32s dstY, Rpp32s dstX, Rpp32f &parCommon, Rpp32f &parY, Rpp32f &parX, Rpp32f9 *perspectiveMatrix_f9, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    dstX -= roiHalfWidth;
    dstY -= roiHalfHeight;
    parX = std::fma(dstX, perspectiveMatrix_f9->data[0], std::fma(dstY, perspectiveMatrix_f9->data[1], perspectiveMatrix_f9->data[2]));
    parY = std::fma(dstX, perspectiveMatrix_f9->data[3], std::fma(dstY, perspectiveMatrix_f9->data[4], perspectiveMatrix_f9->data[5]));
    parCommon = std::fma(dstX, perspectiveMatrix_f9->data[6], std::fma(dstY, perspectiveMatrix_f9->data[7], perspectiveMatrix_f9->data[8]));
}

inline void compute_warp_perspective_src_loc(Rpp32s dstY, Rpp32s dstX, Rpp32f &parCommon, Rpp32f &parY, Rpp32f &parX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f9 *perspectiveMatrix_f9, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    dstX -= roiHalfWidth;
    dstY -= roiHalfHeight;
    parX = std::fma(dstX, perspectiveMatrix_f9->data[0], std::fma(dstY, perspectiveMatrix_f9->data[1], perspectiveMatrix_f9->data[2]));
    parY = std::fma(dstX, perspectiveMatrix_f9->data[3], std::fma(dstY, perspectiveMatrix_f9->data[4], perspectiveMatrix_f9->data[5]));
    parCommon = std::fma(dstX, perspectiveMatrix_f9->data[6], std::fma(dstY, perspectiveMatrix_f9->data[7], perspectiveMatrix_f9->data[8]));
    srcX = (parX/parCommon) + roiHalfWidth;
    srcY = (parY/parCommon) + roiHalfHeight;
}

inline void compute_warp_perspective_src_loc_next_term(Rpp32s dstX, Rpp32f &parCommon, Rpp32f &parY, Rpp32f &parX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f9 *perspectiveMatrix_f9, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    parCommon += perspectiveMatrix_f9->data[6];
    parY += perspectiveMatrix_f9->data[3];   // Used in computation of next srcY locations by adding the delta from previous location
    parX += perspectiveMatrix_f9->data[0];   // Used in computation of next srcX locations by adding the delta from previous location
    srcX = (parX/parCommon) + roiHalfWidth;
    srcY = (parY/parCommon) + roiHalfHeight;
}

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus warp_perspective_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *perspectiveTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f9 *perspectiveMatrix_f9;
        perspectiveMatrix_f9 = (Rpp32f9 *)perspectiveTensor + batchCount;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 16 dst pixels per iteration
        Rpp32s srcLoc[8] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};    // Since 4 dst pixels are processed per iteration

        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pPerspectiveMatrixTerm0 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[0], perspectiveMatrix_f9->data[0] * 2, perspectiveMatrix_f9->data[0] * 3, perspectiveMatrix_f9->data[0] * 4, perspectiveMatrix_f9->data[0] * 5, perspectiveMatrix_f9->data[0] * 6, perspectiveMatrix_f9->data[0] * 7);
        __m256 pPerspectiveMatrixTerm3 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[3], perspectiveMatrix_f9->data[3] * 2, perspectiveMatrix_f9->data[3] * 3, perspectiveMatrix_f9->data[3] * 4, perspectiveMatrix_f9->data[3] * 5, perspectiveMatrix_f9->data[3] * 6, perspectiveMatrix_f9->data[3] * 7);
        __m256 pPerspectiveMatrixTerm6 = _mm256_setr_ps(0, perspectiveMatrix_f9->data[6], perspectiveMatrix_f9->data[6] * 2, perspectiveMatrix_f9->data[6] * 3, perspectiveMatrix_f9->data[6] * 4, perspectiveMatrix_f9->data[6] * 5, perspectiveMatrix_f9->data[6] * 6, perspectiveMatrix_f9->data[6] * 7);
        __m256 pPerspectiveMatrixTerm0Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[0] * 8);
        __m256 pPerspectiveMatrixTerm3Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[3] * 8);
        __m256 pPerspectiveMatrixTerm6Incr = _mm256_set1_ps(perspectiveMatrix_f9->data[6] * 8);
        __m256 roiHalfHeightVec = _mm256_set1_ps(roiHalfHeight);
        __m256 roiHalfWidthVec = _mm256_set1_ps(roiHalfWidth);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);

        // Warp perspective with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                Rpp32f parX, parY, parCommon, srcX, srcY;
                __m256 pParX, pParY, pParCommon, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, parCommon, parY, parX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                pParX = _mm256_add_ps(_mm256_set1_ps(parX), pPerspectiveMatrixTerm0);
                pParY = _mm256_add_ps(_mm256_set1_ps(parY), pPerspectiveMatrixTerm3);
                pParCommon = _mm256_add_ps(_mm256_set1_ps(parCommon), pPerspectiveMatrixTerm6);
                pSrcY = _mm256_add_ps(_mm256_div_ps(pParY, pParCommon), roiHalfHeightVec);
                pSrcX = _mm256_add_ps(_mm256_div_ps(pParX, pParCommon), roiHalfWidthVec);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8pkd3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(pParCommon, pParY, pParX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, roiHalfHeightVec, roiHalfWidthVec);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                parCommon += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                parY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                parX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = (parX/parCommon) + roiHalfWidth;
                srcY = (parY/parCommon) + roiHalfHeight;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NCHW -> NHWC)
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

                int vectorLoopCount = 0;
                Rpp32f parX, parY, parCommon, srcX, srcY;
                __m256 pParX, pParY, pParCommon, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, parCommon, parY, parX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                pParX = _mm256_add_ps(_mm256_set1_ps(parX), pPerspectiveMatrixTerm0);
                pParY = _mm256_add_ps(_mm256_set1_ps(parY), pPerspectiveMatrixTerm3);
                pParCommon = _mm256_add_ps(_mm256_set1_ps(parCommon), pPerspectiveMatrixTerm6);
                pSrcY = _mm256_add_ps(_mm256_div_ps(pParY, pParCommon), roiHalfHeightVec);
                pSrcX = _mm256_add_ps(_mm256_div_ps(pParX, pParCommon), roiHalfWidthVec);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow[3];
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelR, srcLoc, invalidLoad, pRow[0]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelG, srcLoc, invalidLoad, pRow[1]);
                    rpp_simd_load(rpp_generic_nn_load_u8pln1_avx, srcPtrChannelB, srcLoc, invalidLoad, pRow[2]);
                    rpp_simd_store(rpp_store24_u8pln3_to_u8pkd3_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(pParCommon, pParY, pParX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, roiHalfHeightVec, roiHalfWidthVec);
                    dstPtrTemp += vectorIncrementPkd;
                }
                parCommon += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                parY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                parX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = (parX/parCommon) + roiHalfWidth;
                srcY = (parY/parCommon) + roiHalfHeight;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f parX, parY, parCommon, srcX, srcY;
                __m256 pParX, pParY, pParCommon, pSrcX, pSrcY;
                compute_warp_perspective_src_loc_params(i, vectorLoopCount, parCommon, parY, parX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                pParX = _mm256_add_ps(_mm256_set1_ps(parX), pPerspectiveMatrixTerm0);
                pParY = _mm256_add_ps(_mm256_set1_ps(parY), pPerspectiveMatrixTerm3);
                pParCommon = _mm256_add_ps(_mm256_set1_ps(parCommon), pPerspectiveMatrixTerm6);
                pSrcY = _mm256_add_ps(_mm256_div_ps(pParY, pParCommon), roiHalfHeightVec);
                pSrcX = _mm256_add_ps(_mm256_div_ps(pParX, pParCommon), roiHalfWidthVec);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256i pRow;
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8_to_u8_avx, dstPtrTemp, pRow);
                    compute_warp_perspective_src_loc_next_term_avx(pParCommon, pParY, pParX, pSrcY, pSrcX, pPerspectiveMatrixTerm6Incr, pPerspectiveMatrixTerm3Incr, pPerspectiveMatrixTerm0Incr, roiHalfHeightVec, roiHalfWidthVec);
                    dstPtrTemp += vectorIncrementPkd;
                }
                parCommon += (perspectiveMatrix_f9->data[6] * vectorLoopCount);
                parY += (perspectiveMatrix_f9->data[3] * vectorLoopCount);
                parX += (perspectiveMatrix_f9->data[0] * vectorLoopCount);
                srcX = (parX/parCommon) + roiHalfWidth;
                srcY = (parY/parCommon) + roiHalfHeight;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f parX, parY, parCommon, srcX, srcY;
                compute_warp_perspective_src_loc(i, vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}