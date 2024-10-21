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

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp32u alignedLength = bufferLength & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration


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
                compute_warp_perspective_src_loc(i, vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
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
                compute_warp_perspective_src_loc(i, vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
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
                __m128 pSrcX, pSrcY;
                compute_warp_perspective_src_loc(i, vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_generic_nn_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    compute_warp_perspective_src_loc_next_term(vectorLoopCount, parCommon, parY, parX, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Warp perspective with fused output-layout toggle (NCHW -> NCHW)
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
                compute_warp_perspective_src_loc(i, vectorLoopCount, srcY, srcX, perspectiveMatrix_f9, roiHalfHeight, roiHalfWidth);
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