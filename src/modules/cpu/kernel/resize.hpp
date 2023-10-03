/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"


/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;   // Align dst width to process 4 dst pixels per iteration
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc;
        Rpp32s srcLocationColumnArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTemp;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTempB++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u * dstPtrTemp, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempR, srcLocationColumnArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempG, srcLocationColumnArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempB, srcLocationColumnArray, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempChn, srcLocationColumnArray, pRow);
                        rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for ( ;vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8u)*(srcPtrTempChn + srcLocationColumn);
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

RppStatus resize_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;   // Align dst width to process 4 dst pixels per iteration
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc;
        Rpp32s srcLocationColumnArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTemp;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTempB++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f * dstPtrTemp, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[4];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempR, srcLocationColumnArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempG, srcLocationColumnArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempB, srcLocationColumnArray, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    __m128 pRow;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    rpp_simd_load(rpp_load4_f32_to_f32, (srcPtrTemp + srcLocationColumn), &pRow);
                    rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, &pRow);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempChn, srcLocationColumnArray, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp32f)*(srcPtrTempChn + srcLocationColumn);
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

RppStatus resize_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;   // Align dst width to process 4 dst pixels per iteration
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc;
        Rpp32s srcLocationColumnArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTemp;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTempB++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s * dstPtrTemp, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempR, srcLocationColumnArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempG, srcLocationColumnArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempB, srcLocationColumnArray, pRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempChn, srcLocationColumnArray, pRow);
                        rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for ( ;vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8s)*(srcPtrTempChn + srcLocationColumn);
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

RppStatus resize_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;
        Rpp32s srcLocationColumnArray[4] = {0};
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with 3 channel inputs and outputs
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

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcLocationRow = srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempR = srcPtrRowR + srcLocationRow;
                srcPtrTempG = srcPtrRowG + srcLocationRow;
                srcPtrTempB = srcPtrRowB + srcLocationRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR = (Rpp16f)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTempG = (Rpp16f)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTempB = (Rpp16f)*(srcPtrTempB + srcLocationColumn);
                    dstPtrTempR = dstPtrTempR + dstDescPtr->strides.wStride;
                    dstPtrTempG = dstPtrTempG + dstDescPtr->strides.wStride;
                    dstPtrTempB = dstPtrTempB + dstDescPtr->strides.wStride;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with single channel inputs and outputs
        else
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp16f)*(srcPtrTemp + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

/************* BILINEAR INTERPOLATION *************/

RppStatus resize_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8u *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation

                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8u *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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

RppStatus resize_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the col row location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);    // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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

RppStatus resize_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp16f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);    // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp16f *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);    // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst); // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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

RppStatus resize_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8s *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32s maxHeightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32s maxWidthLimit = (roi.xywhROI.roiWidth - 1) * srcDescPtr->strides.wStride;
        Rpp32s maxWidthLimitMinusStride = maxWidthLimit - srcDescPtr->strides.wStride;
        Rpp32s kernelSize = 2;
        Rpp32f kernelRadius = 1.0f; // kernelSize / 2
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32f hOffset = (hRatio - 1) * 0.5f - kernelRadius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - kernelRadius;
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256i pxMaxSrcLoc = _mm256_set1_epi32(maxWidthLimitMinusStride);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8], pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp8s *srcRowPtrsForInterp[2];     // kernelSize(2)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, maxWidthLimit, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp += dstDescPtr->c;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSize[batchCount].height; dstLocRow++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp8s *srcRowPtrsForInterp[6];     // kernelSize(2) * numOfPlanes(3)
                compute_resize_bilinear_src_loc_and_weights(dstLocRow, hRatio, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_bilinear_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, maxHeightLimit, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    __m256i pxSrcLoc;
                    compute_resize_bilinear_src_loc_and_weights_avx(pDstLoc, pWRatio, srcLocationColumnArray, &pWeightParams[2], pxSrcLoc, pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc, pxSrcLoc, pxMaxSrcLoc, maxWidthLimitMinusStride); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_bilinear_src_loc_and_weights(vectorLoopCount, wRatio, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, maxWidthLimit, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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

template <typename T>
RppStatus resize_separable_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f * tempPtr,
                                       RpptDescPtr tempDescPtr,
                                       RpptImagePatchPtr dstImgSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams,
                                       RpptInterpolationType interpolationType,
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

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Filter vFilter(interpolationType, roi.xywhROI.roiHeight, dstImgSize[batchCount].height, hRatio);    // Initialize vertical resampling filter
        Filter hFilter(interpolationType, roi.xywhROI.roiWidth, dstImgSize[batchCount].width, wRatio);      // Initialize Horizontal resampling filter
        Rpp32f hOffset = (hRatio - 1) * 0.5f - vFilter.radius;
        Rpp32f wOffset = (wRatio - 1) * 0.5f - hFilter.radius;

        Rpp32s rowIndex[dstImgSize[batchCount].height], colIndex[dstImgSize[batchCount].width];
        Rpp32f rowCoeffs[dstImgSize[batchCount].height * vFilter.size];
        Rpp32f colCoeffs[((dstImgSize[batchCount].width + 3) & ~3) * hFilter.size]; // Buffer size is made a multiple of 4 inorder to allocate sufficient memory for Horizontal coefficients

        // Pre-compute row index and coefficients
        for(int indexCount = 0, coeffCount = 0; indexCount < dstImgSize[batchCount].height; indexCount++, coeffCount += vFilter.size)
        {
            Rpp32f weightParam;
            compute_resize_src_loc(indexCount, hRatio, rowIndex[indexCount], weightParam, hOffset);
            compute_row_coefficients(interpolationType, vFilter, weightParam, &rowCoeffs[coeffCount]);
        }
        // Pre-compute col index and coefficients
        for(int indexCount = 0, coeffCount = 0; indexCount < dstImgSize[batchCount].width; indexCount++)
        {
            Rpp32f weightParam;
            compute_resize_src_loc(indexCount, wRatio, colIndex[indexCount], weightParam, wOffset, srcDescPtr->strides.wStride);
            coeffCount = (indexCount % 4 == 0) ? (indexCount * hFilter.size) : coeffCount + 1;
            compute_col_coefficients(interpolationType, hFilter, weightParam, &colCoeffs[coeffCount], srcDescPtr->strides.wStride);
        }

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32f * tempPtrImage = tempPtr + batchCount * tempDescPtr->strides.nStride;
        srcPtrImage = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);

        RpptImagePatch srcImgSize;
        srcImgSize.width = roi.xywhROI.roiWidth;
        srcImgSize.height = roi.xywhROI.roiHeight;

        // The intermediate result from Vertical Resampling will have the src width and dst height
        RpptImagePatch tempImgSize;
        tempImgSize.width = roi.xywhROI.roiWidth;
        tempImgSize.height = dstImgSize[batchCount].height;

        compute_separable_vertical_resample(srcPtrImage, tempPtrImage, srcDescPtr, tempDescPtr, srcImgSize, tempImgSize, rowIndex, rowCoeffs, vFilter);
        compute_separable_horizontal_resample(tempPtrImage, dstPtrImage, tempDescPtr, dstDescPtr, tempImgSize, dstImgSize[batchCount], colIndex, colCoeffs, hFilter);
    }

    return RPP_SUCCESS;
}
