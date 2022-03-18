#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

RppStatus resize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptImagePatchPtr dstImgSize,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }
        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roiPtr->xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roiPtr->xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roiPtr->xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roiPtr->xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
        __m256 pDstLocInit = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
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

                Rpp8u *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);
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

                Rpp8u *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;            // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;            // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;            // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;            // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;            // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, pSrc + 4);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, pSrc + 8);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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

                Rpp8u *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);

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
                Rpp8u *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;   // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;           // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;           // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;           // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;           // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc);
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                        rpp_simd_store(rpp_store4_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);
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

RppStatus resize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptImagePatchPtr dstImgSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }
        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roiPtr->xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roiPtr->xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roiPtr->xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roiPtr->xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
        __m256 pDstLocInit = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
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

                Rpp32f *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);
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

                Rpp32f *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;            // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;            // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;            // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;            // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;            // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, pSrc + 4);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, pSrc + 8);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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

                Rpp32f *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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

                Rpp32f *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;   // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;           // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;           // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;           // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;           // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc);
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                        rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTempChn, pDst);
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);
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

RppStatus resize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp16f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptImagePatchPtr dstImgSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }
        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roiPtr->xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roiPtr->xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roiPtr->xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roiPtr->xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
        __m256 pDstLocInit = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
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

                Rpp16f *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);
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

                Rpp16f *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;            // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;            // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;            // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;            // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;            // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, pSrc + 4);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, pSrc + 8);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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

                Rpp16f *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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

                Rpp16f *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;  // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;          // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;          // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;          // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;          // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;          // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc);
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                        rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTempChn, pDst);
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);
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

RppStatus resize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptImagePatchPtr dstImgSize,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = srcDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = srcDescPtr->h;

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtr;

        if (&roiTensorPtrSrc[batchCount] == NULL)
        {
            roiPtr = roiPtrDefault;
        }
        else
        {
            RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];

            RpptROI roiImage;
            RpptROIPtr roiPtrImage;

            if (roiType == RpptRoiType::LTRB)
            {
                roiPtrImage = &roiImage;
                compute_xywh_from_ltrb_host(roiPtrInput, roiPtrImage);
            }
            else if (roiType == RpptRoiType::XYWH)
            {
                roiPtrImage = roiPtrInput;
            }

            roiPtr = &roi;
            compute_roi_boundary_check_host(roiPtrImage, roiPtr, roiPtrDefault);
        }
        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roiPtr->xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roiPtr->xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roiPtr->xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roiPtr->xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~7;
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
        __m256 pDstLocInit = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
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

                Rpp8s *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);
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

                Rpp8s *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;            // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;            // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;            // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;            // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;            // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, pSrc + 4);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, pSrc + 8);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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

                Rpp8s *srcRowPtrsForInterp[2];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;    // srcPtrTopRow for bilinear interpolation
                srcRowPtrsForInterp[1]  = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRow for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);
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
                Rpp8s *srcRowPtrsForInterp[6];
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);
                srcRowPtrsForInterp[0] = srcPtrChannel + srcLocationRow * srcDescPtr->strides.hStride;   // srcPtrTopRowR for bilinear interpolation
                srcRowPtrsForInterp[1] = srcRowPtrsForInterp[0] + srcDescPtr->strides.hStride;           // srcPtrBottomRowR for bilinear interpolation
                srcRowPtrsForInterp[2] = srcRowPtrsForInterp[0] + srcDescPtr->strides.cStride;           // srcPtrTopRowG for bilinear interpolation
                srcRowPtrsForInterp[3] = srcRowPtrsForInterp[1] + srcDescPtr->strides.cStride;           // srcPtrBottomRowG for bilinear interpolation
                srcRowPtrsForInterp[4] = srcRowPtrsForInterp[2] + srcDescPtr->strides.cStride;           // srcPtrTopRowB for bilinear interpolation
                srcRowPtrsForInterp[5] = srcRowPtrsForInterp[3] + srcDescPtr->strides.cStride;           // srcPtrBottomRowB for bilinear interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset);
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc);
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);
                        rpp_simd_store(rpp_store4_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);
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
