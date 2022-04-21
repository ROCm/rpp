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
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
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
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset); // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4]);
                    rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8]);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_u8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation

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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_u8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store4_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
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
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4]);
                    rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8]);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[4];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f32pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the col row location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f32pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc);    // Load input pixels required for bilinear interpolation
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
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
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
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);  // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst);    // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4]);
                    rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8]);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_f16pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp16f *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_f16pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc);    // Load input pixels required for bilinear interpolation
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
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = (hRatio - 1) * 0.5f;
        Rpp32f wOffset = (wRatio - 1) * 0.5f;
        Rpp32s kernelSize = 2;
        Rpp32s noOfCoeffs = 4; // kernelSize * kernelSize
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
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);

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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);                              // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);                             // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[0], srcLocationColumnArray, pSrc);  // Load input pixels required for bilinear interpolation
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[2], srcLocationColumnArray, &pSrc[4]);
                    rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[4], srcLocationColumnArray, &pSrc[8]);
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset, true);   // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);      // Compute Bilinear coefficients
                    rpp_simd_load(rpp_bilinear_load_i8pkd3_to_f32pln3_avx, srcRowPtrsForInterp, srcLocationColumnArray, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store12_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
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
                compute_resize_src_loc(dstLocRow, hRatio, heightLimit, srcLocationRow, &weightParams[0], hOffset);  // Compute the src row location correspoding to the dst row location
                compute_src_row_ptrs_for_interpolation_pln(srcRowPtrsForInterp, srcPtrChannel, srcLocationRow, srcDescPtr); // Compute the src row pointers for interpolation
                pWeightParams[0] = _mm256_set1_ps(weightParams[0]);
                pWeightParams[1]  = _mm256_set1_ps(weightParams[1]);
                pDstLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;

                    __m256 pSrc[4], pDst;
                    compute_resize_src_loc_avx(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, &pWeightParams[2], pWOffset); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients_avx(pWeightParams, pBilinearCoeffs);          // Compute Bilinear coefficients

                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        rpp_simd_load(rpp_bilinear_load_i8pln1_to_f32pln1_avx, &srcRowPtrsForInterp[c * kernelSize], srcLocationColumnArray, pSrc); // Load input pixels required for bilinear interpolation
                        compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst);     // Compute Bilinear interpolation
                        rpp_simd_store(rpp_store4_f32pln1_to_i8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
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

RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
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
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        Rpp32s widthLimitChanneled = widthLimit * 3;
        Rpp32f weightParams[2];
        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pDstLocInit = _mm_setr_ps(0, 1, 2, 3);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc, pWeightParams[2];
        Rpp32s srcLocCF[4] = {0};

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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128i pRow;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset, true);
                    rpp_simd_load(rpp_nn_load_u8pkd3, srcPtrTemp, srcLocCF, pRow);
                    rpp_simd_store(rpp_store4_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp8u)*(srcPtrTemp + srcLocationColumnFloor);
                    *dstPtrTempG++ = (Rpp8u)*(srcPtrTemp + srcLocationColumnFloor + 1);
                    *dstPtrTempB++ = (Rpp8u)*(srcPtrTemp + srcLocationColumnFloor + 2);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128i pRow[3];
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset);
                    rpp_simd_load(rpp_nn_load_u8pln1, srcPtrTempR, srcLocCF, pRow[0]);
                    rpp_simd_load(rpp_nn_load_u8pln1, srcPtrTempG, srcLocCF, pRow[1]);
                    rpp_simd_load(rpp_nn_load_u8pln1, srcPtrTempB, srcLocCF, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempR + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempG + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempB + srcLocationColumnFloor);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128i pRow;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset, true);
                    rpp_simd_load(rpp_nn_load_u8pkd3, srcPtrTemp, srcLocCF, pRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumnFloor + 1);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumnFloor + 2);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_nn_load_u8pln1, srcPtrTempChn, srcLocCF, pRow);
                        rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp+=4;
                }
                for ( ;vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8u)*(srcPtrTempChn + srcLocationColumnFloor);
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
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        Rpp32s widthLimitChanneled = widthLimit * 3;
        Rpp32f weightParams[2];
        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;
        Rpp32s srcLocCF[4] = {0};
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pDstLocInit = _mm_setr_ps(0, 1, 2, 3);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWeightParams[2], pDstLoc;
        __m128 pWOffset = _mm_set1_ps(wOffset);

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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, &weightParams[0], hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pRow[3];
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, &pWeightParams[0], pWOffset, true);
                    rpp_simd_load(rpp_nn_load_f32pkd3_to_f32pln3, srcPtrTemp, srcLocCF, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, &weightParams[0], wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp32f)*(srcPtrTemp + srcLocationColumnFloor);
                    *dstPtrTempG++ = (Rpp32f)*(srcPtrTemp + srcLocationColumnFloor + 1);
                    *dstPtrTempB++ = (Rpp32f)*(srcPtrTemp + srcLocationColumnFloor + 2);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, &weightParams[0], hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128 pRow[4];
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, &pWeightParams[0], pWOffset);
                    rpp_simd_load(rpp_nn_load_f32pln1, srcPtrTempR, srcLocCF, pRow[0]);
                    rpp_simd_load(rpp_nn_load_f32pln1, srcPtrTempG, srcLocCF, pRow[1]);
                    rpp_simd_load(rpp_nn_load_f32pln1, srcPtrTempB, srcLocCF, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, &weightParams[0], wOffset);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempR + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempG + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempB + srcLocationColumnFloor);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, &weightParams[0], hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    __m128 pRow;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, &weightParams[0], wOffset, srcDescPtr->strides.wStride);
                    rpp_simd_load(rpp_load4_f32_to_f32, (srcPtrTemp + srcLocationColumnFloor), &pRow);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, &weightParams[0], hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_nn_load_f32pln1, srcPtrTempChn, srcLocCF, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, &pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp+=4;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, &weightParams[0], wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp32f)*(srcPtrTempChn + srcLocationColumnFloor);
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
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        Rpp32s widthLimitChanneled = widthLimit * 3;
        Rpp32f weightParams[2];
        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pDstLocInit = _mm_setr_ps(0, 1, 2, 3);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc, pWeightParams[2];
        Rpp32s srcLocCF[4] = {0};

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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128i pRow;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset, true);
                    rpp_simd_load(rpp_nn_load_i8pkd3, srcPtrTemp, srcLocCF, pRow);
                    rpp_simd_store(rpp_store4_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += 4;
                    dstPtrTempG += 4;
                    dstPtrTempB += 4;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp8s)*(srcPtrTemp + srcLocationColumnFloor);
                    *dstPtrTempG++ = (Rpp8s)*(srcPtrTemp + srcLocationColumnFloor + 1);
                    *dstPtrTempB++ = (Rpp8s)*(srcPtrTemp + srcLocationColumnFloor + 2);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128i pRow[3];
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset);
                    rpp_simd_load(rpp_nn_load_i8pln1, srcPtrTempR, srcLocCF, pRow[0]);
                    rpp_simd_load(rpp_nn_load_i8pln1, srcPtrTempG, srcLocCF, pRow[1]);
                    rpp_simd_load(rpp_nn_load_i8pln1, srcPtrTempB, srcLocCF, pRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempR + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempG + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempB + srcLocationColumnFloor);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    __m128i pRow;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset, true);
                    rpp_simd_load(rpp_nn_load_i8pkd3, srcPtrTemp, srcLocCF, pRow);
                    rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTemp, pRow);
                    dstPtrTemp += 12;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumnFloor);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumnFloor + 1);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumnFloor + 2);
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, weightParams, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                pDstLoc = pDstLocInit;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 4)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocCF, pWeightParams, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_nn_load_i8pln1, srcPtrTempChn, srcLocCF, pRow);
                        rpp_simd_store(rpp_store4_i8_to_i8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp+=4;
                }
                for ( ;vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, weightParams, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8s)*(srcPtrTempChn + srcLocationColumnFloor);
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
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roiPtr->xywhROI.xy.y * srcDescPtr->strides.hStride) + (roiPtr->xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
        Rpp32s widthLimitChanneled = widthLimit * 3;
        Rpp32f weightParams[2];
        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;
        Rpp32s srcLocCF[4] = {0};

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if (srcDescPtr->c == 3)
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
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, &weightParams[0], hOffset);
                srcLocationRowFloor = srcLocationRowFloor * srcDescPtr->strides.hStride;
                srcPtrTempR = srcPtrRowR + srcLocationRowFloor;
                srcPtrTempG = srcPtrRowG + srcLocationRowFloor;
                srcPtrTempB = srcPtrRowB + srcLocationRowFloor;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, &weightParams[0], wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR = (Rpp16f)*(srcPtrTempR + srcLocationColumnFloor);
                    *dstPtrTempG = (Rpp16f)*(srcPtrTempG + srcLocationColumnFloor);
                    *dstPtrTempB = (Rpp16f)*(srcPtrTempB + srcLocationColumnFloor);
                    dstPtrTempR = dstPtrTempR + dstDescPtr->strides.wStride;
                    dstPtrTempG = dstPtrTempG + dstDescPtr->strides.wStride;
                    dstPtrTempB = dstPtrTempB + dstDescPtr->strides.wStride;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_src_loc(i, hRatio, heightLimit, srcLocationRowFloor, &weightParams[0], hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRowFloor * srcDescPtr->strides.hStride;
                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumnFloor, &weightParams[0], wOffset);
                    *dstPtrTemp++ = (Rpp16f)*(srcPtrTemp + srcLocationColumnFloor);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}