#include "rppdefs.h"
#include "cpu/rpp_cpu_simd.hpp"
#include "cpu/rpp_cpu_common.hpp"

RppStatus resize_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8u *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSizes,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
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

        compute_dst_size_cap_host(&dstImgSizes[batchCount], dstDescPtr);     // Check if the dstImgSize exceeds dst buffer size
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSizes[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSizes[batchCount].height));
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
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSizes[batchCount].width & ~7;   // Align dst width to process 8 dst pixels per iteration
        __m256 pWRatio = _mm256_set1_ps(wRatio);
        __m256 pWOffset = _mm256_set1_ps(wOffset);
        __m256 pWidthLimit = _mm256_set1_ps((float)widthLimit);
        __m256 pWeightParams[noOfCoeffs], pBilinearCoeffs[noOfCoeffs], pDstLoc;
        Rpp32f weightParams[noOfCoeffs], bilinearCoeffs[noOfCoeffs];
        Rpp32s srcLocationColumnArray[8] = {0};     // Since 8 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32f invStdDev = 1.0 / stdDev;
        Rpp32u mirrorFlag = mirrorTensor[batchCount];

        __m256 pRMNParams[2];
        pRMNParams[0] = _mm256_set1_ps(mean);
        pRMNParams[1] = _mm256_set1_ps(invStdDev);

        // Resize Mirror Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < dstImgSizes[batchCount].height; dstLocRow++)
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
                    compute_rmn_12_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSizes[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, dstPtrTempR, dstPtrTempG, dstPtrTempB);   // Compute Bilinear interpolation
                    *dstPtrTempR = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*dstPtrTempR - mean) * invStdDev)));
                    *dstPtrTempG = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*dstPtrTempG - mean) * invStdDev)));
                    *dstPtrTempB = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*dstPtrTempB - mean) * invStdDev)));
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < dstImgSizes[batchCount].height; dstLocRow++)
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
                    compute_rmn_12_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);    // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSizes[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pln(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (dstPtrTemp[0] - mean) * invStdDev)));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (dstPtrTemp[1]- mean) * invStdDev)));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (dstPtrTemp[2] - mean) * invStdDev)));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSizes[batchCount].height; dstLocRow++)
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
                    compute_rmn_12_host(pDst, pRMNParams);
                    rpp_simd_store(rpp_store12_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst);   // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSizes[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset, srcDescPtr->strides.wStride); // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    compute_bilinear_interpolation_3c_pkd(srcRowPtrsForInterp, srcLocationColumn, bilinearCoeffs, &dstPtrTemp[0], &dstPtrTemp[1], &dstPtrTemp[2]);  // Compute Bilinear interpolation
                    dstPtrTemp[0] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (dstPtrTemp[0] - mean) * invStdDev)));
                    dstPtrTemp[1] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (dstPtrTemp[1]- mean) * invStdDev)));
                    dstPtrTemp[2] = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (dstPtrTemp[2] - mean) * invStdDev)));
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize Mirror Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for(int dstLocRow = 0; dstLocRow < dstImgSizes[batchCount].height; dstLocRow++)
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
                        compute_rmn_4_host(&pDst, pRMNParams);
                        rpp_simd_store(rpp_store4_f32pln1_to_u8pln1_avx, dstPtrTempChn, pDst);  // Store dst pixels
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSizes[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, &weightParams[2], wOffset);  // Compute the src col location correspoding to the dst col location
                    compute_bilinear_coefficients(weightParams, bilinearCoeffs);    // Compute Bilinear coefficients
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        compute_bilinear_interpolation_1c(&srcRowPtrsForInterp[c * kernelSize], srcLocationColumn, bilinearCoeffs, dstPtrTempChn);  // Compute Bilinear interpolation
                        *dstPtrTempChn = (Rpp8u) RPPPIXELCHECK((((Rpp32f) (*dstPtrTempChn - mean) * invStdDev)));
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


RppStatus resize_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp32f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSizes,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams layoutParams)
{
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

//     omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         Rpp32u horizontalFlag = horizontalTensor[batchCount];
//         Rpp32u verticalFlag = verticalTensor[batchCount];
        
//         Rpp32f *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

//         Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
//         Rpp32f *srcPtrChannel, *dstPtrChannel;
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = (bufferLength / 24) * 24;
//         Rpp32u vectorIncrement = 24;
//         Rpp32u vectorIncrementPerChannel = 8;
//         auto load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_avx;
//         auto load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_avx;
//         auto load8Fn = &rpp_load8_f32_to_f32_avx;

//         if(horizontalFlag == 0 && verticalFlag == 0)
//         {
//             srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x  * layoutParams.bufferMultiplier);
//         }
//         else if(horizontalFlag == 1 && verticalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
//             load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
//             load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
//             load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
//         }
//         else if(horizontalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
//             load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
//             load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
//             load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
//         }
//         else if(verticalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
//         }
        
//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
//         if ((horizontalFlag == 0) && (verticalFlag == 0) && (srcDescPtr->layout == dstDescPtr->layout))
//         {
//             Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp32f);
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp32f *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }

//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }

//         // flip with fused output-layout toggle (NHWC -> NCHW)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[6];
//                     srcPtrTemp -=  (vectorIncrement * horizontalFlag);
                    
//                     rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);     // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    
//                     srcPtrTemp += (vectorIncrement * (1 - horizontalFlag));
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     srcPtrTemp -=  (3 * horizontalFlag);
//                     *dstPtrTempR = srcPtrTemp[0];
//                     *dstPtrTempG = srcPtrTemp[1];
//                     *dstPtrTempB = srcPtrTemp[2];

//                     srcPtrTemp += (3 * (1 - horizontalFlag));
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }

//                 if(verticalFlag)
//                     srcPtrRow -= srcDescPtr->strides.hStride;
//                 else
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
//             srcPtrRowR = srcPtrChannel;
//             srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
//             srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
//                 srcPtrTempR = srcPtrRowR;
//                 srcPtrTempG = srcPtrRowG;
//                 srcPtrTempB = srcPtrRowB;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 p[6];
//                     srcPtrTempR -= (vectorIncrementPerChannel * horizontalFlag);
//                     srcPtrTempG -= (vectorIncrementPerChannel * horizontalFlag);
//                     srcPtrTempB -= (vectorIncrementPerChannel * horizontalFlag);

//                     rpp_simd_load(load24FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

//                     srcPtrTempR += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     srcPtrTempG += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     srcPtrTempB += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     srcPtrTempR -= (horizontalFlag);
//                     srcPtrTempG -= (horizontalFlag);
//                     srcPtrTempB -= (horizontalFlag);

//                     dstPtrTemp[0] = *srcPtrTempR;
//                     dstPtrTemp[1] = *srcPtrTempG;
//                     dstPtrTemp[2] = *srcPtrTempB;

//                     srcPtrTempR += ((1 - horizontalFlag));
//                     srcPtrTempG += ((1 - horizontalFlag));
//                     srcPtrTempB += ((1 - horizontalFlag));
//                     dstPtrTemp += 3;
//                 }

//                 if(verticalFlag)
//                 {
//                     srcPtrRowR -= srcDescPtr->strides.hStride;
//                     srcPtrRowG -= srcDescPtr->strides.hStride;
//                     srcPtrRowB -= srcDescPtr->strides.hStride;
//                 }
//                 else
//                 {
//                     srcPtrRowR += srcDescPtr->strides.hStride;
//                     srcPtrRowG += srcDescPtr->strides.hStride;
//                     srcPtrRowB += srcDescPtr->strides.hStride;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp32f *srcPtrRow, *dstPtrRow;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp32f *srcPtrTemp, *dstPtrTemp;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[6];
//                     srcPtrTemp -= (vectorIncrement * horizontalFlag);

//                     rpp_simd_load(load24FnPkdPln, srcPtrTemp, p);    // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

//                     srcPtrTemp += (vectorIncrement * (1 - horizontalFlag));
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     srcPtrTemp -= (3 * horizontalFlag);

//                     dstPtrTemp[0] = srcPtrTemp[0];
//                     dstPtrTemp[1] = srcPtrTemp[1];
//                     dstPtrTemp[2] = srcPtrTemp[2];

//                     srcPtrTemp += (3 * (1 - horizontalFlag));
//                     dstPtrTemp += 3;
//                 }

//                 if(verticalFlag)
//                 {
//                     srcPtrRow -= srcDescPtr->strides.hStride;
//                 }
//                 else
//                 {
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
//         else
//         {
//             Rpp32u alignedLength = (bufferLength / 8) * 8;
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp32f *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     Rpp32f *srcPtrTemp, *dstPtrTemp;
//                     srcPtrTemp = srcPtrRow;
//                     dstPtrTemp = dstPtrRow;

//                     int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                     {
//                         __m256 p[2];

//                         srcPtrTemp -= (vectorIncrementPerChannel * horizontalFlag);

//                         rpp_simd_load(load8Fn, srcPtrTemp, p);    // simd loads
//                         rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

//                         srcPtrTemp += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                         dstPtrTemp += vectorIncrementPerChannel;
//                     }
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         srcPtrTemp -= horizontalFlag;

//                         *dstPtrTemp = *srcPtrTemp;

//                         srcPtrTemp += (1 - horizontalFlag);
//                         dstPtrTemp++;
//                     }
//                     if(verticalFlag)
//                         srcPtrRow -= srcDescPtr->strides.hStride;
//                     else
//                         srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }
//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }
//     }

    return RPP_SUCCESS;
}

RppStatus resize_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp16f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSizes,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams layoutParams)
{
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

//     omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         Rpp32u horizontalFlag = horizontalTensor[batchCount];
//         Rpp32u verticalFlag = verticalTensor[batchCount];
        
//         Rpp16f *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

//         Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
//         Rpp16f *srcPtrChannel, *dstPtrChannel;
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = (bufferLength / 24) * 24;
//         Rpp32u vectorIncrement = 24;
//         Rpp32u vectorIncrementPerChannel = 8;
//         auto load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_avx;
//         auto load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_avx;
//         auto load8Fn = &rpp_load8_f32_to_f32_avx;

//         if(horizontalFlag == 0 && verticalFlag == 0)
//         {
//             srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x  * layoutParams.bufferMultiplier);
//         }
//         else if(horizontalFlag == 1 && verticalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
//             load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
//             load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
//             load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
//         }
//         else if(horizontalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
//             load24FnPkdPln = &rpp_load24_f32pkd3_to_f32pln3_mirror_avx;
//             load24FnPlnPln = &rpp_load24_f32pln3_to_f32pln3_mirror_avx;
//             load8Fn = &rpp_load8_f32_to_f32_mirror_avx;
//         }
//         else if(verticalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
//         }
        
//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
//         if ((horizontalFlag == 0) && (verticalFlag == 0) && (srcDescPtr->layout == dstDescPtr->layout))
//         {
//             Rpp32u copyLengthInBytes = bufferLength * sizeof(Rpp16f);
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp16f *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }

//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }

//         // flip with fused output-layout toggle (NHWC -> NCHW)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[6];
//                     srcPtrTemp -=  (vectorIncrement * horizontalFlag);

//                     Rpp32f srcPtrTemp_ps[24];
//                     Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
//                     for(int cnt = 0; cnt < vectorIncrement; cnt++)
//                         srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

//                     rpp_simd_load(load24FnPkdPln, srcPtrTemp_ps, p);     // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
                    
//                     for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                     {
//                         dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
//                         dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
//                         dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
//                     }

//                     srcPtrTemp += (vectorIncrement * (1 - horizontalFlag));
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     srcPtrTemp -=  (3 * horizontalFlag);
//                     *dstPtrTempR = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
//                     *dstPtrTempG = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
//                     *dstPtrTempB = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);

//                     srcPtrTemp += (3 * (1 - horizontalFlag));
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }

//                 if(verticalFlag)
//                     srcPtrRow -= srcDescPtr->strides.hStride;
//                 else
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
//             srcPtrRowR = srcPtrChannel;
//             srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
//             srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
//                 srcPtrTempR = srcPtrRowR;
//                 srcPtrTempG = srcPtrRowG;
//                 srcPtrTempB = srcPtrRowB;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 p[6];
//                     srcPtrTempR -= (vectorIncrementPerChannel * horizontalFlag);
//                     srcPtrTempG -= (vectorIncrementPerChannel * horizontalFlag);
//                     srcPtrTempB -= (vectorIncrementPerChannel * horizontalFlag);

//                     Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
//                     Rpp32f dstPtrTemp_ps[25];
//                     for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                     {
//                         srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
//                         srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
//                         srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
//                     }

//                     rpp_simd_load(load24FnPlnPln, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

//                     for(int cnt = 0; cnt < vectorIncrement; cnt++)
//                         dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
//                     srcPtrTempR += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     srcPtrTempG += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     srcPtrTempB += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     srcPtrTempR -= (horizontalFlag);
//                     srcPtrTempG -= (horizontalFlag);
//                     srcPtrTempB -= (horizontalFlag);

//                     dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempR);
//                     dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempG);
//                     dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTempB);

//                     srcPtrTempR += ((1 - horizontalFlag));
//                     srcPtrTempG += ((1 - horizontalFlag));
//                     srcPtrTempB += ((1 - horizontalFlag));
//                     dstPtrTemp += 3;
//                 }

//                 if(verticalFlag)
//                 {
//                     srcPtrRowR -= srcDescPtr->strides.hStride;
//                     srcPtrRowG -= srcDescPtr->strides.hStride;
//                     srcPtrRowB -= srcDescPtr->strides.hStride;
//                 }
//                 else
//                 {
//                     srcPtrRowR += srcDescPtr->strides.hStride;
//                     srcPtrRowG += srcDescPtr->strides.hStride;
//                     srcPtrRowB += srcDescPtr->strides.hStride;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp16f *srcPtrRow, *dstPtrRow;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp16f *srcPtrTemp, *dstPtrTemp;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[6];
//                     srcPtrTemp -= (vectorIncrement * horizontalFlag);

//                     Rpp32f srcPtrTemp_ps[24], dstPtrTemp_ps[25];
//                     for(int cnt = 0; cnt < vectorIncrement; cnt++)
//                         srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

//                     rpp_simd_load(load24FnPkdPln, srcPtrTemp_ps, p);    // simd loads
//                     rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores

//                     for(int cnt = 0; cnt < vectorIncrement; cnt++)
//                         dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];                    
//                     srcPtrTemp += (vectorIncrement * (1 - horizontalFlag));
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     srcPtrTemp -= (3 * horizontalFlag);

//                     dstPtrTemp[0] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[0]);
//                     dstPtrTemp[1] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[1]);
//                     dstPtrTemp[2] = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)srcPtrTemp[2]);

//                     srcPtrTemp += (3 * (1 - horizontalFlag));
//                     dstPtrTemp += 3;
//                 }

//                 if(verticalFlag)
//                 {
//                     srcPtrRow -= srcDescPtr->strides.hStride;
//                 }
//                 else
//                 {
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
//         else
//         {
//             Rpp32u alignedLength = (bufferLength / 8) * 8;
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp16f *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     Rpp16f *srcPtrTemp, *dstPtrTemp;
//                     srcPtrTemp = srcPtrRow;
//                     dstPtrTemp = dstPtrRow;

//                     int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                     {
//                         __m256 p[2];
//                         srcPtrTemp -= (vectorIncrementPerChannel * horizontalFlag);

//                         Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
//                         for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                             srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

//                         rpp_simd_load(load8Fn, srcPtrTemp_ps, p);    // simd loads
//                         rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores

//                         for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
//                             dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];

//                         srcPtrTemp += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                         dstPtrTemp += vectorIncrementPerChannel;
//                     }
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         srcPtrTemp -= horizontalFlag;

//                         *dstPtrTemp = (Rpp16f) RPPPIXELCHECKF32((Rpp32f)*srcPtrTemp);

//                         srcPtrTemp += (1 - horizontalFlag);
//                         dstPtrTemp++;
//                     }
//                     if(verticalFlag)
//                         srcPtrRow -= srcDescPtr->strides.hStride;
//                     else
//                         srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }
//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }
//     }

    return RPP_SUCCESS;
}

RppStatus resize_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8s *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSizes,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams)
{
//     RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

//     omp_set_dynamic(0);
// #pragma omp parallel for num_threads(dstDescPtr->n)
//     for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
//     {
//         RpptROI roi;
//         RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
//         compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

//         Rpp32u horizontalFlag = horizontalTensor[batchCount];
//         Rpp32u verticalFlag = verticalTensor[batchCount];
        
//         Rpp8s *srcPtrImage, *dstPtrImage;
//         srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
//         dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

//         Rpp32s bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
//         Rpp8s *srcPtrChannel, *dstPtrChannel;
//         dstPtrChannel = dstPtrImage;

//         Rpp32u alignedLength = (bufferLength / 48) * 48;
//         Rpp32u vectorIncrement = 48;
//         Rpp32u vectorIncrementPerChannel = 16;
//         auto load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_avx;
//         auto load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_avx;
//         auto load16Fn = &rpp_load16_i8_to_f32_avx;

//         if(horizontalFlag == 0 && verticalFlag == 0)
//         {
//             srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x  * layoutParams.bufferMultiplier);
//         }
//         else if(horizontalFlag == 1 && verticalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
//             load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_mirror_avx;
//             load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_mirror_avx;
//             load16Fn = &rpp_load16_i8_to_f32_mirror_avx;
//         }
//         else if(horizontalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + ((roi.xywhROI.xy.x + roi.xywhROI.roiWidth) * layoutParams.bufferMultiplier);
//             load48FnPkdPln = &rpp_load48_i8pkd3_to_f32pln3_mirror_avx;
//             load48FnPlnPln = &rpp_load48_i8pln3_to_f32pln3_mirror_avx;
//             load16Fn = &rpp_load16_i8_to_f32_mirror_avx;
//         }
//         else if(verticalFlag == 1)
//         {
//             srcPtrChannel = srcPtrImage + ((roi.xywhROI.xy.y + roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
//         }
        
//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW and  horizontalflag = 0 and verticalflag = 0)
//         if ((horizontalFlag == 0) && (verticalFlag == 0) && (srcDescPtr->layout == dstDescPtr->layout))
//         {
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp8s *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     memcpy(dstPtrRow, srcPtrRow, bufferLength);
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }

//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }

//         // flip with fused output-layout toggle (NHWC -> NCHW)
//         if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
//         {
//             Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRowR = dstPtrChannel;
//             dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
//             dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTempR = dstPtrRowR;
//                 dstPtrTempG = dstPtrRowG;
//                 dstPtrTempB = dstPtrRowB;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[6];
//                     srcPtrTemp -=  (vectorIncrement * horizontalFlag);
                    
//                     rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);     // simd loads
//                     rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    
//                     srcPtrTemp += (vectorIncrement * (1 - horizontalFlag));
//                     dstPtrTempR += vectorIncrementPerChannel;
//                     dstPtrTempG += vectorIncrementPerChannel;
//                     dstPtrTempB += vectorIncrementPerChannel;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     srcPtrTemp -=  (3 * horizontalFlag);
//                     *dstPtrTempR = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[0]));
//                     *dstPtrTempG = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[1]));
//                     *dstPtrTempB = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[2]));

//                     srcPtrTemp += (3 * (1 - horizontalFlag));
//                     dstPtrTempR++;
//                     dstPtrTempG++;
//                     dstPtrTempB++;
//                 }

//                 if(verticalFlag)
//                     srcPtrRow -= srcDescPtr->strides.hStride;
//                 else
//                     srcPtrRow += srcDescPtr->strides.hStride;
//                 dstPtrRowR += dstDescPtr->strides.hStride;
//                 dstPtrRowG += dstDescPtr->strides.hStride;
//                 dstPtrRowB += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip with fused output-layout toggle (NCHW -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
//             srcPtrRowR = srcPtrChannel;
//             srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
//             srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
//                 srcPtrTempR = srcPtrRowR;
//                 srcPtrTempG = srcPtrRowG;
//                 srcPtrTempB = srcPtrRowB;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                 {
//                     __m256 p[6];
//                     srcPtrTempR -= (vectorIncrementPerChannel * horizontalFlag);
//                     srcPtrTempG -= (vectorIncrementPerChannel * horizontalFlag);
//                     srcPtrTempB -= (vectorIncrementPerChannel * horizontalFlag);

//                     rpp_simd_load(load48FnPlnPln, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
//                     rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

//                     srcPtrTempR += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     srcPtrTempG += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     srcPtrTempB += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                 {
//                     srcPtrTempR -= (horizontalFlag);
//                     srcPtrTempG -= (horizontalFlag);
//                     srcPtrTempB -= (horizontalFlag);

//                     dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempR));
//                     dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempG));
//                     dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTempB));

//                     srcPtrTempR += ((1 - horizontalFlag));
//                     srcPtrTempG += ((1 - horizontalFlag));
//                     srcPtrTempB += ((1 - horizontalFlag));
//                     dstPtrTemp += 3;
//                 }

//                 if(verticalFlag)
//                 {
//                     srcPtrRowR -= srcDescPtr->strides.hStride;
//                     srcPtrRowG -= srcDescPtr->strides.hStride;
//                     srcPtrRowB -= srcDescPtr->strides.hStride;
//                 }
//                 else
//                 {
//                     srcPtrRowR += srcDescPtr->strides.hStride;
//                     srcPtrRowG += srcDescPtr->strides.hStride;
//                     srcPtrRowB += srcDescPtr->strides.hStride;
//                 }
//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC)
//         else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
//         {
//             Rpp8s *srcPtrRow, *dstPtrRow;
//             srcPtrRow = srcPtrChannel;
//             dstPtrRow = dstPtrChannel;

//             for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//             {
//                 Rpp8s *srcPtrTemp, *dstPtrTemp;
//                 srcPtrTemp = srcPtrRow;
//                 dstPtrTemp = dstPtrRow;

//                 int vectorLoopCount = 0;
//                 for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
//                 {
//                     __m256 p[6];
//                     srcPtrTemp -= (vectorIncrement * horizontalFlag);

//                     rpp_simd_load(load48FnPkdPln, srcPtrTemp, p);    // simd loads
//                     rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores

//                     srcPtrTemp += (vectorIncrement * (1 - horizontalFlag));
//                     dstPtrTemp += vectorIncrement;
//                 }
//                 for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
//                 {
//                     srcPtrTemp -= (3 * horizontalFlag);

//                     dstPtrTemp[0] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[0]));
//                     dstPtrTemp[1] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[1]));
//                     dstPtrTemp[2] = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (srcPtrTemp[2]));

//                     srcPtrTemp += (3 * (1 - horizontalFlag));
//                     dstPtrTemp += 3;
//                 }

//                 if(verticalFlag)
//                     srcPtrRow -= srcDescPtr->strides.hStride;
//                 else
//                     srcPtrRow += srcDescPtr->strides.hStride;

//                 dstPtrRow += dstDescPtr->strides.hStride;
//             }
//         }

//         // flip without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
//         else
//         {
//             Rpp32u alignedLength = (bufferLength / 16) * 16;
//             for(int c = 0; c < layoutParams.channelParam; c++)
//             {
//                 Rpp8s *srcPtrRow, *dstPtrRow;
//                 srcPtrRow = srcPtrChannel;
//                 dstPtrRow = dstPtrChannel;

//                 for(int i = 0; i < roi.xywhROI.roiHeight; i++)
//                 {
//                     Rpp8s *srcPtrTemp, *dstPtrTemp;
//                     srcPtrTemp = srcPtrRow;
//                     dstPtrTemp = dstPtrRow;

//                     int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
//                     {
//                         __m256 p[2];

//                         srcPtrTemp -= (vectorIncrementPerChannel * horizontalFlag);

//                         rpp_simd_load(load16Fn, srcPtrTemp, p);    // simd loads
//                         rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores

//                         srcPtrTemp += (vectorIncrementPerChannel * (1 - horizontalFlag));
//                         dstPtrTemp += vectorIncrementPerChannel;
//                     }
//                     for (; vectorLoopCount < bufferLength; vectorLoopCount++)
//                     {
//                         srcPtrTemp -= horizontalFlag;
//                         *dstPtrTemp = (Rpp8s) RPPPIXELCHECKI8((Rpp32f) (*srcPtrTemp));

//                         srcPtrTemp += (1 - horizontalFlag);
//                         dstPtrTemp++;
//                     }
//                     if(verticalFlag)
//                         srcPtrRow -= srcDescPtr->strides.hStride;
//                     else
//                         srcPtrRow += srcDescPtr->strides.hStride;
//                     dstPtrRow += dstDescPtr->strides.hStride;
//                 }
//                 srcPtrChannel += srcDescPtr->strides.cStride;
//                 dstPtrChannel += dstDescPtr->strides.cStride;
//             }
//         }
//     }

    return RPP_SUCCESS;
}