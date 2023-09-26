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

RppStatus gaussian_noise_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptGenericDescPtr srcGenericDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptGenericDescPtr dstGenericDescPtr,
                                                   Rpp32f *meanTensor,
                                                   Rpp32f *stdDevTensor,
                                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                   RpptROI3DPtr roiGenericPtrSrc,
                                                   RpptRoi3DType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle)
{
    RpptROI3D roiDefault;
    if(srcGenericDescPtr->layout==RpptLayout::NCDHW)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[4], (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2]};
    else if(srcGenericDescPtr->layout==RpptLayout::NDHWC)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2], (Rpp32s)srcGenericDescPtr->dims[1]};
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32f mean = meanTensor[batchCount];
        Rpp32f stdDev = stdDevTensor[batchCount];
        Rpp32u offset = batchCount * srcGenericDescPtr->strides[0];
        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;
        RpptXorwowStateBoxMuller xorwowState;
#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_avx(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);

        __m256 pGaussianNoiseParams[2];
        pGaussianNoiseParams[0] = _mm256_set1_ps(mean);
        pGaussianNoiseParams[1] = _mm256_set1_ps(stdDev);
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        __m128i pxXorwowStateX[5], pxXorwowStateCounter;
        rpp_host_rng_xorwow_state_offsetted_sse(xorwowInitialStatePtr, xorwowState, offset, pxXorwowStateX, &pxXorwowStateCounter);

        __m128 pGaussianNoiseParams[2];
        pGaussianNoiseParams[0] = _mm_set1_ps(mean);
        pGaussianNoiseParams[1] = _mm_set1_ps(stdDev);
#endif

        // Gaussian Noise without fused output-layout toggle (NHWC -> NHWC)
        if((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        {
            srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);

            Rpp32f *srcPtrDepth, *dstPtrDepth;
            srcPtrDepth = srcPtrChannel;
            dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrDepth;
                dstPtrRow = dstPtrDepth;
                for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
#if __AVX2__
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                                // simd loads
                        compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                              // simd stores
#else
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);                                    // simd loads
                        compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                                  // simd stores
#endif
                        srcPtrTemp += vectorIncrement;
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        dstPtrTemp[0] = compute_gaussian_noise_1_host(srcPtrTemp[0], &xorwowState, mean, stdDev);
                        dstPtrTemp[1] = compute_gaussian_noise_1_host(srcPtrTemp[1], &xorwowState, mean, stdDev);
                        dstPtrTemp[2] = compute_gaussian_noise_1_host(srcPtrTemp[2], &xorwowState, mean, stdDev);

                        srcPtrTemp += 3;
                        dstPtrTemp += 3;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtrDepth += srcGenericDescPtr->strides[1];
                dstPtrDepth += dstGenericDescPtr->strides[1];
            }
        }

        // Gaussian Noise without fused output-layout toggle (NCDHW -> NCDHW)
        else if ((srcGenericDescPtr->dims[1] == 3) && (srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        {
            srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);

            Rpp32f *srcPtrDepthR, *srcPtrDepthG, *srcPtrDepthB, *dstPtrDepthR, *dstPtrDepthG, *dstPtrDepthB;
            srcPtrDepthR = srcPtrChannel;
            srcPtrDepthG = srcPtrDepthR + srcGenericDescPtr->strides[1];
            srcPtrDepthB = srcPtrDepthG + srcGenericDescPtr->strides[1];
            dstPtrDepthR = dstPtrChannel;
            dstPtrDepthG = dstPtrDepthR + dstGenericDescPtr->strides[1];
            dstPtrDepthB = dstPtrDepthG + dstGenericDescPtr->strides[1];
            for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
            {
                Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
                srcPtrRowR = srcPtrDepthR;
                srcPtrRowG = srcPtrDepthG;
                srcPtrRowB = srcPtrDepthB;
                dstPtrRowR = dstPtrDepthR;
                dstPtrRowG = dstPtrDepthG;
                dstPtrRowB = dstPtrDepthB;
                for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                {
                    Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;
                    dstPtrTempR = dstPtrRowR;
                    dstPtrTempG = dstPtrRowG;
                    dstPtrTempB = dstPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
#if __AVX2__
                        __m256 p[3];
                        rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);     // simd loads
                        compute_gaussian_noise_24_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                        rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);   // simd stores
#else
                        __m128 p[4];
                        rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);         // simd loads
                        compute_gaussian_noise_12_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                        rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);       // simd stores
#endif
                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTempR++ = compute_gaussian_noise_1_host(*srcPtrTempR++, &xorwowState, mean, stdDev);
                        *dstPtrTempG++ = compute_gaussian_noise_1_host(*srcPtrTempG++, &xorwowState, mean, stdDev);
                        *dstPtrTempB++ = compute_gaussian_noise_1_host(*srcPtrTempB++, &xorwowState, mean, stdDev);
                    }
                    srcPtrRowR += srcGenericDescPtr->strides[3];
                    srcPtrRowG += srcGenericDescPtr->strides[3];
                    srcPtrRowB += srcGenericDescPtr->strides[3];
                    dstPtrRowR += srcGenericDescPtr->strides[3];
                    dstPtrRowG += srcGenericDescPtr->strides[3];
                    dstPtrRowB += srcGenericDescPtr->strides[3];
                }
                srcPtrDepthR += srcGenericDescPtr->strides[2];
                srcPtrDepthG += srcGenericDescPtr->strides[2];
                srcPtrDepthB += srcGenericDescPtr->strides[2];
                dstPtrDepthR += srcGenericDescPtr->strides[2];
                dstPtrDepthG += srcGenericDescPtr->strides[2];
                dstPtrDepthB += srcGenericDescPtr->strides[2];
            }
        }

        // Gaussian Noise without fused output-layout toggle single channel (NCDHW -> NCDHW)
        else if ((srcGenericDescPtr->dims[1] == 1) && (srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        {
            srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
#if __AVX2__
            alignedLength = bufferLength & ~7;
#else
            alignedLength = bufferLength & ~3;
#endif
            Rpp32u vectorIncrementPerChannelDouble = 2 * vectorIncrementPerChannel;
            Rpp32f *srcPtrDepth, *dstPtrDepth;
            srcPtrDepth = srcPtrChannel;
            dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrDepth;
                dstPtrRow = dstPtrDepth;
                for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannelDouble)
                    {
#if __AVX2__
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_f32_to_f32_avx, srcPtrTemp, p);                                        // simd loads
                        compute_gaussian_noise_16_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams); // gaussian_noise adjustment
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);                                      // simd stores
#else
                        __m128 p[2];
                        rpp_simd_load(rpp_load8_f32_to_f32, srcPtrTemp, p);                                             // simd loads
                        compute_gaussian_noise_8_host(p, pxXorwowStateX, &pxXorwowStateCounter, pGaussianNoiseParams);  // gaussian_noise adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32, dstPtrTemp, p);                                           // simd stores
#endif
                        srcPtrTemp += vectorIncrementPerChannelDouble;
                        dstPtrTemp += vectorIncrementPerChannelDouble;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = compute_gaussian_noise_1_host(*srcPtrTemp++, &xorwowState, mean, stdDev);
                    }
                    srcPtrRow += srcGenericDescPtr->strides[3];
                    dstPtrRow += dstGenericDescPtr->strides[3];
                }
                srcPtrDepth += srcGenericDescPtr->strides[2];
                dstPtrDepth += dstGenericDescPtr->strides[2];
            }
        }
    }

    return RPP_SUCCESS;
}