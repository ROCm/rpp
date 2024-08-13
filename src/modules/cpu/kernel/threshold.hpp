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

RppStatus threshold_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *minTensor,
                                      Rpp32f *maxTensor,
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

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
       
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NHWC)
        {
            Rpp32u batchIndex = batchCount * 3;
            Rpp32f channelMinThreshold[3] = { minTensor[batchIndex], minTensor[batchIndex + 1], minTensor[batchIndex + 2] };
            Rpp32f channelMaxThreshold[3] = { maxTensor[batchIndex], maxTensor[batchIndex + 1], maxTensor[batchIndex + 2] };
#if __AVX2__
            __m256 pThresholdParams[6];
            pThresholdParams[0] = _mm256_set1_ps(channelMinThreshold[0]);
            pThresholdParams[1] = _mm256_set1_ps(channelMaxThreshold[0]);
            pThresholdParams[2] = _mm256_set1_ps(channelMinThreshold[1]);
            pThresholdParams[3] = _mm256_set1_ps(channelMaxThreshold[1]);
            pThresholdParams[4] = _mm256_set1_ps(channelMinThreshold[2]);
            pThresholdParams[5] = _mm256_set1_ps(channelMaxThreshold[2]);
#endif
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p); // simd loads
                    compute_color_48_to_threshold_48_host(p, pThresholdParams); 	// threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);       // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(srcPtrTemp[0]);
                    pixelG = static_cast<Rpp32f>(srcPtrTemp[1]);
                    pixelB = static_cast<Rpp32f>(srcPtrTemp[2]);
                    channelCheck[0] = ((pixelR >= channelMinThreshold[0]) &&  (pixelR <= channelMaxThreshold[0]));
                    channelCheck[1] = ((pixelG >= channelMinThreshold[1]) &&  (pixelG <= channelMaxThreshold[1]));
                    channelCheck[2] = ((pixelB >= channelMinThreshold[2]) &&  (pixelB <= channelMaxThreshold[2]));
                    dstPtrTemp[0] = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 255 : 0;
                    dstPtrTemp[1] = dstPtrTemp[0];
                    dstPtrTemp[2] = dstPtrTemp[0];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u batchIndex = batchCount * 3;
            Rpp32f channelMinThreshold[3] = { minTensor[batchIndex], minTensor[batchIndex + 1], minTensor[batchIndex + 2] };
            Rpp32f channelMaxThreshold[3] = { maxTensor[batchIndex], maxTensor[batchIndex + 1], maxTensor[batchIndex + 2] };
#if __AVX2__
            __m256 pThresholdParams[6];
            pThresholdParams[0] = _mm256_set1_ps(channelMinThreshold[0]);
            pThresholdParams[1] = _mm256_set1_ps(channelMaxThreshold[0]);
            pThresholdParams[2] = _mm256_set1_ps(channelMinThreshold[1]);
            pThresholdParams[3] = _mm256_set1_ps(channelMaxThreshold[1]);
            pThresholdParams[4] = _mm256_set1_ps(channelMinThreshold[2]);
            pThresholdParams[5] = _mm256_set1_ps(channelMaxThreshold[2]);
#endif
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + srcDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + srcDescPtr->strides.cStride;
            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_color_48_to_threshold_48_host(p, pThresholdParams);                                 // threshold adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);                                   // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif 
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixelR, pixelG, pixelB;
                    bool channelCheck[3];
                    pixelR = static_cast<Rpp32f>(*srcPtrTempR);
                    pixelG = static_cast<Rpp32f>(*srcPtrTempG);
                    pixelB = static_cast<Rpp32f>(*srcPtrTempB);
                    channelCheck[0] = ((pixelR >= channelMinThreshold[0]) &&  (pixelR <= channelMaxThreshold[0]));
                    channelCheck[1] = ((pixelG >= channelMinThreshold[1]) &&  (pixelG <= channelMaxThreshold[1]));
                    channelCheck[2] = ((pixelB >= channelMinThreshold[2]) &&  (pixelB <= channelMaxThreshold[2]));
                    *dstPtrTempR = (channelCheck[0] && channelCheck[1] && channelCheck[2]) ? 255 : 0;
                    *dstPtrTempG = *dstPtrTempR;
                    *dstPtrTempB = *dstPtrTempR; 
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f minThreshold, maxThreshold;
            minThreshold = minTensor[batchCount];
            maxThreshold = maxTensor[batchCount];
#if __AVX2__
            __m256 pThresholdParams[2];
            pThresholdParams[0] = _mm256_set1_ps(minThreshold);
            pThresholdParams[1] = _mm256_set1_ps(maxThreshold);
#endif

            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);      // simd loads
                    compute_threshold_16_host(p, pThresholdParams);              // threshold adjustment
                    rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f pixel = static_cast<Rpp32f>(*srcPtrTemp);
                    *dstPtrTemp = (pixel < minThreshold) ? 0 : ((pixel > maxThreshold) ? 0 : 255);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}