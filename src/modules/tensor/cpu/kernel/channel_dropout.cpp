/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "host_tensor_executors.hpp"
#include <random>

inline void generate_channel_masks(uint8_t *channelMasks,
                                   Rpp32f *dropProb,
                                   Rpp32u batchSize,
                                   Rpp32u numChannels)
{
    std::mt19937 rng(42); // fixed seed, or std::random_device{}()
    for (int b = 0; b < batchSize; b++)
    {
        std::bernoulli_distribution keepDist(1.0f - dropProb[b]);
        bool anyKept = false;
        int base = b * numChannels;
        for (Rpp32u c = 0; c < numChannels; c++)
        {
            channelMasks[base + c] = keepDist(rng);
            anyKept |= channelMasks[base + c];
        }

        if (!anyKept)
            channelMasks[base + (rng() % numChannels)] = 1;
    }
}

template<typename T>
RppStatus channel_dropout_host_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *dropProb,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Generate channel mask for this batch
    uint8_t *channelMaskHost = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    generate_channel_masks(channelMaskHost, dropProb, dstDescPtr->n, srcDescPtr->c);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        
        uint8_t *maskPtr = channelMaskHost + batchCount * srcDescPtr->c;
        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Channel Dropout with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            T *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    if constexpr (std::is_same<T, Rpp8s>::value)
                    {
                        *dstPtrTempR = maskPtr[0] ? srcPtrTemp[0] : -128;
                        *dstPtrTempG = maskPtr[1] ? srcPtrTemp[1] : -128;
                        *dstPtrTempB = maskPtr[2] ? srcPtrTemp[2] : -128;
                    }
                    else
                    {
                        *dstPtrTempR = maskPtr[0] * srcPtrTemp[0];
                        *dstPtrTempG = maskPtr[1] * srcPtrTemp[1];
                        *dstPtrTempB = maskPtr[2] * srcPtrTemp[2];
                    }

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Channel Dropout with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if constexpr (std::is_same<T, Rpp8s>::value)
                    {
                        dstPtrRow[0] = maskPtr[0] ? *srcPtrTempR : -128;
                        dstPtrRow[1] = maskPtr[1] ? *srcPtrTempG : -128;
                        dstPtrRow[2] = maskPtr[2] ? *srcPtrTempB : -128;
                    }
                    else
                    {
                        dstPtrTemp[0] = maskPtr[0] * *srcPtrTempR;
                        dstPtrTemp[1] = maskPtr[1] * *srcPtrTempG;
                        dstPtrTemp[2] = maskPtr[2] * *srcPtrTempB;
                    }

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Dropout with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            //for better performance Raw C implementation is optimized
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    if constexpr (std::is_same<T, Rpp8s>::value)
                    {
                        dstPtrRow[0] = maskPtr[0] ? srcPtrTemp[0] : -128;
                        dstPtrRow[1] = maskPtr[1] ? srcPtrTemp[1] : -128;
                        dstPtrRow[2] = maskPtr[2] ? srcPtrTemp[2] : -128;
                    }
                    else
                    {
                        dstPtrTemp[0] = maskPtr[0] * srcPtrTemp[0];
                        dstPtrTemp[1] = maskPtr[1] * srcPtrTemp[1];
                        dstPtrTemp[2] = maskPtr[2] * srcPtrTemp[2];
                    }

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Channel Dropout without fused output-layout toggle (NCHW -> NCHW)
        else
        {
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                T *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        if constexpr (std::is_same<T, Rpp8s>::value)
                            *dstPtrTemp = maskPtr[c] ? *srcPtrTemp : -128;
                        else
                            *dstPtrTemp = maskPtr[c] * *srcPtrTemp;
                        
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus channel_dropout_host_tensor<Rpp8u>(Rpp8u*,
                                                      RpptDescPtr,
                                                      Rpp8u*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);
template RppStatus channel_dropout_host_tensor<Rpp32f>(Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       RppLayoutParams,
                                                       rpp::Handle&);
template RppStatus channel_dropout_host_tensor<Rpp16f>(Rpp16f*,
                                                       RpptDescPtr,
                                                       Rpp16f*,
                                                       RpptDescPtr,
                                                       Rpp32f*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       RppLayoutParams,
                                                       rpp::Handle&);
template RppStatus channel_dropout_host_tensor<Rpp8s>(Rpp8s*,
                                                      RpptDescPtr,
                                                      Rpp8s*,
                                                      RpptDescPtr,
                                                      Rpp32f*,
                                                      RpptROIPtr,
                                                      RpptRoiType,
                                                      RppLayoutParams,
                                                      rpp::Handle&);
