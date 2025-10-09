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

/* median filter algorithm explanation for U8 PLN1 3x3 kernel size variant
Let’s consider a 3x32 image input:

x  x  x  x  x  x  x  x  x  x  ..  x  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  x  x  x  x  x  x  x  x  x  ..  x  x

padLength = 1 (kernelSize / 2)

Below steps are followed for computing each output pixel in the ROI:
1. For each pixel location (i, j), collect a 3x3 neighborhood of pixels centered at (i, j)
   - Apply nearest-neighbor padding at borders
   - Extract values into a temporary array of 9 elements

2. Sort the 9 values:
   e.g., for 3x3 window: [2, 4, 3, 1, 5, 6, 3, 7, 2] → sorted → [1, 2, 2, 3, 3, 4, 5, 6, 7]

3. Pick the median (middle) value:
   - median = element at index 4 (zero-based), i.e., value 3 in the above example

4. Assign this median to the output pixel at (i, j)

This process is repeated for each pixel in the ROI.
- For single-channel (PLN1), apply per pixel.
- For multi-channel, median is computed independently per channel.

Note: Unlike box filter, there is no arithmetic averaging or SIMD optimization here due to sorting-based computation.
*/

// Generic median filter implementation
template<typename T>
inline void median_filter_generic_tensor(T *srcPtrTemp, T *dstPtrTemp, Rpp32s rowIdx, Rpp32s colIdx, Rpp32s kernelSizeSquared, Rpp32s padLength, Rpp32s heightLimit, Rpp32s widthLimit, Rpp32s channels, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
   // Temporary buffer to hold kernel window data for all channels
    T blockData[kernelSizeSquared * channels];
    Rpp32s index = 0, medianIndex = kernelSizeSquared / 2;

    // Fill blockData with padded values from the source image using nearest neighbor padding
    for (Rpp32s i = -padLength; i <= padLength; i++)
    {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -padLength; j <= padLength; j++)
        {
            // Clamp the row and column to image boundaries (nearest-neighbor padding)
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));

            // Compute the index for the pixel in the input tensor
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            // Copy pixel values for all channels
            if (channels == 3)
            {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], 3 * sizeof(T));
                index += 3;
            }
            else if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++)
    {
        // Temporary buffer for the current channel's data in the kernel window
        T channelBlock[kernelSizeSquared];

        // Extract channel data from interleaved blockData
        for (Rpp32s i = 0; i < kernelSizeSquared; i++)
            channelBlock[i] = blockData[i * channels + ch];

        // Sort the data to compute median
        std::nth_element(channelBlock, channelBlock + medianIndex, channelBlock + kernelSizeSquared);
        // Assign the median value to the destination tensor
        dstPtrTemp[ch] = channelBlock[medianIndex];
    }
}

// Host function for median filter
template<typename T>
RppStatus median_filter_generic_host_tensor(T *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            T *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            Rpp32u kernelSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams layoutParams,
                                            rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s kernelSizeSquared = kernelSize * kernelSize;
        Rpp32s padLength = kernelSize / 2;

        if((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for(Rpp32s c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for(Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                        dstPtrTemp++;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow = dstPtrChannel;
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr, dstDescPtr);
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow = dstPtrChannel;
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    T *dstPtrTempChn = dstPtrTemp;
                    T *srcPtrTempChn = srcPtrChannel;
                    for (Rpp32s c = 0; c < srcDescPtr->c; c++)
                    {
                        median_filter_generic_tensor(srcPtrTempChn, dstPtrTempChn, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn++;
                    }
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (Rpp32s c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                        dstPtrTemp ++;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}

template RppStatus median_filter_generic_host_tensor<Rpp8u>(Rpp8u*,
                                                            RpptDescPtr,
                                                            Rpp8u*,
                                                            RpptDescPtr,
                                                            Rpp32u,
                                                            RpptROIPtr,
                                                            RpptRoiType,
                                                            RppLayoutParams,
                                                            rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp8s>(Rpp8s*,
                                                            RpptDescPtr,
                                                            Rpp8s*,
                                                            RpptDescPtr,
                                                            Rpp32u,
                                                            RpptROIPtr,
                                                            RpptRoiType,
                                                            RppLayoutParams,
                                                            rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp32f>(Rpp32f*,
                                                             RpptDescPtr,
                                                             Rpp32f*,
                                                             RpptDescPtr,
                                                             Rpp32u,
                                                             RpptROIPtr,
                                                             RpptRoiType,
                                                             RppLayoutParams,
                                                             rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp16f>(Rpp16f*,
                                                             RpptDescPtr,
                                                             Rpp16f*,
                                                             RpptDescPtr,
                                                             Rpp32u,
                                                             RpptROIPtr,
                                                             RpptRoiType,
                                                             RppLayoutParams,
                                                             rpp::Handle&);
