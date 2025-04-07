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

// handle nearest-neighbor padding
template<typename T>
inline void apply_nn_padding(T *srcPtrTemp, T *blockData, int kernelSize, int rowIdx, int colIdx, int height, int width, int channels,  RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    int padLength = kernelSize / 2;
    int index = 0;
    
    for (int i = -padLength; i <= padLength; i++)
    {
        for (int j = -padLength; j <= padLength; j++)
        {
            int row = std::max(0, std::min(rowIdx + i, height - 1));
            int col = std::max(0, std::min(colIdx + j, width - 1));

            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;
            for (int ch = 0; ch < channels; ch++)
                blockData[index++] = srcPtrTemp[srcIdx + ch];
        }
    }
}

// Generic median filter function
template<typename T>
inline void median_filter_generic_tensor(T *srcPtrTemp, T *dstPtrTemp, int rowIdx, int colIdx, int kernelSize, int height, int width, int channels, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    T blockData[kernelSize * kernelSize * channels];
    apply_nn_padding(srcPtrTemp, blockData, kernelSize, rowIdx, colIdx, height, width, channels, srcDescPtr, dstDescPtr);
    
    for (int ch = 0; ch < channels; ch++)
    {
        T channelBlock[kernelSize * kernelSize];
        for (int i = 0; i < kernelSize * kernelSize; i++)
        {
            channelBlock[i] = blockData[i * channels + ch];
        }
        std::sort(channelBlock, channelBlock + (kernelSize * kernelSize));
        dstPtrTemp[ch] = channelBlock[((kernelSize * kernelSize) / 2)];
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
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
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

        if((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for(int c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for(int j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth, 1, srcDescPtr, dstDescPtr);
                        dstPtrTemp++;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += srcDescPtr->strides.cStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                for (int j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth, srcDescPtr->c, srcDescPtr, dstDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow = dstPtrChannel;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                for (int j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    T *dstPtrTempChn = dstPtrTemp;
                    T *srcPtrTempChn = srcPtrChannel;
                    for (int c = 0; c < srcDescPtr->c; c++)
                    {
                        median_filter_generic_tensor(srcPtrTempChn, dstPtrTempChn, i, j, kernelSize, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth, 1, srcDescPtr, dstDescPtr);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn++;
                    }
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (int c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for (int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for (int j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth, 1, srcDescPtr, dstDescPtr);
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
