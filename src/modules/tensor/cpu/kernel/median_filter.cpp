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

        int height = roi.xywhROI.roiHeight;
        int width = roi.xywhROI.roiWidth;
        int channels = srcDescPtr->c;

        if((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for(int c = 0; c < channels; c++)
            {
                T *dstPtrRow;
                dstPtrRow = dstPtrChannel;
                for(int i = 0; i < height; i++)
                {
                    T *dstPtrTemp;
                    dstPtrTemp = dstPtrRow;
                    for(int j = 0; j < width; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, height, width, 1, srcDescPtr, dstDescPtr);
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
            T *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < height; i++)
            {
                T *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                for (int j = 0; j < width; j++)
                {
                    median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, height, width, channels, srcDescPtr, dstDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            for (int i = 0; i < height; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                for (int j = 0; j < width; j++)
                {
                    T *dstPtrTempChn = dstPtrTemp;
                    T *srcPtrTempChn = srcPtrChannel;
                    for (int c = 0; c < channels; c++)
                    {
                        median_filter_generic_tensor(srcPtrTempChn, dstPtrTempChn, i, j, kernelSize, height, width, 1, srcDescPtr, dstDescPtr);
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
            for (int c = 0; c < channels; c++)
            {
                T *dstPtrRow;
                dstPtrRow = dstPtrChannel;
                for (int i = 0; i < height; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for (int j = 0; j < width; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, height, width, 1, srcDescPtr, dstDescPtr);
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
