#include "rppdefs.h"
#include "rpp_cpu_common.hpp"
#include "rpp_cpu_filter.hpp"
#include <algorithm>

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
                blockData[index++] = srcPtrTemp[srcIdx];
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
            // Rpp32u widthPkd = width * channels;
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
                dstPtrRow = dstPtrChannel + c;
                for (int i = 0; i < height; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for (int j = 0; j < width; j++)
                    {
                        median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSize, height, width, 1, srcDescPtr, dstDescPtr);
                        dstPtrTemp += 3;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}
