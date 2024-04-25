#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel --------------------
// Single channel resample support
__global__ void resample_1channel_hip_tensor(float *srcPtr,
                                             float *dstPtr,
                                             float inRateTensor,
                                             float outRateTensor,
                                             int srcLength,
                                             int outEnd,
                                             double scale,
                                             RpptResamplingWindow &window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    int block = 256;
    int outBlock = id_x * block;
    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + block, outEnd);
    double inBlockRaw = outBlock * scale;
    int inBlockRounded = static_cast<int>(inBlockRaw);
    float inPos = inBlockRaw - inBlockRounded;
    float fscale = scale;

    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        int loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);

        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > srcLength)
            loc1 = srcLength - inBlockRounded;
        int locInWindow = loc0;
        float locBegin = locInWindow - inPos;
        float accum = 0.0f;

        for (; locInWindow < loc1; locInWindow++, locBegin++)
        {
            float w = window(locBegin);
            accum += srcPtr[inBlockRounded + locInWindow] * w;
        }
        dstPtr[outPos] = accum;
    }
}

// Generic n channel resample support
__global__ void resample_nchannel_hip_tensor(float *srcPtr,
                                             float *dstPtr,
                                             float inRateTensor,
                                             float outRateTensor,
                                             int srcLength,
                                             int outEnd,
                                             double scale,
                                             RpptResamplingWindow &window,
                                             int numChannels)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    int block = 256;
    int outBlock = id_x * block;
    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + block, outEnd);
    double inBlockRaw = outBlock * scale;
    int inBlockRounded = static_cast<int>(inBlockRaw);
    float inPos = inBlockRaw - inBlockRounded;
    float fscale = scale;
    float tempBuf[3] = {0.0f}; // Considering max channels as 3

    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        int loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);

        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > srcLength)
            loc1 = srcLength - inBlockRounded;
        int locInWindow = loc0;
        float locBegin = locInWindow - inPos;
        int ofs0 = loc0 * numChannels;
        int ofs1 = loc1 * numChannels;

        for (int inOfs = ofs0; inOfs < ofs1; inOfs += numChannels, locBegin++)
        {
            float w = window(locBegin);
            for (int c = 0; c < numChannels; c++)
                tempBuf[c] += srcPtr[(inBlockRounded * numChannels) + inOfs + c] * w;
        }
        int dstLoc = outPos * numChannels;
        for (int c = 0; c < numChannels; c++)
            dstPtr[dstLoc + c] = tempBuf[c];
    }
}

// -------------------- Set 1 - resample kernels executor --------------------

RppStatus hip_exec_resample_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *inRateTensor,
                                   Rpp32f *outRateTensor,
                                   Rpp32s *srcDimsTensor,
                                   RpptResamplingWindow &window,
                                   rpp::Handle& handle)
{
    int batchSize = handle.GetBatchSize();

    for(int i = 0; i < batchSize; i++)
    {
        int inRate = inRateTensor[i];
        int outRate = outRateTensor[i];
        int srcLength = srcDimsTensor[i * 2];
        int numChannels = srcDimsTensor[i * 2 + 1];
        if (inRate == outRate) // No need of Resampling, do a direct memcpy
        {
            hipMemcpy(dstPtr, srcPtr, srcLength * numChannels * sizeof(float), hipMemcpyDeviceToDevice);
        }
        else
        {
            int outEnd = std::ceil(srcLength * outRate / inRate);
            double scale = static_cast<double>(inRate) / outRate;
            int block = 256; // 1 << 8
            int length = (outEnd / block) + 1;

            if (numChannels == 1)
            {
                int globalThreads_x = length;
                int globalThreads_y = 1;
                int globalThreads_z = 1;

                hipLaunchKernelGGL(resample_1channel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + i * srcDescPtr->strides.nStride,
                                   dstPtr + i * dstDescPtr->strides.nStride,
                                   inRate,
                                   outRate,
                                   srcLength,
                                   outEnd,
                                   scale,
                                   window);
            }
            else
            {
                int globalThreads_x = length;
                int globalThreads_y = 1;
                int globalThreads_z = 1;

                hipLaunchKernelGGL(resample_nchannel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + i * srcDescPtr->strides.nStride,
                                   dstPtr + i * dstDescPtr->strides.nStride,
                                   inRate,
                                   outRate,
                                   srcLength,
                                   outEnd,
                                   scale,
                                   window,
                                   numChannels);
            }
        }

    }

    return RPP_SUCCESS;
}
