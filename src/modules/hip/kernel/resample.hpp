#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel --------------------
// Single channel resample support
__global__ void resample_1channel_hip_tensor(float *srcPtr,
                                             float *dstPtr,
                                             int srcLength,
                                             int outEnd,
                                             double scale,
                                             RpptResamplingWindow &window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int outBlock = id_x * hipBlockDim_x;
    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), outEnd);
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
                                             int2 srcDims,
                                             int outEnd,
                                             double scale,
                                             RpptResamplingWindow &window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int outBlock = id_x * hipBlockDim_x;
    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), outEnd);
    double inBlockRaw = outBlock * scale;
    int inBlockRounded = static_cast<int>(inBlockRaw);
    float inPos = inBlockRaw - inBlockRounded;
    float fscale = scale;
    float tempBuf[RPPT_MAX_CHANNELS] = {0.0f}; // Considering max channels as 3

    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        int loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);

        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > srcDims.x)
            loc1 = srcDims.x - inBlockRounded;
        int locInWindow = loc0;
        float locBegin = locInWindow - inPos;
        int2 ofs_i2 = make_int2(loc0, loc1);
        ofs_i2 *= (int2)srcDims.y;
        int idx = inBlockRounded * srcDims.y;

        for (int inOfs = ofs_i2.x; inOfs < ofs_i2.y; inOfs += srcDims.y, locBegin++)
        {
            float w = window(locBegin);
            for (int c = 0; c < srcDims.y; c++)
                tempBuf[c] += srcPtr[idx + inOfs + c] * w;
        }
        int dstLoc = outPos * srcDims.y;
        for (int c = 0; c < srcDims.y; c++)
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
    Rpp32s batchSize = dstDescPtr->n;
    for(int i = 0, j = 0; i < batchSize; i++, j += 2)
    {
        Rpp32f inRate = inRateTensor[i];
        Rpp32f outRate = outRateTensor[i];
        Rpp32s srcLength = srcDimsTensor[j];
        Rpp32s numChannels = srcDimsTensor[j + 1];
        if (inRate == outRate) // No need of Resampling, do a direct memcpy
        {
            CHECK_RETURN_STATUS(hipMemcpyAsync(dstPtr, srcPtr, srcLength * numChannels * sizeof(Rpp32f), hipMemcpyDeviceToDevice, handle.GetStream()));
            CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));
        }
        else
        {
            Rpp32s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp64f scale = static_cast<Rpp64f>(inRate) / outRate;
            Rpp32s length = (outEnd / LOCAL_THREADS_X_1DIM) + 1;
            Rpp32s globalThreads_x = length;
            Rpp32s globalThreads_y = 1;
            Rpp32s globalThreads_z = 1;
            size_t sharedMemSize = (window.lookupSize + (RPPT_MAX_CHANNELS + 1) * LOCAL_THREADS_X_1DIM) * sizeof(Rpp32f);
            if (numChannels == 1)
            {
                hipLaunchKernelGGL(resample_1channel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                                   sharedMemSize,
                                   handle.GetStream(),
                                   srcPtr + i * srcDescPtr->strides.nStride,
                                   dstPtr + i * dstDescPtr->strides.nStride,
                                   srcLength,
                                   outEnd,
                                   scale,
                                   window);
            }
            else
            {
                hipLaunchKernelGGL(resample_nchannel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                                   0,
                                   handle.GetStream(),
                                   srcPtr + i * srcDescPtr->strides.nStride,
                                   dstPtr + i * dstDescPtr->strides.nStride,
                                   make_int2(srcLength, numChannels),
                                   outEnd,
                                   scale,
                                   window);
            }
        }

    }

    return RPP_SUCCESS;
}
