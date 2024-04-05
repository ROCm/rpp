#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel --------------------

__global__ void resample_1channel_hip_tensor(float *srcPtr,
                                             float *dstPtr,
                                             Rpp32f inRateTensor,
                                             Rpp32f outRateTensor,
                                             Rpp32s srcLength,
                                             Rpp32s outEnd,
                                             Rpp64f scale,
                                             RpptResamplingWindow &window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    Rpp32s block = 256;
    int outBlock = id_x * block;
    if (outBlock >= outEnd)
        return;

    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
    Rpp64f inBlockRaw = outBlock * scale;
    Rpp32s inBlockRounded = static_cast<int>(inBlockRaw);
    Rpp32f inPos = inBlockRaw - inBlockRounded;
    Rpp32f fscale = scale;

    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        Rpp32s loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);

        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > srcLength)
            loc1 = srcLength - inBlockRounded;
        Rpp32s locInWindow = loc0;
        Rpp32f locBegin = locInWindow - inPos;
        Rpp32f accum = 0.0f;

        for (; locInWindow < loc1; locInWindow++, locBegin++)
        {
            Rpp32f w = window(locBegin);
            accum += srcPtr[inBlockRounded + locInWindow] * w;
        }
        dstPtr[outPos] = accum;
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
            Rpp32s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp64f scale = static_cast<Rpp64f>(inRate) / outRate;
            Rpp32s block = 256; // 1 << 8
            int length = outEnd / block;

            if (numChannels == 1)
            {
                int globalThreads_x = length;
                int globalThreads_y = 1;
                int globalThreads_z = 1;

                hipLaunchKernelGGL(resample_1channel_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/512), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                                   dim3(512, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
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
        }

    }

    return RPP_SUCCESS;
}
