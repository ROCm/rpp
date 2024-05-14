#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel device helpers  --------------------

__device__ __forceinline__ float resample_hip_compute(float x, float scale, float center, float *lookup, int lookupSize)
{
    float locRaw = x * scale + center;
    int locFloor = std::floor(locRaw);
    float weight = locRaw - locFloor;
    locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
    float current = lookup[locFloor];
    float next = lookup[locFloor + 1];
    return current + weight * (next - current);
}

// -------------------- Set 1 - resample kernels --------------------

__global__ void resample_single_channel_hip_tensor(float *srcPtr,
                                                   float *dstPtr,
                                                   uint2 strides,
                                                   int *srcDimsTensor,
                                                   int *dstDimsTensor,
                                                   float *inRateTensor,
                                                   float *outRateTensor,
                                                   RpptResamplingWindow &window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int dstLength = dstDimsTensor[id_z * 2];
    int srcLength = srcDimsTensor[id_z * 2];
    int outBlock = id_x * hipBlockDim_x;
    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), dstLength);

    if (dstLength != srcLength)
    {
        double scale = static_cast<double>(inRateTensor[id_z]) / outRateTensor[id_z];
        extern __shared__ float windowCoeffs_smem[];
        for (int k = hipThreadIdx_x; k < window.lookupSize; k += hipBlockDim_x)
            windowCoeffs_smem[k] = window.lookup[k];
        __syncthreads();

        if (outBlock >= dstLength)
            return;

        double inBlockRaw = outBlock * scale;
        int inBlockRounded = static_cast<int>(inBlockRaw);
        float inPos = inBlockRaw - inBlockRounded;
        float fscale = scale;
        float *inBlockPtr = srcPtr + id_z * strides.x + inBlockRounded;
        uint dstIdx = id_z * strides.y + outBlock;
        for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale, dstIdx++)
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
                float w = resample_hip_compute(locBegin, window.scale, window.center, windowCoeffs_smem, window.lookupSize);
                accum += inBlockPtr[locInWindow] * w;
            }
            dstPtr[dstIdx] = accum;
        }
    }
    // copy input to output if dstLength is same as srcLength
    else
    {
        if (outBlock >= dstLength)
            return;

        uint srcIdx = id_z * strides.x + outBlock;
        uint dstIdx = id_z * strides.y + outBlock;
        for (int outPos = outBlock; outPos < blockEnd; outPos++, dstIdx++)
            dstPtr[dstIdx] = srcPtr[srcIdx];
    }
}

__global__ void resample_multi_channel_hip_tensor(float *srcPtr,
                                                  float *dstPtr,
                                                  int2 srcDims,
                                                  int outEnd,
                                                  double scale,
                                                  RpptResamplingWindow &window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int outBlock = id_x * hipBlockDim_x;

    extern __shared__ float windowCoeffs_smem[];
    for (int k = hipThreadIdx_x; k < window.lookupSize; k += hipBlockDim_x)
        windowCoeffs_smem[k] = window.lookup[k];
    __syncthreads();

    if (outBlock >= outEnd)
        return;

    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), outEnd);
    float tempBuf[RPPT_MAX_AUDIO_CHANNELS] = {0.0f};
    int length = srcDims.x;
    int channels = srcDims.y;

    double inBlockRaw = outBlock * scale;
    int inBlockRounded = static_cast<int>(inBlockRaw);
    float inPos = inBlockRaw - inBlockRounded;
    float fscale = scale;
    float *inBlockPtr = srcPtr + (inBlockRounded * channels);
    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale)
    {
        int loc0, loc1;
        window.input_range(inPos, &loc0, &loc1);
        if (loc0 + inBlockRounded < 0)
            loc0 = -inBlockRounded;
        if (loc1 + inBlockRounded > length)
            loc1 = length - inBlockRounded;

        float locInWindow = loc0 - inPos;
        int2 offsetLocs_i2 = make_int2(loc0, loc1) * static_cast<int2>(channels);    // offsetted loc0, loc1 values for multi channel case
        for (int offsetLoc = offsetLocs_i2.x; offsetLoc < offsetLocs_i2.y; offsetLoc += channels, locInWindow++)
        {
            float w = resample_hip_compute(locInWindow, window.scale, window.center, windowCoeffs_smem, window.lookupSize);
            for (int c = 0; c < channels; c++)
                tempBuf[c] += inBlockPtr[offsetLoc + c] * w;
        }

        int dstIdx = outPos * channels;
        for (int c = 0; c < channels; c++)
            dstPtr[dstIdx + c] = tempBuf[c];
    }
}


void compute_output_length(Rpp32f *inRateTensor,
                           Rpp32f *outRateTensor,
                           Rpp32s *srcLengthTensor,
                           Rpp32s *dstLengthTensor,
                           uint batchSize)
{
    for (int i = 0, j = 0; i < batchSize; i++, j += 2)
    {
        dstLengthTensor[j] = std::ceil(srcLengthTensor[j] * outRateTensor[i] / inRateTensor[i]);
        dstLengthTensor[j + 1] = srcLengthTensor[j + 1];
    }
}

// -------------------- Set 2 - resample kernels executor --------------------

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
    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = 1;
    int globalThreads_z = dstDescPtr->n;

    int *dstDimsTensor = reinterpret_cast<int *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.uintmem);
    compute_output_length(inRateTensor, outRateTensor, srcDimsTensor, dstDimsTensor, dstDescPtr->n);

    size_t sharedMemSize = (window.lookupSize * sizeof(Rpp32f));
    hipLaunchKernelGGL(resample_single_channel_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       sharedMemSize,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       make_uint2(srcDescPtr->strides.nStride, dstDescPtr->strides.nStride),
                       srcDimsTensor,
                       dstDimsTensor,
                       inRateTensor,
                       outRateTensor,
                       window);

    return RPP_SUCCESS;
}
