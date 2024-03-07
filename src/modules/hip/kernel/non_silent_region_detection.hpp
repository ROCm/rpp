#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ float square(float value)
{
    return (value * value);
}

__device__ void PrefixSum(float *buf, uint size)
{

}

__global__ void moving_mean_square_hip_tensor(float *srcPtr,
                                              uint nStride,
                                              float *mmsBuffer,
                                              int *srcLengthTensor,
                                              uint logicalBlock,
                                              uint windowLength,
                                              uint pow2)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_x];
    uint batchStride = id_z * nStride;
    float windowFactor = 1.0f / windowLength;

    extern __shared__ float squaredPrefixSum_smem[];
    for(uint blockStart = hipBlockIdx_x * logicalBlock; blockStart < srcLength; blockStart += nStride)
    {
        float *inBlockPtr = srcPtr + batchStride + blockStart;
        float *outBlockPtr = mmsBuffer + batchStride + blockStart;
        uint logicalBlockSize = min(logicalBlock, srcLength - logicalBlock);
        float *extendedBlockStart = inBlockPtr - windowLength;
        float *extendedBlockEnd = inBlockPtr + logicalBlockSize;

        // load input data to shared memory
        for(uint pos = hipThreadIdx_x; pos < pow2; pos += hipBlockDim_x)
        {
            float val = 0.0f;
            float *extendedBlockPtr = extendedBlockStart + pos;
            if (extendedBlockPtr >= inBlockPtr && extendedBlockPtr < extendedBlockEnd)
                val = *extendedBlockPtr;
            squaredPrefixSum_smem[hipThreadIdx_x] = square(val);
        }
        __syncthreads();

        // compute prefix sum
        PrefixSum(squaredPrefixSum_smem, pow2);
        __syncthreads();

        // compute the mms value here
        for(int pos = hipThreadIdx_x; pos < logicalBlockSize; pos += hipBlockDim_x)
        {
            float x = inBlockPtr[pos];
            float outVal = square(x) + squaredPrefixSum_smem[windowLength + pos] - squaredPrefixSum_smem[pos + 1];
            outBlockPtr[pos] = outVal * windowFactor;
        }
    }
}

RppStatus hip_exec_non_silent_region_detection_tensor(Rpp32f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp32s *srcLengthTensor,
                                                      Rpp32f *detectedIndexTensor,
                                                      Rpp32f *detectionLengthTensor,
                                                      Rpp32f cutOffDB,
                                                      Rpp32s windowLength,
                                                      Rpp32f referencePower,
                                                      Rpp32s resetInterval,
                                                      rpp::Handle& handle)
{
    // allocate memory for MMS buffer
    Rpp32f *mmsBuffer;
    hipMalloc(&(mmsBuffer), srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f));

    // launch kernel to compute the values needed for MMS buffer
    int globalThreads_x = 1;
    int globalThreads_y = 1;
    int globalThreads_z = handle.GetBatchSize();
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;

    uint logicalBlock, pow2;
    hipLaunchKernelGGL(moving_mean_square_hip_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       1000,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       mmsBuffer,
                       srcLengthTensor,
                       logicalBlock,
                       windowLength,
                       pow2);

    hipFree(mmsBuffer);
    return RPP_SUCCESS;
}