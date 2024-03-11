#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ float square(float value)
{
    return (value * value);
}

__host__ __device__ int shm_pos(int pos)
{
    return pos + (pos >> 5);
}

__device__ void PrefixSum(float *buf, uint pow2)
{
    int offset = 1;
    int tid = hipThreadIdx_x;

    for (int d = pow2 >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        for (int idx = tid; idx < d; idx += hipBlockDim_x)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            buf[shm_pos(bi)] += buf[shm_pos(ai)];
        }
        offset <<= 1;
    }

    if (tid == 0)
    {
        int last = pow2 - 1;
        buf[shm_pos(last)] = 0;
    }

    for (int d = 1; d < pow2; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        for (int idx = tid; idx < d; idx += hipBlockDim_x)
        {
            int shm_pos_ai = shm_pos(offset * (2 * idx + 1) - 1);
            int shm_pos_bi = shm_pos(offset * (2 * idx + 2) - 1);
            auto t = buf[shm_pos_ai];
            buf[shm_pos_ai] = buf[shm_pos_bi];
            buf[shm_pos_bi] += t;
        }
    }
  __syncthreads();
}

int prev_pow2(int n)
{
  int pow2 = 1;
  while (n - pow2 > pow2)
    pow2 += pow2;

  return pow2;
}

int next_pow2(int n)
{
  int pow2 = 1;
  while (n > pow2)
    pow2 += pow2;

  return pow2;
}

__global__ void moving_mean_square_hip_tensor(float *srcPtr,
                                              uint nStride,
                                              float *mmsBuffer,
                                              int *srcLengthTensor,
                                              int logicalBlock,
                                              int windowLength,
                                              int pow2)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];
    uint batchStride = id_z * nStride;
    uint gridStride = hipGridDim_x * logicalBlock;
    float windowFactor = 1.0f / windowLength;
    float *input = srcPtr + batchStride;

    extern __shared__ char smem[];
    float *squaredPrefixSum_smem = reinterpret_cast<float *>(smem);
    for(int blockStart = hipBlockIdx_x * logicalBlock; blockStart < srcLength; blockStart += gridStride)
    {
        float *inBlockPtr = srcPtr + batchStride + blockStart;
        float *outBlockPtr = mmsBuffer + batchStride + blockStart;
        int logicalBlockSize = min(logicalBlock, srcLength - blockStart);
        float *extendedBlockStart = inBlockPtr - windowLength;
        float *extendedBlockEnd = inBlockPtr + logicalBlockSize;

        // load input data to shared memory
        for(int pos = hipThreadIdx_x; pos < pow2; pos += hipBlockDim_x)
        {
            float val = 0.0f;
            auto extendedBlockPtr = extendedBlockStart + pos;
            if (extendedBlockPtr >= input && extendedBlockPtr < extendedBlockEnd)
                val = *extendedBlockPtr;
            squaredPrefixSum_smem[shm_pos(pos)] = square(val);
        }

        // compute prefix sum
        PrefixSum(squaredPrefixSum_smem, pow2);

        // compute the mms value here
        for(int pos = hipThreadIdx_x; pos < logicalBlockSize; pos += hipBlockDim_x)
        {
            float x = inBlockPtr[pos];
            float outVal = square(x) + squaredPrefixSum_smem[shm_pos(windowLength + pos)] - squaredPrefixSum_smem[shm_pos(pos + 1)];
            outBlockPtr[pos] = outVal * windowFactor;
        }
    }
}

__global__ void final_reduction_hip_tensor(int *minBuffer,
                                           int *maxBuffer,
                                           float *beginTensor,
                                           float *lengthTensor,
                                           int *srcLengthTensor,
                                           int maxLength,
                                           int windowLength)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcLengthTensor[id_z];

    __shared__ int min_smem[256];
    __shared__ int max_smem[256];
    min_smem[hipThreadIdx_x] = srcLength;
    max_smem[hipThreadIdx_x] = 0;

    if (id_x >= maxLength)
        return;

    int minIndex = srcLength;
    int maxIndex = 0;
    uint srcIdx;
    while (id_x < maxLength)
    {
        srcIdx = id_z * maxLength + id_x;
        minIndex = min(minIndex, minBuffer[srcIdx]);
        maxIndex = max(maxIndex, maxBuffer[srcIdx]);
        id_x += hipBlockDim_x;
    }
    min_smem[hipThreadIdx_x] = minIndex;
    max_smem[hipThreadIdx_x] = maxIndex;
    __syncthreads();

    // do reduction on min_smem and max_smem
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            min_smem[hipThreadIdx_x] = min(min_smem[hipThreadIdx_x], min_smem[hipThreadIdx_x + threadMax]);
            max_smem[hipThreadIdx_x] = max(max_smem[hipThreadIdx_x], max_smem[hipThreadIdx_x + threadMax]);
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        int beginIdx = min_smem[0];
        int endIdx = max_smem[0];
        if(beginIdx == srcLength || endIdx == 0)
        {
            beginTensor[id_z] = 0;
            lengthTensor[id_z] = 0;
        }
        else
        {
            int detectBegin = beginIdx;
            int detectEnd = endIdx - beginIdx + 1;
            if(detectBegin != 0 && detectEnd != 0)
            {
                int newBegin = max(detectBegin - (windowLength - 1), 0);
                detectEnd += detectBegin - newBegin;
                detectBegin = newBegin;
            }
            beginTensor[id_z] = detectBegin;
            lengthTensor[id_z] = detectEnd;
        }
    }
}

__global__ void find_region_hip_tensor(float *mmsBuffer,
                                       uint nStride,
                                       int *minBuffer,
                                       int *maxBuffer,
                                       float cutoffDB,
                                       int *srcLengthTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];

    __shared__ int min_smem[256];
    __shared__ int max_smem[256];
    min_smem[hipThreadIdx_x] = srcLength;
    max_smem[hipThreadIdx_x] = 0;

    if (id_x >= srcLength)
        return;

    uint srcIdx = (id_z * nStride) + id_x;
    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(mmsBuffer + srcIdx, &src_f8);
    int minIndex = srcLength;
    int maxIndex = 0;
    for(int i = 0; i < 8; i++)
    {
        if (src_f8.f1[i] >= cutoffDB)
        {
            maxIndex = i;
            if(minIndex == srcLength)
                minIndex = i;
        }
    }
    min_smem[hipThreadIdx_x] = id_x + minIndex;
    max_smem[hipThreadIdx_x] = id_x + maxIndex;
    __syncthreads();

    // do reduction on min_smem and max_smem
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            min_smem[hipThreadIdx_x] = fminf(min_smem[hipThreadIdx_x], min_smem[hipThreadIdx_x + threadMax]);
            max_smem[hipThreadIdx_x] = fmaxf(max_smem[hipThreadIdx_x], max_smem[hipThreadIdx_x + threadMax]);
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        int idx = id_z * hipGridDim_x + hipBlockIdx_x;
        minBuffer[idx] = min_smem[0];
        maxBuffer[idx] = max_smem[0];
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
    hipHostMalloc(&(mmsBuffer), srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f));

    int maxSharedMemoryInBytes = 32000; // 32 KB
    int maxSharedMemoryElements = maxSharedMemoryInBytes / sizeof(Rpp32f);
    int kSharedMemBanks = 32;
    int pow2 = prev_pow2(maxSharedMemoryElements * kSharedMemBanks / (kSharedMemBanks + 1));

    if (resetInterval > 0 && resetInterval < pow2)
    {
        auto p = prev_pow2(resetInterval);
        auto n = next_pow2(resetInterval);
        if (p > windowLength)
            pow2 = p;
        else if (n < pow2)
            pow2 = n;
    }

    int sharedMemorySizeInBytes = shm_pos(pow2) * sizeof(Rpp32f);
    int logicalBlock = pow2 - windowLength;

    if (logicalBlock <= 0)
    {
        std::cout << "Invalid logical block! " << std::endl;
        return RPP_ERROR;
    }
    if (sharedMemorySizeInBytes > maxSharedMemoryInBytes)
    {
        std::cout << "Can't compute the requested running sum, due to shared memory restrictions" << std::endl;
        return RPP_ERROR;
    }

    // launch kernel to compute the values needed for MMS buffer
    int globalThreads_x = std::min<int>(ceil(static_cast<float>(srcDescPtr->strides.nStride) / 32), 1024);
    int globalThreads_y = 1;
    int globalThreads_z = handle.GetBatchSize();
    int localThreads_x = 512;
    int localThreads_y = 1;
    int localThreads_z = 1;

    hipLaunchKernelGGL(moving_mean_square_hip_tensor,
                       dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       sharedMemorySizeInBytes,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       mmsBuffer,
                       srcLengthTensor,
                       logicalBlock,
                       windowLength,
                       pow2);
    hipStreamSynchronize(handle.GetStream());

    // std::cout << "printing GPU moving mean square values: " << std::endl;
    // for(int i = 0; i < srcLengthTensor[0]; i++)
    //     std::cout << mmsBuffer[i] << std::endl;

    const Rpp32f cutOff = std::pow(10.0f, cutOffDB * 0.1f);
    bool referenceMax = (referencePower == 0.0f);

    // TODO - needs to be changed for handling for multi batchsize and kernel for max compute
    Rpp32s srcLength = srcLengthTensor[0];
    Rpp32f base = (referenceMax) ? *std::max_element(mmsBuffer, mmsBuffer + srcLength) : referencePower;
    Rpp32f cutOffMag = base * cutOff;

    // launch kernel to find partial min and max values
    int *minBuffer = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.maskArr.floatmem);
    int *maxBuffer = minBuffer + 45000; // ((700000 / (256 * 8)) * 128) where 700000 is the max audio length in librispeech
    int numBlocks = ceil(static_cast<float>(srcDescPtr->strides.nStride) / (256 * 8));
    hipLaunchKernelGGL(find_region_hip_tensor,
                       dim3(numBlocks, 1, globalThreads_z),
                       dim3(256, 1, 1),
                       0,
                       handle.GetStream(),
                       mmsBuffer,
                       srcDescPtr->strides.nStride,
                       minBuffer,
                       maxBuffer,
                       cutOffMag,
                       srcLengthTensor);
    hipStreamSynchronize(handle.GetStream());
    hipLaunchKernelGGL(final_reduction_hip_tensor,
                       dim3(1, 1, globalThreads_z),
                       dim3(256, 1, 1),
                       0,
                       handle.GetStream(),
                       minBuffer,
                       maxBuffer,
                       detectedIndexTensor,
                       detectionLengthTensor,
                       srcLengthTensor,
                       numBlocks,
                       windowLength);
    hipStreamSynchronize(handle.GetStream());

    hipHostFree(mmsBuffer);
    return RPP_SUCCESS;
}