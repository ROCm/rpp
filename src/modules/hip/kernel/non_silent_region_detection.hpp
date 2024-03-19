#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 -  moving mean square kernel device helpers --------------------

__host__ __device__ int smem_pos(int pos)
{
    return pos + (pos >> 5);
}

__device__ float square(float value)
{
    return (value * value);
}

__device__ void PrefixSum(float *input, uint bufferLength)
{
    int offset = 1;
    int tid = hipThreadIdx_x;

    for (int d = bufferLength >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        for (int idx = tid; idx < d; idx += hipBlockDim_x)
        {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            input[smem_pos(bi)] += input[smem_pos(ai)];
        }
        offset <<= 1;
    }

    if (tid == 0)
    {
        int last = bufferLength - 1;
        input[smem_pos(last)] = 0;
    }

    for (int d = 1; d < bufferLength; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();
        for (int idx = tid; idx < d; idx += hipBlockDim_x)
        {
            int smem_pos_ai = smem_pos(offset * (2 * idx + 1) - 1);
            int smem_pos_bi = smem_pos(offset * (2 * idx + 2) - 1);
            auto t = input[smem_pos_ai];
            input[smem_pos_ai] = input[smem_pos_bi];
            input[smem_pos_bi] += t;
        }
    }
  __syncthreads();
}

// -------------------- Set 1 -  moving mean square compute kernel --------------------

__global__ void moving_mean_square_hip_tensor(float *srcPtr,
                                              uint nStride,
                                              float *mmsArr,
                                              int *srcLengthTensor,
                                              int outputTileLength,
                                              int windowLength,
                                              float windowFactor,
                                              int inputTileLength)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];
    uint batchStride = id_z * nStride;
    float *input = srcPtr + batchStride;

    extern __shared__ char smem[];
    float *squaredPrefixSum_smem = reinterpret_cast<float *>(smem);
    int blockStart = hipBlockIdx_x * outputTileLength;

    if (blockStart >= srcLength)
        return;

    float *inBlockPtr = srcPtr + batchStride + blockStart;
    float *outBlockPtr = mmsArr + batchStride + blockStart;
    int validOutputTileLength = min(outputTileLength, srcLength - blockStart);
    float *extendedBlockStart = inBlockPtr - windowLength;
    float *extendedBlockEnd = inBlockPtr + validOutputTileLength;

    // load input data to shared memory
    for(int pos = hipThreadIdx_x; pos < inputTileLength; pos += hipBlockDim_x)
    {
        float val = 0.0f;
        auto extendedBlockPtr = extendedBlockStart + pos;
        if (extendedBlockPtr >= input && extendedBlockPtr < extendedBlockEnd)
            val = *extendedBlockPtr;
        squaredPrefixSum_smem[smem_pos(pos)] = square(val);
    }

    // compute prefix sum
    PrefixSum(squaredPrefixSum_smem, inputTileLength);

    // compute the mms value here
    for(int pos = hipThreadIdx_x; pos < validOutputTileLength; pos += hipBlockDim_x)
    {
        float x = inBlockPtr[pos];
        float outVal = square(x) + squaredPrefixSum_smem[smem_pos(windowLength + pos)] - squaredPrefixSum_smem[smem_pos(pos + 1)];
        outBlockPtr[pos] = outVal * windowFactor;
    }
}

// -------------------- Set 2 -  kernels for finding cutoffmag value  --------------------

__global__ void max_reduction_hip_tensor(float *srcPtr,
                                         uint nStride,
                                         float *maxArr,
                                         int *srcLengthTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];

    uint srcIdx = id_z * nStride;
    __shared__ float max_smem[512];
    max_smem[hipThreadIdx_x] = srcPtr[srcIdx];

    if (id_x >= srcLength)
        return;

    srcIdx += id_x;
    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    rpp_hip_math_max8(&src_f8, &max_smem[hipThreadIdx_x]);
    __syncthreads();

    // do reduction on min_smem and max_smem
    for (int threadMax = 256; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            max_smem[hipThreadIdx_x] = fmaxf(max_smem[hipThreadIdx_x], max_smem[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        int dstIdx = id_z * hipGridDim_x + hipBlockIdx_x;
        maxArr[dstIdx] = max_smem[0];
    }
}

__global__ void cutoffmag_hip_tensor(float *srcPtr,
                                     int maxLength,
                                     float *cutOffMagPtr,
                                     float cutOff,
                                     float referencePower,
                                     bool referenceMax)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // if referenceMax is set to true, perform final max reduction on srcPtr and compute cutOffMag
    if(referenceMax)
    {
        uint srcIdx = id_z * maxLength;
        __shared__ float max_smem[256];
        max_smem[hipThreadIdx_x] = srcPtr[srcIdx];

        if (id_x >= maxLength)
            return;

        srcIdx += id_x;
        float maxVal = srcPtr[srcIdx];
        while (id_x < maxLength)
        {
            maxVal = fmaxf(maxVal, srcPtr[srcIdx]);
            id_x += hipBlockDim_x;
            srcIdx += hipBlockDim_x;
        }
        max_smem[hipThreadIdx_x] = maxVal;
        __syncthreads();

        // do reduction on min_smem and max_smem
        for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                max_smem[hipThreadIdx_x] = max(max_smem[hipThreadIdx_x], max_smem[hipThreadIdx_x + threadMax]);
            __syncthreads();
        }

        // cutOffMag = max(srcPtr) * cutoff;
        if (hipThreadIdx_x == 0)
            cutOffMagPtr[id_z] = max_smem[0] * cutOff;
    }
    else
    {
        if (hipThreadIdx_x == 0)
            cutOffMagPtr[id_z] = referencePower * cutOff;
    }
}

// -------------------- Set 3 -  kernels for finding begin and length of NSR in inputs --------------------

__global__ void find_region_hip_tensor(float *srcPtr,
                                       uint nStride,
                                       int *beginTensor,
                                       int *lengthTensor,
                                       float *cutOffMagPtr,
                                       int *srcLengthTensor,
                                       float windowLength)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint srcLength = srcLengthTensor[id_z];
    float cutOffMag = cutOffMagPtr[id_z];

    __shared__ int beginResult;
    __shared__ int endResult;
    beginResult = srcLength;
    endResult = 0;
    __syncthreads();

    int beginIdx = srcLength;
    int endIdx = 0;
    for (int i = id_x; i < srcLength; i += hipBlockDim_x)
    {
        uint srcIdx = id_z * nStride + i;
        if (srcPtr[srcIdx] >= cutOffMag)
        {
            beginIdx = i;
            atomicMin(&beginResult, beginIdx);
            if(beginResult != srcLength)
                break;
        }
    }
    for (int i = id_x; i < srcLength; i += hipBlockDim_x)
    {
        uint srcIdx = id_z * nStride + srcLength - 1 - i;
        if (srcPtr[srcIdx] >= cutOffMag)
        {
            endIdx = srcLength - 1 - i;
            atomicMax(&endResult, endIdx);
            if(endResult != 0)
                break;
        }
    }
    if(hipThreadIdx_x == 0)
    {
        if(beginResult == srcLength || endResult == 0)
        {
            beginTensor[id_z] = 0;
            lengthTensor[id_z] = 0;
        }
        else
        {
            int detectBegin = beginResult;
            int detectEnd = endResult - beginResult + 1;
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

// -------------------- Set 4 -  host helpers for kernel executor --------------------

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

// -------------------- Set 5 - non silent region kernels executor --------------------

RppStatus hip_exec_non_silent_region_detection_tensor(Rpp32f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp32s *srcLengthTensor,
                                                      Rpp32s *detectedIndexTensor,
                                                      Rpp32s *detectionLengthTensor,
                                                      Rpp32f cutOffDB,
                                                      Rpp32s windowLength,
                                                      Rpp32f referencePower,
                                                      Rpp32s resetInterval,
                                                      rpp::Handle& handle)
{
    // allocate temporary memory for MMS Array
    Rpp32f *mmsArr;
    hipMalloc(&(mmsArr), srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f));

    int maxSharedMemoryInBytes = 32000; // 32 KB
    int maxSharedMemoryElements = maxSharedMemoryInBytes / sizeof(Rpp32f);
    int kSharedMemBanks = 32;
    int inputTileLength = prev_pow2(maxSharedMemoryElements * kSharedMemBanks / (kSharedMemBanks + 1));

    if (resetInterval > 0 && resetInterval < inputTileLength)
    {
        auto p = prev_pow2(resetInterval);
        auto n = next_pow2(resetInterval);
        if (p > windowLength)
            inputTileLength = p;
        else if (n < inputTileLength)
            inputTileLength = n;
    }

    int sharedMemorySizeInBytes = smem_pos(inputTileLength) * sizeof(Rpp32f);
    int outputTileLength = inputTileLength - windowLength;
    float windowFactor = 1.0f / windowLength;

    if (outputTileLength <= 0)
    {
        std::cout << "Invalid output tile length! " << std::endl;
        return RPP_ERROR;
    }
    if (sharedMemorySizeInBytes > maxSharedMemoryInBytes)
    {
        std::cout << "Cannot compute the requested moving mean square, due to shared memory restrictions" << std::endl;
        return RPP_ERROR;
    }

    // launch kernel to compute the values needed for MMS Array
    int globalThreads_x = ceil(static_cast<float>(srcDescPtr->strides.nStride) / outputTileLength);
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
                       mmsArr,
                       srcLengthTensor,
                       outputTileLength,
                       windowLength,
                       windowFactor,
                       inputTileLength);
    hipStreamSynchronize(handle.GetStream());

    const Rpp32f cutOff = std::pow(10.0f, cutOffDB * 0.1f);
    bool referenceMax = (referencePower == 0.0f);
    Rpp32f *partialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;

    int numBlocksPerSample = ceil(static_cast<float>(srcDescPtr->strides.nStride) / (512 * 8));
    int cutOffMagKernelBlockSize = 1;
    if (referenceMax)
    {
        // compute max value in MMS buffer
        hipLaunchKernelGGL(max_reduction_hip_tensor,
                           dim3(numBlocksPerSample, 1, globalThreads_z),
                           dim3(512, 1, 1),
                           0,
                           handle.GetStream(),
                           mmsArr,
                           srcDescPtr->strides.nStride,
                           partialMaxArr,
                           srcLengthTensor);
        hipStreamSynchronize(handle.GetStream());

        cutOffMagKernelBlockSize = 256;
    }
    // find the cutoff value in magnitude
    hipLaunchKernelGGL(cutoffmag_hip_tensor,
                       dim3(1, 1, globalThreads_z),
                       dim3(cutOffMagKernelBlockSize, 1, 1),
                       0,
                       handle.GetStream(),
                       partialMaxArr,
                       numBlocksPerSample,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       cutOff,
                       referencePower,
                       referenceMax);
    hipStreamSynchronize(handle.GetStream());

    // find the begin and length values of NSR in inputs
    hipLaunchKernelGGL(find_region_hip_tensor,
                       dim3(1, 1, globalThreads_z),
                       dim3(1024, 1, 1),
                       0,
                       handle.GetStream(),
                       mmsArr,
                       srcDescPtr->strides.nStride,
                       detectedIndexTensor,
                       detectionLengthTensor,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       srcLengthTensor,
                       windowLength);
    hipStreamSynchronize(handle.GetStream());

    hipFree(mmsArr);
    return RPP_SUCCESS;
}