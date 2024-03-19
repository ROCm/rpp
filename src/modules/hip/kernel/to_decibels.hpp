#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - to decibels kernel --------------------

__global__ void to_decibels_tensor(float *srcPtr,
                                   uint2 srcStridesNH,
                                   float *dstPtr,
                                   uint2 dstStridesNH,
                                   RpptImagePatchPtr srcDims,
                                   float minRatio,
                                   float multiplier,
                                   float referenceMagnitude,
                                   float *maxValues)
{

}

// -------------------- Set 1 -  kernels for finding max value in input --------------------

__global__ void max_reduction_hip_tensor(float *srcPtr,
                                         uint2 srcStridesNH,
                                         RpptImagePatchPtr srcDims,
                                         uint numDims,
                                         float *maxArr)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // for 1D input
    if (numDims == 1)
    {
        uint srcLength = srcDims[id_z].height;
        uint srcIdx = id_z * srcStridesNH.x;
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
    // for 2D input
    else if (numDims == 2)
    {
        __shared__ float partialMax_smem[16][16];                                   // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
        uint srcIdx = (id_z * srcStridesNH.x);
        float *partialMaxRowPtr_smem = &partialMax_smem[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
        partialMaxRowPtr_smem[hipThreadIdx_x] = srcPtr[srcIdx];                     // initialization of LDS to start value using all 16 x 16 threads

        if ((id_y >= srcDims[id_z].height) || (id_x >= srcDims[id_z].width))
            return;

        srcIdx += ((id_y * srcStridesNH.y) + id_x);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        rpp_hip_math_max8(&src_f8, &partialMaxRowPtr_smem[hipThreadIdx_x]);
        __syncthreads();                                                        // syncthreads after max compute

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
        {
            if (hipThreadIdx_x < threadMax)
                partialMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialMaxRowPtr_smem[hipThreadIdx_x], partialMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
            __syncthreads();
        }

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialMaxRowPtr_smem[0] = fmaxf(partialMaxRowPtr_smem[0], partialMaxRowPtr_smem[increment]);
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
                maxArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialMaxRowPtr_smem[0];
        }
    }
}

__global__ void final_reduction_hip_tensor(float *srcPtr,
                                           int maxLength,
                                           float *maxPtr)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;


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

    if (hipThreadIdx_x == 0)
        maxPtr[id_z] = max_smem[0];
}

// -------------------- Set 2 - to decibels kernels executor --------------------

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr srcDims,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle)
{
    Rpp32u numDims = srcDescPtr->numDims - 1;   // exclude batchSize from input dims

    // Calculate the intermediate values needed for DB conversion
    Rpp32f minRatio = std::pow(10, cutOffDB / multiplier);
    if(minRatio == 0.0f)
        minRatio = std::nextafter(0.0f, 1.0f);

    const Rpp32f log10Factor = 0.3010299956639812;      //1 / std::log(10);
    multiplier *= log10Factor;

    // calculate max in input if referenceMagnitude = 0
    if(!referenceMagnitude)
    {
        Rpp32f *partialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        int numBlocksPerSample;
        if (numDims == 1)
        {
            numBlocksPerSample = ceil(static_cast<float>(srcDescPtr->strides.nStride) / (512 * 8));
            hipLaunchKernelGGL(max_reduction_hip_tensor,
                               dim3(numBlocksPerSample, 1, handle.GetBatchSize()),
                               dim3(512, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, 1),
                               srcDims,
                               numDims,
                               partialMaxArr);
            hipStreamSynchronize(handle.GetStream());
        }
        else if (numDims == 2)
        {
            int globalThreads_x = (srcDescPtr->w + 7) >> 3;
            int globalThreads_y = srcDescPtr->h;
            int globalThreads_z = handle.GetBatchSize();
            int gridDim_x = ceil(static_cast<float>(globalThreads_x)/LOCAL_THREADS_X);
            int gridDim_y = ceil(static_cast<float>(globalThreads_y)/LOCAL_THREADS_Y);
            int gridDim_z = ceil(static_cast<float>(globalThreads_z)/LOCAL_THREADS_Z);
            numBlocksPerSample = gridDim_x * gridDim_y * gridDim_z;
            hipLaunchKernelGGL(max_reduction_hip_tensor,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               srcDims,
                               numDims,
                               partialMaxArr);
            hipStreamSynchronize(handle.GetStream());
        }
        // find the cutoff value in magnitude
        hipLaunchKernelGGL(final_reduction_hip_tensor,
                           dim3(1, 1,  handle.GetBatchSize()),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMaxArr,
                           numBlocksPerSample,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);
        hipStreamSynchronize(handle.GetStream());
    }

    return RPP_SUCCESS;
}