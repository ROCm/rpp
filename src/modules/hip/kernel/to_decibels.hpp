#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - device helpers for to_decibels kernel  --------------------

__device__ void to_decibels_hip_compute(d_float8 *src_f8, d_float8 *dst_f8, double minRatio, float multiplier, float inverseMagnitude)
{
    dst_f8->f1[0] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[0]) * inverseMagnitude)));
    dst_f8->f1[1] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[1]) * inverseMagnitude)));
    dst_f8->f1[2] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[2]) * inverseMagnitude)));
    dst_f8->f1[3] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[3]) * inverseMagnitude)));
    dst_f8->f1[4] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[4]) * inverseMagnitude)));
    dst_f8->f1[5] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[5]) * inverseMagnitude)));
    dst_f8->f1[6] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[6]) * inverseMagnitude)));
    dst_f8->f1[7] = multiplier * log2(max(minRatio, (static_cast<double>(src_f8->f1[7]) * inverseMagnitude)));
}


// -------------------- Set 1 - to decibels kernel --------------------

__global__ void to_decibels_1d_hip_tensor(float *srcPtr,
                                          uint srcStride,
                                          float *dstPtr,
                                          uint dstStride,
                                          RpptImagePatchPtr srcDims,
                                          double minRatio,
                                          float multiplier,
                                          float *inverseMagnitudeTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcDims[id_z].height)
        return;

    uint srcIdx = (id_z * srcStride) + id_x;
    float inverseMagnitude = inverseMagnitudeTensor[id_z];

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    to_decibels_hip_compute(&src_f8, &dst_f8, minRatio, multiplier, inverseMagnitude);

    uint dstIdx = (id_z * dstStride) + id_x;
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

__global__ void to_decibels_2d_hip_tensor(float *srcPtr,
                                          uint2 srcStridesNH,
                                          float *dstPtr,
                                          uint2 dstStridesNH,
                                          RpptImagePatchPtr srcDims,
                                          double minRatio,
                                          float multiplier,
                                          float *inverseMagnitudeTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcDims[id_z].width || id_y >= srcDims[id_z].height)
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    float inverseMagnitude = inverseMagnitudeTensor[id_z];

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    to_decibels_hip_compute(&src_f8, &dst_f8, minRatio, multiplier, inverseMagnitude);

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

// -------------------- Set 2 -  kernels for finding inverse magnitude value --------------------

__global__ void inverse_magnitude_hip_tensor(float *srcPtr,
                                             int maxLength,
                                             bool computeMax,
                                             float *inverseMagnitudeTensor)

{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Do final reduction on block wise max
    if (computeMax)
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

        if (hipThreadIdx_x == 0)
            inverseMagnitudeTensor[id_z] = 1.f / max_smem[0];
    }
    else
    {
        inverseMagnitudeTensor[id_z] = 1.0f;
    }
}

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

// -------------------- Set 3 - to decibels kernels executor --------------------

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

    int globalThreads_z = handle.GetBatchSize();

    // calculate max in input if referenceMagnitude = 0
    Rpp32f *partialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
    int numBlocksPerSample;

    // find the invReferenceMagnitude value
    bool computeMax = (!referenceMagnitude);
    if(computeMax)
    {
        if (numDims == 1)
        {
            numBlocksPerSample = ceil(static_cast<float>((srcDescPtr->strides.nStride + 7) >> 3) / 512);
            hipLaunchKernelGGL(max_reduction_hip_tensor,
                               dim3(numBlocksPerSample, 1, globalThreads_z),
                               dim3(512, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, 1),
                               srcDims,
                               numDims,
                               partialMaxArr);
        }
        else if (numDims == 2)
        {
            int gridDim_x = ceil(static_cast<float>((srcDescPtr->strides.hStride + 7) >> 3)/LOCAL_THREADS_X);
            int gridDim_y = ceil(static_cast<float>(srcDescPtr->h)/LOCAL_THREADS_Y);
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
        }
        hipStreamSynchronize(handle.GetStream());
    }
    Rpp32u blockSize = (computeMax) ? 256: 1;
    hipLaunchKernelGGL(inverse_magnitude_hip_tensor,
                       dim3(1, 1,  globalThreads_z),
                       dim3(blockSize, 1, 1),
                       0,
                       handle.GetStream(),
                       partialMaxArr,
                       numBlocksPerSample,
                       computeMax,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);
    hipStreamSynchronize(handle.GetStream());

    // launch kernel for todecibels
    if (numDims == 1)
    {
        int globalThreads_x = (srcDescPtr->strides.nStride + 7) >> 3;
        int globalThreads_y = 1;
        hipLaunchKernelGGL(to_decibels_1d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           srcDims,
                           static_cast<double>(minRatio),
                           multiplier,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);
    }
    else if (numDims == 2)
    {
        int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = srcDescPtr->h;
        hipLaunchKernelGGL(to_decibels_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           srcDims,
                           static_cast<double>(minRatio),
                           multiplier,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);
    }

    return RPP_SUCCESS;
}