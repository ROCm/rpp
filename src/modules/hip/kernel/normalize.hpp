#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

#define MAX_SHARED_MEMORY_SIZE 1024

// -------------------- Set 0 - normalization kernels device helpers --------------------

__device__ __forceinline__ uint rpp_hip_mod(uint a, uint b)
{
    return (a % b);
}

__device__ __forceinline__ void normalize_check_and_store(float outVal, uchar* dst)
{
    outVal = fmax(fminf(outVal, 255), 0);
    *dst = static_cast<uchar>(outVal);
}

__device__ __forceinline__ void normalize_check_and_store(float outVal, schar* dst)
{
    outVal = fmax(fminf(outVal, 127), -128);
    *dst = static_cast<schar>(outVal);
}

__device__ __forceinline__ void normalize_check_and_store(float outVal, float* dst)
{
    *dst = outVal;
}

__device__ __forceinline__ void normalize_check_and_store(float outVal, half* dst)
{
    *dst = static_cast<half>(outVal);
}

// -------------------- Set 1 - normalization kernel host helpers --------------------

void normalize_setup(Rpp32u *roiTensor, Rpp32u batchSize, Rpp32u numDims,
                     Rpp32u axisMask, Rpp32u &maxParamVolume)
{
    maxParamVolume = 1;
    uint axisSet[RPPT_MAX_DIMS];
    for(int i = 0; i < numDims; i++)
        axisSet[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;

    for(uint i = 0; i < batchSize; i++)
    {
        // calculate the max param volume
        Rpp32u paramVolume = 1;
        Rpp32u *roi = &roiTensor[numDims * 2 * i + numDims];
        for(uint j = 0; j < numDims; j++)
            paramVolume *= (axisSet[j]) ? 1 : roi[j];
        maxParamVolume = std::max(maxParamVolume, paramVolume);
    }
}

void normalize_setup_nd(Rpp32u *roiTensor, Rpp32u batchSize, Rpp32u numDims, Rpp32u axisMask,
                        Rpp32u *paramShapeTensor, Rpp32u *paramStridesTensor, Rpp32u &maxParamVolume)
{
    maxParamVolume = 1;
    uint axisSet[RPPT_MAX_DIMS];
    for(int i = 0; i < numDims; i++)
        axisSet[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;

    for(uint i = 0; i < batchSize; i++)
    {
        // calculate the param shape and param volume based on the axis mask
        Rpp32u paramVolume = 1;
        Rpp32u *roi = &roiTensor[numDims * 2 * i + numDims];
        Rpp32u *paramShape = &paramShapeTensor[i * numDims];
        for(uint j = 0; j < numDims; j++)
        {
            paramShape[j] = (axisSet[j]) ? 1 : roi[j];
            paramVolume *= paramShape[j];
        }
        maxParamVolume = std::max(maxParamVolume, paramVolume);

        // calculate the param strides from the param shape
        Rpp32u *paramStrides = &paramStridesTensor[i * numDims];
        Rpp32u val = 1;
        for(uint j = numDims - 1; j > 0; j--)
        {
            paramStrides[j] = val;
            val *= paramShape[j];
        }
        paramStrides[0] = val;
    }
}

// -------------------- Set 2 - normalization kernels --------------------

template <typename T>
__global__ void normalize_2d_hip_tensor(T *srcPtr,
                                        uint2 srcStridesNH,
                                        T *dstPtr,
                                        uint2 dstStridesNH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float2 scaleAndShift,
                                        uint *roiTensor,
                                        uint2 maxParamVolumeAndAxisMask,
                                        bool computeStdDev)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint *roi = &roiTensor[id_z * 4];
    uint yBegin = roi[0];
    uint xBegin = roi[1];
    uint height = roi[2];
    uint width = roi[3];

    if (id_x >= width || id_y >= height)
        return;

    uint maxParamVolume = maxParamVolumeAndAxisMask.x;
    uint axisMask = maxParamVolumeAndAxisMask.y;
    uint paramIndex = id_z * maxParamVolume;
    // update paramIndex based on axisMask value
    if(axisMask == 1)
        paramIndex += id_x;
    else if(axisMask == 2)
        paramIndex += id_y;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + yBegin) * srcStridesNH.y) + id_x + xBegin;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float scale = scaleAndShift.x;
    float shift = scaleAndShift.y;
    float invStdDev;
    if(computeStdDev)
    {
        float stdDevSquare = stdDev * stdDev;
        invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    }
    else
    {
        invStdDev = (stdDev == 0.0f) ? 1.0f : (1.0f / stdDev);
    }
    float outVal = fmaf((static_cast<float>(srcPtr[srcIdx]) - mean), invStdDev, shift);
    normalize_check_and_store(outVal, &dstPtr[dstIdx]);
}

template <typename T>
__global__ void normalize_3d_hip_tensor(T *srcPtr,
                                        uint2 srcStridesDH,
                                        T *dstPtr,
                                        uint2 dstStridesDH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float2 scaleAndShift,
                                        uint *roiTensor,
                                        uint axisMask,
                                        bool computeStdDev)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // lengthZ

    uint *roi = roiTensor;
    uint zBegin = roi[0];
    uint yBegin = roi[1];
    uint xBegin = roi[2];
    uint lengthZ = roi[3];
    uint lengthY = roi[4];
    uint lengthX = roi[5];

    if (id_x >= lengthX || id_y >= lengthY || id_z >= lengthZ)
        return;

    uint paramIndex = 0;
    // update paramIndex based on axisMask value
    if(axisMask == 1)
        paramIndex += id_y * lengthX + id_x;
    else if(axisMask == 2)
        paramIndex += id_z * lengthX + id_x;
    else if(axisMask == 4)
        paramIndex += id_z * lengthY + id_y;
    else if(axisMask == 3)
        paramIndex += id_x;
    else if(axisMask == 5)
        paramIndex += id_y;
    else if(axisMask == 6)
        paramIndex += id_z;

    uint srcIdx = ((id_z + zBegin) * srcStridesDH.x) + ((id_y + yBegin) * srcStridesDH.y) + id_x + xBegin;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x;
    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float scale = scaleAndShift.x;
    float shift = scaleAndShift.y;
    float invStdDev;
    if(computeStdDev)
    {
        float stdDevSquare = stdDev * stdDev;
        invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    }
    else
    {
        invStdDev = (stdDev == 0.0f) ? 1.0f : (1.0f / stdDev);
    }
    float outVal = fmaf((static_cast<float>(srcPtr[srcIdx]) - mean), invStdDev, shift);
    normalize_check_and_store(outVal, &dstPtr[dstIdx]);
}

template <typename T>
__global__ void normalize_nd_hip_tensor(T *srcPtr,
                                        uint *srcMaxDims,
                                        uint *srcStrides,
                                        T *dstPtr,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        float2 scaleAndShift,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint2 maxParamVolumeAndBufferLength,
                                        uint numDims,
                                        bool computeStdDev)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint maxBufferLength = maxParamVolumeAndBufferLength.y;

    if(id_x >= maxBufferLength)
        return;

    uint *begin = &roiTensor[id_z * numDims * 2];
    uint *length = &roiTensor[id_z * numDims * 2 + numDims];
    uint *paramShape = &paramShapeTensor[id_z * numDims];
    uint *paramStrides = &paramStridesTensor[id_z * numDims];
    uint maxParamVolume = maxParamVolumeAndBufferLength.x;
    uint srcIdx = id_z * maxBufferLength;

    uint paramIndex = id_z * maxParamVolume;
    for (int i = 0; i < numDims; i++)
    {
        uint coord = id_x / srcStrides[i] % srcMaxDims[i];
        srcIdx += ((begin[i] + coord) * srcStrides[i]);
        if(coord >= length[i])
            return;
        paramIndex += (maxParamVolume != 1) ? (rpp_hip_mod(coord, paramShape[i]) * paramStrides[i]) : 0;
    }

    float mean = meanTensor[paramIndex];
    float stdDev = stdDevTensor[paramIndex];
    float scale = scaleAndShift.x;
    float shift = scaleAndShift.y;
    float invStdDev;
    if(computeStdDev)
    {
        float stdDevSquare = stdDev * stdDev;
        invStdDev = stdDevSquare ? rsqrtf(stdDevSquare) * scale : 0;
    }
    else
    {
        invStdDev = (stdDev == 0.0f) ? 1.0f : (1.0f / stdDev);
    }
    uint dstIdx = id_z * maxBufferLength + id_x;
    float outVal = fmaf((static_cast<float>(srcPtr[srcIdx]) - mean), invStdDev, shift);
    normalize_check_and_store(outVal, &dstPtr[dstIdx]);
}

// -------------------- Set 3 - mean and stddev compute kernels device helpers --------------------

__device__ __forceinline__ void reduction_sum_x_hip(float *partialSum_smem)
{
    for(uint threadMax = hipBlockDim_x / 2; threadMax >= 1; threadMax /= 2)
    {
        if(hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }
}

// -------------------- Set 4 - mean compute kernels (reduction stage 1) --------------------

template <typename T>
__global__ void compute_mean_2d_hip_tensor(T *srcPtr,
                                           uint2 srcStridesNH,
                                           float *meanTensor,
                                           float *partialSumTensor,
                                           uint *roiTensor,
                                           uint maxParamVolume,
                                           uint axisMask)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 4];
    uint yBegin = roi[0];
    uint xBegin = roi[1];
    uint height = roi[2];
    uint width = roi[3];

    // perform column wise sum
    if(axisMask == 1)
    {
        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        uint srcIdx = (id_z * srcStridesNH.x) + (yBegin * srcStridesNH.y) + (id_x + xBegin);
        uint dstIdx = id_z * maxParamVolume + id_x;
        if(id_x < width)
        {
            float accum = 0.0f;
            for(int i = 0; i < height; i++)
            {
                accum += static_cast<float>(srcPtr[srcIdx]);
                srcIdx += srcStridesNH.y;
            }
            meanTensor[dstIdx] = accum / static_cast<float>(height);
        }
    }
    // perform row wise sum
    else if(axisMask == 2)
    {
        id_x *= 8;
        __shared__ float partialRowSum_smem[256];
        partialRowSum_smem[hipThreadIdx_x] = 0.0f;

        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        int xAlignedLength =  width & ~7;      // alignedLength for vectorized global loads
        int xDiff = width - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + yBegin) * srcStridesNH.y) + (id_x + xBegin);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        if (id_x + 8 > width)
        {
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        }
        src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
        partialRowSum_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                              src_f8.f1[1] +
                                              src_f8.f1[2] +
                                              src_f8.f1[3]);                    // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();

        // Now do block level reduction sum
        reduction_sum_x_hip(partialRowSum_smem);

        // Final store to dst
        if(hipThreadIdx_x == 0)
        {
            uint paramIndex = (id_z * hipGridDim_y * hipGridDim_x) + (id_y * hipGridDim_x) + hipBlockIdx_x;
            partialSumTensor[paramIndex] = partialRowSum_smem[0];
        }
    }
    else if(axisMask == 3)
    {
        id_x *= 8;
        __shared__ float partialSum_smem[16][16];                               // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
        float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];     // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                           // initialization of Shared to 0.0f using all 16 x 16 threads

        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        int xAlignedLength =  width & ~7;       // alignedLength for vectorized global loads
        int xDiff = width - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + yBegin) * srcStridesNH.y) + (id_x + xBegin);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        if (id_x + 8 > width)
        {
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        }
        src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
        partialSumRowPtr_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                                src_f8.f1[1] +
                                                src_f8.f1[2] +
                                                src_f8.f1[3]);                 // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();                                                       // syncthreads after Shared load

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        reduction_sum_x_hip(partialSumRowPtr_smem);

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
                partialSumTensor[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumRowPtr_smem[0];
        }
    }
}

template <typename T>
__global__ void compute_mean_3d_hip_tensor(T *srcPtr,
                                           uint3 srcStridesNZY,
                                           float *meanTensor,
                                           uint *roiTensor,
                                           float *partialSumTensor,
                                           uint maxParamVolume,
                                           uint axisMask)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 6];
    uint zBegin = roi[0];
    uint yBegin = roi[1];
    uint xBegin = roi[2];
    uint lengthZ = roi[3];
    uint lengthY = roi[4];
    uint lengthX = roi[5];

    // compute mean along z direction
    if(axisMask == 1)
    {
        if(id_x >= lengthX || id_y >= lengthY)
            return;

        uint srcIdx = (id_z * srcStridesNZY.x) + (zBegin * srcStridesNZY.y) + ((id_y + yBegin) * srcStridesNZY.z) + (id_x + xBegin);
        uint dstIdx = id_z * maxParamVolume + id_y * lengthX + id_x;
        float accum = 0.0f;
        for(uint i = 0; i < lengthZ; i++)
        {
            accum += static_cast<float>(srcPtr[srcIdx]);
            srcIdx += srcStridesNZY.y;
        }
        meanTensor[dstIdx] = accum / static_cast<float>(lengthZ);
    }
    // compute mean along y direction
    else if(axisMask == 2)
    {
        if(id_x >= lengthX || id_y >= lengthZ)
            return;

        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + (yBegin * srcStridesNZY.z) + (id_x + xBegin);
        uint dstIdx = id_z * maxParamVolume +  id_y * lengthX + id_x;
        float accum = 0.0f;
        for(uint i = 0; i < lengthY; i++)
        {
            accum += static_cast<float>(srcPtr[srcIdx]);
            srcIdx += srcStridesNZY.z;
        }
        meanTensor[dstIdx] = accum / static_cast<float>(lengthY);
    }
    // compute mean along x direction
    else if(axisMask == 4)
    {
        if(id_x >= lengthY || id_y >= lengthZ)
            return;

        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((id_x + yBegin) * srcStridesNZY.z) + xBegin;
        d_float8 accum_f8;
        accum_f8.f4[0] = (float4)0.0f;
        accum_f8.f4[1] = (float4)0.0f;
        for(int i = 0; i < lengthX; i += 8)
        {
            d_float8 src_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
            if (i + 8 > lengthX)
            {
                int xDiff = i + 8 - lengthX;
                for(int i = xDiff; i < 8; i++)
                    src_f8.f1[i] = 0.0f;
            }
            accum_f8.f4[0] += src_f8.f4[0];
            accum_f8.f4[1] += src_f8.f4[1];
            srcIdx += 8;
        }
        accum_f8.f4[0] += accum_f8.f4[1];
        accum_f8.f1[0] = (accum_f8.f1[0] + accum_f8.f1[1] + accum_f8.f1[2] + accum_f8.f1[3]);
        uint dstIdx = id_z * maxParamVolume + id_y * lengthY + id_x;
        meanTensor[dstIdx] = accum_f8.f1[0] / static_cast<float>(lengthX);
    }
     // compute mean along z-y direction
    else if(axisMask == 3)
    {
        for(uint xIndex = 0; xIndex < lengthX; xIndex++)
        {
            __shared__ float partialSum_smem[16][16];
            float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];              // float pointer to beginning of each row in Shared
            partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                                    // initialization of Shared to 0.0f using all 16 x 16 threads

            if ((id_x >= lengthY) || (id_y >= lengthZ))
            {
                return;
            }

            uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((id_x + yBegin) * srcStridesNZY.z) + (xBegin + xIndex);
            partialSumRowPtr_smem[hipThreadIdx_x] = static_cast<float>(srcPtr[srcIdx]);
            __syncthreads();                                                                      // syncthreads after Shared load

            // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
            for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
            {
                if (hipThreadIdx_x < threadMax)
                    partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
                __syncthreads();
            }

            if (hipThreadIdx_x == 0)
            {
                // Reduction of 16 floats on 16 threads per block in z dimension
                for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
                {
                    if (hipThreadIdx_y < threadMax)
                        partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                    __syncthreads();
                }

                // Final store to dst
                if (hipThreadIdx_y == 0)
                {
                    uint dstIdx = (id_z * srcStridesNZY.z * hipGridDim_y * hipGridDim_x) + (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x) + (xIndex * hipGridDim_y * hipGridDim_x);
                    partialSumTensor[dstIdx] = partialSumRowPtr_smem[0];
                }
            }
            __syncthreads();
        }
    }
    // compute mean along y-x direction
    else if(axisMask == 6)
    {
        __shared__ float partialSum_smem[256];
        partialSum_smem[hipThreadIdx_x] = 0.0f;
        __syncthreads();

        if(id_x >= lengthY || id_y >= lengthZ)
            return;

        uint maxLengthZ = srcStridesNZY.x / srcStridesNZY.y;
        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((id_x + yBegin) * srcStridesNZY.z) + xBegin;
        d_float8 accum_f8;
        accum_f8.f4[0] = (float4)0.0f;
        accum_f8.f4[1] = (float4)0.0f;
        for(int i = 0; i < lengthX; i += 8)
        {
            d_float8 src_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
            if (i + 8 > lengthX)
            {
                int xDiff = lengthX - i;
                for(int j = xDiff; j < 8; j++)
                    src_f8.f1[j] = 0.0f;
            }
            accum_f8.f4[0] += src_f8.f4[0];
            accum_f8.f4[1] += src_f8.f4[1];
            srcIdx += 8;
        }
        accum_f8.f4[0] += accum_f8.f4[1];
        accum_f8.f1[0] = (accum_f8.f1[0] + accum_f8.f1[1] + accum_f8.f1[2] + accum_f8.f1[3]);
        partialSum_smem[hipThreadIdx_x] = accum_f8.f1[0];
        __syncthreads();

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        reduction_sum_x_hip(partialSum_smem);

        if(hipThreadIdx_x == 0)
        {
            uint dstIdx = (id_z * maxLengthZ * hipGridDim_x) + hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x;
            partialSumTensor[dstIdx] = partialSum_smem[0];
        }
    }
    // compute mean along z-x direction
    else if(axisMask == 5)
    {
        __shared__ float partialSum_smem[32];
        partialSum_smem[hipThreadIdx_x] = 0.0f;

        if(hipBlockIdx_x >= lengthY)
            return;

        uint dstIdx = id_z * maxParamVolume +  hipBlockIdx_x;
        float accum = 0.0f;
        for (uint i = 0; i < lengthZ; i++)
        {
            uint tid_x = hipThreadIdx_x;
            uint srcIdx = (id_z * srcStridesNZY.x) + ((i + zBegin) * srcStridesNZY.y) + ((hipBlockIdx_x + yBegin) * srcStridesNZY.z) + xBegin;
            while (tid_x < lengthX)
            {
                accum += static_cast<float>(srcPtr[srcIdx + tid_x]);
                tid_x += hipBlockDim_x;
            }
        }
        partialSum_smem[hipThreadIdx_x] = accum;
        __syncthreads();

        // perform reduction on shared memory sums
        reduction_sum_x_hip(partialSum_smem);

        if(hipThreadIdx_x == 0)
            meanTensor[dstIdx] = partialSum_smem[0] / static_cast<float>(lengthX * lengthZ);
    }
    // compute mean along z-y-x direction
    else if(axisMask == 7)
    {
        id_x *= 8;
        __shared__ float partialSum_smem[16][16];
        float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];              // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                                    // initialization of Shared to 0.0f using all 16 x 16 threads

        uint xIndex = id_x % srcStridesNZY.z;
        uint yIndex = id_x / srcStridesNZY.z;
        if ((xIndex >= lengthX) || (yIndex >= lengthY) || (id_y >= lengthZ))
        {
            return;
        }

        int xAlignedLength =  lengthX & ~7;       // alignedLength for vectorized global loads
        int xDiff = lengthX - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((yIndex + yBegin) * srcStridesNZY.z) + (xIndex + xBegin);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        if (xIndex + 8 > lengthX)
        {
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        }
        src_f8.f4[0] += src_f8.f4[1];
        partialSumRowPtr_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                                 src_f8.f1[1] +
                                                 src_f8.f1[2] +
                                                 src_f8.f1[3]);
        __syncthreads();

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        reduction_sum_x_hip(partialSumRowPtr_smem);

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
            {
                uint dstIdx = (id_z * hipGridDim_y * hipGridDim_x) + (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x);
                partialSumTensor[dstIdx] = partialSumRowPtr_smem[0];
            }
        }
    }
}

template <typename T>
__global__ void compute_mean_nd_hip_tensor(T *srcPtr,
                                           uint *srcMaxDims,
                                           uint *srcStrides,
                                           float *meanTensor,
                                           uint *roiTensor,
                                           uint *paramShapeTensor,
                                           uint *paramStridesTensor,
                                           uint maxParamVolume,
                                           uint numDims,
                                           uint maxBufferLength)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *begin = &roiTensor[id_z * numDims * 2];
    uint *length = &roiTensor[id_z * numDims * 2 + numDims];
    uint *paramShape = &paramShapeTensor[id_z * numDims];
    uint *paramStrides = &paramStridesTensor[id_z * numDims];
    uint srcIdx = id_z * maxBufferLength;
    uint paramBase = id_z * maxParamVolume;
    uint paramIndex = 0;

    if(maxParamVolume > MAX_SHARED_MEMORY_SIZE)
    {
        if(id_x >= maxBufferLength)
            return;

        // validate if id_x is within the roi of input and compute paramIndex if valid
        for (int i = 0; i < numDims; i++)
        {
            uint coord = id_x / srcStrides[i] % srcMaxDims[i];
            srcIdx += ((begin[i] + coord) * srcStrides[i]);
            if(coord >= length[i])
                return;
            paramIndex += (maxParamVolume > 1) ? (rpp_hip_mod(coord, paramShape[i]) * paramStrides[i]) : 0;
        }
        atomicAdd(&meanTensor[paramBase + paramIndex], static_cast<float>(srcPtr[srcIdx]));
    }
    else
    {

        if(id_x >= (hipBlockDim_x * hipGridDim_x))
            return;

        // if number of means needed to compute is within in the max shared memory size
        // use shared memory for atomic addition to reduce global memory traffic
        bool isValid = true;
        for (int i = 0; i < numDims; i++)
        {
            uint coord = id_x / srcStrides[i] % srcMaxDims[i];
            srcIdx += ((begin[i] + coord) * srcStrides[i]);
            if(coord >= length[i])
            {
                isValid = false;
                break;
            }
            paramIndex += (maxParamVolume > 1) ? (rpp_hip_mod(coord, paramShape[i]) * paramStrides[i]) : 0;
        }

        extern __shared__ float sh_mem[];
        sh_mem[hipThreadIdx_x] = 0.0f;
        __syncthreads();

        if(isValid && id_x < maxBufferLength)
            atomicAdd(&sh_mem[paramIndex], static_cast<float>(srcPtr[srcIdx]));
        __syncthreads();

        if (hipThreadIdx_x < maxParamVolume)
            atomicAdd(&meanTensor[paramBase + hipThreadIdx_x], sh_mem[hipThreadIdx_x]);
    }
}

// -------------------- Set 5 - stddev compute kernels (reduction stage 1) --------------------

template <typename T>
__global__ void compute_stddev_2d_hip_tensor(T *srcPtr,
                                             uint2 srcStridesNH,
                                             float *meanTensor,
                                             float *stdDevTensor,
                                             float *partialSumTensor,
                                             uint *roiTensor,
                                             uint maxParamVolume,
                                             uint axisMask)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 4];
    uint yBegin = roi[0];
    uint xBegin = roi[1];
    uint height = roi[2];
    uint width = roi[3];

    // perform column wise stddev
    if(axisMask == 1)
    {
        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        uint srcIdx = (id_z * srcStridesNH.x) + (yBegin * srcStridesNH.y) + (id_x + xBegin);
        uint paramIndex = id_z * maxParamVolume + id_x;
        float mean = meanTensor[paramIndex];
        if(id_x < width)
        {
            float accum = 0.0f;
            for(int i = 0; i < height; i++)
            {
                float val = (static_cast<float>(srcPtr[srcIdx]) - mean);
                accum += (val * val);
                srcIdx += srcStridesNH.y;
            }
            stdDevTensor[paramIndex] = sqrtf(accum / static_cast<float>(height));
        }
    }
    // perform row wise sum
    else if(axisMask == 2)
    {
        id_x *= 8;
        __shared__ float partialRowSum_smem[256];
        partialRowSum_smem[hipThreadIdx_x] = 0.0f;

        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        int xAlignedLength =  width & ~7;      // alignedLength for vectorized global loads
        int xDiff = width - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + yBegin) * srcStridesNH.y) + (id_x + xBegin);

        uint paramIndex = id_z * maxParamVolume + id_y;
        float mean = meanTensor[paramIndex];
        float4 mean_f4 = static_cast<float4>(mean);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        rpp_hip_math_subtract8_const(&src_f8, &src_f8, mean_f4);
        rpp_hip_math_multiply8(&src_f8, &src_f8, &src_f8);

        if (id_x + 8 > width)
        {
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        }
        src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
        partialRowSum_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                              src_f8.f1[1] +
                                              src_f8.f1[2] +
                                              src_f8.f1[3]);                    // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();

        // Now do block level reduction sum
        reduction_sum_x_hip(partialRowSum_smem);

        // Final store to dst
        if(hipThreadIdx_x == 0)
        {
            uint paramIndex = (id_z * hipGridDim_y * hipGridDim_x) + (id_y * hipGridDim_x) + hipBlockIdx_x;
            partialSumTensor[paramIndex] = partialRowSum_smem[0];
        }
    }
    else if(axisMask == 3)
    {
        id_x *= 8;
        __shared__ float partialSum_smem[16][16];                               // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
        float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];     // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                           // initialization of Shared to 0.0f using all 16 x 16 threads

        if ((id_y >= height) || (id_x >= width))
        {
            return;
        }

        int xAlignedLength =  width & ~7;       // alignedLength for vectorized global loads
        int xDiff = width - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + yBegin) * srcStridesNH.y) + (id_x + xBegin);

        float mean = meanTensor[id_z];
        float4 mean_f4 = static_cast<float4>(mean);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        rpp_hip_math_subtract8_const(&src_f8, &src_f8, mean_f4);
        rpp_hip_math_multiply8(&src_f8, &src_f8, &src_f8);
        if (id_x + 8 > width)
        {
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        }
        src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
        partialSumRowPtr_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                                src_f8.f1[1] +
                                                src_f8.f1[2] +
                                                src_f8.f1[3]);                 // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();                                                        // syncthreads after Shared load

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        reduction_sum_x_hip(partialSumRowPtr_smem);

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
                partialSumTensor[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumRowPtr_smem[0];
        }
    }
}

template <typename T>
__global__ void compute_stddev_3d_hip_tensor(T *srcPtr,
                                             uint3 srcStridesNZY,
                                             float *meanTensor,
                                             float *stdDevTensor,
                                             uint *roiTensor,
                                             float *partialSumTensor,
                                             uint maxParamVolume,
                                             uint axisMask)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 6];
    uint zBegin = roi[0];
    uint yBegin = roi[1];
    uint xBegin = roi[2];
    uint lengthZ = roi[3];
    uint lengthY = roi[4];
    uint lengthX = roi[5];

    // compute stddev along z direction
    if(axisMask == 1)
    {
        if(id_x >= lengthX || id_y >= lengthY)
            return;

        uint srcIdx = (id_z * srcStridesNZY.x) + (zBegin * srcStridesNZY.y) + ((id_y + yBegin) * srcStridesNZY.z) + (id_x + xBegin);
        uint paramIndex = id_z * maxParamVolume + id_y * lengthX + id_x;
        float mean = meanTensor[paramIndex];
        float accum = 0.0f;
        for(uint i = 0; i < lengthZ; i++)
        {
            float val = (static_cast<float>(srcPtr[srcIdx]) - mean);
            accum += (val * val);
            srcIdx += srcStridesNZY.y;
        }
        stdDevTensor[paramIndex] = sqrtf(accum / static_cast<float>(lengthZ));
    }
    // compute stddev along y direction
    else if(axisMask == 2)
    {
        if(id_x >= lengthX || id_y >= lengthZ)
            return;

        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + (yBegin * srcStridesNZY.z) + (id_x + xBegin);
        uint paramIndex = id_z * maxParamVolume +  id_y * lengthX + id_x;
        float mean = meanTensor[paramIndex];
        float accum = 0.0f;
        for(uint i = 0; i < lengthY; i++)
        {
            float val = (static_cast<float>(srcPtr[srcIdx]) - mean);
            accum += (val * val);
            srcIdx += srcStridesNZY.z;
        }
        stdDevTensor[paramIndex] = sqrtf(accum / static_cast<float>(lengthY));
    }
    // compute stddev along x direction
    else if(axisMask == 4)
    {
        if(id_x >= lengthY || id_y >= lengthZ)
            return;

        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((id_x + yBegin) * srcStridesNZY.z) + xBegin;
        uint paramIndex = id_z * maxParamVolume + id_y * lengthY + id_x;
        float mean = meanTensor[paramIndex];
        float4 mean_f4 = static_cast<float4>(mean);
        d_float8 accum_f8;
        accum_f8.f4[0] = (float4)0.0f;
        accum_f8.f4[1] = (float4)0.0f;
        for(int i = 0; i < lengthX; i += 8)
        {
            d_float8 src_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
            rpp_hip_math_subtract8_const(&src_f8, &src_f8, mean_f4);
            rpp_hip_math_multiply8(&src_f8, &src_f8, &src_f8);
            if (i + 8 > lengthX)
            {
                int xDiff = i + 8 - lengthX;
                for(int i = xDiff; i < 8; i++)
                    src_f8.f1[i] = 0.0f;
            }
            accum_f8.f4[0] += src_f8.f4[0];
            accum_f8.f4[1] += src_f8.f4[1];
            srcIdx += 8;
        }
        accum_f8.f4[0] += accum_f8.f4[1];
        accum_f8.f1[0] = (accum_f8.f1[0] + accum_f8.f1[1] + accum_f8.f1[2] + accum_f8.f1[3]);

        stdDevTensor[paramIndex] =  sqrtf(accum_f8.f1[0] / static_cast<float>(lengthX));
    }
    // compute stddev along z-y direction
    else if(axisMask == 3)
    {
        for(uint xIndex = 0; xIndex < lengthX; xIndex++)
        {
            __shared__ float partialSum_smem[16][16];
            float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];              // float pointer to beginning of each row in Shared
            partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                                    // initialization of Shared to 0.0f using all 16 x 16 threads

            if ((id_x >= lengthY) || (id_y >= lengthZ))
            {
                return;
            }

            uint paramIndex = id_z * maxParamVolume + xIndex;
            float mean = meanTensor[paramIndex];
            uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((id_x + yBegin) * srcStridesNZY.z) + (xBegin + xIndex);
            float val = static_cast<float>(srcPtr[srcIdx]) - mean;
            partialSumRowPtr_smem[hipThreadIdx_x] = (val * val);
            __syncthreads();                                                                      // syncthreads after Shared load

            // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
            for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
            {
                if (hipThreadIdx_x < threadMax)
                    partialSumRowPtr_smem[hipThreadIdx_x] += partialSumRowPtr_smem[hipThreadIdx_x + threadMax];
                __syncthreads();
            }

            if (hipThreadIdx_x == 0)
            {
                // Reduction of 16 floats on 16 threads per block in z dimension
                for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
                {
                    if (hipThreadIdx_y < threadMax)
                        partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                    __syncthreads();
                }

                // Final store to dst
                if (hipThreadIdx_y == 0)
                {
                    uint dstIdx = (id_z * srcStridesNZY.z * hipGridDim_y * hipGridDim_x) + (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x) + (xIndex * hipGridDim_y * hipGridDim_x);
                    partialSumTensor[dstIdx] = partialSumRowPtr_smem[0];
                }
            }
            __syncthreads();
        }
    }
    // compute stddev along y-x direction
    else if(axisMask == 6)
    {
        __shared__ float partialSum_smem[256];
        partialSum_smem[hipThreadIdx_x] = 0.0f;
        __syncthreads();

        if(id_x >= lengthY || id_y >= lengthZ)
            return;

        uint maxLengthZ = srcStridesNZY.x / srcStridesNZY.y;
        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((id_x + yBegin) * srcStridesNZY.z) + xBegin;

        uint paramIndex = id_z * maxParamVolume + id_y;
        float mean = meanTensor[paramIndex];
        float4 mean_f4 = static_cast<float4>(mean);

        d_float8 accum_f8;
        accum_f8.f4[0] = (float4)0.0f;
        accum_f8.f4[1] = (float4)0.0f;
        for(int i = 0; i < lengthX; i += 8)
        {
            d_float8 src_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
            rpp_hip_math_subtract8_const(&src_f8, &src_f8, mean_f4);
            rpp_hip_math_multiply8(&src_f8, &src_f8, &src_f8);
            if (i + 8 > lengthX)
            {
                int xDiff = lengthX - i;
                for(int j = xDiff; j < 8; j++)
                    src_f8.f1[j] = 0.0f;
            }
            accum_f8.f4[0] += src_f8.f4[0];
            accum_f8.f4[1] += src_f8.f4[1];
            srcIdx += 8;
        }
        accum_f8.f4[0] += accum_f8.f4[1];
        accum_f8.f1[0] = (accum_f8.f1[0] + accum_f8.f1[1] + accum_f8.f1[2] + accum_f8.f1[3]);
        partialSum_smem[hipThreadIdx_x] = accum_f8.f1[0];
        __syncthreads();

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        reduction_sum_x_hip(partialSum_smem);

        if(hipThreadIdx_x == 0)
        {
            uint dstIdx = (id_z * maxLengthZ * hipGridDim_x) + hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x;
            partialSumTensor[dstIdx] = partialSum_smem[0];
        }
    }
    // compute stddev along z-x direction
    else if(axisMask == 5)
    {
        __shared__ float partialSum_smem[32];
        partialSum_smem[hipThreadIdx_x] = 0.0f;

        if(hipBlockIdx_x >= lengthY)
            return;

        uint paramIndex = id_z * maxParamVolume +  hipBlockIdx_x;
        float mean = meanTensor[paramIndex];
        float accum = 0.0f;
        for (uint i = 0; i < lengthZ; i++)
        {
            uint tid_x = hipThreadIdx_x;
            uint srcIdx = (id_z * srcStridesNZY.x) + ((i + zBegin) * srcStridesNZY.y) + ((hipBlockIdx_x + yBegin) * srcStridesNZY.z) + xBegin;
            while (tid_x < lengthX)
            {
                float val = (static_cast<float>(srcPtr[srcIdx + tid_x]) - mean);
                accum += (val * val);
                tid_x += hipBlockDim_x;
            }
        }
        partialSum_smem[hipThreadIdx_x] = accum;
        __syncthreads();

        // perform reduction on shared memory sums
        reduction_sum_x_hip(partialSum_smem);

        if(hipThreadIdx_x == 0)
            stdDevTensor[paramIndex] = sqrtf(partialSum_smem[0] / static_cast<float>(lengthX * lengthZ));
    }
    // compute stddev along z-y-x direction
    else if(axisMask == 7)
    {
       id_x *= 8;
        __shared__ float partialSum_smem[16][16];
        float *partialSumRowPtr_smem = &partialSum_smem[hipThreadIdx_y][0];              // float pointer to beginning of each row in Shared
        partialSumRowPtr_smem[hipThreadIdx_x] = 0.0f;                                    // initialization of Shared to 0.0f using all 16 x 16 threads

        uint xIndex = id_x % srcStridesNZY.z;
        uint yIndex = id_x / srcStridesNZY.z;
        if ((xIndex >= lengthX) || (yIndex >= lengthY) || (id_y >= lengthZ))
        {
            return;
        }

        int xAlignedLength =  lengthX & ~7;       // alignedLength for vectorized global loads
        int xDiff = lengthX - xAlignedLength;    // difference between roiWidth and alignedLength
        uint srcIdx = (id_z * srcStridesNZY.x) + ((id_y + zBegin) * srcStridesNZY.y) + ((yIndex + yBegin) * srcStridesNZY.z) + (xIndex + xBegin);

        uint paramIndex = id_z * maxParamVolume;
        float mean = meanTensor[paramIndex];
        float4 mean_f4 = static_cast<float4>(mean);

        d_float8 src_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
        rpp_hip_math_subtract8_const(&src_f8, &src_f8, mean_f4);
        rpp_hip_math_multiply8(&src_f8, &src_f8, &src_f8);

        if (xIndex + 8 > lengthX)
        {
            for(int i = xDiff; i < 8; i++)
                src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
        }
        src_f8.f4[0] += src_f8.f4[1];
        partialSumRowPtr_smem[hipThreadIdx_x] = (src_f8.f1[0] +
                                                 src_f8.f1[1] +
                                                 src_f8.f1[2] +
                                                 src_f8.f1[3]);                 // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
        __syncthreads();                                                        // syncthreads after Shared load

        // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
        reduction_sum_x_hip(partialSumRowPtr_smem);

        if (hipThreadIdx_x == 0)
        {
            // Reduction of 16 floats on 16 threads per block in y dimension
            for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
            {
                if (hipThreadIdx_y < threadMax)
                    partialSumRowPtr_smem[0] += partialSumRowPtr_smem[increment];
                __syncthreads();
            }

            // Final store to dst
            if (hipThreadIdx_y == 0)
            {
                uint dstIdx = (id_z * hipGridDim_y * hipGridDim_x) + (hipBlockIdx_y * hipGridDim_x + hipBlockIdx_x);
                partialSumTensor[dstIdx] = partialSumRowPtr_smem[0];
            }
        }
    }
}

template <typename T>
__global__ void compute_stddev_nd_hip_tensor(T *srcPtr,
                                             uint *srcMaxDims,
                                             uint *srcStrides,
                                             float *meanTensor,
                                             float *stdDevTensor,
                                             uint *roiTensor,
                                             uint *paramShapeTensor,
                                             uint *paramStridesTensor,
                                             uint maxParamVolume,
                                             uint numDims,
                                             uint maxBufferLength)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *begin = &roiTensor[id_z * numDims * 2];
    uint *length = &roiTensor[id_z * numDims * 2 + numDims];
    uint *paramShape = &paramShapeTensor[id_z * numDims];
    uint *paramStrides = &paramStridesTensor[id_z * numDims];
    uint srcIdx = id_z * maxBufferLength;
    uint paramBase = id_z * maxParamVolume;
    uint paramIndex = 0;

    if(maxParamVolume > MAX_SHARED_MEMORY_SIZE)
    {
        if(id_x >= maxBufferLength)
            return;

        // validate if id_x is within the roi of input and compute paramIndex if valid
        for (int i = 0; i < numDims; i++)
        {
            uint coord = id_x / srcStrides[i] % srcMaxDims[i];
            srcIdx += ((begin[i] + coord) * srcStrides[i]);
            if(coord >= length[i])
                return;
            paramIndex += (maxParamVolume > 1) ? (rpp_hip_mod(coord, paramShape[i]) * paramStrides[i]) : 0;
        }
        float val = static_cast<float>(srcPtr[srcIdx]) - meanTensor[paramBase + paramIndex];
        atomicAdd(&stdDevTensor[paramBase + paramIndex], (val * val));
    }
    else
    {

        if(id_x >= (hipBlockDim_x * hipGridDim_x))
            return;

        // if number of means needed to compute is within in the max shared memory size
        // use shared memory for atomic addition to reduce global memory traffic
        bool isValid = true;
        for (int i = 0; i < numDims; i++)
        {
            uint coord = id_x / srcStrides[i] % srcMaxDims[i];
            srcIdx += ((begin[i] + coord) * srcStrides[i]);
            if(coord >= length[i])
            {
                isValid = false;
                break;
            }
            paramIndex += (maxParamVolume > 1) ? (rpp_hip_mod(coord, paramShape[i]) * paramStrides[i]) : 0;
        }

        extern __shared__ float sh_mem[];
        sh_mem[hipThreadIdx_x] = 0.0f;
        __syncthreads();

        if(isValid && id_x < maxBufferLength)
        {
            float val = static_cast<float>(srcPtr[srcIdx]) - meanTensor[paramBase + paramIndex];
            atomicAdd(&sh_mem[paramIndex], (val * val));
        }
        __syncthreads();

        if (hipThreadIdx_x < maxParamVolume)
            atomicAdd(&stdDevTensor[paramBase + hipThreadIdx_x], sh_mem[hipThreadIdx_x]);
    }
}

// -------------------- Set 6 - mean and stddev compute kernels (reduction stage 2) --------------------

__global__ void reduce_final_result_hip(float *partialSumTensor,
                                        uint numPartialSums,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        bool isMean,
                                        uint *roiTensor,
                                        uint axisMask,
                                        uint numDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    uint *roi = &roiTensor[id_z * numDims * 2 + numDims];

    uint meanFactor;
    if(numDims == 3)
    {
        uint lengthZ = roi[0];
        uint lengthY = roi[1];
        uint lengthX = roi[2];

        if(axisMask == 3)
            meanFactor = lengthZ * lengthY;
        else if(axisMask == 6)
            meanFactor = lengthY * lengthX;
        else if(axisMask == 7)
            meanFactor = lengthZ * lengthY * lengthX;
    }
    else if(numDims == 2)
    {
        uint lengthY = roi[0];
        uint lengthX = roi[1];

        if(axisMask == 2)
            meanFactor = lengthX;
        else if(axisMask == 3)
            meanFactor = lengthY * lengthX;
    }

    __shared__ float partialSum_smem[16];
    partialSum_smem[hipThreadIdx_x] = 0.0f;

    float accum = 0.0f;
    while(id_x < numPartialSums)
    {
        uint srcIdx = (id_z * hipGridDim_y * numPartialSums) + (id_y * numPartialSums) + id_x;
        accum += partialSumTensor[srcIdx];
        id_x += hipBlockDim_x;
    }
    partialSum_smem[hipThreadIdx_x] = accum;
    __syncthreads();

    // Now do block level reduction sum
    reduction_sum_x_hip(partialSum_smem);

    // Final store to dst
    if(hipThreadIdx_x == 0)
    {
        if(isMean)
            meanTensor[id_z * hipGridDim_y + id_y] = partialSum_smem[0] / meanFactor;
        else
            stdDevTensor[id_z * hipGridDim_y + id_y] = sqrtf(partialSum_smem[0] / meanFactor);
    }
}

__global__ void final_reduction_nd_hip_tensor(float *meanTensor,
                                              float *stdDevTensor,
                                              uint *paramShapeTensor,
                                              uint *roiTensor,
                                              uint numDims,
                                              uint maxParamVolume,
                                              bool isMean)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *paramShape = &paramShapeTensor[id_z * numDims];
    uint *roi = &roiTensor[id_z * numDims * 2 + numDims];

    uint divisionFactor = 1;
    uint paramVolume = 1;
    for(int i = 0; i < numDims; i++)
    {
        paramVolume *= paramShape[i];
        if(paramShape[i] == 1)
            divisionFactor *= roi[i];
    }

    if(id_x >= paramVolume)
        return;

    uint paramIndex = id_z * maxParamVolume + id_x;
    if(isMean)
        meanTensor[paramIndex] = meanTensor[paramIndex] / divisionFactor;
    else
        stdDevTensor[paramIndex] = sqrtf(stdDevTensor[paramIndex] / divisionFactor);
}

// -------------------- Set 7 - mean and stddev compute kernels launch helpers --------------------

void set_kernel_launch_config_2d(RpptGenericDescPtr srcGenericDescPtr,
                                 int &globalThreads_x,
                                 int &globalThreads_y,
                                 int &globalThreads_z,
                                 int &localThreads_x,
                                 int &localThreads_y,
                                 int &localThreads_z,
                                 Rpp32u axisMask,
                                 Rpp32f *partialSumArr,
                                 rpp::Handle& handle)
{
    switch (axisMask)
    {
        // compute along Y direction
        case 1:
        {
            localThreads_x = 256;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = static_cast<int>(ceil((float)srcGenericDescPtr->dims[2] / localThreads_x));
            globalThreads_y = 1;
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along X direction
        case 2:
        {
            localThreads_x = 256;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)((srcGenericDescPtr->dims[2] + 7) >> 3) / 256));
            globalThreads_y = srcGenericDescPtr->dims[1];
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = srcGenericDescPtr->dims[0] * srcGenericDescPtr->dims[1] * globalThreads_x;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
        // compute along XY direction
        case 3:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)((srcGenericDescPtr->dims[2] + 7) >> 3) / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
    }
}

void set_kernel_launch_config_3d(RpptGenericDescPtr srcGenericDescPtr,
                                 int &globalThreads_x,
                                 int &globalThreads_y,
                                 int &globalThreads_z,
                                 int &localThreads_x,
                                 int &localThreads_y,
                                 int &localThreads_z,
                                 Rpp32u axisMask,
                                 Rpp32f *partialSumArr,
                                 rpp::Handle& handle)
{
    switch (axisMask)
    {
        // compute along Z direction
        case 1:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[3] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along Y direction
        case 2:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[3] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along YZ direction
        case 3:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
        // compute along X direction
        case 4:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along XZ direction
        case 5:
        {
            localThreads_x = 32;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = srcGenericDescPtr->dims[2];
            globalThreads_y = 1;
            globalThreads_z = srcGenericDescPtr->dims[0];
            break;
        }
        // compute along XY direction
        case 6:
        {
            localThreads_x = 256;
            localThreads_y = 1;
            localThreads_z = 1;
            globalThreads_x = static_cast<int> (ceil((float)srcGenericDescPtr->dims[2] / localThreads_x));
            globalThreads_y = srcGenericDescPtr->dims[1];
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
        // compute along XYZ direction
        case 7:
        {
            localThreads_x = 16;
            localThreads_y = 16;
            localThreads_z = 1;
            Rpp32u numValues = (srcGenericDescPtr->dims[2] * srcGenericDescPtr->dims[3] + 7) >> 3;
            globalThreads_x = static_cast<int> (ceil((float)numValues / localThreads_x));
            globalThreads_y = static_cast<int> (ceil((float)srcGenericDescPtr->dims[1] / localThreads_y));
            globalThreads_z = srcGenericDescPtr->dims[0];

            Rpp32u partialSumArrLength = globalThreads_x * globalThreads_y * globalThreads_z;
            hipMemsetAsync(partialSumArr, 0, partialSumArrLength * sizeof(Rpp32f), handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
            break;
        }
    }
}

// -------------------- Set 8 - mean and stddev compute kernels executor --------------------

template <typename T>
RppStatus hip_exec_compute_mean_stddev_tensor(T *srcPtr,
                                              RpptGenericDescPtr srcGenericDescPtr,
                                              Rpp32f *meanTensor,
                                              Rpp32f *stdDevTensor,
                                              bool isMean,
                                              Rpp32u *roiTensor,
                                              Rpp32u axisMask,
                                              Rpp32u numDims,
                                              Rpp32u maxParamVolume,
                                              Rpp32u *paramShape,
                                              Rpp32u *paramStrides,
                                              rpp::Handle& handle)
{
    Rpp32f *partialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
    Rpp32u partialSumArrLength, partialSumBlocksPerSample;

    int globalThreads_x, globalThreads_y, globalThreads_z;
    int localThreads_x, localThreads_y, localThreads_z;
    // based on number of dimensions call the corresponding kernel
    if (numDims == 2)
    {
        // set the block and grid configuration based on axisMask
        set_kernel_launch_config_2d(srcGenericDescPtr, globalThreads_x, globalThreads_y, globalThreads_z,
                                    localThreads_x, localThreads_y, localThreads_z, axisMask,
                                    partialSumArr, handle);

        if(isMean)
        {
            hipLaunchKernelGGL(compute_mean_2d_hip_tensor,
                               dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                               meanTensor,
                               partialSumArr,
                               roiTensor,
                               maxParamVolume,
                               axisMask);
        }
        else
        {
            hipLaunchKernelGGL(compute_stddev_2d_hip_tensor,
                               dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                               meanTensor,
                               stdDevTensor,
                               partialSumArr,
                               roiTensor,
                               maxParamVolume,
                               axisMask);
        }

        if(axisMask == 2)
        {
            partialSumBlocksPerSample = globalThreads_x;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               stdDevTensor,
                               isMean,
                               roiTensor,
                               axisMask,
                               numDims);
        }
        else if(axisMask == 3)
        {
            partialSumBlocksPerSample = globalThreads_x * globalThreads_y;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), 1, globalThreads_z),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               stdDevTensor,
                               isMean,
                               roiTensor,
                               axisMask,
                               numDims);
        }
    }
    else if (numDims == 3)
    {
        // set the block and grid configuration based on axisMask
        set_kernel_launch_config_3d(srcGenericDescPtr, globalThreads_x, globalThreads_y, globalThreads_z,
                                    localThreads_x, localThreads_y, localThreads_z, axisMask,
                                    partialSumArr, handle);

        if(isMean)
        {
            hipLaunchKernelGGL(compute_mean_3d_hip_tensor,
                               dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               meanTensor,
                               roiTensor,
                               partialSumArr,
                               maxParamVolume,
                               axisMask);
        }
        else
        {
            hipLaunchKernelGGL(compute_stddev_3d_hip_tensor,
                               dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               meanTensor,
                               stdDevTensor,
                               roiTensor,
                               partialSumArr,
                               maxParamVolume,
                               axisMask);
        }

        // perform final reduction on block wise sums for below cases
        // reduce on YZ partial sums
        if(axisMask == 3)
        {
            partialSumBlocksPerSample = globalThreads_x * globalThreads_y;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), srcGenericDescPtr->dims[3], srcGenericDescPtr->dims[0]),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               stdDevTensor,
                               isMean,
                               roiTensor,
                               axisMask,
                               numDims);
        }
        // reduce on XY partial sums
        if(axisMask == 6)
        {
            partialSumBlocksPerSample = globalThreads_x;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), srcGenericDescPtr->dims[1], srcGenericDescPtr->dims[0]),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               stdDevTensor,
                               isMean,
                               roiTensor,
                               axisMask,
                               numDims);
        }
        // reduce on XYZ block partial sums
        else if(axisMask == 7)
        {
            partialSumBlocksPerSample = globalThreads_x * globalThreads_y;
            hipLaunchKernelGGL(reduce_final_result_hip,
                               dim3(ceil((float)partialSumBlocksPerSample/16), 1, srcGenericDescPtr->dims[0]),
                               dim3(16, 1, 1),
                               0,
                               handle.GetStream(),
                               partialSumArr,
                               partialSumBlocksPerSample,
                               meanTensor,
                               stdDevTensor,
                               isMean,
                               roiTensor,
                               axisMask,
                               numDims);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        globalThreads_x = srcGenericDescPtr->strides[0];
        globalThreads_y = 1;
        globalThreads_z = srcGenericDescPtr->dims[0];
        Rpp32u batchSize = globalThreads_z;

        // allocate tensor for src strides
        Rpp32u *srcMaxDims = &srcGenericDescPtr->dims[1];
        Rpp32u *srcStrides = &srcGenericDescPtr->strides[1];

        Rpp32u shared_memory_size = 0;
        Rpp32u block_size = 1024;
        if(maxParamVolume <= MAX_SHARED_MEMORY_SIZE)
        {
            if(maxParamVolume <= 32)
                shared_memory_size = 32;
            else if(maxParamVolume <= 64)
                shared_memory_size = 64;
            else if(maxParamVolume <= 128)
                shared_memory_size = 128;
            else if(maxParamVolume <= 256)
                shared_memory_size = 256;
            else if(maxParamVolume <= 512)
                shared_memory_size = 512;
            else
                shared_memory_size = MAX_SHARED_MEMORY_SIZE;
            block_size = shared_memory_size;
        }

        if(isMean)
        {
            hipLaunchKernelGGL(compute_mean_nd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/block_size), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                               dim3(block_size, 1, 1),
                               shared_memory_size,
                               handle.GetStream(),
                               srcPtr,
                               srcMaxDims,
                               srcStrides,
                               meanTensor,
                               roiTensor,
                               paramShape,
                               paramStrides,
                               maxParamVolume,
                               numDims,
                               srcGenericDescPtr->strides[0]);
        }
        else
        {
            hipLaunchKernelGGL(compute_stddev_nd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/block_size), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                               dim3(block_size, 1, 1),
                               shared_memory_size,
                               handle.GetStream(),
                               srcPtr,
                               srcMaxDims,
                               srcStrides,
                               meanTensor,
                               stdDevTensor,
                               roiTensor,
                               paramShape,
                               paramStrides,
                               maxParamVolume,
                               numDims,
                               srcGenericDescPtr->strides[0]);
        }
        hipLaunchKernelGGL(final_reduction_nd_hip_tensor,
                           dim3(ceil((float)maxParamVolume/1024), 1, globalThreads_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           meanTensor,
                           stdDevTensor,
                           paramShape,
                           roiTensor,
                           numDims,
                           maxParamVolume,
                           isMean);
    }
    hipStreamSynchronize(handle.GetStream());
    return RPP_SUCCESS;
}

// -------------------- Set 9 - normalization kernel executor --------------------

template <typename T>
RppStatus hip_exec_normalize_tensor(T *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    T *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u axisMask,
                                    Rpp32f *meanTensor,
                                    Rpp32f *stdDevTensor,
                                    Rpp32u computeMean,
                                    Rpp32u computeStdDev,
                                    Rpp32f scale,
                                    Rpp32f shift,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    Rpp32u batchSize = srcGenericDescPtr->dims[0];
    Rpp32u numDims = srcGenericDescPtr->numDims - 1; // exclude batchsize from input dims

    // create buffer for paramShape and paramStride
    Rpp32u *paramShape, *paramStrides;
    paramShape =  handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem;
    paramStrides = handle.GetInitHandle()->mem.mgpu.scratchBuf.uintmem + (batchSize * numDims);

    // do initial preprocessing and fill the values for paramShape and paramStrides
    Rpp32u maxParamVolume;
    if(numDims == 2 || numDims == 3)
        normalize_setup(roiTensor, batchSize, numDims,
                        axisMask, maxParamVolume);
    else
        normalize_setup_nd(roiTensor, batchSize, numDims, axisMask,
                           paramShape, paramStrides, maxParamVolume);

    if((computeMean == 0) && (computeStdDev == 0))
        maxParamVolume = 0;

    // if computeMean is set compute mean values by processing over input based on axisMask values
    if(computeMean)
        hip_exec_compute_mean_stddev_tensor(srcPtr, srcGenericDescPtr, meanTensor, stdDevTensor, true,
                                            roiTensor, axisMask, numDims, maxParamVolume,
                                            paramShape, paramStrides, handle);
    if(computeStdDev)
        hip_exec_compute_mean_stddev_tensor(srcPtr, srcGenericDescPtr, meanTensor, stdDevTensor, false,
                                            roiTensor, axisMask, numDims, maxParamVolume,
                                            paramShape, paramStrides, handle);

    // based on number of dimensions call the corresponding kernel
    if (numDims == 2)
    {
        // NHW
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(normalize_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                           dstPtr,
                           make_uint2(dstGenericDescPtr->strides[0], dstGenericDescPtr->strides[1]),
                           meanTensor,
                           stdDevTensor,
                           make_float2(scale, shift),
                           roiTensor,
                           make_uint2(maxParamVolume, axisMask),
                           computeStdDev);
    }
    else if (numDims == 3)
    {
        // NDHW
        int globalThreads_x = dstGenericDescPtr->dims[3];
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(normalize_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               &meanTensor[batchCount * maxParamVolume],
                               &stdDevTensor[batchCount * maxParamVolume],
                               make_float2(scale, shift),
                               &roiTensor[batchCount * 6],
                               axisMask,
                               computeStdDev);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        // allocate tensor for src strides
        Rpp32u *srcMaxDims = &srcGenericDescPtr->dims[1];
        Rpp32u *srcStrides = &srcGenericDescPtr->strides[1];
        hipLaunchKernelGGL(normalize_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y), ceil((float)globalThreads_z)),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcMaxDims,
                           srcStrides,
                           dstPtr,
                           meanTensor,
                           stdDevTensor,
                           make_float2(scale, shift),
                           roiTensor,
                           paramShape,
                           paramStrides,
                           make_uint2(maxParamVolume, srcGenericDescPtr->strides[0]),
                           numDims,
                           computeStdDev);
    }

    return RPP_SUCCESS;
}