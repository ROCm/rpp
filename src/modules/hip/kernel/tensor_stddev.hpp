#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// ----------------------- Helper Functions --------------------------

__device__ void stddev_hip_compute(uchar *srcPtr, float src, float *dst) { *dst = src; }
__device__ void stddev_hip_compute(float *srcPtr, float src, float *dst) { *dst = src * 255; }
__device__ void stddev_hip_compute(signed char *srcPtr, float src, float *dst) { *dst = src; }
__device__ void stddev_hip_compute(half *srcPtr, float src, float *dst) { *dst = src * 255; }

// -------------------- Set 0 - Reduction Stage 2 --------------------

template <typename T>
__global__ void tensor_stddev_grid_result_hip(T *inputSrcPtr,
                                              float *srcPtr,
                                              uint xBufferLength,
                                              float *dstPtr,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialVar_smem[1024];                         // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialVar_smem[hipThreadIdx_x] = 0.0f;                         // initialization of _smem to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local memory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialVar_smem[hipThreadIdx_x] += (src_f8.f1[0] +
                                        src_f8.f1[1] +
                                        src_f8.f1[2] +
                                        src_f8.f1[3]);              // perform small work of reducing float4s to float using 1024 x 1 threads and store in _smem
    __syncthreads();                                                // syncthreads after _smem load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialVar_smem[hipThreadIdx_x] += partialVar_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[id_z].xywhROI.roiHeight * roiTensorPtrSrc[id_z].xywhROI.roiWidth;
        stddev_hip_compute(inputSrcPtr, sqrt(partialVar_smem[0] / totalElements), &dstPtr[id_z]);
    }
}

template <typename T>
__global__ void tensor_stddev_grid_3channel_result_hip(T *inputSrcPtr,
                                                       float *srcPtr,
                                                       uint xBufferLength,
                                                       float4 *dstPtr_f4,
                                                       bool flag,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    /* Stores individual channel Variations computed from channel Means to compute Stddev of individual channels*/
    __shared__ float partialRVar_smem[1024];                                     // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGVar_smem[1024];
    __shared__ float partialBVar_smem[1024];
    partialRVar_smem[hipThreadIdx_x] = 0.0f;                                     // initialization of _smem to 0 using all 1024 x 1 threads
    partialGVar_smem[hipThreadIdx_x] = 0.0f;
    partialBVar_smem[hipThreadIdx_x] = 0.0f;

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                                     // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                                  // difference between bufferLength and alignedLength
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);   // load 24 pixels to local memory
    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                  // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRVar_smem[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                        src_f24.f8[0].f1[1] +
                                        src_f24.f8[0].f1[2] +
                                        src_f24.f8[0].f1[3]);                    // perform small work of reducing R float4s to float using 1024 threads and store in _smem
    partialGVar_smem[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                        src_f24.f8[1].f1[1] +
                                        src_f24.f8[1].f1[2] +
                                        src_f24.f8[1].f1[3]);                    // perform small work of reducing G float4s to float using 1024 threads and store in _smem
    partialBVar_smem[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                        src_f24.f8[2].f1[1] +
                                        src_f24.f8[2].f1[2] +
                                        src_f24.f8[2].f1[3]);                    // perform small work of reducing B float4s to float using 1024 threads and store in _smem

    __syncthreads();                                                             // syncthreads after _smem load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVar_smem[hipThreadIdx_x] += partialRVar_smem[hipThreadIdx_x + threadMax];
            partialGVar_smem[hipThreadIdx_x] += partialGVar_smem[hipThreadIdx_x + threadMax];
            partialBVar_smem[hipThreadIdx_x] += partialBVar_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[id_z].xywhROI.roiHeight * roiTensorPtrSrc[id_z].xywhROI.roiWidth;
        float var = partialRVar_smem[0] + partialGVar_smem[0] + partialBVar_smem[0];
        int idx = id_z * 4;
        if (!flag)
        {
            stddev_hip_compute(inputSrcPtr, sqrt(partialRVar_smem[0] / totalElements), &dstPtr_f4->x);
            stddev_hip_compute(inputSrcPtr, sqrt(partialGVar_smem[0] / totalElements), &dstPtr_f4->y);
            stddev_hip_compute(inputSrcPtr, sqrt(partialBVar_smem[0] / totalElements), &dstPtr_f4->z);
        }
        else
            stddev_hip_compute(inputSrcPtr, sqrt(var  / (totalElements * 3)), &dstPtr_f4->w);
    }
}

// -------------------- Set 1 - Reduction Stage 1 --------------------

// Compute complete tensor variance
template <typename T, typename U>
__global__ void tensor_variance_pln1_hip(T *srcPtr,
                                         uint2 srcStridesNH,
                                         U *tensorVarArr,
                                         Rpp32f *mean,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float4 mean_f4 = (float4)mean[id_z];

    __shared__ float partialVar_smem[16][16];                               // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialVarRowPtr_smem = &partialVar_smem[hipThreadIdx_y][0];    // float pointer to beginning of each row in _smem
    partialVarRowPtr_smem[hipThreadIdx_x] = 0.0f;                          // initialization of _smem to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8, temp_f8, tempSq_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f8, &temp_f8, mean_f4);               // subtract mean from each pixel
    rpp_hip_math_multiply8(&temp_f8, &temp_f8, &tempSq_f8);                 // square the temporary value

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            tempSq_f8.f1[i] = 0.0f;
    tempSq_f8.f4[0] += tempSq_f8.f4[1];                                     // perform small work of vectorized float4 addition
    partialVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f8.f1[0] +
                                             tempSq_f8.f1[1] +
                                             tempSq_f8.f1[2] +
                                             tempSq_f8.f1[3]);
    __syncthreads();                                                        // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialVarRowPtr_smem[hipThreadIdx_x] += partialVarRowPtr_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialVarRowPtr_smem[0] += partialVarRowPtr_smem[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            tensorVarArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialVarRowPtr_smem[0];
    }
}

// Compute individual channel variance
template <typename T, typename U>
__global__ void channelwise_variance_pln3_hip(T *srcPtr,
                                              uint3 srcStridesNCH,
                                              U *tensorVarArr,
                                              float4 *mean_f4,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from channel Means to compute Stddev of individual channels*/
    __shared__ float partialRVar_smem[16][16];                                                   // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGVar_smem[16][16];
    __shared__ float partialBVar_smem[16][16];
    float *partialRVarRowPtr_smem = &partialRVar_smem[hipThreadIdx_y][0];                       // float pointer to beginning of each row in _smem
    float *partialGVarRowPtr_smem = &partialGVar_smem[hipThreadIdx_y][0];
    float *partialBVarRowPtr_smem = &partialBVar_smem[hipThreadIdx_y][0];
    partialRVarRowPtr_smem[hipThreadIdx_x] = 0.0f;                                              // initialization of _smem to 0 using all 16 x 16 threads
    partialGVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialBVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    float4 meanR_f4 = (float4)mean_f4[id_z].x;
    float4 meanG_f4 = (float4)mean_f4[id_z].y;
    float4 meanB_f4 = (float4)mean_f4[id_z].z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24, tempChannel_f24, tempChannelSquared_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24); // load 24 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f24.f8[0], &tempChannel_f24.f8[0], meanR_f4);                  // subtract mean from each pixel
    rpp_hip_math_multiply8(&tempChannel_f24.f8[0], &tempChannel_f24.f8[0], &tempChannelSquared_f24.f8[0]);          // square the temporary value
    rpp_hip_math_subtract8_const(&src_f24.f8[1], &tempChannel_f24.f8[1], meanG_f4);
    rpp_hip_math_multiply8(&tempChannel_f24.f8[1], &tempChannel_f24.f8[1], &tempChannelSquared_f24.f8[1]);
    rpp_hip_math_subtract8_const(&src_f24.f8[2], &tempChannel_f24.f8[2], meanB_f4);
    rpp_hip_math_multiply8(&tempChannel_f24.f8[2], &tempChannel_f24.f8[2], &tempChannelSquared_f24.f8[2]);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempChannelSquared_f24.f8[0].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[1].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[2].f1[i] = 0.0f;
        }
    }
    tempChannelSquared_f24.f8[0].f4[0] += tempChannelSquared_f24.f8[0].f4[1];                                       // perform small work of vectorized float4 addition
    tempChannelSquared_f24.f8[1].f4[0] += tempChannelSquared_f24.f8[1].f4[1];
    tempChannelSquared_f24.f8[2].f4[0] += tempChannelSquared_f24.f8[2].f4[1];
    partialRVarRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24.f8[0].f1[0] +
                                              tempChannelSquared_f24.f8[0].f1[1] +
                                              tempChannelSquared_f24.f8[0].f1[2] +
                                              tempChannelSquared_f24.f8[0].f1[3]);                        // perform small work of reducing R float4s to float using 16 x 16 threads and store in _smem
    partialGVarRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24.f8[1].f1[0] +
                                              tempChannelSquared_f24.f8[1].f1[1] +
                                              tempChannelSquared_f24.f8[1].f1[2] +
                                              tempChannelSquared_f24.f8[1].f1[3]);                        // perform small work of reducing G float4s to float using 16 x 16 threads and store in _smem
    partialBVarRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24.f8[2].f1[0] +
                                              tempChannelSquared_f24.f8[2].f1[1] +
                                              tempChannelSquared_f24.f8[2].f1[2] +
                                              tempChannelSquared_f24.f8[2].f1[3]);                        // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem

    __syncthreads();                                                                            // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVarRowPtr_smem[hipThreadIdx_x] += partialRVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGVarRowPtr_smem[hipThreadIdx_x] += partialGVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBVarRowPtr_smem[hipThreadIdx_x] += partialBVarRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRVarRowPtr_smem[0] += partialRVarRowPtr_smem[increment];
                partialGVarRowPtr_smem[0] += partialGVarRowPtr_smem[increment];
                partialBVarRowPtr_smem[0] += partialBVarRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorVarArr[idx] = partialRVarRowPtr_smem[0];
            tensorVarArr[idx + 1] = partialGVarRowPtr_smem[0];
            tensorVarArr[idx + 2] = partialBVarRowPtr_smem[0];
        }
    }
}

// Compute complete tensor variance
template <typename T, typename U>
__global__ void tensor_variance_pln3_hip(T *srcPtr,
                                         uint3 srcStridesNCH,
                                         U *tensorVarArr,
                                         float4 *mean_f4,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from Image Mean to compute Stddev of Image*/
    __shared__ float partialChannelRVar_smem[16][16];
    __shared__ float partialChannelGVar_smem[16][16];
    __shared__ float partialChannelBVar_smem[16][16];
    float *partialChannelRVarRowPtr_smem = &partialChannelRVar_smem[hipThreadIdx_y][0];
    float *partialChannelGVarRowPtr_smem = &partialChannelGVar_smem[hipThreadIdx_y][0];
    float *partialChannelBVarRowPtr_smem = &partialChannelBVar_smem[hipThreadIdx_y][0];
    partialChannelRVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialChannelGVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialChannelBVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    float4 meanImage_f4 = (float4)mean_f4[id_z].w;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24, temp_f24, tempSq_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24); // load 24 pixels to local memory
    rpp_hip_math_subtract24_const(&src_f24, &temp_f24, meanImage_f4);                           // subtract mean from each pixel
    rpp_hip_math_multiply24(&temp_f24, &temp_f24, &tempSq_f24);                                 // square the temporary value

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempSq_f24.f8[0].f1[i] = 0.0f;
            tempSq_f24.f8[1].f1[i] = 0.0f;
            tempSq_f24.f8[2].f1[i] = 0.0f;
        }
    }
    tempSq_f24.f8[0].f4[0] += tempSq_f24.f8[0].f4[1];                                           // perform small work of vectorized float4 addition
    tempSq_f24.f8[1].f4[0] += tempSq_f24.f8[1].f4[1];
    tempSq_f24.f8[2].f4[0] += tempSq_f24.f8[2].f4[1];
    partialChannelRVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f24.f8[0].f1[0] +
                                                     tempSq_f24.f8[0].f1[1] +
                                                     tempSq_f24.f8[0].f1[2] +
                                                     tempSq_f24.f8[0].f1[3]);                   // perform small work of reducing R float4s to float using 16 x 16 threads and store in _smem
    partialChannelGVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f24.f8[1].f1[0] +
                                                     tempSq_f24.f8[1].f1[1] +
                                                     tempSq_f24.f8[1].f1[2] +
                                                     tempSq_f24.f8[1].f1[3]);                   // perform small work of reducing G float4s to float using 16 x 16 threads and store in _smem
    partialChannelBVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f24.f8[2].f1[0] +
                                                     tempSq_f24.f8[2].f1[1] +
                                                     tempSq_f24.f8[2].f1[2] +
                                                     tempSq_f24.f8[2].f1[3]);                   // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem                      // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem

    __syncthreads();                                                                            // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialChannelRVarRowPtr_smem[hipThreadIdx_x] += partialChannelRVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialChannelGVarRowPtr_smem[hipThreadIdx_x] += partialChannelGVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialChannelBVarRowPtr_smem[hipThreadIdx_x] += partialChannelBVarRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialChannelRVarRowPtr_smem[0] += partialChannelRVarRowPtr_smem[increment];
                partialChannelGVarRowPtr_smem[0] += partialChannelGVarRowPtr_smem[increment];
                partialChannelBVarRowPtr_smem[0] += partialChannelBVarRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorVarArr[idx] = partialChannelRVarRowPtr_smem[0];
            tensorVarArr[idx + 1] = partialChannelGVarRowPtr_smem[0];
            tensorVarArr[idx + 2] = partialChannelBVarRowPtr_smem[0];
        }
    }
}

// Compute individual channel variance
template <typename T, typename U>
__global__ void channelwise_variance_pkd3_hip(T *srcPtr,
                                              uint2 srcStridesNH,
                                              U *tensorVarArr,
                                              float4 *mean_f4,
                                              RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRVar_smem[16][16];                                          // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGVar_smem[16][16];
    __shared__ float partialBVar_smem[16][16];
    float *partialRVarRowPtr_smem = &partialRVar_smem[hipThreadIdx_y][0];              // float pointer to beginning of each row in _smem
    float *partialGVarRowPtr_smem = &partialGVar_smem[hipThreadIdx_y][0];
    float *partialBVarRowPtr_smem = &partialBVar_smem[hipThreadIdx_y][0];
    partialRVarRowPtr_smem[hipThreadIdx_x] = 0.0f;                                     // initialization of _smem to 0 using all 16 x 16 threads
    partialGVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialBVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    float4 meanR_f4 = (float4)mean_f4[id_z].x;
    float4 meanG_f4 = (float4)mean_f4[id_z].y;
    float4 meanB_f4 = (float4)mean_f4[id_z].z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                  // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;               // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24, tempChannel_f24, tempChannelSquared_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);         // load 24 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f24.f8[0], &tempChannel_f24.f8[0], meanR_f4);         // subtract mean from each pixel
    rpp_hip_math_multiply8(&tempChannel_f24.f8[0], &tempChannel_f24.f8[0], &tempChannelSquared_f24.f8[0]); // square the temporary value
    rpp_hip_math_subtract8_const(&src_f24.f8[1], &tempChannel_f24.f8[1], meanG_f4);
    rpp_hip_math_multiply8(&tempChannel_f24.f8[1], &tempChannel_f24.f8[1], &tempChannelSquared_f24.f8[1]);
    rpp_hip_math_subtract8_const(&src_f24.f8[2], &tempChannel_f24.f8[2], meanB_f4);
    rpp_hip_math_multiply8(&tempChannel_f24.f8[2], &tempChannel_f24.f8[2], &tempChannelSquared_f24.f8[2]);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                             // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempChannelSquared_f24.f8[0].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[1].f1[i] = 0.0f;
            tempChannelSquared_f24.f8[2].f1[i] = 0.0f;
        }
    }
    tempChannelSquared_f24.f8[0].f4[0] += tempChannelSquared_f24.f8[0].f4[1];                              // perform small work of vectorized float4 addition
    tempChannelSquared_f24.f8[1].f4[0] += tempChannelSquared_f24.f8[1].f4[1];
    tempChannelSquared_f24.f8[2].f4[0] += tempChannelSquared_f24.f8[2].f4[1];
    partialRVarRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24.f8[0].f1[0] +
                                              tempChannelSquared_f24.f8[0].f1[1] +
                                              tempChannelSquared_f24.f8[0].f1[2] +
                                              tempChannelSquared_f24.f8[0].f1[3]);               // perform small work of reducing R float4s to float using 16 x 16 threads and store in _smem
    partialGVarRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24.f8[1].f1[0] +
                                              tempChannelSquared_f24.f8[1].f1[1] +
                                              tempChannelSquared_f24.f8[1].f1[2] +
                                              tempChannelSquared_f24.f8[1].f1[3]);               // perform small work of reducing G float4s to float using 16 x 16 threads and store in _smem
    partialBVarRowPtr_smem[hipThreadIdx_x] = (tempChannelSquared_f24.f8[2].f1[0] +
                                              tempChannelSquared_f24.f8[2].f1[1] +
                                              tempChannelSquared_f24.f8[2].f1[2] +
                                              tempChannelSquared_f24.f8[2].f1[3]);               // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem

    __syncthreads();                                                                   // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVarRowPtr_smem[hipThreadIdx_x] += partialRVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialGVarRowPtr_smem[hipThreadIdx_x] += partialGVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialBVarRowPtr_smem[hipThreadIdx_x] += partialBVarRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialRVarRowPtr_smem[0] += partialRVarRowPtr_smem[increment];
                partialGVarRowPtr_smem[0] += partialGVarRowPtr_smem[increment];
                partialBVarRowPtr_smem[0] += partialBVarRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorVarArr[idx] = partialRVarRowPtr_smem[0];
            tensorVarArr[idx + 1] = partialGVarRowPtr_smem[0];
            tensorVarArr[idx + 2] = partialBVarRowPtr_smem[0];
        }
    }
}

// Compute complete tensor variance
template <typename T, typename U>
__global__ void tensor_variance_pkd3_hip(T *srcPtr,
                                         uint2 srcStridesNH,
                                         U *tensorVarArr,
                                         float4 *mean_f4,
                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from Image Mean to compute Stddev of Image*/
    __shared__ float partialChannelRVar_smem[16][16];
    __shared__ float partialChannelGVar_smem[16][16];
    __shared__ float partialChannelBVar_smem[16][16];
    float *partialChannelRVarRowPtr_smem = &partialChannelRVar_smem[hipThreadIdx_y][0];
    float *partialChannelGVarRowPtr_smem = &partialChannelGVar_smem[hipThreadIdx_y][0];
    float *partialChannelBVarRowPtr_smem = &partialChannelBVar_smem[hipThreadIdx_y][0];
    partialChannelRVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialChannelGVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    partialChannelBVarRowPtr_smem[hipThreadIdx_x] = 0.0f;
    float4 meanImage_f4 = (float4)mean_f4[id_z].w;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24, temp_f24, tempSq_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);                  // load 24 pixels to local memory
    rpp_hip_math_subtract24_const(&src_f24, &temp_f24, meanImage_f4);                           // subtract mean from each pixel
    rpp_hip_math_multiply24(&temp_f24, &temp_f24, &tempSq_f24);                                 // square the temporary value

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempSq_f24.f8[0].f1[i] = 0.0f;
            tempSq_f24.f8[1].f1[i] = 0.0f;
            tempSq_f24.f8[2].f1[i] = 0.0f;
        }
    }
    tempSq_f24.f8[0].f4[0] += tempSq_f24.f8[0].f4[1];                                           // perform small work of vectorized float4 addition
    tempSq_f24.f8[1].f4[0] += tempSq_f24.f8[1].f4[1];
    tempSq_f24.f8[2].f4[0] += tempSq_f24.f8[2].f4[1];
    partialChannelRVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f24.f8[0].f1[0] +
                                                     tempSq_f24.f8[0].f1[1] +
                                                     tempSq_f24.f8[0].f1[2] +
                                                     tempSq_f24.f8[0].f1[3]);                   // perform small work of reducing R float4s to float using 16 x 16 threads and store in _smem
    partialChannelGVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f24.f8[1].f1[0] +
                                                     tempSq_f24.f8[1].f1[1] +
                                                     tempSq_f24.f8[1].f1[2] +
                                                     tempSq_f24.f8[1].f1[3]);                   // perform small work of reducing G float4s to float using 16 x 16 threads and store in _smem
    partialChannelBVarRowPtr_smem[hipThreadIdx_x] = (tempSq_f24.f8[2].f1[0] +
                                                     tempSq_f24.f8[2].f1[1] +
                                                     tempSq_f24.f8[2].f1[2] +
                                                     tempSq_f24.f8[2].f1[3]);                   // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem                      // perform small work of reducing B float4s to float using 16 x 16 threads and store in _smem

    __syncthreads();                                                                            // syncthreads after _smem load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialChannelRVarRowPtr_smem[hipThreadIdx_x] += partialChannelRVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialChannelGVarRowPtr_smem[hipThreadIdx_x] += partialChannelGVarRowPtr_smem[hipThreadIdx_x + threadMax];
            partialChannelBVarRowPtr_smem[hipThreadIdx_x] += partialChannelBVarRowPtr_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
            {
                partialChannelRVarRowPtr_smem[0] += partialChannelRVarRowPtr_smem[increment];
                partialChannelGVarRowPtr_smem[0] += partialChannelGVarRowPtr_smem[increment];
                partialChannelBVarRowPtr_smem[0] += partialChannelBVarRowPtr_smem[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorVarArr[idx] = partialChannelRVarRowPtr_smem[0];
            tensorVarArr[idx + 1] = partialChannelGVarRowPtr_smem[0];
            tensorVarArr[idx + 2] = partialChannelBVarRowPtr_smem[0];
        }
    }
}

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_tensor_stddev(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp32f *imageStddevArr,
                                 Rpp32f *meanTensor,
                                 int flag,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = srcDescPtr->n;
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialVarArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *tensorPartialVarArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_variance_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialVarArr,
                           meanTensor,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_stddev_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           tensorPartialVarArr,
                           gridDim_x * gridDim_y,
                           imageStddevArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialVarArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *tensorPartialVarArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
        int index = globalThreads_z * 4;
        float4 mean_f4 = make_float4(meanTensor[index], meanTensor[index + 1], meanTensor[index + 2], meanTensor[index + 3]);
        if (!flag)
        {
            hipLaunchKernelGGL(channelwise_variance_pln3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               flag,
                               roiTensorPtrSrc);
        }
        if(flag == 1)
        {
            hipLaunchKernelGGL(tensor_variance_pln3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               flag,
                               roiTensorPtrSrc);
        }
        if(flag == 2)
        {
            hipLaunchKernelGGL(channelwise_variance_pln3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               0, //setting flag to 0 here to compute individual channel stddev
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
            hipLaunchKernelGGL(tensor_variance_pln3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               1, //setting flag to 1 here to compute image stddev
                               roiTensorPtrSrc);
        }

    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u tensorPartialVarArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *tensorPartialVarArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
        int index = globalThreads_z * 4;
        float4 mean_f4 = make_float4(meanTensor[index], meanTensor[index + 1], meanTensor[index + 2], meanTensor[index + 3]);
        if(!flag)
        {
            hipLaunchKernelGGL(channelwise_variance_pkd3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               flag,
                               roiTensorPtrSrc);
        }
        if(flag == 1)
        {
            hipLaunchKernelGGL(tensor_variance_pkd3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               flag,
                               roiTensorPtrSrc);
        }
        if(flag == 2)
        {
            hipLaunchKernelGGL(channelwise_variance_pkd3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               0, //setting flag to 0 here to compute individual channel stddev
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipMemsetAsync(tensorPartialVarArr, 0, tensorPartialVarArrLength * sizeof(float), handle.GetStream());
            hipLaunchKernelGGL(tensor_variance_pkd3_hip,
                               dim3(gridDim_x, gridDim_y, gridDim_z),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               tensorPartialVarArr,
                               &mean_f4,
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(tensor_stddev_grid_3channel_result_hip,
                               dim3(1, 1, gridDim_z),
                               dim3(1024, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               tensorPartialVarArr,
                               gridDim_x * gridDim_y,
                               reinterpret_cast<float4*>(imageStddevArr),
                               1, //setting flag to 1 here to compute image stddev
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
