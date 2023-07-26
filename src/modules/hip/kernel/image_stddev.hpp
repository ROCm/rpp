#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// ----------------------- Helper Functions --------------------------
__device__ void stddev_hip_compute(uchar *srcPtr, float src, float *dst)
{
    *dst = src;
}

__device__ void stddev_hip_compute(float *srcPtr, float src, float *dst)
{
    *dst = src * 255;
}

__device__ void stddev_hip_compute(signed char *srcPtr, float src, float *dst)
{
    *dst = src;
}

__device__ void stddev_hip_compute(half *srcPtr, float src, float *dst)
{
    *dst = src * 255;
}

// -------------------- Set 0 - Reduction Stage 2 --------------------
template <typename T>
__global__ void image_stddev_grid_result_tensor(T *inputSrcPtr,
                                                float *srcPtr,
                                                uint xBufferLength,
                                                float *dstPtr,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialVarLDS[1024];                           // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialVarLDS[hipThreadIdx_x] = 0.0f;                           // initialization of LDS to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local memory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialVarLDS[hipThreadIdx_x] += (src_f8.f1[0] +
                                      src_f8.f1[1] +
                                      src_f8.f1[2] +
                                      src_f8.f1[3]);                // perform small work of reducing float4s to float using 1024 x 1 threads and store in LDS
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialVarLDS[hipThreadIdx_x] += partialVarLDS[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[id_z].xywhROI.roiHeight * roiTensorPtrSrc[id_z].xywhROI.roiWidth;
        stddev_hip_compute(inputSrcPtr, sqrt(partialVarLDS[0] / totalElements), &dstPtr[id_z]);
    }
}

template <typename T>
__global__ void image_stddev_grid_3channel_result_tensor(T *inputSrcPtr,
                                                         float *srcPtr,
                                                         uint xBufferLength,
                                                         float *dstPtr,
                                                         bool flag,
                                                         RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    /* Stores individual channel Variations computed from channel Means to compute Stddev of individual channels*/
    __shared__ float partialRVarLDS[1024];                                       // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGVarLDS[1024];
    __shared__ float partialBVarLDS[1024];
    partialRVarLDS[hipThreadIdx_x] = 0.0f;                                       // initialization of LDS to 0 using all 1024 x 1 threads
    partialGVarLDS[hipThreadIdx_x] = 0.0f;
    partialBVarLDS[hipThreadIdx_x] = 0.0f;

    if (id_x >= xBufferLength)
    {
        return;
    }

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
    partialRVarLDS[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                      src_f24.f8[0].f1[1] +
                                      src_f24.f8[0].f1[2] +
                                      src_f24.f8[0].f1[3]);                      // perform small work of reducing R float4s to float using 1024 threads and store in LDS
    partialGVarLDS[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                      src_f24.f8[1].f1[1] +
                                      src_f24.f8[1].f1[2] +
                                      src_f24.f8[1].f1[3]);                      // perform small work of reducing G float4s to float using 1024 threads and store in LDS
    partialBVarLDS[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                      src_f24.f8[2].f1[1] +
                                      src_f24.f8[2].f1[2] +
                                      src_f24.f8[2].f1[3]);                      // perform small work of reducing B float4s to float using 1024 threads and store in LDS

    __syncthreads();                                                             // syncthreads after LDS load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVarLDS[hipThreadIdx_x] += partialRVarLDS[hipThreadIdx_x + threadMax];
            partialGVarLDS[hipThreadIdx_x] += partialGVarLDS[hipThreadIdx_x + threadMax];
            partialBVarLDS[hipThreadIdx_x] += partialBVarLDS[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[id_z].xywhROI.roiHeight * roiTensorPtrSrc[id_z].xywhROI.roiWidth;
        float var = partialRVarLDS[0] + partialGVarLDS[0] + partialBVarLDS[0];
        int idx = id_z * 4;
        if (!flag)
        {
            stddev_hip_compute(inputSrcPtr, sqrt(partialRVarLDS[0] / totalElements), &dstPtr[idx]);
            stddev_hip_compute(inputSrcPtr, sqrt(partialGVarLDS[0] / totalElements), &dstPtr[idx + 1]);
            stddev_hip_compute(inputSrcPtr, sqrt(partialBVarLDS[0] / totalElements), &dstPtr[idx + 2]);
        }
        else
            stddev_hip_compute(inputSrcPtr, sqrt(var  / (totalElements * 3)), &dstPtr[idx + 3]);
    }
}

// -------------------- Set 1 - Reduction Stage 1 --------------------

template <typename T, typename U>
__global__ void image_var_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      U *imageVarArr,
                                      Rpp32f *mean,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float4 mean_f4 = (float4)mean[id_z];

    __shared__ float partialVarLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialVarLDSRowPtr = &partialVarLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialVarLDSRowPtr[hipThreadIdx_x] = 0.0f;                             // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8, temp_f8, tempSq_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f8, &temp_f8, mean_f4);               //subtract mean from each pixel
    rpp_hip_math_multiply8(&temp_f8, &temp_f8, &tempSq_f8);                 //square the temporary value

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            tempSq_f8.f1[i] = 0.0f;
    tempSq_f8.f4[0] += tempSq_f8.f4[1];                                     // perform small work of vectorized float4 addition
    partialVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f8.f1[0] +
                                           tempSq_f8.f1[1] +
                                           tempSq_f8.f1[2] +
                                           tempSq_f8.f1[3]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialVarLDSRowPtr[hipThreadIdx_x] += partialVarLDSRowPtr[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialVarLDSRowPtr[0] += partialVarLDSRowPtr[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            imageVarArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialVarLDSRowPtr[0];
    }
}

template <typename T, typename U>
__global__ void channel_var_pln3_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        U *imageVarArr,
                                        Rpp32f *mean,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from channel Means to compute Stddev of individual channels*/
    __shared__ float partialRVarLDS[16][16];                                                    // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGVarLDS[16][16];
    __shared__ float partialBVarLDS[16][16];
    float *partialRVarLDSRowPtr = &partialRVarLDS[hipThreadIdx_y][0];                           // float pointer to beginning of each row in LDS
    float *partialGVarLDSRowPtr = &partialGVarLDS[hipThreadIdx_y][0];
    float *partialBVarLDSRowPtr = &partialBVarLDS[hipThreadIdx_y][0];
    partialRVarLDSRowPtr[hipThreadIdx_x] = 0.0f;                                                // initialization of LDS to 0 using all 16 x 16 threads
    partialGVarLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialBVarLDSRowPtr[hipThreadIdx_x] = 0.0f;

    int index       = id_z * 4;
    float4 meanR_f4 = (float4)mean[index];
    float4 meanG_f4 = (float4)mean[index + 1];
    float4 meanB_f4 = (float4)mean[index + 2];

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24, tempCh_f24, tempChSq_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24); // load 24 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f24.f8[0], &tempCh_f24.f8[0], meanR_f4);                  //subtract mean from each pixel
    rpp_hip_math_multiply8(&tempCh_f24.f8[0], &tempCh_f24.f8[0], &tempChSq_f24.f8[0]);          //square the temporary value
    rpp_hip_math_subtract8_const(&src_f24.f8[1], &tempCh_f24.f8[1], meanG_f4);
    rpp_hip_math_multiply8(&tempCh_f24.f8[1], &tempCh_f24.f8[1], &tempChSq_f24.f8[1]);
    rpp_hip_math_subtract8_const(&src_f24.f8[2], &tempCh_f24.f8[2], meanB_f4);
    rpp_hip_math_multiply8(&tempCh_f24.f8[2], &tempCh_f24.f8[2], &tempChSq_f24.f8[2]);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempChSq_f24.f8[0].f1[i] = 0.0f;
            tempChSq_f24.f8[1].f1[i] = 0.0f;
            tempChSq_f24.f8[2].f1[i] = 0.0f;
        }
    }
    tempChSq_f24.f8[0].f4[0] += tempChSq_f24.f8[0].f4[1];                                       // perform small work of vectorized float4 addition
    tempChSq_f24.f8[1].f4[0] += tempChSq_f24.f8[1].f4[1];
    tempChSq_f24.f8[2].f4[0] += tempChSq_f24.f8[2].f4[1];
    partialRVarLDSRowPtr[hipThreadIdx_x] = (tempChSq_f24.f8[0].f1[0] +
                                            tempChSq_f24.f8[0].f1[1] +
                                            tempChSq_f24.f8[0].f1[2] +
                                            tempChSq_f24.f8[0].f1[3]);                          // perform small work of reducing R float4s to float using 16 x 16 threads and store in LDS
    partialGVarLDSRowPtr[hipThreadIdx_x] = (tempChSq_f24.f8[1].f1[0] +
                                            tempChSq_f24.f8[1].f1[1] +
                                            tempChSq_f24.f8[1].f1[2] +
                                            tempChSq_f24.f8[1].f1[3]);                          // perform small work of reducing G float4s to float using 16 x 16 threads and store in LDS
    partialBVarLDSRowPtr[hipThreadIdx_x] = (tempChSq_f24.f8[2].f1[0] +
                                            tempChSq_f24.f8[2].f1[1] +
                                            tempChSq_f24.f8[2].f1[2] +
                                            tempChSq_f24.f8[2].f1[3]);                          // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS

    __syncthreads();                                                                            // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVarLDSRowPtr[hipThreadIdx_x] += partialRVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialGVarLDSRowPtr[hipThreadIdx_x] += partialGVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialBVarLDSRowPtr[hipThreadIdx_x] += partialBVarLDSRowPtr[hipThreadIdx_x + threadMax];
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
                partialRVarLDSRowPtr[0] += partialRVarLDSRowPtr[increment];
                partialGVarLDSRowPtr[0] += partialGVarLDSRowPtr[increment];
                partialBVarLDSRowPtr[0] += partialBVarLDSRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageVarArr[idx] = partialRVarLDSRowPtr[0];
            imageVarArr[idx + 1] = partialGVarLDSRowPtr[0];
            imageVarArr[idx + 2] = partialBVarLDSRowPtr[0];
        }
    }
}

template <typename T, typename U>
__global__ void image_var_pln3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      U *imageVarArr,
                                      Rpp32f *mean,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from Image Mean to compute Stddev of Image*/
    __shared__ float partialImageRVarLDS[16][16];
    __shared__ float partialImageGVarLDS[16][16];
    __shared__ float partialImageBVarLDS[16][16];
    float *partialImageRVarLDSRowPtr = &partialImageRVarLDS[hipThreadIdx_y][0];
    float *partialImageGVarLDSRowPtr = &partialImageGVarLDS[hipThreadIdx_y][0];
    float *partialImageBVarLDSRowPtr = &partialImageBVarLDS[hipThreadIdx_y][0];
    partialImageRVarLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialImageGVarLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialImageBVarLDSRowPtr[hipThreadIdx_x] = 0.0f;

    int index           = (id_z * 4) + 3;
    float4 meanImage_f4 = (float4)mean[index];

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24, temp_f24, tempSq_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24); // load 24 pixels to local memory
    rpp_hip_math_subtract24_const(&src_f24, &temp_f24, meanImage_f4);                           //subtract mean from each pixel
    rpp_hip_math_multiply24(&temp_f24, &temp_f24, &tempSq_f24);                                 //square the temporary value

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
    partialImageRVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f24.f8[0].f1[0] +
                                                 tempSq_f24.f8[0].f1[1] +
                                                 tempSq_f24.f8[0].f1[2] +
                                                 tempSq_f24.f8[0].f1[3]);                       // perform small work of reducing R float4s to float using 16 x 16 threads and store in LDS
    partialImageGVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f24.f8[1].f1[0] +
                                                 tempSq_f24.f8[1].f1[1] +
                                                 tempSq_f24.f8[1].f1[2] +
                                                 tempSq_f24.f8[1].f1[3]);                       // perform small work of reducing G float4s to float using 16 x 16 threads and store in LDS
    partialImageBVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f24.f8[2].f1[0] +
                                                 tempSq_f24.f8[2].f1[1] +
                                                 tempSq_f24.f8[2].f1[2] +
                                                 tempSq_f24.f8[2].f1[3]);                       // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS                      // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS

    __syncthreads();                                                                            // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialImageRVarLDSRowPtr[hipThreadIdx_x] += partialImageRVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialImageGVarLDSRowPtr[hipThreadIdx_x] += partialImageGVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialImageBVarLDSRowPtr[hipThreadIdx_x] += partialImageBVarLDSRowPtr[hipThreadIdx_x + threadMax];
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
                partialImageRVarLDSRowPtr[0] += partialImageRVarLDSRowPtr[increment];
                partialImageGVarLDSRowPtr[0] += partialImageGVarLDSRowPtr[increment];
                partialImageBVarLDSRowPtr[0] += partialImageBVarLDSRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageVarArr[idx] = partialImageRVarLDSRowPtr[0];
            imageVarArr[idx + 1] = partialImageGVarLDSRowPtr[0];
            imageVarArr[idx + 2] = partialImageBVarLDSRowPtr[0];
        }
    }
}

template <typename T, typename U>
__global__ void channel_var_pkd3_tensor(T *srcPtr,
                                        uint2 srcStridesNH,
                                        U *imageVarArr,
                                        Rpp32f *mean,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRVarLDS[16][16];                                           // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGVarLDS[16][16];
    __shared__ float partialBVarLDS[16][16];
    float *partialRVarLDSRowPtr = &partialRVarLDS[hipThreadIdx_y][0];                  // float pointer to beginning of each row in LDS
    float *partialGVarLDSRowPtr = &partialGVarLDS[hipThreadIdx_y][0];
    float *partialBVarLDSRowPtr = &partialBVarLDS[hipThreadIdx_y][0];
    partialRVarLDSRowPtr[hipThreadIdx_x] = 0.0f;                                       // initialization of LDS to 0 using all 16 x 16 threads
    partialGVarLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialBVarLDSRowPtr[hipThreadIdx_x] = 0.0f;

    int index       = id_z * 4;
    float4 meanR_f4 = (float4)mean[index];
    float4 meanG_f4 = (float4)mean[index + 1];
    float4 meanB_f4 = (float4)mean[index + 2];

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                  // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;               // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24, tempCh_f24, tempChSq_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);         // load 24 pixels to local memory
    rpp_hip_math_subtract8_const(&src_f24.f8[0], &tempCh_f24.f8[0], meanR_f4);         //subtract mean from each pixel
    rpp_hip_math_multiply8(&tempCh_f24.f8[0], &tempCh_f24.f8[0], &tempChSq_f24.f8[0]); //square the temporary value
    rpp_hip_math_subtract8_const(&src_f24.f8[1], &tempCh_f24.f8[1], meanG_f4);
    rpp_hip_math_multiply8(&tempCh_f24.f8[1], &tempCh_f24.f8[1], &tempChSq_f24.f8[1]);
    rpp_hip_math_subtract8_const(&src_f24.f8[2], &tempCh_f24.f8[2], meanB_f4);
    rpp_hip_math_multiply8(&tempCh_f24.f8[2], &tempCh_f24.f8[2], &tempChSq_f24.f8[2]);

    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                             // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            tempChSq_f24.f8[0].f1[i] = 0.0f;
            tempChSq_f24.f8[1].f1[i] = 0.0f;
            tempChSq_f24.f8[2].f1[i] = 0.0f;
        }
    }
    tempChSq_f24.f8[0].f4[0] += tempChSq_f24.f8[0].f4[1];                              // perform small work of vectorized float4 addition
    tempChSq_f24.f8[1].f4[0] += tempChSq_f24.f8[1].f4[1];
    tempChSq_f24.f8[2].f4[0] += tempChSq_f24.f8[2].f4[1];
    partialRVarLDSRowPtr[hipThreadIdx_x] = (tempChSq_f24.f8[0].f1[0] +
                                            tempChSq_f24.f8[0].f1[1] +
                                            tempChSq_f24.f8[0].f1[2] +
                                            tempChSq_f24.f8[0].f1[3]);                 // perform small work of reducing R float4s to float using 16 x 16 threads and store in LDS
    partialGVarLDSRowPtr[hipThreadIdx_x] = (tempChSq_f24.f8[1].f1[0] +
                                            tempChSq_f24.f8[1].f1[1] +
                                            tempChSq_f24.f8[1].f1[2] +
                                            tempChSq_f24.f8[1].f1[3]);                 // perform small work of reducing G float4s to float using 16 x 16 threads and store in LDS
    partialBVarLDSRowPtr[hipThreadIdx_x] = (tempChSq_f24.f8[2].f1[0] +
                                            tempChSq_f24.f8[2].f1[1] +
                                            tempChSq_f24.f8[2].f1[2] +
                                            tempChSq_f24.f8[2].f1[3]);                 // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS

    __syncthreads();                                                                   // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRVarLDSRowPtr[hipThreadIdx_x] += partialRVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialGVarLDSRowPtr[hipThreadIdx_x] += partialGVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialBVarLDSRowPtr[hipThreadIdx_x] += partialBVarLDSRowPtr[hipThreadIdx_x + threadMax];
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
                partialRVarLDSRowPtr[0] += partialRVarLDSRowPtr[increment];
                partialGVarLDSRowPtr[0] += partialGVarLDSRowPtr[increment];
                partialBVarLDSRowPtr[0] += partialBVarLDSRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageVarArr[idx] = partialRVarLDSRowPtr[0];
            imageVarArr[idx + 1] = partialGVarLDSRowPtr[0];
            imageVarArr[idx + 2] = partialBVarLDSRowPtr[0];
        }
    }
}

template <typename T, typename U>
__global__ void image_var_pkd3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      U *imageVarArr,
                                      Rpp32f *mean,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    /* Stores individual channel Variations computed from Image Mean to compute Stddev of Image*/
    __shared__ float partialImageRVarLDS[16][16];
    __shared__ float partialImageGVarLDS[16][16];
    __shared__ float partialImageBVarLDS[16][16];
    float *partialImageRVarLDSRowPtr = &partialImageRVarLDS[hipThreadIdx_y][0];
    float *partialImageGVarLDSRowPtr = &partialImageGVarLDS[hipThreadIdx_y][0];
    float *partialImageBVarLDSRowPtr = &partialImageBVarLDS[hipThreadIdx_y][0];
    partialImageRVarLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialImageGVarLDSRowPtr[hipThreadIdx_x] = 0.0f;
    partialImageBVarLDSRowPtr[hipThreadIdx_x] = 0.0f;

    int index           = (id_z * 4) + 3;
    float4 meanImage_f4 = (float4)mean[index];

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24, temp_f24, tempSq_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);                  // load 24 pixels to local memory
    rpp_hip_math_subtract24_const(&src_f24, &temp_f24, meanImage_f4);                           //subtract mean from each pixel
    rpp_hip_math_multiply24(&temp_f24, &temp_f24, &tempSq_f24);                                 //square the temporary value

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
    partialImageRVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f24.f8[0].f1[0] +
                                                 tempSq_f24.f8[0].f1[1] +
                                                 tempSq_f24.f8[0].f1[2] +
                                                 tempSq_f24.f8[0].f1[3]);                       // perform small work of reducing R float4s to float using 16 x 16 threads and store in LDS
    partialImageGVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f24.f8[1].f1[0] +
                                                 tempSq_f24.f8[1].f1[1] +
                                                 tempSq_f24.f8[1].f1[2] +
                                                 tempSq_f24.f8[1].f1[3]);                       // perform small work of reducing G float4s to float using 16 x 16 threads and store in LDS
    partialImageBVarLDSRowPtr[hipThreadIdx_x] = (tempSq_f24.f8[2].f1[0] +
                                                 tempSq_f24.f8[2].f1[1] +
                                                 tempSq_f24.f8[2].f1[2] +
                                                 tempSq_f24.f8[2].f1[3]);                       // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS                      // perform small work of reducing B float4s to float using 16 x 16 threads and store in LDS

    __syncthreads();                                                                            // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialImageRVarLDSRowPtr[hipThreadIdx_x] += partialImageRVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialImageGVarLDSRowPtr[hipThreadIdx_x] += partialImageGVarLDSRowPtr[hipThreadIdx_x + threadMax];
            partialImageBVarLDSRowPtr[hipThreadIdx_x] += partialImageBVarLDSRowPtr[hipThreadIdx_x + threadMax];
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
                partialImageRVarLDSRowPtr[0] += partialImageRVarLDSRowPtr[increment];
                partialImageGVarLDSRowPtr[0] += partialImageGVarLDSRowPtr[increment];
                partialImageBVarLDSRowPtr[0] += partialImageBVarLDSRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageVarArr[idx] = partialImageRVarLDSRowPtr[0];
            imageVarArr[idx + 1] = partialImageGVarLDSRowPtr[0];
            imageVarArr[idx + 2] = partialImageBVarLDSRowPtr[0];
        }
    }
}

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_image_stddev_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       U *imageStddevArr,
                                       int flag,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/localThreads_x);
    int gridDim_y = (int) ceil((float)globalThreads_y/localThreads_y);
    int gridDim_z = (int) ceil((float)globalThreads_z/localThreads_z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u imagePartialVarArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *imagePartialVarArr;
        imagePartialVarArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(image_var_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialVarArr,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(image_stddev_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           imagePartialVarArr,
                           gridDim_x * gridDim_y,
                           imageStddevArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u imagePartialVarArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *imagePartialVarArr;
        imagePartialVarArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        if(!flag)
        {
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(channel_var_pln3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            flag,
                            roiTensorPtrSrc);
        }
        if(flag == 1)
        {
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_var_pln3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            flag,
                            roiTensorPtrSrc);
        }
        if(flag == 2)
        {
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(channel_var_pln3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            0, //setting flag to 0 here to compute individual channel stddev
                            roiTensorPtrSrc);
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_var_pln3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            1, //setting flag to 1 here to compute image stddev
                            roiTensorPtrSrc);
        }

    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u imagePartialVarArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *imagePartialVarArr;
        imagePartialVarArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        if(!flag)
        {
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(channel_var_pkd3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            flag,
                            roiTensorPtrSrc);
        }
        if(flag == 1)
        {
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_var_pkd3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            flag,
                            roiTensorPtrSrc);
        }
        if(flag == 2)
        {
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(channel_var_pkd3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            0, //setting flag to 0 here to compute individual channel stddev
                            roiTensorPtrSrc);
            hipMemsetAsync(imagePartialVarArr, 0, imagePartialVarArrLength * sizeof(float));
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_var_pkd3_tensor,
                            dim3(gridDim_x, gridDim_y, gridDim_z),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            imagePartialVarArr,
                            handle.GetInitHandle()->mem.mgpu.float4Arr[0].floatmem,
                            roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            hipLaunchKernelGGL(image_stddev_grid_3channel_result_tensor,
                            dim3(1, 1, gridDim_z),
                            dim3(1024, 1, 1),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            imagePartialVarArr,
                            gridDim_x * gridDim_y,
                            imageStddevArr,
                            1, //setting flag to 1 here to compute image stddev
                            roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
