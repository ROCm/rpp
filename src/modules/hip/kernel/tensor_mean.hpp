#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------

__global__ void tensor_mean_grid_result(float *srcPtr,
                                        uint xBufferLength,
                                        float *dstPtr,
                                        RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSumShared[1024];                        // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialSumShared[hipThreadIdx_x] = 0.0f;                        // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
    {
        return;
    }

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                    // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                   // perform small work of vectorized float4 addition
    partialSumShared[hipThreadIdx_x] += (src_f8.f1[0] +
                                         src_f8.f1[1] +
                                         src_f8.f1[2] +
                                         src_f8.f1[3]);             // perform small work of reducing float4s to float using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumShared[hipThreadIdx_x] += partialSumShared[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        dstPtr[hipBlockIdx_z] = partialSumShared[0] / totalElements;
    }
}

__global__ void tensor_mean_grid_3channel_result(float *srcPtr,
                                                 uint xBufferLength,
                                                 float *dstPtr,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRSumShared[1024];                                    // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGSumShared[1024];
    __shared__ float partialBSumShared[1024];
    partialRSumShared[hipThreadIdx_x] = 0.0f;                                    // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSumShared[hipThreadIdx_x] = 0.0f;
    partialBSumShared[hipThreadIdx_x] = 0.0f;

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
    partialRSumShared[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                         src_f24.f8[0].f1[1] +
                                         src_f24.f8[0].f1[2] +
                                         src_f24.f8[0].f1[3]);                   // perform small work of reducing R float4s to float using 1024 threads and store in Shared
    partialGSumShared[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                         src_f24.f8[1].f1[1] +
                                         src_f24.f8[1].f1[2] +
                                         src_f24.f8[1].f1[3]);                   // perform small work of reducing G float4s to float using 1024 threads and store in Shared
    partialBSumShared[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                         src_f24.f8[2].f1[1] +
                                         src_f24.f8[2].f1[2] +
                                         src_f24.f8[2].f1[3]);                   // perform small work of reducing B float4s to float using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumShared[hipThreadIdx_x] += partialRSumShared[hipThreadIdx_x + threadMax];
            partialGSumShared[hipThreadIdx_x] += partialGSumShared[hipThreadIdx_x + threadMax];
            partialBSumShared[hipThreadIdx_x] += partialBSumShared[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        float sum = partialRSumShared[0] + partialGSumShared[0] + partialBSumShared[0];
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = partialRSumShared[0] / totalElements;
        dstPtr[idx + 1] = partialGSumShared[0] / totalElements;
        dstPtr[idx + 2] = partialBSumShared[0] / totalElements;
        dstPtr[idx + 3] = sum  / (totalElements * 3);
    }
}

// -------------------- Set 1 - Reduction Stage 1 --------------------

template <typename T, typename U>
__global__ void tensor_sum_pln1(T *srcPtr,
                                uint2 srcStridesNH,
                                U *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialSumShared[16][16];                              // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialSumSharedRowPtr = &partialSumShared[hipThreadIdx_y][0];   // float pointer to beginning of each row in Shared
    partialSumSharedRowPtr[hipThreadIdx_x] = 0.0f;                          // initialization of Shared to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f
    src_f8.f4[0] += src_f8.f4[1];                                           // perform small work of vectorized float4 addition
    partialSumSharedRowPtr[hipThreadIdx_x] = (src_f8.f1[0] +
                                              src_f8.f1[1] +
                                              src_f8.f1[2] +
                                              src_f8.f1[3]);                // perform small work of reducing float4s to float using 16 x 16 threads and store in Shared
    __syncthreads();                                                        // syncthreads after Shared load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSumSharedRowPtr[hipThreadIdx_x] += partialSumSharedRowPtr[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialSumSharedRowPtr[0] += partialSumSharedRowPtr[increment];
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            tensorSumArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialSumSharedRowPtr[0];
    }
}

template <typename T, typename U>
__global__ void tensor_sum_pln3(T *srcPtr,
                                uint3 srcStridesNCH,
                                U *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRSumShared[16][16];                                                    // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGSumShared[16][16];
    __shared__ float partialBSumShared[16][16];
    float *partialRSumSharedRowPtr = &partialRSumShared[hipThreadIdx_y][0];                        // float pointer to beginning of each row in Shared
    float *partialGSumSharedRowPtr = &partialGSumShared[hipThreadIdx_y][0];
    float *partialBSumSharedRowPtr = &partialBSumShared[hipThreadIdx_y][0];
    partialRSumSharedRowPtr[hipThreadIdx_x] = 0.0f;                                                // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumSharedRowPtr[hipThreadIdx_x] = 0.0f;
    partialBSumSharedRowPtr[hipThreadIdx_x] = 0.0f;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;                           // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;                        // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24); // load 24 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                                      // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                                 // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRSumSharedRowPtr[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                               src_f24.f8[0].f1[1] +
                                               src_f24.f8[0].f1[2] +
                                               src_f24.f8[0].f1[3]);                            // perform small work of reducing R float4s to float using 16 x 16 threads and store in Shared
    partialGSumSharedRowPtr[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                               src_f24.f8[1].f1[1] +
                                               src_f24.f8[1].f1[2] +
                                               src_f24.f8[1].f1[3]);                            // perform small work of reducing G float4s to float using 16 x 16 threads and store in Shared
    partialBSumSharedRowPtr[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                               src_f24.f8[2].f1[1] +
                                               src_f24.f8[2].f1[2] +
                                               src_f24.f8[2].f1[3]);                            // perform small work of reducing B float4s to float using 16 x 16 threads and store in Shared

    __syncthreads();                                                                            // syncthreads after Shared load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumSharedRowPtr[hipThreadIdx_x] += partialRSumSharedRowPtr[hipThreadIdx_x + threadMax];
            partialGSumSharedRowPtr[hipThreadIdx_x] += partialGSumSharedRowPtr[hipThreadIdx_x + threadMax];
            partialBSumSharedRowPtr[hipThreadIdx_x] += partialBSumSharedRowPtr[hipThreadIdx_x + threadMax];
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
                partialRSumSharedRowPtr[0] += partialRSumSharedRowPtr[increment];
                partialGSumSharedRowPtr[0] += partialGSumSharedRowPtr[increment];
                partialBSumSharedRowPtr[0] += partialBSumSharedRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumSharedRowPtr[0];
            tensorSumArr[idx + 1] = partialGSumSharedRowPtr[0];
            tensorSumArr[idx + 2] = partialBSumSharedRowPtr[0];
        }
    }
}

template <typename T, typename U>
__global__ void tensor_sum_pkd3(T *srcPtr,
                                uint2 srcStridesNH,
                                U *tensorSumArr,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRSumShared[16][16];                                        // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGSumShared[16][16];
    __shared__ float partialBSumShared[16][16];
    float *partialRSumSharedRowPtr = &partialRSumShared[hipThreadIdx_y][0];            // float pointer to beginning of each row in Shared
    float *partialGSumSharedRowPtr = &partialGSumShared[hipThreadIdx_y][0];
    float *partialBSumSharedRowPtr = &partialBSumShared[hipThreadIdx_y][0];
    partialRSumSharedRowPtr[hipThreadIdx_x] = 0.0f;                                    // initialization of Shared to 0 using all 16 x 16 threads
    partialGSumSharedRowPtr[hipThreadIdx_x] = 0.0f;
    partialBSumSharedRowPtr[hipThreadIdx_x] = 0.0f;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;               // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;            // difference between roiWidth and alignedLength
    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);      // load 24 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                          // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = 0.0f;
            src_f24.f8[1].f1[i] = 0.0f;
            src_f24.f8[2].f1[i] = 0.0f;
        }
    }
    src_f24.f8[0].f4[0] += src_f24.f8[0].f4[1];                                     // perform small work of vectorized float4 addition
    src_f24.f8[1].f4[0] += src_f24.f8[1].f4[1];
    src_f24.f8[2].f4[0] += src_f24.f8[2].f4[1];
    partialRSumSharedRowPtr[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                               src_f24.f8[0].f1[1] +
                                               src_f24.f8[0].f1[2] +
                                               src_f24.f8[0].f1[3]);                // perform small work of reducing R float4s to float using 16 x 16 threads and store in Shared
    partialGSumSharedRowPtr[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                               src_f24.f8[1].f1[1] +
                                               src_f24.f8[1].f1[2] +
                                               src_f24.f8[1].f1[3]);                // perform small work of reducing G float4s to float using 16 x 16 threads and store in Shared
    partialBSumSharedRowPtr[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                               src_f24.f8[2].f1[1] +
                                               src_f24.f8[2].f1[2] +
                                               src_f24.f8[2].f1[3]);                // perform small work of reducing B float4s to float using 16 x 16 threads and store in Shared

    __syncthreads();                                                                // syncthreads after Shared load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSumSharedRowPtr[hipThreadIdx_x] += partialRSumSharedRowPtr[hipThreadIdx_x + threadMax];
            partialGSumSharedRowPtr[hipThreadIdx_x] += partialGSumSharedRowPtr[hipThreadIdx_x + threadMax];
            partialBSumSharedRowPtr[hipThreadIdx_x] += partialBSumSharedRowPtr[hipThreadIdx_x + threadMax];
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
                partialRSumSharedRowPtr[0] += partialRSumSharedRowPtr[increment];
                partialGSumSharedRowPtr[0] += partialGSumSharedRowPtr[increment];
                partialBSumSharedRowPtr[0] += partialBSumSharedRowPtr[increment];
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            tensorSumArr[idx] = partialRSumSharedRowPtr[0];
            tensorSumArr[idx + 1] = partialGSumSharedRowPtr[0];
            tensorSumArr[idx + 2] = partialBSumSharedRowPtr[0];
        }
    }
}

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_mean(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               U *tensorMeanArr,
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
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *tensorPartialSumArr;
        tensorPartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(float));
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_result,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           tensorPartialSumArr,
                           gridDim_x * gridDim_y,
                           tensorMeanArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *tensorPartialSumArr;
        tensorPartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(float));
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           tensorPartialSumArr,
                           gridDim_x * gridDim_y,
                           tensorMeanArr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *tensorPartialSumArr;
        tensorPartialSumArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(float));
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result,
                           dim3(1, 1, gridDim_z),
                           dim3(1024, 1, 1),
                           0,
                           handle.GetStream(),
                           tensorPartialSumArr,
                           gridDim_x * gridDim_y,
                           tensorMeanArr,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}
