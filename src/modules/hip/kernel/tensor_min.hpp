#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------

template <typename T>
__global__ void tensor_min_grid_3channel_result_hip(float *srcPtr,
                                                    uint xBufferLength,
                                                    T *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRMin_smem[256];                           // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialGMin_smem[256];                           // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialBMin_smem[256];                           // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength) * 3;
    float srcRefR = srcPtr[srcIdx];                                 // get starting value of R channel
    float srcRefG = srcPtr[srcIdx + 1];                             // get starting value of G channel
    float srcRefB = srcPtr[srcIdx + 2];                             // get starting value of B channel

    partialRMin_smem[hipThreadIdx_x] = srcRefR;                       // initialization of LDS for R channel to srcRefR using all 256 x 1 threads
    partialGMin_smem[hipThreadIdx_x] = srcRefG;                       // initialization of LDS for G channel to srcRefG using all 256 x 1 threads
    partialBMin_smem[hipThreadIdx_x] = srcRefB;                       // initialization of LDS for B channel to srcRefB using all 256 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    srcIdx += id_x * 3;

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);           // load 24 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = srcRefR;                                              // local memory reset of invalid values (from the vectorized global load) to srcRefR
            src_f24.f8[1].f1[i] = srcRefG;  	                                        // local memory reset of invalid values (from the vectorized global load) to srcRefG
            src_f24.f8[2].f1[i] = srcRefB;                                              // local memory reset of invalid values (from the vectorized global load) to srcRefB
        }
    }

    rpp_hip_math_min8(&src_f24.f8[0], &partialRMin_smem[hipThreadIdx_x]);
    rpp_hip_math_min8(&src_f24.f8[1], &partialGMin_smem[hipThreadIdx_x]);
    rpp_hip_math_min8(&src_f24.f8[2], &partialBMin_smem[hipThreadIdx_x]);
    __syncthreads();                                                                    // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMin_smem[hipThreadIdx_x] = fminf(partialRMin_smem[hipThreadIdx_x], partialRMin_smem[hipThreadIdx_x + threadMax]);
            partialGMin_smem[hipThreadIdx_x] = fminf(partialGMin_smem[hipThreadIdx_x], partialGMin_smem[hipThreadIdx_x + threadMax]);
            partialBMin_smem[hipThreadIdx_x] = fminf(partialBMin_smem[hipThreadIdx_x], partialBMin_smem[hipThreadIdx_x + threadMax]);
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int dstIdx = hipBlockIdx_z * 4;
        dstPtr[dstIdx] = (T) partialRMin_smem[0];
        dstPtr[dstIdx + 1] = (T) partialGMin_smem[0];
        dstPtr[dstIdx + 2] = (T) partialBMin_smem[0];
        dstPtr[dstIdx + 3] = (T) (fminf(fminf(partialRMin_smem[0], partialGMin_smem[0]), partialBMin_smem[0]));
    }
}

template <typename T>
__global__ void tensor_min_grid_result_hip(float *srcPtr,
                                           uint xBufferLength,
                                           T *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialMin_smem[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength);
    float srcRef = srcPtr[srcIdx];
    partialMin_smem[hipThreadIdx_x] = srcRef;                         // initialization of LDS to srcRef using all 256 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    srcIdx += id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;                                  // local memory reset of invalid values (from the vectorized global load) to srcRef

    rpp_hip_math_min8(&src_f8, &partialMin_smem[hipThreadIdx_x]);
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMin_smem[hipThreadIdx_x] = fminf(partialMin_smem[hipThreadIdx_x], partialMin_smem[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = (T) (partialMin_smem[0]);
}


// -------------------- Set 1 - Reduction Stage 1 --------------------

template <typename T>
__global__ void tensor_min_pkd3_hip(T *srcPtr,
                                    uint2 srcStridesNH,
                                    float *minArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block for R channel
    __shared__ float partialGMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block for G channel
    __shared__ float partialBMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block for B channel

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRefR = srcPtr[srcIdx];                                          // get starting value of R channel
    float srcRefG = srcPtr[srcIdx + 1];                                      // get starting value of G channel
    float srcRefB = srcPtr[srcIdx + 2];                                      // get starting value of B channel

    float *partialRMinRowPtr_smem = &partialRMin_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS for R Channel
    float *partialGMinRowPtr_smem = &partialGMin_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS for G Channel
    float *partialBMinRowPtr_smem = &partialBMin_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS for B Channel

    partialRMinRowPtr_smem[hipThreadIdx_x] = srcRefR;                          // initialization of LDS for R channel to srcRefR using all 16 x 16 threads
    partialGMinRowPtr_smem[hipThreadIdx_x] = srcRefG;                          // initialization of LDS for G channel to srcRefG using all 16 x 16 threads
    partialBMinRowPtr_smem[hipThreadIdx_x] = srcRefB;                          // initialization of LDS for B channel to srcRefB using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        minArr[idx] = srcRefR;
        minArr[idx + 1] = srcRefG;
        minArr[idx + 2] = srcRefB;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);           // load 24 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = srcRefR;
            src_f24.f8[1].f1[i] = srcRefG;
            src_f24.f8[2].f1[i] = srcRefB;
        }
    }

    rpp_hip_math_min8(&src_f24.f8[0], &partialRMinRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_min8(&src_f24.f8[1], &partialGMinRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_min8(&src_f24.f8[2], &partialBMinRowPtr_smem[hipThreadIdx_x]);
    __syncthreads();

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMinRowPtr_smem[hipThreadIdx_x] = fminf(partialRMinRowPtr_smem[hipThreadIdx_x], partialRMinRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialGMinRowPtr_smem[hipThreadIdx_x] = fminf(partialGMinRowPtr_smem[hipThreadIdx_x], partialGMinRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialBMinRowPtr_smem[hipThreadIdx_x] = fminf(partialBMinRowPtr_smem[hipThreadIdx_x], partialBMinRowPtr_smem[hipThreadIdx_x + threadMax]);
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
                partialRMinRowPtr_smem[0] = fminf(partialRMinRowPtr_smem[0], partialRMinRowPtr_smem[increment]);
                partialGMinRowPtr_smem[0] = fminf(partialGMinRowPtr_smem[0], partialGMinRowPtr_smem[increment]);
                partialBMinRowPtr_smem[0] = fminf(partialBMinRowPtr_smem[0], partialBMinRowPtr_smem[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            minArr[idx] = partialRMinRowPtr_smem[0];
            minArr[idx + 1] = partialGMinRowPtr_smem[0];
            minArr[idx + 2] = partialBMinRowPtr_smem[0];
        }
    }
}

template <typename T>
__global__ void tensor_min_pln3_hip(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    float *minArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialBMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNCH.x);
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + srcStridesNCH.y];
    float srcRefB = srcPtr[srcIdx + 2 * srcStridesNCH.y];

    float *partialRMinRowPtr_smem = &partialRMin_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialGMinRowPtr_smem = &partialGMin_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialBMinRowPtr_smem = &partialBMin_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS

    partialRMinRowPtr_smem[hipThreadIdx_x] = srcRefR;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialGMinRowPtr_smem[hipThreadIdx_x] = srcRefG;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialBMinRowPtr_smem[hipThreadIdx_x] = srcRefB;                          // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        minArr[idx] = srcRefR;
        minArr[idx + 1] = srcRefG;
        minArr[idx + 2] = srcRefB;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;        // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;     // difference between roiWidth and alignedLength
    srcIdx += ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = srcRefR;                                   // local memory reset of invalid values (from the vectorized global load) to srcRefR
            src_f24.f8[1].f1[i] = srcRefG;                                   // local memory reset of invalid values (from the vectorized global load) to srcRefG
            src_f24.f8[2].f1[i] = srcRefB;                                   // local memory reset of invalid values (from the vectorized global load) to srcRefB
        }
    }

    rpp_hip_math_min8(&src_f24.f8[0], &partialRMinRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_min8(&src_f24.f8[1], &partialGMinRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_min8(&src_f24.f8[2], &partialBMinRowPtr_smem[hipThreadIdx_x]);
    __syncthreads();                                                         // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMinRowPtr_smem[hipThreadIdx_x] = fminf(partialRMinRowPtr_smem[hipThreadIdx_x], partialRMinRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialGMinRowPtr_smem[hipThreadIdx_x] = fminf(partialGMinRowPtr_smem[hipThreadIdx_x], partialGMinRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialBMinRowPtr_smem[hipThreadIdx_x] = fminf(partialBMinRowPtr_smem[hipThreadIdx_x], partialBMinRowPtr_smem[hipThreadIdx_x + threadMax]);
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
                partialRMinRowPtr_smem[0] = fminf(partialRMinRowPtr_smem[0], partialRMinRowPtr_smem[increment]);
                partialGMinRowPtr_smem[0] = fminf(partialGMinRowPtr_smem[0], partialGMinRowPtr_smem[increment]);
                partialBMinRowPtr_smem[0] = fminf(partialBMinRowPtr_smem[0], partialBMinRowPtr_smem[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            minArr[idx] = partialRMinRowPtr_smem[0];
            minArr[idx + 1] = partialGMinRowPtr_smem[0];
            minArr[idx + 2] = partialBMinRowPtr_smem[0];
        }
    }
}

template <typename T>
__global__ void tensor_min_pln1_hip(T *srcPtr,
                                    uint2 srcStridesNH,
                                    float *minArr,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialMin_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = srcPtr[srcIdx];
    float *partialMinRowPtr_smem = &partialMin_smem[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialMinRowPtr_smem[hipThreadIdx_x] = srcRef;                           // initialization of LDS to srcRefR using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        minArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = srcRef;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    srcIdx += ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;                                          // local memory reset of invalid values (from the vectorized global load) to srcRef

    rpp_hip_math_min8(&src_f8, &partialMinRowPtr_smem[hipThreadIdx_x]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMinRowPtr_smem[hipThreadIdx_x] = fminf(partialMinRowPtr_smem[hipThreadIdx_x], partialMinRowPtr_smem[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialMinRowPtr_smem[0] = fminf(partialMinRowPtr_smem[0], partialMinRowPtr_smem[increment]);
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            minArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialMinRowPtr_smem[0];
    }
}


// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_min(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *minArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialMinArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *partialMinArr;
        partialMinArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(partialMinArr, 0, partialMinArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_min_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialMinArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_min_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMinArr,
                           gridDim_x * gridDim_y,
                           minArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialMinArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *partialMinArr;
        partialMinArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(partialMinArr, 0, partialMinArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_min_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           partialMinArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_min_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMinArr,
                           gridDim_x * gridDim_y,
                           minArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u partialMinArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *partialMinArr;
        partialMinArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemsetAsync(partialMinArr, 0, partialMinArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_min_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialMinArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_min_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMinArr,
                           gridDim_x * gridDim_y,
                           minArr);
    }

    return RPP_SUCCESS;
}