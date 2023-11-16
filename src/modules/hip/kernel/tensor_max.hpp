#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------

template <typename T>
__global__ void tensor_max_grid_3channel_result_hip(float *srcPtr,
                                                uint xBufferLength,
                                                T *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRMax_smem[256];                           // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialGMax_smem[256];                           // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialBMax_smem[256];                           // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength) * 3;
    float srcRefR = srcPtr[srcIdx];                                 // get starting value of R channel
    float srcRefG = srcPtr[srcIdx + 1];                             // get starting value of G channel
    float srcRefB = srcPtr[srcIdx + 2];                             // get starting value of B channel

    partialRMax_smem[hipThreadIdx_x] = srcRefR;                       // initialization of LDS for R channel to srcRefR using all 256 x 1 threads
    partialGMax_smem[hipThreadIdx_x] = srcRefG;                       // initialization of LDS for G channel to srcRefG using all 256 x 1 threads
    partialBMax_smem[hipThreadIdx_x] = srcRefB;                       // initialization of LDS for B channel to srcRefB using all 256 x 1 threads

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

    rpp_hip_math_max8(&src_f24.f8[0], &partialRMax_smem[hipThreadIdx_x]);
    rpp_hip_math_max8(&src_f24.f8[1], &partialGMax_smem[hipThreadIdx_x]);
    rpp_hip_math_max8(&src_f24.f8[2], &partialBMax_smem[hipThreadIdx_x]);
    __syncthreads();                                                                    // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMax_smem[hipThreadIdx_x] = fmaxf(partialRMax_smem[hipThreadIdx_x], partialRMax_smem[hipThreadIdx_x + threadMax]);
            partialGMax_smem[hipThreadIdx_x] = fmaxf(partialGMax_smem[hipThreadIdx_x], partialGMax_smem[hipThreadIdx_x + threadMax]);
            partialBMax_smem[hipThreadIdx_x] = fmaxf(partialBMax_smem[hipThreadIdx_x], partialBMax_smem[hipThreadIdx_x + threadMax]);
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int dstIdx = hipBlockIdx_z * 4;
        dstPtr[dstIdx] = (T) partialRMax_smem[0];
        dstPtr[dstIdx + 1] = (T) partialGMax_smem[0];
        dstPtr[dstIdx + 2] = (T) partialBMax_smem[0];
        dstPtr[dstIdx + 3] = (T) (fmaxf(fmaxf(partialRMax_smem[0], partialGMax_smem[0]), partialBMax_smem[0]));
    }
}

template <typename T>
__global__ void tensor_max_grid_result_hip(float *srcPtr,
                                       uint xBufferLength,
                                       T *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialMax_smem[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength);
    float srcRef = srcPtr[srcIdx];
    partialMax_smem[hipThreadIdx_x] = srcRef;                         // initialization of LDS to srcRef using all 256 x 1 threads

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

    rpp_hip_math_max8(&src_f8, &partialMax_smem[hipThreadIdx_x]);
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMax_smem[hipThreadIdx_x] = fmaxf(partialMax_smem[hipThreadIdx_x], partialMax_smem[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = (T) (partialMax_smem[0]);
}


// -------------------- Set 1 - Reduction Stage 1 --------------------

template <typename T>
__global__ void tensor_max_pkd3_hip(T *srcPtr,
                                uint2 srcStridesNH,
                                float *maxArr,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block for R channel
    __shared__ float partialGMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block for G channel
    __shared__ float partialBMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block for B channel

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRefR = srcPtr[srcIdx];                                          // get starting value of R channel
    float srcRefG = srcPtr[srcIdx + 1];                                      // get starting value of G channel
    float srcRefB = srcPtr[srcIdx + 2];                                      // get starting value of B channel

    float *partialRMaxRowPtr_smem = &partialRMax_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS for R Channel
    float *partialGMaxRowPtr_smem = &partialGMax_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS for G Channel
    float *partialBMaxRowPtr_smem = &partialBMax_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS for B Channel

    partialRMaxRowPtr_smem[hipThreadIdx_x] = srcRefR;                          // initialization of LDS for R channel to srcRefR using all 16 x 16 threads
    partialGMaxRowPtr_smem[hipThreadIdx_x] = srcRefG;                          // initialization of LDS for G channel to srcRefG using all 16 x 16 threads
    partialBMaxRowPtr_smem[hipThreadIdx_x] = srcRefB;                          // initialization of LDS for B channel to srcRefB using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        maxArr[idx] = srcRefR;
        maxArr[idx + 1] = srcRefG;
        maxArr[idx + 2] = srcRefB;
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

    rpp_hip_math_max8(&src_f24.f8[0], &partialRMaxRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_max8(&src_f24.f8[1], &partialGMaxRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_max8(&src_f24.f8[2], &partialBMaxRowPtr_smem[hipThreadIdx_x]);
    __syncthreads();

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialRMaxRowPtr_smem[hipThreadIdx_x], partialRMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialGMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialGMaxRowPtr_smem[hipThreadIdx_x], partialGMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialBMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialBMaxRowPtr_smem[hipThreadIdx_x], partialBMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
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
                partialRMaxRowPtr_smem[0] = fmaxf(partialRMaxRowPtr_smem[0], partialRMaxRowPtr_smem[increment]);
                partialGMaxRowPtr_smem[0] = fmaxf(partialGMaxRowPtr_smem[0], partialGMaxRowPtr_smem[increment]);
                partialBMaxRowPtr_smem[0] = fmaxf(partialBMaxRowPtr_smem[0], partialBMaxRowPtr_smem[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            maxArr[idx] = partialRMaxRowPtr_smem[0];
            maxArr[idx + 1] = partialGMaxRowPtr_smem[0];
            maxArr[idx + 2] = partialBMaxRowPtr_smem[0];
        }
    }
}

template <typename T>
__global__ void tensor_max_pln3_hip(T *srcPtr,
                                uint3 srcStridesNCH,
                                float *maxArr,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialBMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNCH.x);
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + srcStridesNCH.y];
    float srcRefB = srcPtr[srcIdx + 2 * srcStridesNCH.y];

    float *partialRMaxRowPtr_smem = &partialRMax_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialGMaxRowPtr_smem = &partialGMax_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialBMaxRowPtr_smem = &partialBMax_smem[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS

    partialRMaxRowPtr_smem[hipThreadIdx_x] = srcRefR;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialGMaxRowPtr_smem[hipThreadIdx_x] = srcRefG;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialBMaxRowPtr_smem[hipThreadIdx_x] = srcRefB;                          // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        maxArr[idx] = srcRefR;
        maxArr[idx + 1] = srcRefG;
        maxArr[idx + 2] = srcRefB;
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

    rpp_hip_math_max8(&src_f24.f8[0], &partialRMaxRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_max8(&src_f24.f8[1], &partialGMaxRowPtr_smem[hipThreadIdx_x]);
    rpp_hip_math_max8(&src_f24.f8[2], &partialBMaxRowPtr_smem[hipThreadIdx_x]);
    __syncthreads();                                                         // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialRMaxRowPtr_smem[hipThreadIdx_x], partialRMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialGMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialGMaxRowPtr_smem[hipThreadIdx_x], partialGMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
            partialBMaxRowPtr_smem[hipThreadIdx_x] = fmaxf(partialBMaxRowPtr_smem[hipThreadIdx_x], partialBMaxRowPtr_smem[hipThreadIdx_x + threadMax]);
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
                partialRMaxRowPtr_smem[0] = fmaxf(partialRMaxRowPtr_smem[0], partialRMaxRowPtr_smem[increment]);
                partialGMaxRowPtr_smem[0] = fmaxf(partialGMaxRowPtr_smem[0], partialGMaxRowPtr_smem[increment]);
                partialBMaxRowPtr_smem[0] = fmaxf(partialBMaxRowPtr_smem[0], partialBMaxRowPtr_smem[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            maxArr[idx] = partialRMaxRowPtr_smem[0];
            maxArr[idx + 1] = partialGMaxRowPtr_smem[0];
            maxArr[idx + 2] = partialBMaxRowPtr_smem[0];
        }
    }
}

template <typename T>
__global__ void tensor_max_pln1_hip(T *srcPtr,
                                uint2 srcStridesNH,
                                float *maxArr,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialMax_smem[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = srcPtr[srcIdx];
    float *partialMaxRowPtr_smem = &partialMax_smem[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialMaxRowPtr_smem[hipThreadIdx_x] = srcRef;                           // initialization of LDS to srcRefR using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        maxArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = srcRef;
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

    rpp_hip_math_max8(&src_f8, &partialMaxRowPtr_smem[hipThreadIdx_x]);
    __syncthreads();                                                        // syncthreads after LDS load

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


// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_tensor_max(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *maxArr,
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
        Rpp32u partialMaxArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *partialMaxArr;
        partialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(partialMaxArr, 0, partialMaxArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(tensor_max_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialMaxArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(tensor_max_grid_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMaxArr,
                           gridDim_x * gridDim_y,
                           maxArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u partialMaxArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *partialMaxArr;
        partialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(partialMaxArr, 0, partialMaxArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(tensor_max_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           partialMaxArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(tensor_max_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMaxArr,
                           gridDim_x * gridDim_y,
                           maxArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u partialMaxArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *partialMaxArr;
        partialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(partialMaxArr, 0, partialMaxArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(tensor_max_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           partialMaxArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(tensor_max_grid_3channel_result_hip,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           partialMaxArr,
                           gridDim_x * gridDim_y,
                           maxArr);
    }

    return RPP_SUCCESS;
}