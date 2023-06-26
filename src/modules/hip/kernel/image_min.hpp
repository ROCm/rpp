#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - image_min reduction stage 2 --------------------

// template <typename T>
__global__ void image_min_grid_3channel_result_tensor(float *srcPtr,
                                                      uint xBufferLength,
                                                      float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRMinLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialGMinLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialBMinLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength) * 3;
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + 1];
    float srcRefB = srcPtr[srcIdx + 2];

    partialRMinLDS[hipThreadIdx_x] = srcRefR;                         // initialization of LDS to 0 using all 256 x 1 threads
    partialGMinLDS[hipThreadIdx_x] = srcRefG;                         // initialization of LDS to 0 using all 256 x 1 threads
    partialBMinLDS[hipThreadIdx_x] = srcRefB;                         // initialization of LDS to 0 using all 256 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                          // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                       // difference between bufferLength and alignedLength
    srcIdx += id_x * 3;

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);           // load 24 pixels to local mmemory
    if (id_x + 8 > xBufferLength)                                                        // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = srcRefR;
            src_f24.f8[1].f1[i] = srcRefG;
            src_f24.f8[2].f1[i] = srcRefB;
        }
    }

    partialRMinLDS[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[0].f1[0], src_f24.f8[0].f1[1]), src_f24.f8[0].f1[2]), src_f24.f8[0].f1[3]), src_f24.f8[0].f1[4]), src_f24.f8[0].f1[5]), src_f24.f8[0].f1[6]), src_f24.f8[0].f1[7]);
    partialGMinLDS[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[1].f1[0], src_f24.f8[1].f1[1]), src_f24.f8[1].f1[2]), src_f24.f8[1].f1[3]), src_f24.f8[1].f1[4]), src_f24.f8[1].f1[5]), src_f24.f8[1].f1[6]), src_f24.f8[1].f1[7]);
    partialBMinLDS[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[2].f1[0], src_f24.f8[2].f1[1]), src_f24.f8[2].f1[2]), src_f24.f8[2].f1[3]), src_f24.f8[2].f1[4]), src_f24.f8[2].f1[5]), src_f24.f8[2].f1[6]), src_f24.f8[2].f1[7]);
    __syncthreads();                                                                    // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMinLDS[hipThreadIdx_x] = fminf(partialRMinLDS[hipThreadIdx_x], partialRMinLDS[hipThreadIdx_x + threadMax]);
            partialGMinLDS[hipThreadIdx_x] = fminf(partialGMinLDS[hipThreadIdx_x], partialGMinLDS[hipThreadIdx_x + threadMax]);
            partialBMinLDS[hipThreadIdx_x] = fminf(partialBMinLDS[hipThreadIdx_x], partialBMinLDS[hipThreadIdx_x + threadMax]);
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int dstIdx = hipBlockIdx_z * 4;
        dstPtr[dstIdx] = partialRMinLDS[0];
        dstPtr[dstIdx + 1] = partialGMinLDS[0];
        dstPtr[dstIdx + 2] = partialBMinLDS[0];
        dstPtr[dstIdx + 3] = (fminf(fminf(partialRMinLDS[0], partialGMinLDS[0]), partialBMinLDS[0]));
    }
}

// template <typename T>
__global__ void image_min_grid_result_tensor(float *srcPtr,
                                             uint xBufferLength,
                                             float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialMinLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength);
    float srcRef = srcPtr[srcIdx];
    partialMinLDS[hipThreadIdx_x] = srcRef;                         // initialization of LDS to 0 using all 256 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    srcIdx += id_x;

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;                                  // local memory reset of invalid values (from the vectorized global load) to 0.0f

    partialMinLDS[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f8.f1[0], src_f8.f1[1]), src_f8.f1[2]), src_f8.f1[3]), src_f8.f1[4]), src_f8.f1[5]), src_f8.f1[6]), src_f8.f1[7]);
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMinLDS[hipThreadIdx_x] = fminf(partialMinLDS[hipThreadIdx_x], partialMinLDS[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = partialMinLDS[0];
}


// -------------------- Set 0 - image_min reduction stage 1 --------------------

template <typename T>
__global__ void image_min_pkd3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      float *imageMinArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMinLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGMinLDS[16][16];
    __shared__ float partialBMinLDS[16][16];

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + 1];
    float srcRefB = srcPtr[srcIdx + 2];

    float *partialRMinLDSRowPtr = &partialRMinLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    float *partialGMinLDSRowPtr = &partialGMinLDS[hipThreadIdx_y][0];
    float *partialBMinLDSRowPtr = &partialBMinLDS[hipThreadIdx_y][0];

    partialRMinLDSRowPtr[hipThreadIdx_x] = srcRefR;                           // initialization of LDS to 0 using all 16 x 16 threads
    partialGMinLDSRowPtr[hipThreadIdx_x] = srcRefG;
    partialBMinLDSRowPtr[hipThreadIdx_x] = srcRefB;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        imageMinArr[idx] = srcRefR;
        imageMinArr[idx + 1] = srcRefG;
        imageMinArr[idx + 2] = srcRefB;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);

    d_float24 src_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);           // load 24 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                               // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = srcRefR;
            src_f24.f8[1].f1[i] = srcRefG;
            src_f24.f8[2].f1[i] = srcRefB;
        }
    }

    partialRMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[0].f1[0], src_f24.f8[0].f1[1]), src_f24.f8[0].f1[2]), src_f24.f8[0].f1[3]), src_f24.f8[0].f1[4]), src_f24.f8[0].f1[5]), src_f24.f8[0].f1[6]), src_f24.f8[0].f1[7]);
    partialGMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[1].f1[0], src_f24.f8[1].f1[1]), src_f24.f8[1].f1[2]), src_f24.f8[1].f1[3]), src_f24.f8[1].f1[4]), src_f24.f8[1].f1[5]), src_f24.f8[1].f1[6]), src_f24.f8[1].f1[7]);
    partialBMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[2].f1[0], src_f24.f8[2].f1[1]), src_f24.f8[2].f1[2]), src_f24.f8[2].f1[3]), src_f24.f8[2].f1[4]), src_f24.f8[2].f1[5]), src_f24.f8[2].f1[6]), src_f24.f8[2].f1[7]);
    __syncthreads();

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMinLDSRowPtr[hipThreadIdx_x] = fminf(partialRMinLDSRowPtr[hipThreadIdx_x], partialRMinLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialGMinLDSRowPtr[hipThreadIdx_x] = fminf(partialGMinLDSRowPtr[hipThreadIdx_x], partialGMinLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialBMinLDSRowPtr[hipThreadIdx_x] = fminf(partialBMinLDSRowPtr[hipThreadIdx_x], partialBMinLDSRowPtr[hipThreadIdx_x + threadMax]);
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
                partialRMinLDSRowPtr[0] = fminf(partialRMinLDSRowPtr[0], partialRMinLDSRowPtr[increment]);
                partialGMinLDSRowPtr[0] = fminf(partialGMinLDSRowPtr[0], partialGMinLDSRowPtr[increment]);
                partialBMinLDSRowPtr[0] = fminf(partialBMinLDSRowPtr[0], partialBMinLDSRowPtr[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageMinArr[idx] = partialRMinLDSRowPtr[0];
            imageMinArr[idx + 1] = partialGMinLDSRowPtr[0];
            imageMinArr[idx + 2] = partialBMinLDSRowPtr[0];
        }
    }
}

template <typename T>
__global__ void image_min_pln3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      float *imageMinArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMinLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGMinLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialBMinLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNCH.x);
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + srcStridesNCH.y];
    float srcRefB = srcPtr[srcIdx + 2 * srcStridesNCH.y];

    float *partialRMinLDSRowPtr = &partialRMinLDS[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialGMinLDSRowPtr = &partialGMinLDS[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialBMinLDSRowPtr = &partialBMinLDS[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS

    partialRMinLDSRowPtr[hipThreadIdx_x] = srcRefR;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialGMinLDSRowPtr[hipThreadIdx_x] = srcRefG;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialBMinLDSRowPtr[hipThreadIdx_x] = srcRefB;                          // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        imageMinArr[idx] = srcRefR;
        imageMinArr[idx + 1] = srcRefG;
        imageMinArr[idx + 2] = srcRefB;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    srcIdx += ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float24 src_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                  // local memory reset of invalid values (from the vectorized global load) to srcRef
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_f24.f8[0].f1[i] = srcRefR;
            src_f24.f8[1].f1[i] = srcRefG;
            src_f24.f8[2].f1[i] = srcRefB;
        }
    }

    partialRMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[0].f1[0], src_f24.f8[0].f1[1]), src_f24.f8[0].f1[2]), src_f24.f8[0].f1[3]), src_f24.f8[0].f1[4]), src_f24.f8[0].f1[5]), src_f24.f8[0].f1[6]), src_f24.f8[0].f1[7]);
    partialGMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[1].f1[0], src_f24.f8[1].f1[1]), src_f24.f8[1].f1[2]), src_f24.f8[1].f1[3]), src_f24.f8[1].f1[4]), src_f24.f8[1].f1[5]), src_f24.f8[1].f1[6]), src_f24.f8[1].f1[7]);
    partialBMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f24.f8[2].f1[0], src_f24.f8[2].f1[1]), src_f24.f8[2].f1[2]), src_f24.f8[2].f1[3]), src_f24.f8[2].f1[4]), src_f24.f8[2].f1[5]), src_f24.f8[2].f1[6]), src_f24.f8[2].f1[7]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMinLDSRowPtr[hipThreadIdx_x] = fminf(partialRMinLDSRowPtr[hipThreadIdx_x], partialRMinLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialGMinLDSRowPtr[hipThreadIdx_x] = fminf(partialGMinLDSRowPtr[hipThreadIdx_x], partialGMinLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialBMinLDSRowPtr[hipThreadIdx_x] = fminf(partialBMinLDSRowPtr[hipThreadIdx_x], partialBMinLDSRowPtr[hipThreadIdx_x + threadMax]);
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
                partialRMinLDSRowPtr[0] = fminf(partialRMinLDSRowPtr[0], partialRMinLDSRowPtr[increment]);
                partialGMinLDSRowPtr[0] = fminf(partialGMinLDSRowPtr[0], partialGMinLDSRowPtr[increment]);
                partialBMinLDSRowPtr[0] = fminf(partialBMinLDSRowPtr[0], partialBMinLDSRowPtr[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageMinArr[idx] = partialRMinLDSRowPtr[0];
            imageMinArr[idx + 1] = partialGMinLDSRowPtr[0];
            imageMinArr[idx + 2] = partialBMinLDSRowPtr[0];
        }
    }
}

template <typename T>
__global__ void image_min_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      float *imageMinArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialMinLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = srcPtr[srcIdx];
    float *partialMinLDSRowPtr = &partialMinLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialMinLDSRowPtr[hipThreadIdx_x] = srcRef;                             // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        imageMinArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = srcRef;
        return;
    }

    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength
    srcIdx += ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                  // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;
    }

    partialMinLDSRowPtr[hipThreadIdx_x] = fminf(fminf(fminf(fminf(fminf(fminf(fminf(src_f8.f1[0], src_f8.f1[1]), src_f8.f1[2]), src_f8.f1[3]), src_f8.f1[4]), src_f8.f1[5]), src_f8.f1[6]), src_f8.f1[7]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMinLDSRowPtr[hipThreadIdx_x] = fminf(partialMinLDSRowPtr[hipThreadIdx_x], partialMinLDSRowPtr[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialMinLDSRowPtr[0] = fminf(partialMinLDSRowPtr[0], partialMinLDSRowPtr[increment]);
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            imageMinArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialMinLDSRowPtr[0];
    }
}


// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_image_min_tensor(T *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    U *imageMinArr,
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
        Rpp32u imagePartialMinArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *imagePartialMinArr;
        imagePartialMinArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialMinArr, 0, imagePartialMinArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(image_min_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMinArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(image_min_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMinArr,
                           gridDim_x * gridDim_y,
                           imageMinArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u imagePartialMinArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *imagePartialMinArr;
        imagePartialMinArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialMinArr, 0, imagePartialMinArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(image_min_pln3_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           imagePartialMinArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_min_grid_3channel_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMinArr,
                           gridDim_x * gridDim_y,
                           imageMinArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u imagePartialMinArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *imagePartialMinArr;
        imagePartialMinArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialMinArr, 0, imagePartialMinArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(image_min_pkd3_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMinArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(image_min_grid_3channel_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMinArr,
                           gridDim_x * gridDim_y,
                           imageMinArr);
    }

    return RPP_SUCCESS;
}