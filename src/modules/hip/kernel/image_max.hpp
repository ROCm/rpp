#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - image_max reduction stage 2 --------------------

// template <typename T>
__global__ void image_max_grid_3channel_result_tensor(float *srcPtr,
                                                      uint xBufferLength,
                                                      float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRMaxLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialGMaxLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    __shared__ float partialBMaxLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength) * 3;
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + 1];
    float srcRefB = srcPtr[srcIdx + 2];

    partialRMaxLDS[hipThreadIdx_x] = srcRefR;                         // initialization of LDS to 0 using all 256 x 1 threads
    partialGMaxLDS[hipThreadIdx_x] = srcRefG;                         // initialization of LDS to 0 using all 256 x 1 threads
    partialBMaxLDS[hipThreadIdx_x] = srcRefB;                         // initialization of LDS to 0 using all 256 x 1 threads

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

    partialRMaxLDS[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[0].f1[0], src_f24.f8[0].f1[1]), src_f24.f8[0].f1[2]), src_f24.f8[0].f1[3]), src_f24.f8[0].f1[4]), src_f24.f8[0].f1[5]), src_f24.f8[0].f1[6]), src_f24.f8[0].f1[7]);
    partialGMaxLDS[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[1].f1[0], src_f24.f8[1].f1[1]), src_f24.f8[1].f1[2]), src_f24.f8[1].f1[3]), src_f24.f8[1].f1[4]), src_f24.f8[1].f1[5]), src_f24.f8[1].f1[6]), src_f24.f8[1].f1[7]);
    partialBMaxLDS[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[2].f1[0], src_f24.f8[2].f1[1]), src_f24.f8[2].f1[2]), src_f24.f8[2].f1[3]), src_f24.f8[2].f1[4]), src_f24.f8[2].f1[5]), src_f24.f8[2].f1[6]), src_f24.f8[2].f1[7]);
    __syncthreads();                                                                    // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMaxLDS[hipThreadIdx_x] = fmaxf(partialRMaxLDS[hipThreadIdx_x], partialRMaxLDS[hipThreadIdx_x + threadMax]);
            partialGMaxLDS[hipThreadIdx_x] = fmaxf(partialGMaxLDS[hipThreadIdx_x], partialGMaxLDS[hipThreadIdx_x + threadMax]);
            partialBMaxLDS[hipThreadIdx_x] = fmaxf(partialBMaxLDS[hipThreadIdx_x], partialBMaxLDS[hipThreadIdx_x + threadMax]);
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int dstIdx = hipBlockIdx_z * 4;
        dstPtr[dstIdx] = partialRMaxLDS[0];
        dstPtr[dstIdx + 1] = partialGMaxLDS[0];
        dstPtr[dstIdx + 2] = partialBMaxLDS[0];
        dstPtr[dstIdx + 3] = (fmaxf(fmaxf(partialRMaxLDS[0], partialGMaxLDS[0]), partialBMaxLDS[0]));
    }
}

// template <typename T>
__global__ void image_max_grid_result_tensor(float *srcPtr,
                                             uint xBufferLength,
                                             float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialMaxLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block

    uint srcIdx = (id_z * xBufferLength);
    float srcRef = srcPtr[srcIdx];
    partialMaxLDS[hipThreadIdx_x] = srcRef;                         // initialization of LDS to 0 using all 256 x 1 threads

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

    partialMaxLDS[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f8.f1[0], src_f8.f1[1]), src_f8.f1[2]), src_f8.f1[3]), src_f8.f1[4]), src_f8.f1[5]), src_f8.f1[6]), src_f8.f1[7]);
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMaxLDS[hipThreadIdx_x] = fmaxf(partialMaxLDS[hipThreadIdx_x], partialMaxLDS[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = partialMaxLDS[0];
}


// -------------------- Set 0 - image_max reduction stage 1 --------------------

template <typename T>
__global__ void image_max_pkd3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      float *imageMaxArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMaxLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGMaxLDS[16][16];
    __shared__ float partialBMaxLDS[16][16];

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + 1];
    float srcRefB = srcPtr[srcIdx + 2];

    float *partialRMaxLDSRowPtr = &partialRMaxLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    float *partialGMaxLDSRowPtr = &partialGMaxLDS[hipThreadIdx_y][0];
    float *partialBMaxLDSRowPtr = &partialBMaxLDS[hipThreadIdx_y][0];

    partialRMaxLDSRowPtr[hipThreadIdx_x] = srcRefR;                           // initialization of LDS to 0 using all 16 x 16 threads
    partialGMaxLDSRowPtr[hipThreadIdx_x] = srcRefG;
    partialBMaxLDSRowPtr[hipThreadIdx_x] = srcRefB;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        imageMaxArr[idx] = srcRefR;
        imageMaxArr[idx + 1] = srcRefG;
        imageMaxArr[idx + 2] = srcRefB;
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

    partialRMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[0].f1[0], src_f24.f8[0].f1[1]), src_f24.f8[0].f1[2]), src_f24.f8[0].f1[3]), src_f24.f8[0].f1[4]), src_f24.f8[0].f1[5]), src_f24.f8[0].f1[6]), src_f24.f8[0].f1[7]);
    partialGMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[1].f1[0], src_f24.f8[1].f1[1]), src_f24.f8[1].f1[2]), src_f24.f8[1].f1[3]), src_f24.f8[1].f1[4]), src_f24.f8[1].f1[5]), src_f24.f8[1].f1[6]), src_f24.f8[1].f1[7]);
    partialBMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[2].f1[0], src_f24.f8[2].f1[1]), src_f24.f8[2].f1[2]), src_f24.f8[2].f1[3]), src_f24.f8[2].f1[4]), src_f24.f8[2].f1[5]), src_f24.f8[2].f1[6]), src_f24.f8[2].f1[7]);
    __syncthreads();

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialRMaxLDSRowPtr[hipThreadIdx_x], partialRMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialGMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialGMaxLDSRowPtr[hipThreadIdx_x], partialGMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialBMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialBMaxLDSRowPtr[hipThreadIdx_x], partialBMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
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
                partialRMaxLDSRowPtr[0] = fmaxf(partialRMaxLDSRowPtr[0], partialRMaxLDSRowPtr[increment]);
                partialGMaxLDSRowPtr[0] = fmaxf(partialGMaxLDSRowPtr[0], partialGMaxLDSRowPtr[increment]);
                partialBMaxLDSRowPtr[0] = fmaxf(partialBMaxLDSRowPtr[0], partialBMaxLDSRowPtr[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageMaxArr[idx] = partialRMaxLDSRowPtr[0];
            imageMaxArr[idx + 1] = partialGMaxLDSRowPtr[0];
            imageMaxArr[idx + 2] = partialBMaxLDSRowPtr[0];
        }
    }
}

template <typename T>
__global__ void image_max_pln3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      float *imageMaxArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialRMaxLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialGMaxLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    __shared__ float partialBMaxLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNCH.x);
    float srcRefR = srcPtr[srcIdx];
    float srcRefG = srcPtr[srcIdx + srcStridesNCH.y];
    float srcRefB = srcPtr[srcIdx + 2 * srcStridesNCH.y];

    float *partialRMaxLDSRowPtr = &partialRMaxLDS[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialGMaxLDSRowPtr = &partialGMaxLDS[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS
    float *partialBMaxLDSRowPtr = &partialBMaxLDS[hipThreadIdx_y][0];        // float pointer to beginning of each row in LDS

    partialRMaxLDSRowPtr[hipThreadIdx_x] = srcRefR;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialGMaxLDSRowPtr[hipThreadIdx_x] = srcRefG;                          // initialization of LDS to 0 using all 16 x 16 threads
    partialBMaxLDSRowPtr[hipThreadIdx_x] = srcRefB;                          // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
        imageMaxArr[idx] = srcRefR;
        imageMaxArr[idx + 1] = srcRefG;
        imageMaxArr[idx + 2] = srcRefB;
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

    partialRMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[0].f1[0], src_f24.f8[0].f1[1]), src_f24.f8[0].f1[2]), src_f24.f8[0].f1[3]), src_f24.f8[0].f1[4]), src_f24.f8[0].f1[5]), src_f24.f8[0].f1[6]), src_f24.f8[0].f1[7]);
    partialGMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[1].f1[0], src_f24.f8[1].f1[1]), src_f24.f8[1].f1[2]), src_f24.f8[1].f1[3]), src_f24.f8[1].f1[4]), src_f24.f8[1].f1[5]), src_f24.f8[1].f1[6]), src_f24.f8[1].f1[7]);
    partialBMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f24.f8[2].f1[0], src_f24.f8[2].f1[1]), src_f24.f8[2].f1[2]), src_f24.f8[2].f1[3]), src_f24.f8[2].f1[4]), src_f24.f8[2].f1[5]), src_f24.f8[2].f1[6]), src_f24.f8[2].f1[7]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialRMaxLDSRowPtr[hipThreadIdx_x], partialRMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialGMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialGMaxLDSRowPtr[hipThreadIdx_x], partialGMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
            partialBMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialBMaxLDSRowPtr[hipThreadIdx_x], partialBMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
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
                partialRMaxLDSRowPtr[0] = fmaxf(partialRMaxLDSRowPtr[0], partialRMaxLDSRowPtr[increment]);
                partialGMaxLDSRowPtr[0] = fmaxf(partialGMaxLDSRowPtr[0], partialGMaxLDSRowPtr[increment]);
                partialBMaxLDSRowPtr[0] = fmaxf(partialBMaxLDSRowPtr[0], partialBMaxLDSRowPtr[increment]);
            }
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
        {
            int idx = ((hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x) * 3;
            imageMaxArr[idx] = partialRMaxLDSRowPtr[0];
            imageMaxArr[idx + 1] = partialGMaxLDSRowPtr[0];
            imageMaxArr[idx + 2] = partialBMaxLDSRowPtr[0];
        }
    }
}

template <typename T>
__global__ void image_max_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      float *imageMaxArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    __shared__ float partialMaxLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = srcPtr[srcIdx];
    float *partialMaxLDSRowPtr = &partialMaxLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialMaxLDSRowPtr[hipThreadIdx_x] = srcRef;                             // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        imageMaxArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = srcRef;
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

    partialMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f8.f1[0], src_f8.f1[1]), src_f8.f1[2]), src_f8.f1[3]), src_f8.f1[4]), src_f8.f1[5]), src_f8.f1[6]), src_f8.f1[7]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMaxLDSRowPtr[hipThreadIdx_x] = fmaxf(partialMaxLDSRowPtr[hipThreadIdx_x], partialMaxLDSRowPtr[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialMaxLDSRowPtr[0] = fmaxf(partialMaxLDSRowPtr[0], partialMaxLDSRowPtr[increment]);
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            imageMaxArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialMaxLDSRowPtr[0];
    }
}


// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_image_max_tensor(T *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    U *imageMaxArr,
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
        Rpp32u imagePartialMaxArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *imagePartialMaxArr;
        imagePartialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialMaxArr, 0, imagePartialMaxArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(image_max_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMaxArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(image_max_grid_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMaxArr,
                           gridDim_x * gridDim_y,
                           imageMaxArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u imagePartialMaxArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *imagePartialMaxArr;
        imagePartialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialMaxArr, 0, imagePartialMaxArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(image_max_pln3_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           imagePartialMaxArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_max_grid_3channel_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMaxArr,
                           gridDim_x * gridDim_y,
                           imageMaxArr);
    }
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32u imagePartialMaxArrLength = gridDim_x * gridDim_y * gridDim_z * 3;
        float *imagePartialMaxArr;
        imagePartialMaxArr = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;
        hipMemset(imagePartialMaxArr, 0, imagePartialMaxArrLength * sizeof(float));
        hipDeviceSynchronize();

        hipLaunchKernelGGL(image_max_pkd3_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMaxArr,
                           roiTensorPtrSrc);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(image_max_grid_3channel_result_tensor,
                           dim3(1, 1, gridDim_z),
                           dim3(256, 1, 1),
                           0,
                           handle.GetStream(),
                           imagePartialMaxArr,
                           gridDim_x * gridDim_y,
                           imageMaxArr);
    }

    return RPP_SUCCESS;
}