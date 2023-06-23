#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void image_min_grid_result_tensor(float *srcPtr,
                                             uint xBufferLength,
                                             float *dstPtr)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    if (id_x >= xBufferLength)
    {
        return;
    }

    uint srcIdx = (id_z * xBufferLength);
    float srcRef = srcPtr[srcIdx];
    srcIdx += id_x;

    __shared__ float partialMinLDS[256];                            // 1024 floats of src reduced to 256 in a 256 x 1 thread block
    partialMinLDS[hipThreadIdx_x] = srcRef;                         // initialization of LDS to 0 using all 256 x 1 threads

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;                                  // local memory reset of invalid values (from the vectorized global load) to 0.0f

    partialMinLDS[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f8.f1[0], src_f8.f1[1]), src_f8.f1[2]), src_f8.f1[3]), src_f8.f1[4]), src_f8.f1[5]), src_f8.f1[6]), src_f8.f1[7]);
    __syncthreads();                                                // syncthreads after LDS load

    // Reduction of 256 floats on 256 threads per block in x dimension
    for (int threadMax = 128; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMinLDS[hipThreadIdx_x] = fmaxf(partialMinLDS[hipThreadIdx_x], partialMinLDS[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
        dstPtr[hipBlockIdx_z] = partialMinLDS[0];
}

template <typename T, typename U>
__global__ void image_min_pln1_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      U *imageMinArr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx = (id_z * srcStridesNH.x);
    float srcRef = (float)srcPtr[srcIdx];

    __shared__ float partialMinLDS[16][16];                                 // 16 rows of src, 128 reduced cols of src in a 16 x 16 thread block
    float *partialMinLDSRowPtr = &partialMinLDS[hipThreadIdx_y][0];         // float pointer to beginning of each row in LDS
    partialMinLDSRowPtr[hipThreadIdx_x] = srcRef;                             // initialization of LDS to 0 using all 16 x 16 threads

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    srcIdx += ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int xAlignedLength = roiTensorPtrSrc[id_z].xywhROI.roiWidth & ~7;       // alignedLength for vectorized global loads
    int xDiff = roiTensorPtrSrc[id_z].xywhROI.roiWidth - xAlignedLength;    // difference between roiWidth and alignedLength

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)                  // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = srcRef;
    }

    partialMinLDSRowPtr[hipThreadIdx_x] = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(src_f8.f1[0], src_f8.f1[1]), src_f8.f1[2]), src_f8.f1[3]), src_f8.f1[4]), src_f8.f1[5]), src_f8.f1[6]), src_f8.f1[7]);
    __syncthreads();                                                        // syncthreads after LDS load

    // Reduction of 16 floats on 16 threads per block in x dimension (for every y dimension)
    for (int threadMax = 8; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialMinLDSRowPtr[hipThreadIdx_x] = fmaxf(partialMinLDSRowPtr[hipThreadIdx_x], partialMinLDSRowPtr[hipThreadIdx_x + threadMax]);
        __syncthreads();
    }

    if (hipThreadIdx_x == 0)
    {
        // Reduction of 16 floats on 16 threads per block in y dimension
        for (int threadMax = 8, increment = 128; threadMax >= 1; threadMax /= 2, increment /= 2)
        {
            if (hipThreadIdx_y < threadMax)
                partialMinLDSRowPtr[0] = fmaxf(partialMinLDSRowPtr[0], partialMinLDSRowPtr[increment]);
            __syncthreads();
        }

        // Final store to dst
        if (hipThreadIdx_y == 0)
            imageMinArr[(hipBlockIdx_z * hipGridDim_y + hipBlockIdx_y) * hipGridDim_x + hipBlockIdx_x] = partialMinLDSRowPtr[0];
    }
}

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

        hipLaunchKernelGGL(image_min_pln1_tensor,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           imagePartialMinArr,
                           roiTensorPtrSrc);
        hipLaunchKernelGGL(image_min_grid_result_tensor,
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