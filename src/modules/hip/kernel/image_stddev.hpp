#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------

__global__ void image_stddev_grid_result_tensor(float *srcPtr,
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
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);   // load 8 pixels to local mmemory
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
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        dstPtr[hipBlockIdx_z] = sqrt(partialVarLDS[0] / totalElements);
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
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);           // load 8 pixels to local mmemory
    if (id_x + 8 > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        for(int i = xDiff; i < 8; i++)
            src_f8.f1[i] = 0.0f;                                            // local memory reset of invalid values (from the vectorized global load) to 0.0f

    rpp_hip_math_subtract8_const(&src_f8, &temp_f8, mean_f4);               //subtract mean from each pixel
    rpp_hip_math_multiply8(&temp_f8, &temp_f8, &tempSq_f8);                 //square the temporary value
    tempSq_f8.f4[0] += tempSq_f8.f4[1];                                     // perform small work of vectorized float4 addition
    partialVarLDSRowPtr[hipThreadIdx_x] += (tempSq_f8.f1[0] +
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

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T, typename U>
RppStatus hip_exec_image_stddev_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       U *imageStddevArr,
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
                           imagePartialVarArr,
                           gridDim_x * gridDim_y,
                           imageStddevArr,
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}
