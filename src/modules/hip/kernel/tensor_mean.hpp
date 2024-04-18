#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduction.hpp"

// -------------------- Set 0 - Reduction Stage 2 --------------------
__global__ void tensor_mean_grid_result_hip(Rpp32u *srcPtr,
                                            uint xBufferLength,
                                            float *dstPtr,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ uint partialSum_smem[1024];                          // 8192 uints of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0;                            // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_uint8 src_ui8;
    *reinterpret_cast<d_uint8_s *>(&src_ui8) = *reinterpret_cast<d_uint8_s *>(srcPtr + srcIdx);

    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_ui8.ui1[i] = 0;                                     // local memory reset of invalid values (from the vectorized global load) to 0

    src_ui8.ui4[0] += src_ui8.ui4[1];                               // perform small work of vectorized uint4 addition
    partialSum_smem[hipThreadIdx_x] += (src_ui8.ui1[0] +
                                        src_ui8.ui1[1] +
                                        src_ui8.ui1[2] +
                                        src_ui8.ui1[3]);            // perform small work of reducing uint32s to uint using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 uints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        dstPtr[hipBlockIdx_z] = static_cast<float>(partialSum_smem[0]) / totalElements;
    }
}

__global__ void tensor_mean_grid_result_hip(Rpp32s *srcPtr,
                                            uint xBufferLength,
                                            float *dstPtr,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ int partialSum_smem[1024];                           // 8192 ints of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0;                            // initialization of Shared to 0 using all 1024 x 1 threads

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                        // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                     // difference between bufferLength and alignedLength
    uint srcIdx = (id_z * xBufferLength) + id_x;

    d_int8 src_i8;
    *reinterpret_cast<d_int8_s *>(&src_i8) = *reinterpret_cast<d_int8_s *>(srcPtr + srcIdx);

    if (id_x + 8 > xBufferLength)
        for(int i = xDiff; i < 8; i++)
            src_i8.i1[i] = 0;                                       // local memory reset of invalid values (from the vectorized global load) to 0.0f

    src_i8.i4[0] += src_i8.i4[1];
    partialSum_smem[hipThreadIdx_x] += (src_i8.i1[0] +
                                        src_i8.i1[1] +
                                        src_i8.i1[2] +
                                        src_i8.i1[3]);              // perform small work of reducing uint32s to uint using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 ints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        dstPtr[hipBlockIdx_z] = static_cast<float>(partialSum_smem[0]) / totalElements;
    }
}

__global__ void tensor_mean_grid_result_hip(float *srcPtr,
                                            uint xBufferLength,
                                            float *dstPtr,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialSum_smem[1024];                         // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    partialSum_smem[hipThreadIdx_x] = 0.0f;                         // initialization of Shared to 0 using all 1024 x 1 threads

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
    partialSum_smem[hipThreadIdx_x] += (src_f8.f1[0] +
                                        src_f8.f1[1] +
                                        src_f8.f1[2] +
                                        src_f8.f1[3]);              // perform small work of reducing float4s to float using 1024 x 1 threads and store in Shared
    __syncthreads();                                                // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
            partialSum_smem[hipThreadIdx_x] += partialSum_smem[hipThreadIdx_x + threadMax];
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        dstPtr[hipBlockIdx_z] = partialSum_smem[0] / totalElements;
    }
}

__global__ void tensor_mean_grid_3channel_result_hip(Rpp32u *srcPtr,
                                                     uint xBufferLength,
                                                     float *dstPtr,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ uint partialRSum_smem[1024];                                      // 8192 uints of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ uint partialGSum_smem[1024];
    __shared__ uint partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0;                                        // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0;
    partialBSum_smem[hipThreadIdx_x] = 0;

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                                     // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                                  // difference between bufferLength and alignedLength
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_uint24 src_ui24;
    *reinterpret_cast<d_uint24_s *>(&src_ui24) = *reinterpret_cast<d_uint24_s *>(srcPtr + srcIdx);
    rpp_hip_layouttoggle24_pkd3_to_pln3(reinterpret_cast<d_uint24_s *>(&src_ui24));

    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_ui24.ui8[0].ui1[i] = 0;
            src_ui24.ui8[1].ui1[i] = 0;
            src_ui24.ui8[2].ui1[i] = 0;
        }
    }
    src_ui24.ui8[0].ui4[0] += src_ui24.ui8[0].ui4[1];
    src_ui24.ui8[1].ui4[0] += src_ui24.ui8[1].ui4[1];
    src_ui24.ui8[2].ui4[0] += src_ui24.ui8[2].ui4[1];
    partialRSum_smem[hipThreadIdx_x] = (src_ui24.ui8[0].ui1[0] +
                                        src_ui24.ui8[0].ui1[1] +
                                        src_ui24.ui8[0].ui1[2] +
                                        src_ui24.ui8[0].ui1[3]);                 // perform small work of reducing R uint32s to uint using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_ui24.ui8[1].ui1[0] +
                                        src_ui24.ui8[1].ui1[1] +
                                        src_ui24.ui8[1].ui1[2] +
                                        src_ui24.ui8[1].ui1[3]);                 // perform small work of reducing G uint32s to uint using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_ui24.ui8[2].ui1[0] +
                                        src_ui24.ui8[2].ui1[1] +
                                        src_ui24.ui8[2].ui1[2] +
                                        src_ui24.ui8[2].ui1[3]);                 // perform small work of reducing B uint32s to uint using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 uints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        float sum = static_cast<float>(partialRSum_smem[0] + partialGSum_smem[0] + partialBSum_smem[0]);
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = static_cast<float>(partialRSum_smem[0]) / totalElements;
        dstPtr[idx + 1] = static_cast<float>(partialGSum_smem[0]) / totalElements;
        dstPtr[idx + 2] = static_cast<float>(partialBSum_smem[0]) / totalElements;
        dstPtr[idx + 3] = sum  / (totalElements * 3);
    }
}

__global__ void tensor_mean_grid_3channel_result_hip(Rpp32s *srcPtr,
                                                     uint xBufferLength,
                                                     float *dstPtr,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ int partialRSum_smem[1024];                                       // 8192 uints of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ int partialGSum_smem[1024];
    __shared__ int partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0;                                        // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0;
    partialBSum_smem[hipThreadIdx_x] = 0;

    if (id_x >= xBufferLength)
        return;

    int xAlignedLength = xBufferLength & ~7;                                     // alignedLength for vectorized global loads
    int xDiff = xBufferLength - xAlignedLength;                                  // difference between bufferLength and alignedLength
    uint srcIdx = ((id_z * xBufferLength) + id_x) * 3;

    d_int24 src_i24;
    *reinterpret_cast<d_int24_s *>(&src_i24) = *reinterpret_cast<d_int24_s *>(srcPtr + srcIdx);
    rpp_hip_layouttoggle24_pkd3_to_pln3(reinterpret_cast<d_int24_s *>(&src_i24));

    if (id_x + 8 > xBufferLength)                                                // local memory reset of invalid values (from the vectorized global load) to 0.0f
    {
        for(int i = xDiff; i < 8; i++)
        {
            src_i24.i8[0].i1[i] = 0;
            src_i24.i8[1].i1[i] = 0;
            src_i24.i8[2].i1[i] = 0;
        }
    }
    src_i24.i8[0].i4[0] += src_i24.i8[0].i4[1];
    src_i24.i8[1].i4[0] += src_i24.i8[1].i4[1];
    src_i24.i8[2].i4[0] += src_i24.i8[2].i4[1];
    partialRSum_smem[hipThreadIdx_x] = (src_i24.i8[0].i1[0] +
                                        src_i24.i8[0].i1[1] +
                                        src_i24.i8[0].i1[2] +
                                        src_i24.i8[0].i1[3]);                    // perform small work of reducing R int32s to int using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_i24.i8[1].i1[0] +
                                        src_i24.i8[1].i1[1] +
                                        src_i24.i8[1].i1[2] +
                                        src_i24.i8[1].i1[3]);                    // perform small work of reducing G int32s to int using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_i24.i8[2].i1[0] +
                                        src_i24.i8[2].i1[1] +
                                        src_i24.i8[2].i1[2] +
                                        src_i24.i8[2].i1[3]);                    // perform small work of reducing B int32s to int using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 uints on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        float sum = static_cast<float>(partialRSum_smem[0] + partialGSum_smem[0] + partialBSum_smem[0]);
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = static_cast<float>(partialRSum_smem[0]) / totalElements;
        dstPtr[idx + 1] = static_cast<float>(partialGSum_smem[0]) / totalElements;
        dstPtr[idx + 2] = static_cast<float>(partialBSum_smem[0]) / totalElements;
        dstPtr[idx + 3] = sum  / (totalElements * 3);
    }
}

__global__ void tensor_mean_grid_3channel_result_hip(float *srcPtr,
                                                     uint xBufferLength,
                                                     float *dstPtr,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipThreadIdx_x * 8;
    int id_z = hipBlockIdx_z;

    __shared__ float partialRSum_smem[1024];                                     // 8192 floats of src reduced to 1024 in a 1024 x 1 thread block
    __shared__ float partialGSum_smem[1024];
    __shared__ float partialBSum_smem[1024];
    partialRSum_smem[hipThreadIdx_x] = 0.0f;                                     // initialization of Shared to 0 using all 1024 x 1 threads
    partialGSum_smem[hipThreadIdx_x] = 0.0f;
    partialBSum_smem[hipThreadIdx_x] = 0.0f;

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
    partialRSum_smem[hipThreadIdx_x] = (src_f24.f8[0].f1[0] +
                                        src_f24.f8[0].f1[1] +
                                        src_f24.f8[0].f1[2] +
                                        src_f24.f8[0].f1[3]);                    // perform small work of reducing R float4s to float using 1024 threads and store in Shared
    partialGSum_smem[hipThreadIdx_x] = (src_f24.f8[1].f1[0] +
                                        src_f24.f8[1].f1[1] +
                                        src_f24.f8[1].f1[2] +
                                        src_f24.f8[1].f1[3]);                    // perform small work of reducing G float4s to float using 1024 threads and store in Shared
    partialBSum_smem[hipThreadIdx_x] = (src_f24.f8[2].f1[0] +
                                        src_f24.f8[2].f1[1] +
                                        src_f24.f8[2].f1[2] +
                                        src_f24.f8[2].f1[3]);                    // perform small work of reducing B float4s to float using 1024 threads and store in Shared

    __syncthreads();                                                             // syncthreads after Shared load

    // Reduction of 1024 floats on 1024 threads per block in x dimension
    for (int threadMax = 512; threadMax >= 1; threadMax /= 2)
    {
        if (hipThreadIdx_x < threadMax)
        {
            partialRSum_smem[hipThreadIdx_x] += partialRSum_smem[hipThreadIdx_x + threadMax];
            partialGSum_smem[hipThreadIdx_x] += partialGSum_smem[hipThreadIdx_x + threadMax];
            partialBSum_smem[hipThreadIdx_x] += partialBSum_smem[hipThreadIdx_x + threadMax];
        }
        __syncthreads();
    }

    // Final store to dst
    if (hipThreadIdx_x == 0)
    {
        int totalElements = roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiHeight * roiTensorPtrSrc[hipBlockIdx_z].xywhROI.roiWidth;
        float sum = partialRSum_smem[0] + partialGSum_smem[0] + partialBSum_smem[0];
        int idx = hipBlockIdx_z * 4;
        dstPtr[idx] = partialRSum_smem[0] / totalElements;
        dstPtr[idx + 1] = partialGSum_smem[0] / totalElements;
        dstPtr[idx + 2] = partialBSum_smem[0] / totalElements;
        dstPtr[idx + 3] = sum  / (totalElements * 3);
    }
}

// -------------------- Set 2 - Kernel Executors --------------------
// Handle U8 datatype
RppStatus hip_exec_tensor_mean(Rpp8u *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *tensorMeanArr,
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
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        Rpp32u *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(uint), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_result_hip,
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
        Rpp32u *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(Rpp32u), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
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
        Rpp32u *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<Rpp32u*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(Rpp32u), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
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

// Handle I8 datatype
RppStatus hip_exec_tensor_mean(Rpp8s *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *tensorMeanArr,
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
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        Rpp32s *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(Rpp32s), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_result_hip,
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
        Rpp32s *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(Rpp32s), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
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
        Rpp32s *tensorPartialSumArr;
        tensorPartialSumArr = reinterpret_cast<Rpp32s*>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(Rpp32s), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
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

// Handle f16/32 datatype
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

    int globalThreads_x = (srcDescPtr->w + 7) >> 3;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    int gridDim_x = (int) ceil((float)globalThreads_x/LOCAL_THREADS_X);
    int gridDim_y = (int) ceil((float)globalThreads_y/LOCAL_THREADS_Y);
    int gridDim_z = (int) ceil((float)globalThreads_z/LOCAL_THREADS_Z);

    if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
    {
        Rpp32u tensorPartialSumArrLength = gridDim_x * gridDim_y * gridDim_z;
        float *tensorPartialSumArr;
        tensorPartialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(float), handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln1_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_result_hip,
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
        tensorPartialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pln3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
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
        tensorPartialSumArr = handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem;
        hipMemsetAsync(tensorPartialSumArr, 0, tensorPartialSumArrLength * sizeof(float), handle.GetStream());
        hipLaunchKernelGGL(tensor_sum_pkd3_hip,
                           dim3(gridDim_x, gridDim_y, gridDim_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           tensorPartialSumArr,
                           roiTensorPtrSrc);
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(tensor_mean_grid_3channel_result_hip,
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
