#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__device__ void log_hip_compute(T *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    if constexpr (std::is_same<T, schar>::value)
        rpp_hip_math_add8_const(src_f8, src_f8, (float4)128);

    rpp_hip_log(src_f8, dst_f8);
}

template <typename T, typename U>
__global__ void log_generic_hip_tensor(T *srcPtr,
                                       uint *srcStrides,
                                       uint *srcDims,
                                       uint numDims,
                                       U *dstPtr,
                                       uint *dstStrides,
                                       Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // batchsize

    if(id_x >= srcStrides[0])
        return;

    uint *roi = roiTensor + id_y * numDims * 2;
    uint *begin = roi;
    uint *length = &roi[numDims];
    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);
    uint coords[RPPT_MAX_DIMS];

    for (int i = 0; i < numDims; i++)
    {
        coords[i] = (id_x / srcStrides[i]) % srcDims[i];
        if(coords[i] > length[i])
            return;
    }

    for (int i = 0; i < numDims; i++)
    {
        dstIdx += (coords[i] * dstStrides[i]);
        srcIdx += (begin[i] + (coords[i] * srcStrides[i]));
    }

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    log_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T, typename U>
RppStatus hip_exec_log_generic_tensor(T *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      U *dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      uint *roiTensor,
                                      rpp::Handle& handle)
{

    int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
    int globalThreads_y = dstGenericDescPtr->dims[0];
    int globalThreads_z = 1;

    hipLaunchKernelGGL(log_generic_hip_tensor,
                    dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                    dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                    0,
                    handle.GetStream(),
                    srcPtr,
                    srcGenericDescPtr->strides,
                    srcGenericDescPtr->dims + 1,
                    srcGenericDescPtr->numDims - 1,
                    dstPtr,
                    dstGenericDescPtr->strides,
                    roiTensor);

    return RPP_SUCCESS;
}