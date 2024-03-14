#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__device__ void log_hip_compute(T *srcPtr, d_float8 *src_f8, d_float8 *dst_f8)
{
    if constexpr (std::is_same<T, schar>::value)
        rpp_hip_math_add8_const(src_f8, src_f8, (float4)128);

    dst_f8->f1[0] = __logf(fabsf(src_f8->f1[0]));
    dst_f8->f1[1] = __logf(fabsf(src_f8->f1[1]));
    dst_f8->f1[2] = __logf(fabsf(src_f8->f1[2]));
    dst_f8->f1[3] = __logf(fabsf(src_f8->f1[3]));
    dst_f8->f1[4] = __logf(fabsf(src_f8->f1[4]));
    dst_f8->f1[5] = __logf(fabsf(src_f8->f1[5]));
    dst_f8->f1[6] = __logf(fabsf(src_f8->f1[6]));
    dst_f8->f1[7] = __logf(fabsf(src_f8->f1[7]));

    if constexpr (std::is_same<T, schar>::value)
    {
        dst_f8->f4[0] = rpp_hip_pixel_check_0to255(dst_f8->f4[0]) - (float4)128;
        dst_f8->f4[1] = rpp_hip_pixel_check_0to255(dst_f8->f4[1]) - (float4)128;
    }
}

// F16 stores without layout toggle (8 F16 pixels)

__device__ __forceinline__ void rpp_hip_pack_float8_and_store(half *dstPtr, d_float8 *dst_f8)
{
    d_half8 dst_h8;

    dst_h8.h2[0] = __float22half2_rn(make_float2(dst_f8->f1[0], dst_f8->f1[1]));
    dst_h8.h2[1] = __float22half2_rn(make_float2(dst_f8->f1[2], dst_f8->f1[3]));
    dst_h8.h2[2] = __float22half2_rn(make_float2(dst_f8->f1[4], dst_f8->f1[5]));
    dst_h8.h2[3] = __float22half2_rn(make_float2(dst_f8->f1[6], dst_f8->f1[7]));

    *(d_half8 *)dstPtr = dst_h8;
}

// F32 stores without layout toggle (8 F32 pixels)

template <typename T>
__device__ __forceinline__ void rpp_hip_pack_float8_and_store(T *dstPtr, d_float8 *dst_f8)
{
    *(d_float8_s *)dstPtr = *(d_float8_s *)dst_f8;
}

template <typename T1, typename T2>
__global__ void log_generic_hip_tensor(T1 *srcPtr,
                                       uint *srcStrides,
                                       uint *srcDims,
                                       uint srcNumDims,
                                       T2 *dstPtr,
                                       uint *dstStrides,
                                       Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(id_x >= srcStrides[0])
        return;

    uint *roi = &roiTensor[id_y * srcNumDims * 2 + srcNumDims];
    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);

    uint coords[RPPT_MAX_DIMS];
    uint maxInnerDim = roi[srcNumDims - 1];

    if((id_x + 8) > maxInnerDim)
        id_x -= (8 - maxInnerDim);

    for (int i = 0; i < srcNumDims; i++)
        coords[i] = id_x / srcStrides[i] % srcDims[i];

    for (int i = 0; i < srcNumDims; i++)
    {
        dstIdx += (coords[i] * dstStrides[i]);
        srcIdx += (coords[i] * srcStrides[i]);
    }

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    log_hip_compute(srcPtr, &src_f8, &dst_f8);
    rpp_hip_pack_float8_and_store(dstPtr + dstIdx, &dst_f8);
}

template <typename T1, typename T2>
RppStatus hip_exec_log_generic_tensor(T1 *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      T2 *dstPtr,
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