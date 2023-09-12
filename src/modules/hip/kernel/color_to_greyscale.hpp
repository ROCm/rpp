#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void color_to_greyscale_hip_compute(uchar *srcPtr, d_float24 *src_f24, d_float8 *dst_f8, d_float12 *channelWeights_f12)
{
    dst_f8->f4[0] = src_f24->f4[0] * channelWeights_f12->f4[0] + src_f24->f4[2] * channelWeights_f12->f4[1] + src_f24->f4[4] * channelWeights_f12->f4[2];
    dst_f8->f4[1] = src_f24->f4[1] * channelWeights_f12->f4[0] + src_f24->f4[3] * channelWeights_f12->f4[1] + src_f24->f4[5] * channelWeights_f12->f4[2];
}

__device__ void color_to_greyscale_hip_compute(float *srcPtr, d_float24 *src_f24, d_float8 *dst_f8, d_float12 *channelWeights_f12)
{
    dst_f8->f4[0] = src_f24->f4[0] * channelWeights_f12->f4[0] + src_f24->f4[2] * channelWeights_f12->f4[1] + src_f24->f4[4] * channelWeights_f12->f4[2];
    dst_f8->f4[1] = src_f24->f4[1] * channelWeights_f12->f4[0] + src_f24->f4[3] * channelWeights_f12->f4[1] + src_f24->f4[5] * channelWeights_f12->f4[2];
}

__device__ void color_to_greyscale_hip_compute(signed char *srcPtr, d_float24 *src_f24, d_float8 *dst_f8, d_float12 *channelWeights_f12)
{
    dst_f8->f4[0] = src_f24->f4[0] * channelWeights_f12->f4[0] + src_f24->f4[2] * channelWeights_f12->f4[1] + src_f24->f4[4] * channelWeights_f12->f4[2];
    dst_f8->f4[1] = src_f24->f4[1] * channelWeights_f12->f4[0] + src_f24->f4[3] * channelWeights_f12->f4[1] + src_f24->f4[5] * channelWeights_f12->f4[2];
}

__device__ void color_to_greyscale_hip_compute(half *srcPtr, d_float24 *src_f24, d_float8 *dst_f8, d_float12 *channelWeights_f12)
{
    dst_f8->f4[0] = src_f24->f4[0] * channelWeights_f12->f4[0] + src_f24->f4[2] * channelWeights_f12->f4[1] + src_f24->f4[4] * channelWeights_f12->f4[2];
    dst_f8->f4[1] = src_f24->f4[1] * channelWeights_f12->f4[0] + src_f24->f4[3] * channelWeights_f12->f4[1] + src_f24->f4[5] * channelWeights_f12->f4[2];
}

template <typename T>
__global__ void color_to_greyscale_pkd3_pln1_tensor(T *srcPtr,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    float3 channelWeights_f3,
                                                    uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_float12 channelWeights_f12;
    channelWeights_f12.f4[0] = (float4) channelWeights_f3.x;
    channelWeights_f12.f4[1] = (float4) channelWeights_f3.y;
    channelWeights_f12.f4[2] = (float4) channelWeights_f3.z;

    d_float24 src_f24;
    d_float8 dst_f8;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    color_to_greyscale_hip_compute(srcPtr, &src_f24, &dst_f8, &channelWeights_f12);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
__global__ void color_to_greyscale_pln3_pln1_tensor(T *srcPtr,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    float3 channelWeights_f3,
                                                    uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_float12 channelWeights_f12;
    channelWeights_f12.f4[0] = (float4) channelWeights_f3.x;
    channelWeights_f12.f4[1] = (float4) channelWeights_f3.y;
    channelWeights_f12.f4[2] = (float4) channelWeights_f3.z;

    d_float24 src_f24;
    d_float8 dst_f8;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    color_to_greyscale_hip_compute(srcPtr, &src_f24, &dst_f8, &channelWeights_f12);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
RppStatus hip_exec_color_to_greyscale_tensor(T *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             T *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *channelWeights,
                                             rpp::Handle& handle)
{
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (srcDescPtr->layout == RpptLayout::NHWC)
    {
        globalThreads_x = (srcDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(color_to_greyscale_pkd3_pln1_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           make_float3(channelWeights[0], channelWeights[1], channelWeights[2]),
                           make_uint2(srcDescPtr->w, srcDescPtr->h));
    }
    else if (srcDescPtr->layout == RpptLayout::NCHW)
    {
        hipLaunchKernelGGL(color_to_greyscale_pln3_pln1_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           make_float3(channelWeights[0], channelWeights[1], channelWeights[2]),
                           make_uint2(srcDescPtr->w, srcDescPtr->h));
    }

    return RPP_SUCCESS;
}
