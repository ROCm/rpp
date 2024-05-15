#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__device__ __forceinline__ void rpp_hip_load1_glitch(T *srcPtr, uint2 srcStrideCH, float &locSrcX, float &locSrcY, float *dst, int channels)
{
    int srcIdx = locSrcY * srcStrideCH.y + locSrcX * srcStrideCH.x + channels;
    rpp_hip_interpolate1_nearest_neighbor_load_pln1(srcPtr + srcIdx, dst);
}

template <typename T>
__device__ __forceinline__ void rpp_hip_load8_glitch(T *srcPtr, uint2 srcStrideCH, d_float8 *srcX_f8, d_float8 *srcY_f8, d_float8 *dst_f8, int channels)
{
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[0], srcY_f8->f1[0], &(dst_f8->f1[0]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[1], srcY_f8->f1[1], &(dst_f8->f1[1]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[2], srcY_f8->f1[2], &(dst_f8->f1[2]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[3], srcY_f8->f1[3], &(dst_f8->f1[3]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[4], srcY_f8->f1[4], &(dst_f8->f1[4]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[5], srcY_f8->f1[5], &(dst_f8->f1[5]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[6], srcY_f8->f1[6], &(dst_f8->f1[6]), channels);
    rpp_hip_load1_glitch(srcPtr, srcStrideCH, srcX_f8->f1[7], srcY_f8->f1[7], &(dst_f8->f1[7]), channels);
}

__device__ void check_locs(d_float8 &xLocVals, d_float8 &yLocVals, RppiPoint offset, RpptROI roiTensorPtrSrc)
{
    for(int i = 0; i < 8; i++)
    {
        if (xLocVals.f1[i] >= roiTensorPtrSrc.ltrbROI.rb.x || xLocVals.f1[i] < roiTensorPtrSrc.ltrbROI.lt.x || yLocVals.f1[i] >= roiTensorPtrSrc.ltrbROI.rb.y || yLocVals.f1[i] < roiTensorPtrSrc.ltrbROI.lt.y)
        {
            xLocVals.f1[i] -= offset.x;
            yLocVals.f1[i] -= offset.y;
        }
    }
}

__device__ void compute_glitch_locs_hip(int id_x, int id_y, RpptChannelOffsets rgbOffsets, RpptROI roiTensorPtrSrc, d_float24 *srcLocsX_f24, d_float24 *srcLocsY_f24)
{
    float4 increment_f4;
    increment_f4 = make_float4(0.0f, 1.0f, 2.0f, 3.0f);

    srcLocsX_f24->f4[0] = static_cast<float4>(id_x + rgbOffsets.r.x) + increment_f4;
    srcLocsX_f24->f4[1] = srcLocsX_f24->f4[0] + (float4) 4;
    srcLocsY_f24->f4[0] = srcLocsY_f24->f4[1] = static_cast<float4>(id_y + rgbOffsets.r.y);
    check_locs(srcLocsX_f24->f8[0], srcLocsY_f24->f8[0], rgbOffsets.r, roiTensorPtrSrc);

    srcLocsX_f24->f4[2] = static_cast<float4>(id_x + rgbOffsets.g.x) + increment_f4;
    srcLocsX_f24->f4[3] = srcLocsX_f24->f4[2] +(float4) 4;
    srcLocsY_f24->f4[2] = srcLocsY_f24->f4[3]  = static_cast<float4>(id_y + rgbOffsets.g.y);
    check_locs(srcLocsX_f24->f8[1], srcLocsY_f24->f8[1], rgbOffsets.g, roiTensorPtrSrc);

    srcLocsX_f24->f4[4] = static_cast<float4>(id_x + rgbOffsets.b.x) + increment_f4;
    srcLocsX_f24->f4[5] = srcLocsX_f24->f4[4] + (float4) 4;
    srcLocsY_f24->f4[4] = srcLocsY_f24->f4[5] = static_cast<float4>(id_y + rgbOffsets.b.y);
    check_locs(srcLocsX_f24->f8[2], srcLocsY_f24->f8[2], rgbOffsets.b, roiTensorPtrSrc);
}

template <typename T>
__global__ void glitch_pkd_hip_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      RpptChannelOffsets *rgbOffsetsPtr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];
    uint2 srcStrideCH = make_uint2(3, srcStridesNH.y);
    d_float24 dst_f24, srcLocsX_f24, srcLocsY_f24;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &srcLocsX_f24, &srcLocsY_f24);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[0], &srcLocsY_f24.f8[0], &(dst_f24.f8[0]), 0);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[1], &srcLocsY_f24.f8[1], &(dst_f24.f8[1]), 1);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[2], &srcLocsY_f24.f8[2], &(dst_f24.f8[2]), 2);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void glitch_pln_hip_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      RpptChannelOffsets *rgbOffsetsPtr,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];
    uint2 srcStrideCH = make_uint2(1, srcStridesNCH.z);

    d_float24 srcLocsX_f24, srcLocsY_f24;
    d_float8 dst_f8;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &srcLocsX_f24, &srcLocsY_f24);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[0], &srcLocsY_f24.f8[0], &dst_f8, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    srcIdx += srcStridesNCH.y;
    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[1], &srcLocsY_f24.f8[1], &dst_f8, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    srcIdx += srcStridesNCH.y;
    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[2], &srcLocsY_f24.f8[2], &dst_f8, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
__global__ void glitch_pkd3_pln3_hip_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            RpptChannelOffsets *rgbOffsetsPtr,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];
    uint2 srcStrideCH = make_uint2(3, srcStridesNH.y);

    d_float24 srcLocsX_f24, srcLocsY_f24;
    d_float8 dst_f8;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &srcLocsX_f24, &srcLocsY_f24);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[0], &srcLocsY_f24.f8[0], &dst_f8, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[1], &srcLocsY_f24.f8[1], &dst_f8, 1);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[2], &srcLocsY_f24.f8[2], &dst_f8, 2);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
__global__ void glitch_pln3_pkd3_hip_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            RpptChannelOffsets *rgbOffsetsPtr,
                                            RpptROIPtr roiTensorPtrSrc)
{

    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];
    uint2 srcStrideCH = make_uint2(1, srcStridesNCH.z);

    d_float24 dst_f24, srcLocsX_f24, srcLocsY_f24;
    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &srcLocsX_f24, &srcLocsY_f24);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[0], &srcLocsY_f24.f8[0], &(dst_f24.f8[0]), 0);

    srcIdx += srcStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[1], &srcLocsY_f24.f8[1], &(dst_f24.f8[1]), 0);

    srcIdx += srcStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStrideCH, &srcLocsX_f24.f8[2], &srcLocsY_f24.f8[2], &(dst_f24.f8[2]), 0);

    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_glitch_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptChannelOffsets *rgbOffsets,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           rgbOffsets,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pln3_pkd3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           rgbOffsets,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pkd3_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           rgbOffsets,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           rgbOffsets,
                           roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}
