#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void check_locs(float4 &xLocVals, float4 &yLocVals, RppiPoint offset, RpptROI roiTensorPtrSrc)
{
    if (xLocVals.x >= roiTensorPtrSrc.ltrbROI.rb.x || xLocVals.x < roiTensorPtrSrc.ltrbROI.lt.x || yLocVals.x >= roiTensorPtrSrc.ltrbROI.rb.y || yLocVals.x < roiTensorPtrSrc.ltrbROI.lt.y)
    {
        xLocVals.x -= offset.x;
        yLocVals.x -= offset.y;
    }
    if (xLocVals.y >= roiTensorPtrSrc.ltrbROI.rb.x || xLocVals.y < roiTensorPtrSrc.ltrbROI.lt.x || yLocVals.y >= roiTensorPtrSrc.ltrbROI.rb.y || yLocVals.y < roiTensorPtrSrc.ltrbROI.lt.y)
    {
        xLocVals.y -= offset.x;
        yLocVals.y -= offset.y;
    }
    if (xLocVals.z >= roiTensorPtrSrc.ltrbROI.rb.x || xLocVals.z < roiTensorPtrSrc.ltrbROI.lt.x || yLocVals.z >= roiTensorPtrSrc.ltrbROI.rb.y || yLocVals.z < roiTensorPtrSrc.ltrbROI.lt.y)
    {
        xLocVals.z -= offset.x;
        yLocVals.z -= offset.y;
    }
    if (xLocVals.w >= roiTensorPtrSrc.ltrbROI.rb.x || xLocVals.w < roiTensorPtrSrc.ltrbROI.lt.x || yLocVals.w >= roiTensorPtrSrc.ltrbROI.rb.y || yLocVals.w < roiTensorPtrSrc.ltrbROI.lt.y)
    {
        xLocVals.w -= offset.x;
        yLocVals.w -= offset.y;
    }
}

__device__ void compute_glitch_locs_hip(int id_x, int id_y, RpptChannelOffsets rgbOffsets, RpptROI roiTensorPtrSrc, d_float16 *rlocSrc_f16, d_float16 *glocSrc_f16, d_float16 *blocSrc_f16)
{
    d_float8 increment_f8, rLocsX_f8, rLocsY_f8, gLocsX_f8, gLocsY_f8, bLocsX_f8, bLocsY_f8;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);

    rLocsX_f8.f4[0] = static_cast<float4>(id_x + rgbOffsets.r.x) + increment_f8.f4[0];
    rLocsX_f8.f4[1] = rLocsX_f8.f4[0] + (float4) 4;
    rLocsY_f8.f4[0] = rLocsY_f8.f4[1] = static_cast<float4>(id_y + rgbOffsets.r.y);
    check_locs(rLocsX_f8.f4[0], rLocsY_f8.f4[0], rgbOffsets.r, roiTensorPtrSrc);
    check_locs(rLocsX_f8.f4[1], rLocsY_f8.f4[1], rgbOffsets.r, roiTensorPtrSrc);

    gLocsX_f8.f4[0] = static_cast<float4>(id_x + rgbOffsets.g.x) + increment_f8.f4[0];
    gLocsX_f8.f4[1] = gLocsX_f8.f4[0] +(float4) 4;
    gLocsY_f8.f4[0] = gLocsY_f8.f4[1]  = static_cast<float4>(id_y + rgbOffsets.g.y);
    // gLocsY_f8.f4[1] = static_cast<float4>(id_y + rgbOffsets.g.y);
    check_locs(gLocsX_f8.f4[0], gLocsY_f8.f4[0], rgbOffsets.g, roiTensorPtrSrc);
    check_locs(gLocsX_f8.f4[1], gLocsY_f8.f4[1], rgbOffsets.g, roiTensorPtrSrc);

    bLocsX_f8.f4[0] = static_cast<float4>(id_x + rgbOffsets.b.x) + increment_f8.f4[0];
    bLocsX_f8.f4[1] = bLocsX_f8.f4[0] + (float4) 4;
    bLocsY_f8.f4[0] = bLocsY_f8.f4[1] = static_cast<float4>(id_y + rgbOffsets.b.y);
    //  = static_cast<float4>(id_y + rgbOffsets.b.y);
    check_locs(bLocsX_f8.f4[0], bLocsY_f8.f4[0], rgbOffsets.b, roiTensorPtrSrc);
    check_locs(bLocsX_f8.f4[1], bLocsY_f8.f4[1], rgbOffsets.b, roiTensorPtrSrc);

    rlocSrc_f16->f4[0] = rLocsX_f8.f4[0];
    rlocSrc_f16->f4[1] = rLocsX_f8.f4[1];
    rlocSrc_f16->f4[2] = rLocsY_f8.f4[0];
    rlocSrc_f16->f4[3] = rLocsY_f8.f4[1];

    glocSrc_f16->f4[0] = gLocsX_f8.f4[0];
    glocSrc_f16->f4[1] = gLocsX_f8.f4[1];
    glocSrc_f16->f4[2] = gLocsY_f8.f4[0];
    glocSrc_f16->f4[3] = gLocsY_f8.f4[1];

    blocSrc_f16->f4[0] = bLocsX_f8.f4[0];
    blocSrc_f16->f4[1] = bLocsX_f8.f4[1];
    blocSrc_f16->f4[2] = bLocsY_f8.f4[0];
    blocSrc_f16->f4[3] = bLocsY_f8.f4[1];

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

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];
    d_float24 dst_f24;
    d_float16 rlocSrc_f16, glocSrc_f16, blocSrc_f16;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &rlocSrc_f16, &glocSrc_f16, &blocSrc_f16);

    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNH.y, &rlocSrc_f16, &srcRoi_i4, &(dst_f24.f8[0]), 3, 0);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNH.y, &glocSrc_f16, &srcRoi_i4, &(dst_f24.f8[1]), 3, 1);
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNH.y, &blocSrc_f16, &srcRoi_i4, &(dst_f24.f8[2]), 3, 2);

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

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];

    d_float8 dst_f8;
    d_float16 rlocSrc_f16, glocSrc_f16, blocSrc_f16;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &rlocSrc_f16, &glocSrc_f16, &blocSrc_f16);

    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNCH.z, &rlocSrc_f16, &srcRoi_i4, &dst_f8, 1, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    srcIdx += srcStridesNCH.y;
    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNCH.z, &glocSrc_f16, &srcRoi_i4, &dst_f8, 1, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    srcIdx += srcStridesNCH.y;
    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNCH.z, &blocSrc_f16, &srcRoi_i4, &dst_f8, 1, 0);
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

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];

    d_float8 dst_f8;
    d_float16 rlocSrc_f16, glocSrc_f16, blocSrc_f16;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &rlocSrc_f16, &glocSrc_f16, &blocSrc_f16);

    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNH.y, &rlocSrc_f16, &srcRoi_i4, &dst_f8, 3, 0);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNH.y, &glocSrc_f16, &srcRoi_i4, &dst_f8, 3, 1);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    dstIdx += dstStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNH.y, &blocSrc_f16, &srcRoi_i4, &dst_f8, 3, 2);
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

    int4 srcRoi_i4 = *(reinterpret_cast<int4 *>(&roiTensorPtrSrc[id_z]));
    RpptChannelOffsets rgbOffsets = rgbOffsetsPtr[id_z];

    d_float24 dst_f24;
    d_float16 rlocSrc_f16, glocSrc_f16, blocSrc_f16;

    compute_glitch_locs_hip(id_x, id_y, rgbOffsets, roiTensorPtrSrc[id_z], &rlocSrc_f16, &glocSrc_f16, &blocSrc_f16);

    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNCH.z, &rlocSrc_f16, &srcRoi_i4, &(dst_f24.f8[0]), 1, 0);

    srcIdx += srcStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNCH.z, &glocSrc_f16, &srcRoi_i4, &(dst_f24.f8[1]), 1, 0);

    srcIdx += srcStridesNCH.y;
    rpp_hip_load8_glitch(srcPtr + srcIdx, srcStridesNCH.z, &blocSrc_f16, &srcRoi_i4, &(dst_f24.f8[2]), 1, 0);

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
    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

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