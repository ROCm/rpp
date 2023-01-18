#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void non_linear_blend_gaussian_hip_compute(float &multiplier, int2 &halfDimsWH_i2, int2 &idXY_i2, d_float8 *gaussianValue_f8)
{
    float rowLocComponent;
    rowLocComponent = idXY_i2.y - halfDimsWH_i2.y;
    rowLocComponent *= (rowLocComponent * multiplier);

    float4 rowLocComponent_f4 = (float4)rowLocComponent;
    float4 multiplier_f4 = (float4)multiplier;

    d_float8 colLocComponent_f8;
    colLocComponent_f8.f4[0] = make_float4(idXY_i2.x, idXY_i2.x + 1, idXY_i2.x + 2, idXY_i2.x + 3);
    colLocComponent_f8.f4[1] = colLocComponent_f8.f4[0] + (float4)4;
    colLocComponent_f8.f4[0] -= (float4)halfDimsWH_i2.x;
    colLocComponent_f8.f4[1] -= (float4)halfDimsWH_i2.x;
    colLocComponent_f8.f4[0] = (colLocComponent_f8.f4[0] * colLocComponent_f8.f4[0] * multiplier_f4) + rowLocComponent_f4;
    colLocComponent_f8.f4[1] = (colLocComponent_f8.f4[1] * colLocComponent_f8.f4[1] * multiplier_f4) + rowLocComponent_f4;

    gaussianValue_f8->f4[0] = make_float4(expf(colLocComponent_f8.f4[0].x), expf(colLocComponent_f8.f4[0].y), expf(colLocComponent_f8.f4[0].z), expf(colLocComponent_f8.f4[0].w));
    gaussianValue_f8->f4[1] = make_float4(expf(colLocComponent_f8.f4[1].x), expf(colLocComponent_f8.f4[1].y), expf(colLocComponent_f8.f4[1].z), expf(colLocComponent_f8.f4[1].w));
}

__device__ void non_linear_blend_8_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, d_float8 *gaussianValue_f8)
{
    dst_f8->f4[0] = (src1_f8->f4[0] - src2_f8->f4[0]) * gaussianValue_f8->f4[0] + src2_f8->f4[0];
    dst_f8->f4[1] = (src1_f8->f4[1] - src2_f8->f4[1]) * gaussianValue_f8->f4[1] + src2_f8->f4[1];
}

__device__ void non_linear_blend_24_hip_compute(d_float24 *src1_f24, d_float24 *src2_f24, d_float24 *dst_f24, d_float8 *gaussianValue_f8)
{
    non_linear_blend_8_hip_compute(&(src1_f24->f8[0]), &(src2_f24->f8[0]), &(dst_f24->f8[0]), gaussianValue_f8);
    non_linear_blend_8_hip_compute(&(src1_f24->f8[1]), &(src2_f24->f8[1]), &(dst_f24->f8[1]), gaussianValue_f8);
    non_linear_blend_8_hip_compute(&(src1_f24->f8[2]), &(src2_f24->f8[2]), &(dst_f24->f8[2]), gaussianValue_f8);
}

template <typename T>
__global__ void non_linear_blend_pkd_tensor(T *srcPtr1,
                                            T *srcPtr2,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            float *stdDev,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float multiplier = -0.5f / (stdDev[id_z] * stdDev[id_z]);
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);

    d_float24 src1_f24, src2_f24, dst_f24;
    d_float8 gaussianValue_f8;
    non_linear_blend_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx, &src1_f24);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx, &src2_f24);
    non_linear_blend_24_hip_compute(&src1_f24, &src2_f24, &dst_f24, &gaussianValue_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void non_linear_blend_pln_tensor(T *srcPtr1,
                                            T *srcPtr2,
                                            uint3 srcStridesNCH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            int channelsDst,
                                            float *stdDev,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float multiplier = -0.5f / (stdDev[id_z] * stdDev[id_z]);
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);

    d_float8 src1_f8, src2_f8, dst_f8, gaussianValue_f8;
    non_linear_blend_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx, &src2_f8);
    non_linear_blend_8_hip_compute(&src1_f8, &src2_f8, &dst_f8, &gaussianValue_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx, &src2_f8);
        non_linear_blend_8_hip_compute(&src1_f8, &src2_f8, &dst_f8, &gaussianValue_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx, &src2_f8);
        non_linear_blend_8_hip_compute(&src1_f8, &src2_f8, &dst_f8, &gaussianValue_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void non_linear_blend_pkd3_pln3_tensor(T *srcPtr1,
                                                  T *srcPtr2,
                                                  uint2 srcStridesNH,
                                                  T *dstPtr,
                                                  uint3 dstStridesNCH,
                                                  float *stdDev,
                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float multiplier = -0.5f / (stdDev[id_z] * stdDev[id_z]);
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);

    d_float24 src1_f24, src2_f24, dst_f24;
    d_float8 gaussianValue_f8;
    non_linear_blend_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx, &src1_f24);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx, &src2_f24);
    non_linear_blend_24_hip_compute(&src1_f24, &src2_f24, &dst_f24, &gaussianValue_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void non_linear_blend_pln3_pkd3_tensor(T *srcPtr1,
                                                  T *srcPtr2,
                                                  uint3 srcStridesNCH,
                                                  T *dstPtr,
                                                  uint2 dstStridesNH,
                                                  float *stdDev,
                                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float multiplier = -0.5f / (stdDev[id_z] * stdDev[id_z]);
    int2 halfDimsWH_i2 = make_int2(roiTensorPtrSrc[id_z].xywhROI.roiWidth >> 1, roiTensorPtrSrc[id_z].xywhROI.roiHeight >> 1);
    int2 idXY_i2 = make_int2(id_x, id_y);

    d_float24 src1_f24, src2_f24, dst_f24;
    d_float8 gaussianValue_f8;
    non_linear_blend_gaussian_hip_compute(multiplier, halfDimsWH_i2, idXY_i2, &gaussianValue_f8);

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx, srcStridesNCH.y, &src1_f24);
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx, srcStridesNCH.y, &src2_f24);
    non_linear_blend_24_hip_compute(&src1_f24, &src2_f24, &dst_f24, &gaussianValue_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_non_linear_blend_tensor(T *srcPtr1,
                                           T *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;

        hipLaunchKernelGGL(non_linear_blend_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(non_linear_blend_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(non_linear_blend_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(non_linear_blend_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
