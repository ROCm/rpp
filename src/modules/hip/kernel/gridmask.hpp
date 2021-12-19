#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// Gridmask helper - Computing row and column ratios

__device__ void gridmask_ratio_hip_compute(int id_x, int id_y, float2 *rotateRatios, float2 *translateRatios, float2 *gridRowRatio, d_float16 *gridColRatio)
{
    gridRowRatio->x = fmaf(id_y, -rotateRatios->y, -translateRatios->x);
    gridRowRatio->y = fmaf(id_y, rotateRatios->x, -translateRatios->y);

    int id_x_vector[8];
    id_x_vector[0] = id_x;
    id_x_vector[1] = id_x + 1;
    id_x_vector[2] = id_x + 2;
    id_x_vector[3] = id_x + 3;
    id_x_vector[4] = id_x + 4;
    id_x_vector[5] = id_x + 5;
    id_x_vector[6] = id_x + 6;
    id_x_vector[7] = id_x + 7;

    gridColRatio->x.x.x = fmaf(id_x_vector[0], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.x.y = fmaf(id_x_vector[1], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.x.z = fmaf(id_x_vector[2], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.x.w = fmaf(id_x_vector[3], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.y.x = fmaf(id_x_vector[4], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.y.y = fmaf(id_x_vector[5], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.y.z = fmaf(id_x_vector[6], rotateRatios->x, gridRowRatio->x);
    gridColRatio->x.y.w = fmaf(id_x_vector[7], rotateRatios->x, gridRowRatio->x);

    gridColRatio->y.x.x = fmaf(id_x_vector[0], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.x.y = fmaf(id_x_vector[1], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.x.z = fmaf(id_x_vector[2], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.x.w = fmaf(id_x_vector[3], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.y.x = fmaf(id_x_vector[4], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.y.y = fmaf(id_x_vector[5], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.y.z = fmaf(id_x_vector[6], rotateRatios->y, gridRowRatio->y);
    gridColRatio->y.y.w = fmaf(id_x_vector[7], rotateRatios->y, gridRowRatio->y);
}

// Gridmask helpers - Vector masked store computes for 8 pixels per channel (PKD3/PLN3/PLN1)

template <typename T>
__device__ void gridmask_vector_masked_store_hip_compute(T *src, T *dst, d_float16 *gridColRatio, float gridRatio)
{
    dst->x.x = ((gridColRatio->x.x.x >= gridRatio) || (gridColRatio->y.x.x >= gridRatio)) ? src->x.x : dst->x.x;
    dst->x.y = ((gridColRatio->x.x.y >= gridRatio) || (gridColRatio->y.x.y >= gridRatio)) ? src->x.y : dst->x.y;
    dst->x.z = ((gridColRatio->x.x.z >= gridRatio) || (gridColRatio->y.x.z >= gridRatio)) ? src->x.z : dst->x.z;
    dst->x.w = ((gridColRatio->x.x.w >= gridRatio) || (gridColRatio->y.x.w >= gridRatio)) ? src->x.w : dst->x.w;
    dst->y.x = ((gridColRatio->x.y.x >= gridRatio) || (gridColRatio->y.y.x >= gridRatio)) ? src->y.x : dst->y.x;
    dst->y.y = ((gridColRatio->x.y.y >= gridRatio) || (gridColRatio->y.y.y >= gridRatio)) ? src->y.y : dst->y.y;
    dst->y.z = ((gridColRatio->x.y.z >= gridRatio) || (gridColRatio->y.y.z >= gridRatio)) ? src->y.z : dst->y.z;
    dst->y.w = ((gridColRatio->x.y.w >= gridRatio) || (gridColRatio->y.y.w >= gridRatio)) ? src->y.w : dst->y.w;
}

// Gridmask helpers for different data layouts

// PKD3 -> PKD3
__device__ void gridmask_result_pkd3_pkd3_hip_compute(uchar *srcPtr, int srcIdx, uchar *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_uchar24_as_uchar3s src, dst;
    src = *(d_uchar24_as_uchar3s *)&srcPtr[srcIdx];
    dst = *(d_uchar24_as_uchar3s *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_uchar24_as_uchar3s *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pkd3_pkd3_hip_compute(float *srcPtr, int srcIdx, float *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_float24_as_float3s src, dst;
    src = *(d_float24_as_float3s *)&srcPtr[srcIdx];
    dst = *(d_float24_as_float3s *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_float24_as_float3s *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pkd3_pkd3_hip_compute(schar *srcPtr, int srcIdx, schar *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_schar24_as_schar3s src, dst;
    src = *(d_schar24_as_schar3s *)&srcPtr[srcIdx];
    dst = *(d_schar24_as_schar3s *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_schar24_as_schar3s *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pkd3_pkd3_hip_compute(half *srcPtr, int srcIdx, half *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_half24_as_half3s src, dst;
    src = *(d_half24_as_half3s *)&srcPtr[srcIdx];
    dst = *(d_half24_as_half3s *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_half24_as_half3s *)&dstPtr[dstIdx] = dst;
}

// PLN3 -> PLN3
__device__ void gridmask_result_pln3_pln3_hip_compute(uchar *srcPtr, int srcIdx, uint srcStrideC, uchar *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_uchar24 src, dst;
    src.x = *(d_uchar8 *)&srcPtr[srcIdx];
    dst.x = *(d_uchar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst.x;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.y = *(d_uchar8 *)&srcPtr[srcIdx];
    dst.y = *(d_uchar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst.y;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.z = *(d_uchar8 *)&srcPtr[srcIdx];
    dst.z = *(d_uchar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst.z;
}
__device__ void gridmask_result_pln3_pln3_hip_compute(float *srcPtr, int srcIdx, uint srcStrideC, float *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_float24 src, dst;
    src.x = *(d_float8 *)&srcPtr[srcIdx];
    dst.x = *(d_float8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst.x;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.y = *(d_float8 *)&srcPtr[srcIdx];
    dst.y = *(d_float8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst.y;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.z = *(d_float8 *)&srcPtr[srcIdx];
    dst.z = *(d_float8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst.z;
}
__device__ void gridmask_result_pln3_pln3_hip_compute(schar *srcPtr, int srcIdx, uint srcStrideC, schar *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_schar24 src, dst;
    src.x = *(d_schar8 *)&srcPtr[srcIdx];
    dst.x = *(d_schar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst.x;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.y = *(d_schar8 *)&srcPtr[srcIdx];
    dst.y = *(d_schar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst.y;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.z = *(d_schar8 *)&srcPtr[srcIdx];
    dst.z = *(d_schar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst.z;
}
__device__ void gridmask_result_pln3_pln3_hip_compute(half *srcPtr, int srcIdx, uint srcStrideC, half *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_half24_as_halfs src, dst;
    src.x = *(d_half8_as_halfs *)&srcPtr[srcIdx];
    dst.x = *(d_half8_as_halfs *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst.x;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.y = *(d_half8_as_halfs *)&srcPtr[srcIdx];
    dst.y = *(d_half8_as_halfs *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst.y;
    srcIdx += srcStrideC;
    dstIdx += dstStrideC;
    src.z = *(d_half8_as_halfs *)&srcPtr[srcIdx];
    dst.z = *(d_half8_as_halfs *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst.z;
}

// PLN1 -> PLN1
__device__ void gridmask_result_pln1_pln1_hip_compute(uchar *srcPtr, int srcIdx, uchar *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_uchar8 src, dst;
    src = *(d_uchar8 *)&srcPtr[srcIdx];
    dst = *(d_uchar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pln1_pln1_hip_compute(float *srcPtr, int srcIdx, float *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_float8 src, dst;
    src = *(d_float8 *)&srcPtr[srcIdx];
    dst = *(d_float8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pln1_pln1_hip_compute(schar *srcPtr, int srcIdx, schar *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_schar8 src, dst;
    src = *(d_schar8 *)&srcPtr[srcIdx];
    dst = *(d_schar8 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pln1_pln1_hip_compute(half *srcPtr, int srcIdx, half *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_half8_as_halfs src, dst;
    src = *(d_half8_as_halfs *)&srcPtr[srcIdx];
    dst = *(d_half8_as_halfs *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src, &dst, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst;
}

// PKD3 -> PLN3
__device__ void gridmask_result_pkd3_pln3_hip_compute(uchar *srcPtr, int srcIdx, uchar *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_uchar24 src, dst;
    src = *(d_uchar24 *)&srcPtr[srcIdx];
    dst = *(d_uchar24 *)&dstPtr[dstIdx];
    rpp_hip_layouttoggle24_pkd3_to_pln3(&src);
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst.x;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst.y;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_uchar8 *)&dstPtr[dstIdx] = dst.z;
}
__device__ void gridmask_result_pkd3_pln3_hip_compute(float *srcPtr, int srcIdx, float *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_float24 src, dst;
    src = *(d_float24 *)&srcPtr[srcIdx];
    dst = *(d_float24 *)&dstPtr[dstIdx];
    rpp_hip_layouttoggle24_pkd3_to_pln3(&src);
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst.x;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst.y;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_float8 *)&dstPtr[dstIdx] = dst.z;
}
__device__ void gridmask_result_pkd3_pln3_hip_compute(schar *srcPtr, int srcIdx, schar *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_schar24 src, dst;
    src = *(d_schar24 *)&srcPtr[srcIdx];
    dst = *(d_schar24 *)&dstPtr[dstIdx];
    rpp_hip_layouttoggle24_pkd3_to_pln3(&src);
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst.x;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst.y;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_schar8 *)&dstPtr[dstIdx] = dst.z;
}
__device__ void gridmask_result_pkd3_pln3_hip_compute(half *srcPtr, int srcIdx, half *dstPtr, int dstIdx, uint dstStrideC, d_float16 *gridColRatio, float gridRatio)
{
    d_half24_as_halfs src, dst;
    src = *(d_half24_as_halfs *)&srcPtr[srcIdx];
    dst = *(d_half24_as_halfs *)&dstPtr[dstIdx];
    rpp_hip_layouttoggle24_pkd3_to_pln3(&src);
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst.x;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.y, &dst.y, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst.y;
    dstIdx += dstStrideC;
    gridmask_vector_masked_store_hip_compute(&src.z, &dst.z, gridColRatio, gridRatio);
    *(d_half8_as_halfs *)&dstPtr[dstIdx] = dst.z;
}

// PLN3 -> PKD3
__device__ void gridmask_result_pln3_pkd3_hip_compute(uchar *srcPtr, int srcIdx, uint srcStrideC, uchar *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_uchar24 src, dst;
    src = *(d_uchar24 *)&srcPtr[srcIdx];
    dst = *(d_uchar24 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3(&dst);
    *(d_uchar24 *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pln3_pkd3_hip_compute(float *srcPtr, int srcIdx, uint srcStrideC, float *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_float24 src, dst;
    src = *(d_float24 *)&srcPtr[srcIdx];
    dst = *(d_float24 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3(&dst);
    *(d_float24 *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pln3_pkd3_hip_compute(schar *srcPtr, int srcIdx, uint srcStrideC, schar *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_schar24 src, dst;
    src = *(d_schar24 *)&srcPtr[srcIdx];
    dst = *(d_schar24 *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3(&dst);
    *(d_schar24 *)&dstPtr[dstIdx] = dst;
}
__device__ void gridmask_result_pln3_pkd3_hip_compute(half *srcPtr, int srcIdx, uint srcStrideC, half *dstPtr, int dstIdx, d_float16 *gridColRatio, float gridRatio)
{
    d_half24_as_halfs src, dst;
    src = *(d_half24_as_halfs *)&srcPtr[srcIdx];
    dst = *(d_half24_as_halfs *)&dstPtr[dstIdx];
    gridmask_vector_masked_store_hip_compute(&src.x, &dst.x, gridColRatio, gridRatio);
    rpp_hip_layouttoggle24_pln3_to_pkd3(&dst);
    *(d_half24_as_halfs *)&dstPtr[dstIdx] = dst;
}

// Gridmask kernels

template <typename T>
__global__ void gridmask_pkd_tensor(T *srcPtr,
                                    uint2 srcStridesNH,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float2 rotateRatios,
                                    float2 translateRatios,
                                    float gridRatio,
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
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    float2 gridRowRatio;
    d_float16 gridColRatio, gridColRatioFloor;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);
    rpp_hip_math_floor16(&gridColRatio, &gridColRatioFloor);
    rpp_hip_math_subtract16(&gridColRatio, &gridColRatioFloor, &gridColRatio);
    gridmask_result_pkd3_pkd3_hip_compute(srcPtr, srcIdx, dstPtr, dstIdx, &gridColRatio, gridRatio);
}

template <typename T>
__global__ void gridmask_pln_tensor(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    int channelsDst,
                                    float2 rotateRatios,
                                    float2 translateRatios,
                                    float gridRatio,
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

    float2 gridRowRatio;
    d_float16 gridColRatio, gridColRatioFloor;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);
    rpp_hip_math_floor16(&gridColRatio, &gridColRatioFloor);
    rpp_hip_math_subtract16(&gridColRatio, &gridColRatioFloor, &gridColRatio);

    if (channelsDst == 3)
        gridmask_result_pln3_pln3_hip_compute(srcPtr, srcIdx, srcStridesNCH.y, dstPtr, dstIdx, dstStridesNCH.y, &gridColRatio, gridRatio);
    else
        gridmask_result_pln1_pln1_hip_compute(srcPtr, srcIdx, dstPtr, dstIdx, &gridColRatio, gridRatio);
}

template <typename T>
__global__ void gridmask_pkd3_pln3_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float2 rotateRatios,
                                          float2 translateRatios,
                                          float gridRatio,
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

    float2 gridRowRatio;
    d_float16 gridColRatio, gridColRatioFloor;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);
    rpp_hip_math_floor16(&gridColRatio, &gridColRatioFloor);
    rpp_hip_math_subtract16(&gridColRatio, &gridColRatioFloor, &gridColRatio);
    gridmask_result_pkd3_pln3_hip_compute(srcPtr, srcIdx, dstPtr, dstIdx, dstStridesNCH.y, &gridColRatio, gridRatio);
}

template <typename T>
__global__ void gridmask_pln3_pkd3_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float2 rotateRatios,
                                          float2 translateRatios,
                                          float gridRatio,
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
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    float2 gridRowRatio;
    d_float16 gridColRatio, gridColRatioFloor;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);
    rpp_hip_math_floor16(&gridColRatio, &gridColRatioFloor);
    rpp_hip_math_subtract16(&gridColRatio, &gridColRatioFloor, &gridColRatio);
    gridmask_result_pln3_pkd3_hip_compute(srcPtr, srcIdx, srcStridesNCH.y, dstPtr, dstIdx, &gridColRatio, gridRatio);
}

template <typename T>
RppStatus hip_exec_gridmask_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u tileWidth,
                                   Rpp32f gridRatio,
                                   Rpp32f gridAngle,
                                   RpptUintVector2D translateVector,
                                   RpptROIPtr roiTensorPtrSrc,
                                   rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32f tileWidthInv = 1.0f / (Rpp32f)tileWidth;
    float2 rotateRatios = make_float2((cos(gridAngle) * tileWidthInv), (sin(gridAngle) * tileWidthInv));
    float2 translateRatios = make_float2((translateVector.x * tileWidthInv), (translateVector.y * tileWidthInv));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(gridmask_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           rotateRatios,
                           translateRatios,
                           gridRatio,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(gridmask_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           rotateRatios,
                           translateRatios,
                           gridRatio,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(gridmask_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               rotateRatios,
                               translateRatios,
                               gridRatio,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(gridmask_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               rotateRatios,
                               translateRatios,
                               gridRatio,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
