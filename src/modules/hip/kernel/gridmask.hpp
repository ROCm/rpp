#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void gridmask_ratio_hip_compute(int id_x, int id_y, float2 *rotateRatios, float2 *translateRatios, float2 *gridRowRatio, float2 *gridColRatio)
{
    gridRowRatio->x = fmaf(id_y, -rotateRatios->y, -translateRatios->x);
    gridRowRatio->y = fmaf(id_y, rotateRatios->x, -translateRatios->y);
    gridColRatio->x = fmaf(id_x, rotateRatios->x, gridRowRatio->x);
    gridColRatio->y = fmaf(id_x, rotateRatios->y, gridRowRatio->y);
}

__device__ void gridmask_color(uchar &pix){pix = 0;}
__device__ void gridmask_color(float &pix){pix = 0;}
__device__ void gridmask_color(schar &pix){pix = -128;}
__device__ void gridmask_color(half &pix){pix = 0;}

// PKD3 -> PKD3

__device__ void gridmask_result_pkd3_pkd3_hip_compute(uchar *srcPtr, int srcIdx, uchar *dstPtr, int dstIdx, float2 *gridColRatio, float gridRatio)
{
    uchar pix;
    gridmask_color(pix);
    if ((gridColRatio->x - floorf(gridColRatio->x) >= gridRatio) || (gridColRatio->y - floorf(gridColRatio->y) >= gridRatio))
        *(uchar3 *)&dstPtr[dstIdx] = *(uchar3 *)&srcPtr[srcIdx];
    else
        *(uchar3 *)&dstPtr[dstIdx] = (uchar3)pix;
}

__device__ void gridmask_result_pkd3_pkd3_hip_compute(float *srcPtr, int srcIdx, float *dstPtr, int dstIdx, float2 *gridColRatio, float gridRatio)
{
    float pix;
    gridmask_color(pix);
    if ((gridColRatio->x - floorf(gridColRatio->x) >= gridRatio) || (gridColRatio->y - floorf(gridColRatio->y) >= gridRatio))
        *(float3 *)&dstPtr[dstIdx] = *(float3 *)&srcPtr[srcIdx];
    else
        *(float3 *)&dstPtr[dstIdx] = (float3)pix;
}

__device__ void gridmask_result_pkd3_pkd3_hip_compute(schar *srcPtr, int srcIdx, schar *dstPtr, int dstIdx, float2 *gridColRatio, float gridRatio)
{
    schar pix;
    gridmask_color(pix);
    d_schar3 pix_schar3;
    pix_schar3.x = pix_schar3.y = pix_schar3.z = pix;
    if ((gridColRatio->x - floorf(gridColRatio->x) >= gridRatio) || (gridColRatio->y - floorf(gridColRatio->y) >= gridRatio))
        *(d_schar3 *)&dstPtr[dstIdx] = *(d_schar3 *)&srcPtr[srcIdx];
    else
        *(d_schar3 *)&dstPtr[dstIdx] = pix_schar3;
}

__device__ void gridmask_result_pkd3_pkd3_hip_compute(half *srcPtr, int srcIdx, half *dstPtr, int dstIdx, float2 *gridColRatio, float gridRatio)
{
    half pix;
    gridmask_color(pix);
    d_half3 pix_half3;
    pix_half3.x = pix_half3.y = pix_half3.z = 0;
    if ((gridColRatio->x - floorf(gridColRatio->x) >= gridRatio) || (gridColRatio->y - floorf(gridColRatio->y) >= gridRatio))
        *(d_half3 *)&dstPtr[dstIdx] = *(d_half3 *)&srcPtr[srcIdx];
    else
        *(d_half3 *)&dstPtr[dstIdx] = pix_half3;
}

// PLN3 -> PLN3

template <typename T>
__device__ void gridmask_result_pln3_pln3_hip_compute(T *srcPtr, int srcIdx, uint3 *srcStridesNCH, T *dstPtr, int dstIdx, uint3 *dstStridesNCH, float2 *gridColRatio, float gridRatio)
{
    if ((gridColRatio->x - floorf(gridColRatio->x) >= gridRatio) || (gridColRatio->y - floorf(gridColRatio->y) >= gridRatio))
    {
        dstPtr[dstIdx] = srcPtr[srcIdx];
        srcIdx += srcStridesNCH->y;
        dstIdx += dstStridesNCH->y;
        dstPtr[dstIdx] = srcPtr[srcIdx];
        srcIdx += srcStridesNCH->y;
        dstIdx += dstStridesNCH->y;
        dstPtr[dstIdx] = srcPtr[srcIdx];
    }
    else
    {
        T pix;
        gridmask_color(pix);
        dstPtr[dstIdx] = pix;
        srcIdx += srcStridesNCH->y;
        dstIdx += dstStridesNCH->y;
        dstPtr[dstIdx] = pix;
        srcIdx += srcStridesNCH->y;
        dstIdx += dstStridesNCH->y;
        dstPtr[dstIdx] = pix;
    }
}

// PLN1 -> PLN1

template <typename T>
__device__ void gridmask_result_pln1_pln1_hip_compute(T *srcPtr, int srcIdx, T *dstPtr, int dstIdx, float2 *gridColRatio, float gridRatio)
{
    if ((gridColRatio->x - floorf(gridColRatio->x) >= gridRatio) || (gridColRatio->y - floorf(gridColRatio->y) >= gridRatio))
    {
        dstPtr[dstIdx] = srcPtr[srcIdx];
    }
    else
    {
        T pix;
        gridmask_color(pix);
        dstPtr[dstIdx] = pix;
    }
}

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
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    float2 gridRowRatio, gridColRatio;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);
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
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float2 gridRowRatio, gridColRatio;
    gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);

    if (channelsDst == 3)
        gridmask_result_pln3_pln3_hip_compute(srcPtr, srcIdx, &srcStridesNCH, dstPtr, dstIdx, &dstStridesNCH, &gridColRatio, gridRatio);
    else
        gridmask_result_pln1_pln1_hip_compute(srcPtr, srcIdx, dstPtr, dstIdx, &gridColRatio, gridRatio);
}

// template <typename T>
// __global__ void gridmask_pkd3_pln3_tensor(T *srcPtr,
//                                           uint2 srcStridesNH,
//                                           T *dstPtr,
//                                           uint3 dstStridesNCH,
//                                           float2 rotateRatios,
//                                           float2 translateRatios,
//                                           float gridRatio,
//                                           RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

//     float2 gridRowRatio, gridColRatio;
//     gridmask_ratio_hip_compute(id_x, id_y, &rotateRatios, &translateRatios, &gridRowRatio, &gridColRatio);

//     if ((gridColRatio.x - floorf(gridColRatio.x) >= gridRatio) || (gridColRatio.y - floorf(gridColRatio.y) >= gridRatio))
//     {
//         uchar3 src;
//         src = *(uchar3 *)&srcPtr[srcIdx];
//         dstPtr[dstIdx] = srcPtr[srcIdx];
//         srcIdx++;
//         dstIdx += dstStridesNCH.y;
//         dstPtr[dstIdx] = srcPtr[srcIdx];
//         srcIdx += srcStridesNCH.y;
//         dstIdx += dstStridesNCH.y;
//         dstPtr[dstIdx] = srcPtr[srcIdx];

//     }
// }

// template <typename T>
// __global__ void gridmask_pln3_pkd3_tensor(T *srcPtr,
//                                           uint3 srcStridesNCH,
//                                           T *dstPtr,
//                                           uint2 dstStridesNH,
//                                           float2 rotateRatios,
//                                           float2 translateRatios,
//                                           float gridRatio,
//                                           RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

//     float4 alpha_f4 = (float4)(alpha[id_z]);
//     float4 beta_f4 = (float4)(beta[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr, srcIdx, srcStridesNCH.y, &src_f24);
//     gridmask_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &alpha_f4, &beta_f4);
//     gridmask_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &alpha_f4, &beta_f4);
//     gridmask_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
// }

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
    int globalThreads_x = dstDescPtr->w;
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
            // hipLaunchKernelGGL(gridmask_pkd3_pln3_tensor,
            //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
            //                    dim3(localThreads_x, localThreads_y, localThreads_z),
            //                    0,
            //                    handle.GetStream(),
            //                    srcPtr,
            //                    make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
            //                    dstPtr,
            //                    make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
            //                    rotateRatios,
            //                    translateRatios,
            //                    gridRatio,
            //                    roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            // globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            // hipLaunchKernelGGL(gridmask_pln3_pkd3_tensor,
            //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
            //                    dim3(localThreads_x, localThreads_y, localThreads_z),
            //                    0,
            //                    handle.GetStream(),
            //                    srcPtr,
            //                    make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
            //                    dstPtr,
            //                    make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            //                    rotateRatios,
            //                    translateRatios,
            //                    gridRatio,
            //                    roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
