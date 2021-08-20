#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void brightness_hip_compute(uchar *srcPtr, float4 *src_f4, float4 *dst_f4, float4 *alpha_f4, float4 *beta_f4)
{
    *dst_f4 = *src_f4 * *alpha_f4 + *beta_f4;
}

__device__ void brightness_hip_compute(float *srcPtr, float4 *src_f4, float4 *dst_f4, float4 *alpha_f4, float4 *beta_f4)
{
    float4 betaNormFactor_f4 = make_float4(0.0039216, 0.0039216, 0.0039216, 0.0039216);
    *dst_f4 = *src_f4 * *alpha_f4 + *beta_f4 * betaNormFactor_f4;
}

__device__ void brightness_hip_compute(signed char *srcPtr, float4 *src_f4, float4 *dst_f4, float4 *alpha_f4, float4 *beta_f4)
{
    float4 i8Offset_f4 = make_float4(128, 128, 128, 128);
    *dst_f4 = (*src_f4 + i8Offset_f4) * *alpha_f4 + *beta_f4 - i8Offset_f4;
}

template <typename T>
__global__ void brightness_pkd_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int hStrideDst,
                                      float *alpha,
                                      float *beta,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    float4 alpha_f4 = (float4)alpha[id_z];
    float4 beta_f4 = (float4)beta[id_z];

    float4 srcX_f4, srcY_f4, dstX_f4, dstY_f4;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &srcX_f4, &srcY_f4);

    brightness_hip_compute(srcPtr, &srcX_f4, &dstX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcY_f4, &dstY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstX_f4, &dstY_f4);
}

template <typename T>
__global__ void brightness_pln_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int cStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int cStrideDst,
                                      int hStrideDst,
                                      int channelsDst,
                                      float *alpha,
                                      float *beta,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    float4 alpha_f4 = (float4)(alpha[id_z]);
    float4 beta_f4 = (float4)(beta[id_z]);

    float4 srcX_f4, srcY_f4, dstX_f4, dstY_f4;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &srcX_f4, &srcY_f4);

    brightness_hip_compute(srcPtr, &srcX_f4, &dstX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcY_f4, &dstY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstX_f4, &dstY_f4);

    if (channelsDst == 3)
    {
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &srcX_f4, &srcY_f4);

        brightness_hip_compute(srcPtr, &srcX_f4, &dstX_f4, &alpha_f4, &beta_f4);
        brightness_hip_compute(srcPtr, &srcY_f4, &dstY_f4, &alpha_f4, &beta_f4);

        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstX_f4, &dstY_f4);

        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &srcX_f4, &srcY_f4);

        brightness_hip_compute(srcPtr, &srcX_f4, &dstX_f4, &alpha_f4, &beta_f4);
        brightness_hip_compute(srcPtr, &srcY_f4, &dstY_f4, &alpha_f4, &beta_f4);

        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstX_f4, &dstY_f4);
    }
}

template <typename T>
__global__ void brightness_pkd3_pln3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int cStrideDst,
                                            int hStrideDst,
                                            float *alpha,
                                            float *beta,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    float4 alpha_f4 = (float4)alpha[id_z];
    float4 beta_f4 = (float4)beta[id_z];

    uint2 src1 = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += 8;
    uint2 src2 = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += 8;
    uint2 src3 = *((uint2 *)(&srcPtr[srcIdx]));
    uint2 dst;

    float4 src1X, src1Y, src2X, src2Y, src3X, src3Y;
    float4 pixX, pixY;

    src1X = rpp_hip_unpack(src1.x); // [R0|G0|B0|R1]
    src1Y = rpp_hip_unpack(src1.y); // [G1|B1|R2|G2]
    src2X = rpp_hip_unpack(src2.x); // [B2|R3|G3|B3]
    src2Y = rpp_hip_unpack(src2.y); // [R4|G4|B4|R5]
    src3X = rpp_hip_unpack(src3.x); // [G5|B5|R6|G6]
    src3Y = rpp_hip_unpack(src3.y); // [B6|R7|G7|B7]

    pixX.x = src1X.x;
    pixX.y = src1X.w;
    pixX.z = src1Y.z;
    pixX.w = src2X.y;
    pixY.x = src2Y.x;
    pixY.y = src2Y.w;
    pixY.z = src3X.z;
    pixY.w = src3Y.y;

    dst.x = rpp_hip_pack(pixX * alpha_f4 + beta_f4);
    dst.y = rpp_hip_pack(pixY * alpha_f4 + beta_f4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
    dstIdx += cStrideDst;

    pixX.x = src1X.y;
    pixX.y = src1Y.x;
    pixX.z = src1Y.w;
    pixX.w = src2X.z;
    pixY.x = src2Y.y;
    pixY.y = src3X.x;
    pixY.z = src3X.w;
    pixY.w = src3Y.z;

    dst.x = rpp_hip_pack(pixX * alpha_f4 + beta_f4);
    dst.y = rpp_hip_pack(pixY * alpha_f4 + beta_f4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
    dstIdx += cStrideDst;

    pixX.x = src1X.z;
    pixX.y = src1Y.y;
    pixX.z = src2X.x;
    pixX.w = src2X.w;
    pixY.x = src2Y.z;
    pixY.y = src3X.y;
    pixY.z = src3Y.x;
    pixY.w = src3Y.w;

    dst.x = rpp_hip_pack(pixX * alpha_f4 + beta_f4);
    dst.y = rpp_hip_pack(pixY * alpha_f4 + beta_f4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

extern "C" __global__ void brightness_pln3_pkd3_tensor(uchar *srcPtr,
                                                       int nStrideSrc,
                                                       int cStrideSrc,
                                                       int hStrideSrc,
                                                       uchar *dstPtr,
                                                       int nStrideDst,
                                                       int hStrideDst,
                                                       float *alpha,
                                                       float *beta,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;

    uint2 src1 = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += cStrideSrc;
    uint2 src2 = *((uint2 *)(&srcPtr[srcIdx]));
    srcIdx += cStrideSrc;
    uint2 src3 = *((uint2 *)(&srcPtr[srcIdx]));
    uint2 dst;

    float4 alpha_f4 = (float4)(alpha[id_z]);
    float4 beta_f4 = (float4)(beta[id_z]);

    float4 src1X, src1Y, src2X, src2Y, src3X, src3Y;
    float4 pixX, pixY;

    src1X = rpp_hip_unpack(src1.x); // [R0|R1|R2|R3]
    src1Y = rpp_hip_unpack(src1.y); // [R4|R5|R6|R7]
    src2X = rpp_hip_unpack(src2.x); // [G0|G1|G2|G3]
    src2Y = rpp_hip_unpack(src2.y); // [G4|G5|G6|G7]
    src3X = rpp_hip_unpack(src3.x); // [B0|B1|B2|B3]
    src3Y = rpp_hip_unpack(src3.y); // [B4|B5|B6|B7]

    pixX.x = src1X.x;
    pixX.y = src2X.x;
    pixX.z = src3X.x;
    pixX.w = src1X.y;
    pixY.x = src2X.y;
    pixY.y = src3X.y;
    pixY.z = src1X.z;
    pixY.w = src2X.z;

    dst.x = rpp_hip_pack(pixX * alpha_f4 + beta_f4);
    dst.y = rpp_hip_pack(pixY * alpha_f4 + beta_f4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
    dstIdx += 8;

    pixX.x = src3X.z;
    pixX.y = src1X.w;
    pixX.z = src2X.w;
    pixX.w = src3X.w;
    pixY.x = src1Y.x;
    pixY.y = src2Y.x;
    pixY.z = src3Y.x;
    pixY.w = src1Y.y;

    dst.x = rpp_hip_pack(pixX * alpha_f4 + beta_f4);
    dst.y = rpp_hip_pack(pixY * alpha_f4 + beta_f4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
    dstIdx += 8;

    pixX.x = src2Y.y;
    pixX.y = src3Y.y;
    pixX.z = src1Y.z;
    pixX.w = src2Y.z;
    pixY.x = src3Y.z;
    pixY.y = src1Y.w;
    pixY.z = src2Y.w;
    pixY.w = src3Y.w;

    dst.x = rpp_hip_pack(pixX * alpha_f4 + beta_f4);
    dst.y = rpp_hip_pack(pixY * alpha_f4 + beta_f4);

    *((uint2 *)(&dstPtr[dstIdx])) = dst;
}

template <typename T>
RppStatus hip_exec_brightness_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     rpp::Handle& handle)
{
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(brightness_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.hStride,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(brightness_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           roiTensorPtrSrc);
    }
    // else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    // {
    //     if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    //     {
    //         hipLaunchKernelGGL(brightness_pkd3_pln3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.cStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    //     else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    //     {
    //         globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    //         hipLaunchKernelGGL(brightness_pln3_pkd3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.cStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    // }

    return RPP_SUCCESS;
}
