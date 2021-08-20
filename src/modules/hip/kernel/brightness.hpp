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

    float4 src1X_f4, src1Y_f4, src2X_f4, src2Y_f4, src3X_f4, src3Y_f4;
    float4 srcPixX_f4, srcPixY_f4, dstPixX_f4, dstPixY_f4;

    rpp_hip_load24_and_unpack_to_float8(srcPtr, srcIdx, 8, &src1X_f4, &src1Y_f4, &src2X_f4, &src2Y_f4, &src3X_f4, &src3Y_f4);

    srcPixX_f4.x = src1X_f4.x;
    srcPixX_f4.y = src1X_f4.w;
    srcPixX_f4.z = src1Y_f4.z;
    srcPixX_f4.w = src2X_f4.y;
    srcPixY_f4.x = src2Y_f4.x;
    srcPixY_f4.y = src2Y_f4.w;
    srcPixY_f4.z = src3X_f4.z;
    srcPixY_f4.w = src3Y_f4.y;

    brightness_hip_compute(srcPtr, &srcPixX_f4, &dstPixX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcPixY_f4, &dstPixY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstPixX_f4, &dstPixY_f4);

    dstIdx += cStrideDst;

    srcPixX_f4.x = src1X_f4.y;
    srcPixX_f4.y = src1Y_f4.x;
    srcPixX_f4.z = src1Y_f4.w;
    srcPixX_f4.w = src2X_f4.z;
    srcPixY_f4.x = src2Y_f4.y;
    srcPixY_f4.y = src3X_f4.x;
    srcPixY_f4.z = src3X_f4.w;
    srcPixY_f4.w = src3Y_f4.z;

    brightness_hip_compute(srcPtr, &srcPixX_f4, &dstPixX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcPixY_f4, &dstPixY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstPixX_f4, &dstPixY_f4);

    dstIdx += cStrideDst;

    srcPixX_f4.x = src1X_f4.z;
    srcPixX_f4.y = src1Y_f4.y;
    srcPixX_f4.z = src2X_f4.x;
    srcPixX_f4.w = src2X_f4.w;
    srcPixY_f4.x = src2Y_f4.z;
    srcPixY_f4.y = src3X_f4.y;
    srcPixY_f4.z = src3Y_f4.x;
    srcPixY_f4.w = src3Y_f4.w;

    brightness_hip_compute(srcPtr, &srcPixX_f4, &dstPixX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcPixY_f4, &dstPixY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstPixX_f4, &dstPixY_f4);
}

template <typename T>
__global__ void brightness_pln3_pkd3_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int cStrideSrc,
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

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;

    float4 alpha_f4 = (float4)(alpha[id_z]);
    float4 beta_f4 = (float4)(beta[id_z]);

    float4 src1X_f4, src1Y_f4, src2X_f4, src2Y_f4, src3X_f4, src3Y_f4;
    float4 srcPixX_f4, srcPixY_f4, dstPixX_f4, dstPixY_f4;

    rpp_hip_load24_and_unpack_to_float8(srcPtr, srcIdx, cStrideSrc, &src1X_f4, &src1Y_f4, &src2X_f4, &src2Y_f4, &src3X_f4, &src3Y_f4);

    srcPixX_f4.x = src1X_f4.x;
    srcPixX_f4.y = src2X_f4.x;
    srcPixX_f4.z = src3X_f4.x;
    srcPixX_f4.w = src1X_f4.y;
    srcPixY_f4.x = src2X_f4.y;
    srcPixY_f4.y = src3X_f4.y;
    srcPixY_f4.z = src1X_f4.z;
    srcPixY_f4.w = src2X_f4.z;

    brightness_hip_compute(srcPtr, &srcPixX_f4, &dstPixX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcPixY_f4, &dstPixY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstPixX_f4, &dstPixY_f4);

    dstIdx += 8;

    srcPixX_f4.x = src3X_f4.z;
    srcPixX_f4.y = src1X_f4.w;
    srcPixX_f4.z = src2X_f4.w;
    srcPixX_f4.w = src3X_f4.w;
    srcPixY_f4.x = src1Y_f4.x;
    srcPixY_f4.y = src2Y_f4.x;
    srcPixY_f4.z = src3Y_f4.x;
    srcPixY_f4.w = src1Y_f4.y;

    brightness_hip_compute(srcPtr, &srcPixX_f4, &dstPixX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcPixY_f4, &dstPixY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstPixX_f4, &dstPixY_f4);

    dstIdx += 8;

    srcPixX_f4.x = src2Y_f4.y;
    srcPixX_f4.y = src3Y_f4.y;
    srcPixX_f4.z = src1Y_f4.z;
    srcPixX_f4.w = src2Y_f4.z;
    srcPixY_f4.x = src3Y_f4.z;
    srcPixY_f4.y = src1Y_f4.w;
    srcPixY_f4.z = src2Y_f4.w;
    srcPixY_f4.w = src3Y_f4.w;

    brightness_hip_compute(srcPtr, &srcPixX_f4, &dstPixX_f4, &alpha_f4, &beta_f4);
    brightness_hip_compute(srcPtr, &srcPixY_f4, &dstPixY_f4, &alpha_f4, &beta_f4);

    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dstPixX_f4, &dstPixY_f4);
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
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(brightness_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(brightness_pln3_pkd3_tensor,
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
                               dstDescPtr->strides.hStride,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
