#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void contrast_hip_compute(uchar *srcPtr, d_float8 *pix_f8, d_float8 *contrastParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255((pix_f8->f4[0] - contrastParams_f8->f4[1]) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255((pix_f8->f4[1] - contrastParams_f8->f4[1]) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1]);
}

__device__ void contrast_hip_compute(float *srcPtr, d_float8 *pix_f8, d_float8 *contrastParams_f8)
{
    float4 normalizer_f4 = (float4) ONE_OVER_255;

    pix_f8->f4[0] = rpp_hip_pixel_check_0to1((pix_f8->f4[0] - (contrastParams_f8->f4[1] * normalizer_f4)) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1] * normalizer_f4);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1((pix_f8->f4[1] - (contrastParams_f8->f4[1] * normalizer_f4)) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1] * normalizer_f4);
}

__device__ void contrast_hip_compute(schar *srcPtr, d_float8 *pix_f8, d_float8 *contrastParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255((pix_f8->f4[0] + (float4)128 - contrastParams_f8->f4[1]) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1]) - (float4)128;
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255((pix_f8->f4[1] + (float4)128 - contrastParams_f8->f4[1]) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1]) - (float4)128;
}
__device__ void contrast_hip_compute(half *srcPtr, d_float8 *pix_f8, d_float8 *contrastParams_f8)
{
    float4 normalizer_f4 = (float4) ONE_OVER_255;

    pix_f8->f4[0] = rpp_hip_pixel_check_0to1((pix_f8->f4[0] - (contrastParams_f8->f4[1] * normalizer_f4)) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1] * normalizer_f4);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1((pix_f8->f4[1] - (contrastParams_f8->f4[1] * normalizer_f4)) * contrastParams_f8->f4[0] + contrastParams_f8->f4[1] * normalizer_f4);
}

template <typename T>
__global__ void contrast_pkd_hip_tensor(T *srcPtr,
                                    uint2 srcStridesNH,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float *contrastFactor,
                                    float *contrastCenter,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_float8 pix_f8, contrastParams_f8;
    contrastParams_f8.f4[0] = (float4)contrastFactor[id_z];
    contrastParams_f8.f4[1] = (float4)contrastCenter[id_z];

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    contrast_hip_compute(srcPtr, &pix_f8, &contrastParams_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
}

template <typename T>
__global__ void contrast_pln_hip_tensor(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    int channelsDst,
                                    float *contrastFactor,
                                    float *contrastCenter,
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

    d_float8 pix_f8, contrastParams_f8;
    contrastParams_f8.f4[0] = (float4)(contrastFactor[id_z]);
    contrastParams_f8.f4[1] = (float4)(contrastCenter[id_z]);

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    contrast_hip_compute(srcPtr, &pix_f8, &contrastParams_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        contrast_hip_compute(srcPtr, &pix_f8, &contrastParams_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        contrast_hip_compute(srcPtr, &pix_f8, &contrastParams_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
    }
}

template <typename T>
__global__ void contrast_pkd3_pln3_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float *contrastFactor,
                                          float *contrastCenter,
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

    d_float24 pix_f24;
    d_float8 contrastParams_f8;
    contrastParams_f8.f4[0] = (float4)contrastFactor[id_z];
    contrastParams_f8.f4[1] = (float4)contrastCenter[id_z];

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    contrast_hip_compute(srcPtr, &pix_f24.f8[0], &contrastParams_f8);
    contrast_hip_compute(srcPtr, &pix_f24.f8[1], &contrastParams_f8);
    contrast_hip_compute(srcPtr, &pix_f24.f8[2], &contrastParams_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void contrast_pln3_pkd3_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *contrastFactor,
                                          float *contrastCenter,
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

    d_float24 pix_f24;
    d_float8 contrastParams_f8;
    contrastParams_f8.f4[0] = (float4)contrastFactor[id_z];
    contrastParams_f8.f4[1] = (float4)contrastCenter[id_z];

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    contrast_hip_compute(srcPtr, &pix_f24.f8[0], &contrastParams_f8);
    contrast_hip_compute(srcPtr, &pix_f24.f8[1], &contrastParams_f8);
    contrast_hip_compute(srcPtr, &pix_f24.f8[2], &contrastParams_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_contrast_tensor(T *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   T *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(contrast_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(contrast_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(contrast_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(contrast_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
