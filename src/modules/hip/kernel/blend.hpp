#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void blend_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *alpha_f4)
{
    dst_f8->x = (src1_f8->x - src2_f8->x) * *alpha_f4 + src2_f8->x;
    dst_f8->y = (src1_f8->y - src2_f8->y) * *alpha_f4 + src2_f8->y;
}

template <typename T>
__global__ void blend_pkd_tensor(T *srcPtr1,
                                 T *srcPtr2,
                                 int nStrideSrc,
                                 int hStrideSrc,
                                 T *dstPtr,
                                 int nStrideDst,
                                 int hStrideDst,
                                 float *alpha,
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

    d_float8 src1_f8, src2_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr1, srcIdx, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2, srcIdx, &src2_f8);
    blend_hip_compute(&src1_f8, &src2_f8, &dst_f8, &alpha_f4);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
}

template <typename T>
__global__ void blend_pln_tensor(T *srcPtr1,
                                 T *srcPtr2,
                                 int nStrideSrc,
                                 int cStrideSrc,
                                 int hStrideSrc,
                                 T *dstPtr,
                                 int nStrideDst,
                                 int cStrideDst,
                                 int hStrideDst,
                                 int channelsDst,
                                 float *alpha,
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

    d_float8 src1_f8, src2_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr1, srcIdx, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2, srcIdx, &src2_f8);
    blend_hip_compute(&src1_f8, &src2_f8, &dst_f8, &alpha_f4);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr1, srcIdx, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2, srcIdx, &src2_f8);
        blend_hip_compute(&src1_f8, &src2_f8, &dst_f8, &alpha_f4);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr1, srcIdx, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2, srcIdx, &src2_f8);
        blend_hip_compute(&src1_f8, &src2_f8, &dst_f8, &alpha_f4);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void blend_pkd3_pln3_tensor(T *srcPtr1,
                                       T *srcPtr2,
                                       int nStrideSrc,
                                       int hStrideSrc,
                                       T *dstPtr,
                                       int nStrideDst,
                                       int cStrideDst,
                                       int hStrideDst,
                                       float *alpha,
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

    d_float24 src1_f24, src2_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1, srcIdx, &src1_f24);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2, srcIdx, &src2_f24);
    blend_hip_compute(&src1_f24.x, &src2_f24.x, &dst_f24.x, &alpha_f4);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.x);

    dstIdx += cStrideDst;

    blend_hip_compute(&src1_f24.y, &src2_f24.y, &dst_f24.y, &alpha_f4);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.y);

    dstIdx += cStrideDst;

    blend_hip_compute(&src1_f24.z, &src2_f24.z, &dst_f24.z, &alpha_f4);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.z);
}

template <typename T>
__global__ void blend_pln3_pkd3_tensor(T *srcPtr1,
                                       T *srcPtr2,
                                       int nStrideSrc,
                                       int cStrideSrc,
                                       int hStrideSrc,
                                       T *dstPtr,
                                       int nStrideDst,
                                       int hStrideDst,
                                       float *alpha,
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

    d_float24 src1_f24, src2_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr1, srcIdx, cStrideSrc, &src1_f24);
    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr2, srcIdx, cStrideSrc, &src2_f24);
    blend_hip_compute(&src1_f24.x, &src2_f24.x, &dst_f24.x, &alpha_f4);
    blend_hip_compute(&src1_f24.y, &src2_f24.y, &dst_f24.y, &alpha_f4);
    blend_hip_compute(&src1_f24.z, &src2_f24.z, &dst_f24.z, &alpha_f4);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr, dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_blend_tensor(T *srcPtr1,
                                T *srcPtr2,
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
        hipLaunchKernelGGL(blend_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.hStride,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(blend_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(blend_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(blend_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.hStride,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
