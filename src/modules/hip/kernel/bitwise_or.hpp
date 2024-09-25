#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

/* BitwiseOR is logical operation only on U8/I8 types.
   For a Rpp32f precision image (pixel values from 0-1), the BitwiseOR is applied on a 0-255
   range-translated approximation, of the original 0-1 decimal-range image.
   The bitwise operation is applied to the char representation of the raw floating-point data in memory */

template <typename T>
__device__ void bitwise_or_hip_compute(T *srcPtr, d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8)
{
    if constexpr ((std::is_same<T, float>::value) || (std::is_same<T, half>::value))
    {
        rpp_hip_math_multiply8_const(src1_f8, src1_f8, (float4)255);
        rpp_hip_math_multiply8_const(src2_f8, src2_f8, (float4)255);
        rpp_hip_math_bitwiseOr8(src1_f8, src2_f8, dst_f8);
        rpp_hip_math_multiply8_const(dst_f8, dst_f8, (float4)ONE_OVER_255);
    }
    else if constexpr (std::is_same<T, signed char>::value)
    {
        rpp_hip_math_add8_const(src1_f8, src1_f8, (float4)128);
        rpp_hip_math_add8_const(src2_f8, src2_f8, (float4)128);
        rpp_hip_math_bitwiseOr8(src1_f8, src2_f8, dst_f8);
        rpp_hip_math_subtract8_const(dst_f8, dst_f8, (float4)128);
    }
    else
        rpp_hip_math_bitwiseOr8(src1_f8, src2_f8, dst_f8);
}

template <typename T>
__global__ void bitwise_or_pkd_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 src1_f24, src2_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx, &src1_f24);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx, &src2_f24);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[0], &src2_f24.f8[0], &dst_f24.f8[0]);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[1], &src2_f24.f8[1], &dst_f24.f8[1]);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[2], &src2_f24.f8[2], &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void bitwise_or_pln_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
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

    d_float8 src1_f8, src2_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx, &src2_f8);
    bitwise_or_hip_compute(srcPtr1, &src1_f8, &src2_f8, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx, &src2_f8);
        bitwise_or_hip_compute(srcPtr1, &src1_f8, &src2_f8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx, &src2_f8);
        bitwise_or_hip_compute(srcPtr1, &src1_f8, &src2_f8, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void bitwise_or_pkd3_pln3_hip_tensor(T *srcPtr1,
                                                T *srcPtr2,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
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

    d_float24 src1_f24, src2_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr1 + srcIdx, &src1_f24);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr2 + srcIdx, &src2_f24);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[0], &src2_f24.f8[0], &dst_f24.f8[0]);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[1], &src2_f24.f8[1], &dst_f24.f8[1]);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[2], &src2_f24.f8[2], &dst_f24.f8[2]);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void bitwise_or_pln3_pkd3_hip_tensor(T *srcPtr1,
                                                T *srcPtr2,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
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

    d_float24 src1_f24, src2_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr1 + srcIdx, srcStridesNCH.y, &src1_f24);
    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr2 + srcIdx, srcStridesNCH.y, &src2_f24);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[0], &src2_f24.f8[0], &dst_f24.f8[0]);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[1], &src2_f24.f8[1], &dst_f24.f8[1]);
    bitwise_or_hip_compute(srcPtr1, &src1_f24.f8[2], &src2_f24.f8[2], &dst_f24.f8[2]);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_bitwise_or_tensor(T *srcPtr1,
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

    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(bitwise_or_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(bitwise_or_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(bitwise_or_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(bitwise_or_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}