#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// -------------------- Set 0 - resize_mirror_normalize device helpers --------------------

__device__ void resize_mirror_normalize_srclocs_hip_compute(float4 locSrcComponent_f4, d_float8 *locSrcPtr_f8)
{
    d_float8 increment_f8;
    increment_f8.f4[0] = make_float4(0, 1, 2, 3);
    increment_f8.f4[1] = make_float4(4, 5, 6, 7);
    locSrcPtr_f8->f4[0] = locSrcComponent_f4 + increment_f8.f4[0];
    locSrcPtr_f8->f4[1] = locSrcComponent_f4 + increment_f8.f4[1];
}

__device__ void resize_mirror_normalize_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float2 locDst_f2, locSrc_f2;
    float wRatio = float(srcRoiPtr_i4->z) / dstDimsWH->x;
    float hRatio = float(srcRoiPtr_i4->w) / dstDimsWH->y;
    float wOffset = (wRatio - 1) * 0.5f;
    float hOffset = (hRatio - 1) * 0.5f;
    locDst_f2.x = (float) id_x;
    locDst_f2.y = (float) id_y;
    locSrc_f2.x = locDst_f2.x * hRatio + hOffset;
    locSrc_f2.y = locDst_f2.y * wRatio + wOffset;
    resize_mirror_normalize_srclocs_hip_compute((float4)locSrc_f2.x, &(locSrc_f16->f8[0]));    // Compute 8 locSrcX
    resize_mirror_normalize_srclocs_hip_compute((float4)locSrc_f2.y, &(locSrc_f16->f8[1]));    // Compute 8 locSrcY
}

// -------------------- Set 1 - Bilinear Interpolation --------------------

// template <typename T>
// __global__ void resize_mirror_normalize_bilinear_pkd_tensor(T *srcPtr,
//                                                 uint2 srcStridesNH,
//                                                 T *dstPtr,
//                                                 uint2 dstStridesNH,
//                                                 uint2 dstDimsWH,
//                                                 RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

//     int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
//     d_float16 locSrc_f16;
//     resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &locSrc_f16);

//     d_float24 dst_f24;
//     rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
//     rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
// }

template <typename T>
__global__ void resize_mirror_normalize_bilinear_pln_tensor(T *srcPtr,
                                                            uint3 srcStridesNCH,
                                                            T *dstPtr,
                                                            uint3 dstStridesNCH,
                                                            RpptImagePatchPtr dstImgSize,
                                                            int channelsDst,
                                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

// template <typename T>
// __global__ void resize_mirror_normalize_bilinear_pkd3_pln3_tensor(T *srcPtr,
//                                                       uint2 srcStridesNH,
//                                                       T *dstPtr,
//                                                       uint3 dstStridesNCH,
//                                                       uint2 dstDimsWH,
//                                                       d_float6 *affineTensorPtr,
//                                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNH.x);
//     uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

//     d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
//     int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
//     d_float16 locSrc_f16;
//     resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

//     d_float24 dst_f24;
//     rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
//     rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
// }

// template <typename T>
// __global__ void resize_mirror_normalize_bilinear_pln3_pkd3_tensor(T *srcPtr,
//                                                       uint3 srcStridesNCH,
//                                                       T *dstPtr,
//                                                       uint2 dstStridesNH,
//                                                       uint2 dstDimsWH,
//                                                       d_float6 *affineTensorPtr,
//                                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * srcStridesNCH.x);
//     uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

//     d_float6 affineMatrix_f6 = affineTensorPtr[id_z];
//     int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
//     d_float16 locSrc_f16;
//     resize_mirror_normalize_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, &affineMatrix_f6, &locSrc_f16);

//     d_float24 dst_f24;
//     rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
//     rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
// }

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_resize_mirror_normalize_tensor(T *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 T *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSizes,
                                                 RpptInterpolationType interpolationType,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            // hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pkd_tensor,
            //                 dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
            //                 dim3(localThreads_x, localThreads_y, localThreads_z),
            //                 0,
            //                 handle.GetStream(),
            //                 srcPtr,
            //                 make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
            //                 dstPtr,
            //                 make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            //                 make_uint2(dstDescPtr->w, dstDescPtr->h),
            //                 (d_float6 *)affineTensorPtr,
            //                 roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstImgSizes,
                               dstDescPtr->c,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            // if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            // {
            //     hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pkd3_pln3_tensor,
            //                     dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
            //                     dim3(localThreads_x, localThreads_y, localThreads_z),
            //                     0,
            //                     handle.GetStream(),
            //                     srcPtr,
            //                     make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
            //                     dstPtr,
            //                     make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
            //                     make_uint2(dstDescPtr->w, dstDescPtr->h),
            //                     (d_float6 *)affineTensorPtr,
            //                     roiTensorPtrSrc);
            // }
            // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            // {
            //     globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            //     hipLaunchKernelGGL(resize_mirror_normalize_bilinear_pln3_pkd3_tensor,
            //                     dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
            //                     dim3(localThreads_x, localThreads_y, localThreads_z),
            //                     0,
            //                     handle.GetStream(),
            //                     srcPtr,
            //                     make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
            //                     dstPtr,
            //                     make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            //                     make_uint2(dstDescPtr->w, dstDescPtr->h),
            //                     (d_float6 *)affineTensorPtr,
            //                     roiTensorPtrSrc);
            // }
        }
    }

    return RPP_SUCCESS;
}
