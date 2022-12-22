#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void glitch_pln_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      int channelDst,
                                      unsigned int *x_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_r,
                                      unsigned int *y_offset_g,
                                      unsigned int *y_offset_b,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    printf(" \n %d  %d  %d", id_x, id_y, id_z);
    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }
    int x_r, y_r, x_g, y_g, x_b, y_b;

    x_r = id_x + x_offset_r[id_z];
    y_r = id_y + y_offset_r[id_z];

    x_g = id_x + x_offset_g[id_z];
    y_g = id_y + y_offset_g[id_z];

    x_b = id_x + x_offset_b[id_z];
    y_b = id_x + y_offset_b[id_z];

    printf("\n %d %d", x_r, y_r);
    int srcIdx, dstIdx;
    int srcIdx_r, srcIdx_g, srcIdx_b, dstIdx_r, dstIdx_g, dstIdx_b;
    d_float8 srcR_f8, srcG_f8, srcB_f8;
    d_float24 pix_f24;

    srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);

    printf("srcIdx :%d \n", srcIdx);
    printf("dstIdx :%d \n", dstIdx);
    if((y_r >= 0) && (y_r <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        srcIdx_r = (id_z * srcStridesNCH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        printf("srcIdx_r :%d \n", srcIdx_r);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_r, &srcR_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_r, &srcR_f8);

    }

    // if((y_r >= 0) && (y_r <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    // {
    //     srcIdx_g = (id_z * srcStridesNCH.x) + srcStridesNCH.y + ((y_g + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_g + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    //     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_g, &srcG_f8);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_g, &srcR_f8);
    // }

    // if((y_r >= 0) && (y_r <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    // {
    //     srcIdx_b = (id_z * srcStridesNCH.x) + (srcStridesNCH.y * 2) + ((y_b + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_b + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    //     rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_b, &srcB_f8);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_b, &srcR_f8);
    // }
}

template <typename T>
__global__ void glitch_pln3_pkd3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      int channelDst,
                                      unsigned int *x_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_r,
                                      unsigned int *y_offset_g,
                                      unsigned int *y_offset_b,
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
    d_float24 pix_f24, pix_temp;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    //rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);

    int x_r, y_r, x_g, y_g, x_b, y_b;

    x_r = id_x + x_offset_r[id_z];
    y_r = id_y + y_offset_r[id_z];

    x_g = id_x + x_offset_g[id_z];
    y_g = id_y + y_offset_g[id_z];

    x_b = id_x + x_offset_b[id_z];
    y_b = id_x + y_offset_b[id_z];

    if((y_r >= 0) && (y_r <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdx_r = (id_z * srcStridesNCH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx_r, srcStridesNCH.y, &pix_temp);
        pix_f24.f4[0] = pix_temp.f4[0];
        pix_f24.f4[1] = pix_temp.f4[1];
    }
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}


template <typename T>
RppStatus hip_exec_glitch_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *x_offset_r,
                                     Rpp32u *y_offset_r,
                                     Rpp32u *x_offset_g,
                                     Rpp32u *y_offset_g,
                                     Rpp32u *x_offset_b,
                                     Rpp32u *y_offset_b,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);
    std::cerr << "GLITCH TENSOR\n";
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           x_offset_r,
                           y_offset_r,
                           x_offset_g,
                           y_offset_g,
                           x_offset_b,
                           y_offset_b,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pln3_pkd3_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           x_offset_r,
                           y_offset_r,
                           x_offset_g,
                           y_offset_g,
                           x_offset_b,
                           y_offset_b,
                           roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}