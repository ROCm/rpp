#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - kernels for layout toggle (PKD3->PLN3, PLN3->PKD3) --------------------
template <typename T>
__global__ void convert_pkd3_pln3_hip_tensor(T *srcPtr,
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

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 dst_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void convert_pln3_pkd3_hip_tensor(T *srcPtr,
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

    d_float24 dst_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 1 - Erase main kernels --------------------
// Rpp8u datatype
__global__ void erase_pkd_hip_tensor(Rpp8u *dstPtr,
                                     uint2 dstStridesNH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     Rpp8u *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uchar3 pixel;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            temp *= 3;
            pixel.x = colorsTensor[temp];
            pixel.y = colorsTensor[temp + 1];
            pixel.z = colorsTensor[temp + 2];
            *(uchar3 *)(dstPtr + dstIdx) = pixel;
            break;
        }
    }
}

// Rpp8s datatype
__global__ void erase_pkd_hip_tensor(Rpp8s *dstPtr,
                                     uint2 dstStridesNH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     Rpp8s *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    d_schar3_s pixel;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            temp *= 3;
            pixel.sc1[0] = colorsTensor[temp];
            pixel.sc1[1] = colorsTensor[temp + 1];
            pixel.sc1[2] = colorsTensor[temp + 2];
            *(d_schar3_s *)(dstPtr + dstIdx) = pixel;
            break;
        }
    }
}

// Half datatype
__global__ void erase_pkd_hip_tensor(half *dstPtr,
                                     uint2 dstStridesNH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     half *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    d_half3_s pixel;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            temp *= 3;
            pixel.h1[0] = colorsTensor[temp];
            pixel.h1[1] = colorsTensor[temp + 1];
            pixel.h1[2] = colorsTensor[temp + 2];
            *(d_half3_s *)(dstPtr + dstIdx) = pixel;
            break;
        }
    }
}

// Rpp32f datatype
__global__ void erase_pkd_hip_tensor(Rpp32f *dstPtr,
                                     uint2 dstStridesNH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     Rpp32f *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    float3 pixel;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            temp *= 3;
            pixel.x = colorsTensor[temp];
            pixel.y = colorsTensor[temp + 1];
            pixel.z = colorsTensor[temp + 2];
            *(float3 *)(dstPtr + dstIdx) = pixel;
            break;
        }
    }
}

template <typename T>
__global__ void erase_pln_hip_tensor(T *dstPtr,
                                     uint3 dstStridesNCH,
                                     RpptRoiLtrb *anchorBoxInfoTensor,
                                     T *colorsTensor,
                                     Rpp32u *numBoxesTensor,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            *(T *)(dstPtr + dstIdx) = colorsTensor[temp];
            break;
        }
    }
}

template <typename T>
__global__ void erase_pln3_hip_tensor(T *dstPtr,
                                      uint3 dstStridesNCH,
                                      RpptRoiLtrb *anchorBoxInfoTensor,
                                      T *colorsTensor,
                                      Rpp32u *numBoxesTensor,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // check if the co-ordinates is within any user defined box
    for (int i = 0; i < numBoxes; i++)
    {
        int temp = (id_z * numBoxes) + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x && id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            temp *= 3;
            *(T *)(dstPtr + dstIdx) = colorsTensor[temp];
            dstIdx += dstStridesNCH.y;
            *(T *)(dstPtr + dstIdx) = colorsTensor[temp + 1];
            dstIdx += dstStridesNCH.y;
            *(T *)(dstPtr + dstIdx) = colorsTensor[temp + 2];
            break;
        }
    }
}

// -------------------- Set 2 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_erase_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptRoiLtrb *anchorBoxInfoTensor,
                                T *colorsTensor,
                                Rpp32u *numBoxesTensor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(erase_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           colorsTensor,
                           numBoxesTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1)
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(erase_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           colorsTensor,
                           numBoxesTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3)
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(erase_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           colorsTensor,
                           numBoxesTensor,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            globalThreads_x = dstDescPtr->w;
            hipLaunchKernelGGL(erase_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               colorsTensor,
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            globalThreads_x = dstDescPtr->w;
            hipLaunchKernelGGL(erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               colorsTensor,
                               numBoxesTensor,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}