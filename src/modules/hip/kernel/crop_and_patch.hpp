#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"


template <typename T>
__global__ void crop_and_patch_pkd_hip_tensor(T *srcPtr1,
                                              T *srcPtr2,
                                              uint2 srcStridesNH,
                                              T *dstPtr,
                                              uint2 dstStridesNH,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptROIPtr cropTensorPtrSrc,
                                              RpptROIPtr patchTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
        dstPtr[dstIdx + 1] = srcPtr1[srcIdx1 + 1];
        dstPtr[dstIdx + 2] = srcPtr1[srcIdx1 + 2];
    }
    else
    {
        uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
        dstPtr[dstIdx + 1] = srcPtr2[srcIdx2 + 1];
        dstPtr[dstIdx + 2] = srcPtr2[srcIdx2 + 2];
    }
}

template <typename T>
__global__ void crop_and_patch_pln_hip_tensor(T *srcPtr1,
                                              T *srcPtr2,
                                              uint3 srcStridesNCH,
                                              T *dstPtr,
                                              uint3 dstStridesNCH,
                                              int channelsDst,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptROIPtr cropTensorPtrSrc,
                                              RpptROIPtr patchTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNCH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x);
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
        if (channelsDst == 3)
        {
            srcIdx1 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            dstPtr[dstIdx] = srcPtr1[srcIdx1];

            srcIdx1 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            dstPtr[dstIdx] = srcPtr1[srcIdx1];
        }
    }
    else
    {
        uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
        if (channelsDst == 3)
        {
            srcIdx2 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            dstPtr[dstIdx] = srcPtr2[srcIdx2];

            srcIdx2 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            dstPtr[dstIdx] = srcPtr2[srcIdx2];
        }
    }
}

template <typename T>
__global__ void crop_and_patch_pkd3_pln3_hip_tensor(T *srcPtr1,
                                                    T *srcPtr2,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptROIPtr cropTensorPtrSrc,
                                                    RpptROIPtr patchTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
        dstPtr[dstIdx + dstStridesNCH.y] = srcPtr1[srcIdx1 + 1];
        dstPtr[dstIdx + 2 * dstStridesNCH.y] = srcPtr1[srcIdx1 + 2];
    }
    else
    {
        uint srcIdx2 = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
        dstPtr[dstIdx + dstStridesNCH.y] = srcPtr2[srcIdx2 + 1];
        dstPtr[dstIdx + 2 * dstStridesNCH.y] = srcPtr2[srcIdx2 + 2];
    }
}

template <typename T>
__global__ void crop_and_patch_pln3_pkd3_hip_tensor(T *srcPtr1,
                                                    T *srcPtr2,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptROIPtr cropTensorPtrSrc,
                                                    RpptROIPtr patchTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.roiWidth));
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNCH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x);
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
        dstPtr[dstIdx + 1] = srcPtr1[srcIdx1 + srcStridesNCH.y];
        dstPtr[dstIdx + 1] = srcPtr1[srcIdx1 + 2 * srcStridesNCH.y];
    }
    else
    {
        uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
        dstPtr[dstIdx + 1] = srcPtr2[srcIdx2 + srcStridesNCH.y];
        dstPtr[dstIdx + 1] = srcPtr2[srcIdx2 + 2 * srcStridesNCH.y];
    }
}


template <typename T>
RppStatus hip_exec_crop_and_patch_tensor(T *srcPtr1,
                                         T *srcPtr2,
                                         RpptDescPtr srcDescPtr,
                                         T *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptROIPtr cropTensorPtr,
                                         RpptROIPtr patchTensorPtr,
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
        hipLaunchKernelGGL(crop_and_patch_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           cropTensorPtr,
                           patchTensorPtr);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(crop_and_patch_pln_hip_tensor,
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
                           roiTensorPtrSrc,
                           cropTensorPtr,
                           patchTensorPtr);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(crop_and_patch_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc,
                               cropTensorPtr,
                               patchTensorPtr);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(crop_and_patch_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1,
                               srcPtr2,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc,
                               cropTensorPtr,
                               patchTensorPtr);
        }
    }

    return RPP_SUCCESS;
}
