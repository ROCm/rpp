#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"


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
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    // check if the co-ordinates is within the patch region
    bool rowCheck = (id_y >= patchTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y < (patchTensorPtrSrc[id_z].xywhROI.xy.y + patchTensorPtrSrc[id_z].xywhROI.roiHeight));
    bool colCheck = (id_x >= patchTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x < (patchTensorPtrSrc[id_z].xywhROI.xy.x + patchTensorPtrSrc[id_z].xywhROI.roiWidth));
    if(rowCheck && colCheck)
    {
        uint srcIdx1 = (id_z * srcStridesNCH.x) + (id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z + (id_x - patchTensorPtrSrc[id_z].xywhROI.xy.x + cropTensorPtrSrc[id_z].xywhROI.xy.x);
        uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
        uint patchEnd = patchTensorPtrSrc[id_z].xywhROI.xy.x + patchTensorPtrSrc[id_z].xywhROI.roiWidth;

        d_float8 pix_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &pix_f8);
        uint srcIdx2;
        if((id_x + 8) >= patchEnd)
        {
            srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (patchEnd + roiTensorPtrSrc[id_z].xywhROI.xy.x);
            uint tempLoc = srcIdx2;
            for(uint i = patchEnd - id_x; i < 8; i++, tempLoc++)
                pix_f8.f1[i] = static_cast<float>(srcPtr2[tempLoc]);
        }
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        if (channelsDst == 3)
        {
            srcIdx1 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &pix_f8);
            if((id_x + 8) >= patchEnd)
            {
                srcIdx2 += srcStridesNCH.y;
                uint tempLoc = srcIdx2;
                for(uint i = patchEnd - id_x; i < 8; i++, tempLoc++)
                    pix_f8.f1[i] = static_cast<float>(srcPtr2[tempLoc]);
            }
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx1 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &pix_f8);
            if((id_x + 8) >= patchEnd)
            {
                srcIdx2 += srcStridesNCH.y;
                uint tempLoc = srcIdx2;
                for(uint i = patchEnd - id_x; i < 8; i++, tempLoc++)
                    pix_f8.f1[i] = static_cast<float>(srcPtr2[tempLoc]);
            }
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
    else
    {
        uint srcIdx2 = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
        uint patchStart = patchTensorPtrSrc[id_z].xywhROI.xy.x;

        d_float8 pix_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &pix_f8);
        uint srcIdx1, patchBegin;
        if(rowCheck && (id_x + 8) >= patchStart)
        {
            srcIdx1 = (id_z * srcStridesNCH.x) + ((id_y - patchTensorPtrSrc[id_z].xywhROI.xy.y + cropTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + cropTensorPtrSrc[id_z].xywhROI.xy.x;
            patchBegin = patchStart - id_x;
            uint tempLoc = srcIdx1;
            for(uint i = patchBegin; i < 8; i++, tempLoc++)
                pix_f8.f1[i] = static_cast<float>(srcPtr1[tempLoc]);
        }
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        if (channelsDst == 3)
        {
            srcIdx2 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &pix_f8);
            if(rowCheck && (id_x + 8) >= patchStart)
            {
                srcIdx1 += srcStridesNCH.y;
                uint tempLoc = srcIdx1;
                for(uint i = patchBegin; i < 8; i++, tempLoc++)
                    pix_f8.f1[i] = static_cast<float>(srcPtr1[tempLoc]);
            }
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx2 += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;
            rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &pix_f8);
            if(rowCheck && (id_x + 8) >= patchStart)
            {
                srcIdx1 += srcStridesNCH.y;
                uint tempLoc = srcIdx1;
                for(uint i = patchBegin; i < 8; i++, tempLoc++)
                    pix_f8.f1[i] = static_cast<float>(srcPtr1[tempLoc]);
            }
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
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

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
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

    return RPP_SUCCESS;
}
