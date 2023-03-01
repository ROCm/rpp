#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void cmn_hip_compute(uchar *srcPtr, float *dstPtr, d_float8 *pix_f8, d_float8 *cmnParams_f8)
{
    pix_f8->f4[0] = (pix_f8->f4[0] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
    pix_f8->f4[1] = (pix_f8->f4[1] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
}

__device__ void cmn_hip_compute(uchar *srcPtr, half *dstPtr, d_float8 *pix_f8, d_float8 *cmnParams_f8)
{
    pix_f8->f4[0] = (pix_f8->f4[0] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
    pix_f8->f4[1] = (pix_f8->f4[1] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
}

__device__ void cmn_hip_compute(uchar *srcPtr, uchar *dstPtr, d_float8 *pix_f8, d_float8 *cmnParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255((pix_f8->f4[0] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255((pix_f8->f4[1] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1]);
}

__device__ void cmn_hip_compute(float *srcPtr, float *dstPtr, d_float8 *pix_f8, d_float8 *cmnParams_f8)
{
    pix_f8->f4[0] = (pix_f8->f4[0] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
    pix_f8->f4[1] = (pix_f8->f4[1] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
}

__device__ void cmn_hip_compute(schar *srcPtr, schar *dstPtr, d_float8 *pix_f8, d_float8 *cmnParams_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255((pix_f8->f4[0] + (float4)128) * cmnParams_f8->f4[0] +  cmnParams_f8->f4[1]) - (float4)128;
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255((pix_f8->f4[1] + (float4)128) * cmnParams_f8->f4[0] +  cmnParams_f8->f4[1]) - (float4)128;
}
__device__ void cmn_hip_compute(half *srcPtr, half *dstPtr, d_float8 *pix_f8, d_float8 *cmnParams_f8)
{
    pix_f8->f4[0] = (pix_f8->f4[0] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
    pix_f8->f4[1] = (pix_f8->f4[1] * cmnParams_f8->f4[0]) + cmnParams_f8->f4[1];
}

template <typename T, typename U>
__global__ void crop_mirror_normalize_pkd_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 U *dstPtr,
                                                 uint2 dstStridesNH,
                                                 float *offsetTensor,
                                                 float *multiplierTensor,
                                                 unsigned int *mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint srcIdx;
    d_float24 pix_f24;
    if(mirrorTensor[id_z] == 1)
    {
        // Temporary change - To handle the case when trying to load from invalid memory location when width is not a multiple of 8
        // This additional condition will be removed once the changes for adding an additional offset memory to allocated input memory are done in MIVisionX
        if((id_z == 0) && (id_y == 0) && (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        {
            srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3;
            dstIdx -= (id_x + 8 - roiTensorPtrSrc[id_z].xywhROI.roiWidth) * 3;
        }
        else
        {
            srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x - 8) * 3;
        }
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    }
    else
    {
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    }

    int cmnParamLoc = id_z * 3;
    int3 cmnParamLocs = make_int3(cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2);
    d_float8 cmnParamsR_f8, cmnParamsG_f8, cmnParamsB_f8;
    cmnParamsR_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.x];      // Get multiplier for R channel
    cmnParamsR_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.x];          // Get offset for R channel
    cmnParamsG_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.y];  // Get multiplier for G channel
    cmnParamsG_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.y];      // Get offset for G channel
    cmnParamsB_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.z];  // Get multiplier for B channel
    cmnParamsB_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.z];      // Get offset for B channel

    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[0], &cmnParamsR_f8);
    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[1], &cmnParamsG_f8);
    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[2], &cmnParamsB_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T, typename U>
__global__ void crop_mirror_normalize_pln_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 U *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 float *offsetTensor,
                                                 float *multiplierTensor,
                                                 uint *mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    int cmnParamLoc = id_z * channelsDst;
    int3 cmnParamLocs = make_int3(cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2);
    d_float8 pix_f8, cmnParams_f8;
    cmnParams_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.x];  // Get multiplier for R channel
    cmnParams_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.x];      // Get offset for R channel

    if(mirrorTensor[id_z] == 1)
    {
        // Temporary change - To handle the case when trying to load from invalid memory location when width is not a multiple of 8
        // This additional condition will be removed once the changes for adding an additional offset memory to allocated input memory are done in MIVisionX
        if((id_z == 0) && (id_y == 0) && (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        {
            srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + roiTensorPtrSrc[id_z].xywhROI.xy.x;
            dstIdx -= (id_x + 8 - roiTensorPtrSrc[id_z].xywhROI.roiWidth);
        }
        else
        {
            srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x - 8);
        }

        rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
        cmn_hip_compute(srcPtr, dstPtr, &pix_f8, &cmnParams_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3)
        {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            cmnParams_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.y];  // Get multiplier for G channel
            cmnParams_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.y];      // Get offset for G channel

            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            cmn_hip_compute(srcPtr, dstPtr, &pix_f8, &cmnParams_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            cmnParams_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.z];  // Get multiplier for B channel
            cmnParams_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.z];      // Get offset for B channel

            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            cmn_hip_compute(srcPtr, dstPtr, &pix_f8, &cmnParams_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
    else
    {
        srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        cmn_hip_compute(srcPtr, dstPtr, &pix_f8, &cmnParams_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        if (channelsDst == 3)
        {
            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            cmnParams_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.y];  // Get multiplier for G channel
            cmnParams_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.y];      // Get offset for G channel

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            cmn_hip_compute(srcPtr, dstPtr, &pix_f8, &cmnParams_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

            srcIdx += srcStridesNCH.y;
            dstIdx += dstStridesNCH.y;

            cmnParams_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.z];  // Get multiplier for B channel
            cmnParams_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.z];      // Get offset for B channel

            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            cmn_hip_compute(srcPtr, dstPtr, &pix_f8, &cmnParams_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
        }
    }
}

template <typename T, typename U>
__global__ void crop_mirror_normalize_pkd3_pln3_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       U *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       float *offsetTensor,
                                                       float *multiplierTensor,
                                                       uint *mirrorTensor,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint srcIdx;
    d_float24 pix_f24;
    if(mirrorTensor[id_z] == 1)
    {
        // Temporary change - To handle the case when trying to load from invalid memory location when width is not a multiple of 8
        // This additional condition will be removed once the changes for adding an additional offset memory to allocated input memory are done in MIVisionX
        if((id_z == 0) && (id_y == 0) && (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        {
            srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3;
            dstIdx -= (id_x + 8 - roiTensorPtrSrc[id_z].xywhROI.roiWidth);
        }
        else
        {
            srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x - 8) * 3;
        }
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
    }
    else
    {
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    }

    int cmnParamLoc = id_z * 3;
    int3 cmnParamLocs = make_int3(cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2);
    d_float8 cmnParamsR_f8, cmnParamsG_f8, cmnParamsB_f8;
    cmnParamsR_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.x];      // Get multiplier for R channel
    cmnParamsR_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.x];          // Get offset for R channel
    cmnParamsG_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.y];  // Get multiplier for G channel
    cmnParamsG_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.y];      // Get offset for G channel
    cmnParamsB_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.z];  // Get multiplier for B channel
    cmnParamsB_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.z];      // Get offset for B channel

    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[0], &cmnParamsR_f8);
    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[1], &cmnParamsG_f8);
    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[2], &cmnParamsB_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T, typename U>
__global__ void crop_mirror_normalize_pln3_pkd3_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       U *dstPtr,
                                                       uint2 dstStridesNH,
                                                       float *offsetTensor,
                                                       float *multiplierTensor,
                                                       uint *mirrorTensor,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint srcIdx;
    d_float24 pix_f24;
    if(mirrorTensor[id_z] == 1)
    {
        // Temporary change - To handle the case when trying to load from invalid memory location when width is not a multiple of 8
        // This additional condition will be removed once the changes for adding an additional offset memory to allocated input memory are done in MIVisionX
        if((id_z == 0) && (id_y == 0) && (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth)
        {
            srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + roiTensorPtrSrc[id_z].xywhROI.xy.x;
            dstIdx -= (id_x + 8 - roiTensorPtrSrc[id_z].xywhROI.roiWidth) * 3;
        }
        else
        {
            srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x - 8);
        }
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    }
    else
    {
        srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    }

    int cmnParamLoc = id_z * 3;
    int3 cmnParamLocs = make_int3(cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2);
    d_float8 cmnParamsR_f8, cmnParamsG_f8, cmnParamsB_f8;
    cmnParamsR_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.x];      // Get multiplier for R channel
    cmnParamsR_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.x];          // Get offset for R channel
    cmnParamsG_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.y];  // Get multiplier for G channel
    cmnParamsG_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.y];      // Get offset for G channel
    cmnParamsB_f8.f4[0] = (float4)multiplierTensor[cmnParamLocs.z];  // Get multiplier for B channel
    cmnParamsB_f8.f4[1] = (float4)offsetTensor[cmnParamLocs.z];      // Get offset for B channel

    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[0], &cmnParamsR_f8);
    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[1], &cmnParamsG_f8);
    cmn_hip_compute(srcPtr, dstPtr, &pix_f24.f8[2], &cmnParamsB_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T, typename U>
RppStatus hip_exec_crop_mirror_normalize_tensor(T *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                U *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(crop_mirror_normalize_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.float3Arr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.float3Arr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if(srcDescPtr->c == 1)
        {
            hipLaunchKernelGGL(crop_mirror_normalize_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                               roiTensorPtrSrc);
        }
        else if(srcDescPtr->c == 3)
        {
            hipLaunchKernelGGL(crop_mirror_normalize_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               handle.GetInitHandle()->mem.mgpu.float3Arr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.float3Arr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(crop_mirror_normalize_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.float3Arr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.float3Arr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(crop_mirror_normalize_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.float3Arr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.float3Arr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}