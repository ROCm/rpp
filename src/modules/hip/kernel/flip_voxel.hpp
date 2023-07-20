#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void flip_ncdhw_tensor(T *srcPtr,
                                  uint3 srcStridesCDH,
                                  T *dstPtr,
                                  uint3 dstStridesCDH,
                                  int channels,
                                  uint3 mirrorXYZ,
                                  RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;
    uint xFactor =  id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x;
    uint yFactor = (id_y + roiGenericPtrSrc->xyzwhdROI.xyz.y) * srcStridesCDH.z;
    uint zFactor = (id_z + roiGenericPtrSrc->xyzwhdROI.xyz.z) * srcStridesCDH.y;

    if (mirrorXYZ.y)
        yFactor = (roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) * srcStridesCDH.z;
    if (mirrorXYZ.z)
        zFactor = (roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) * srcStridesCDH.y;
    if (mirrorXYZ.x)
    {
        xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8);
        uint srcIdx = zFactor + yFactor + xFactor;
        d_float8 pix_f8;
        for(int c = 0; c < channels; c++)
        {
            rpp_hip_load8_and_unpack_to_float8_mirror(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
            srcIdx += srcStridesCDH.x;
            dstIdx += dstStridesCDH.x;
        }
    }
    else
    {
        uint srcIdx = zFactor + yFactor + xFactor;
        d_float8 pix_f8;
        for(int c = 0; c < channels; c++)
        {
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
            srcIdx += srcStridesCDH.x;
            dstIdx += dstStridesCDH.x;
        }
    }
}

template <typename T>
__global__ void flip_ndhwc_tensor(T *srcPtr,
                                  uint2 srcStridesDH,
                                  T *dstPtr,
                                  uint2 dstStridesDH,
                                  uint3 mirrorXYZ,
                                  RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;
    uint xFactor =  (id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x) * 3;
    uint yFactor = (id_y + roiGenericPtrSrc->xyzwhdROI.xyz.y) * srcStridesDH.y;
    uint zFactor = (id_z + roiGenericPtrSrc->xyzwhdROI.xyz.z) * srcStridesDH.x;

    if (mirrorXYZ.y)
        yFactor = (roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) * srcStridesDH.y;
    if (mirrorXYZ.z)
        zFactor = (roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) * srcStridesDH.x;
    if (mirrorXYZ.x)
    {
        xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8) * 3;
        uint srcIdx = zFactor + yFactor + xFactor;
        d_float24 pix_f24;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
    }
    else
    {
        uint srcIdx = zFactor + yFactor + xFactor;
        d_float24 pix_f24;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
    }
}

RppStatus hip_exec_flip_voxel_tensor(Rpp32f *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32u *horizontalTensor,
                                     Rpp32u *verticalTensor,
                                     Rpp32u *depthTensor,
                                     RpptRoi3DType roiType,
                                     rpp::Handle& handle)
{
    if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[3];               // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[2];               // D - depth (z direction)

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(flip_ncdhw_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                               dstGenericDescPtr->dims[1],
                               make_uint3(horizontalTensor[batchCount], verticalTensor[batchCount], depthTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount]);
        }
    }
    else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];                   // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];                   // D - depth (z direction)

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(flip_ndhwc_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               make_uint3(horizontalTensor[batchCount], verticalTensor[batchCount], depthTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount]);
        }
    }

    return RPP_SUCCESS;
}