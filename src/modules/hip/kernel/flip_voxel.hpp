#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void flip_ncdhw_tensor(T *srcPtr,
                                  uint3 srcStridesCDH,
                                  T *dstPtr,
                                  uint3 dstStridesCDH,
                                  int channels,
                                  uint3 mirrorXYZ,
                                  RpptROI3DPtr roiGenericPtrSrc,
                                  int batchIndex)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    int dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;
    int zFactor = ((mirrorXYZ.z) ? (roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) : (id_z + roiGenericPtrSrc->xyzwhdROI.xyz.z)) * srcStridesCDH.y;
    int yFactor = ((mirrorXYZ.y) ? (roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) : (id_y + roiGenericPtrSrc->xyzwhdROI.xyz.y)) * srcStridesCDH.z;
    int xFactor = id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x;
    if (mirrorXYZ.x)
    {
        // To handle the case when trying to load from invalid memory location when width is not a multiple of 8
        if((batchIndex == 0) && (id_x + 8 > roiGenericPtrSrc->xyzwhdROI.roiWidth))
        {
            bool yCheck = ((mirrorXYZ.y && id_y == roiGenericPtrSrc->xyzwhdROI.roiHeight - 1) || (!mirrorXYZ.y && id_y == 0));
            bool zCheck = ((mirrorXYZ.z && id_z == roiGenericPtrSrc->xyzwhdROI.roiDepth - 1) || (!mirrorXYZ.z && id_z == 0));
            if(yCheck && zCheck)
            {
                xFactor = roiGenericPtrSrc->xyzwhdROI.xyz.x;
                dstIdx -= (id_x + 8 - roiGenericPtrSrc->xyzwhdROI.roiWidth);
            }
            else
            {
                xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8);
            }
        }
        else
        {
            xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8);
        }
        int srcIdx = zFactor + yFactor + xFactor;
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
        int srcIdx = zFactor + yFactor + xFactor;
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
                                  RpptROI3DPtr roiGenericPtrSrc,
                                  int batchIndex)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    int dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;
    int zFactor = ((mirrorXYZ.z) ? (roiGenericPtrSrc->xyzwhdROI.roiDepth - 1 - id_z) : (id_z + roiGenericPtrSrc->xyzwhdROI.xyz.z))  * srcStridesDH.x;
    int yFactor = ((mirrorXYZ.y) ? (roiGenericPtrSrc->xyzwhdROI.roiHeight - 1 - id_y) : (id_y + roiGenericPtrSrc->xyzwhdROI.xyz.y)) * srcStridesDH.y;
    int xFactor =  (id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x) * 3;
    if (mirrorXYZ.x)
    {
        // To handle the case when trying to load from invalid memory location when width is not a multiple of 8
        if((batchIndex == 0) && (id_x + 8 > roiGenericPtrSrc->xyzwhdROI.roiWidth))
        {
            bool yCheck = ((mirrorXYZ.y && id_y == roiGenericPtrSrc->xyzwhdROI.roiHeight - 1) || (!mirrorXYZ.y && id_y == 0));
            bool zCheck = ((mirrorXYZ.z && id_z == roiGenericPtrSrc->xyzwhdROI.roiDepth - 1) || (!mirrorXYZ.z && id_z == 0));
            if(yCheck && zCheck)
            {
                xFactor = roiGenericPtrSrc->xyzwhdROI.xyz.x * 3;
                dstIdx -= (id_x + 8 - roiGenericPtrSrc->xyzwhdROI.roiWidth) * 3;
            }
            else
            {
                xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8) * 3;
            }
        }
        else
        {
            xFactor = (roiGenericPtrSrc->xyzwhdROI.xyz.x + roiGenericPtrSrc->xyzwhdROI.roiWidth - id_x - 8) * 3;
        }
        int srcIdx = zFactor + yFactor + xFactor;
        d_float24 pix_f24;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
    }
    else
    {
        int srcIdx = zFactor + yFactor + xFactor;
        d_float24 pix_f24;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3_mirror(srcPtr + srcIdx, &pix_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
    }
}

template <typename T>
RppStatus hip_exec_flip_voxel_tensor(T *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     T *dstPtr,
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
        int globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[3];               // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[2];               // D - depth (z direction)

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(flip_ncdhw_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                               dstGenericDescPtr->dims[1],
                               make_uint3(horizontalTensor[batchCount], verticalTensor[batchCount], depthTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount],
                               batchCount);
        }
    }
    else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
    {
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];                   // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];                   // D - depth (z direction)

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(flip_ndhwc_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               make_uint3(horizontalTensor[batchCount], verticalTensor[batchCount], depthTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount],
                               batchCount);
        }
    }

    return RPP_SUCCESS;
}