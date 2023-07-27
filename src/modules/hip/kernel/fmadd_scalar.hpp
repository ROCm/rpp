#include <hip/hip_runtime.h>
#include <omp.h>
#include "rpp_hip_common.hpp"

__device__ void fmaf_scalar_hip_compute(d_float8 *val_f8, float2 *fmaddParams_f2)
{
    val_f8->f1[0] = fmaf(val_f8->f1[0], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[1] = fmaf(val_f8->f1[1], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[2] = fmaf(val_f8->f1[2], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[3] = fmaf(val_f8->f1[3], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[4] = fmaf(val_f8->f1[4], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[5] = fmaf(val_f8->f1[5], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[6] = fmaf(val_f8->f1[6], fmaddParams_f2->x, fmaddParams_f2->y);
    val_f8->f1[7] = fmaf(val_f8->f1[7], fmaddParams_f2->x, fmaddParams_f2->y);
}

__device__ void fmaf_scalar_hip_compute(d_float24 *val_f24, float2 *fmaddParams_f2)
{
    fmaf_scalar_hip_compute(&(val_f24->f8[0]), fmaddParams_f2);
    fmaf_scalar_hip_compute(&(val_f24->f8[1]), fmaddParams_f2);
    fmaf_scalar_hip_compute(&(val_f24->f8[2]), fmaddParams_f2);
}

__global__ void fmadd_scalar_ncdhw_tensor(float *srcPtr,
                                          uint3 srcStridesCDH,
                                          float *dstPtr,
                                          uint3 dstStridesCDH,
                                          int channels,
                                          float2 fmaddParams_f2,
                                          RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    uint srcIdx = ((id_z + roiGenericPtrSrc->xyzwhdROI.xyz.z) * srcStridesCDH.y) + ((id_y + roiGenericPtrSrc->xyzwhdROI.xyz.y) * srcStridesCDH.z) + (id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x);
    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;

    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        fmaf_scalar_hip_compute(&val_f8, &fmaddParams_f2);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

__global__ void fmadd_scalar_ndhwc_tensor(float *srcPtr,
                                          uint2 srcStridesDH,
                                          float *dstPtr,
                                          uint2 dstStridesDH,
                                          float2 fmaddParams_f2,
                                          RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= roiGenericPtrSrc->xyzwhdROI.roiDepth) || (id_y >= roiGenericPtrSrc->xyzwhdROI.roiHeight) || (id_x >= roiGenericPtrSrc->xyzwhdROI.roiWidth))
    {
        return;
    }

    uint srcIdx = ((id_z + roiGenericPtrSrc->xyzwhdROI.xyz.z) * srcStridesDH.x) + ((id_y + roiGenericPtrSrc->xyzwhdROI.xyz.y) * srcStridesDH.y) + (id_x + roiGenericPtrSrc->xyzwhdROI.xyz.x) * 3;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;

    d_float24 val_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &val_f24);
    fmaf_scalar_hip_compute(&val_f24, &fmaddParams_f2);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

RppStatus hip_exec_fmadd_scalar_tensor(Rpp32f *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       RpptROI3DPtr roiGenericPtrSrc,
                                       Rpp32f *mulTensor,
                                       Rpp32f *addTensor,
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
            hipLaunchKernelGGL(fmadd_scalar_ncdhw_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                               dstGenericDescPtr->dims[1],
                               make_float2(mulTensor[batchCount], addTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount]);
        }
    }
    else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];               // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];               // D - depth (z direction)

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(fmadd_scalar_ndhwc_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               make_float2(mulTensor[batchCount], addTensor[batchCount]),
                               &roiGenericPtrSrc[batchCount]);
        }
    }

    return RPP_SUCCESS;
}