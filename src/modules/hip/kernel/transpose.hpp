#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// Unvectorized src->dst
/*template <typename T>
__global__ void transpose_generic_hip_tensor(T *srcPtr,
                                             uint *srcStrides,
                                             uint *srcDims,
                                             uint srcNumDims,
                                             T *dstPtr,
                                             uint *dstStrides,
                                             uint *permTensor,
                                             uint *roiTensor)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);
    uint srcCoords[RPPT_MAX_DIMS];

    for (int i = 0; i < srcNumDims; i++)
        srcCoords[i] = id_x / srcStrides[i] % srcDims[i];

    for (int i = 0; i < srcNumDims; i++)
    {
        dstIdx += (srcCoords[permTensor[i]] * dstStrides[i]);
        srcIdx += (srcCoords[i] * srcStrides[i]);
    }

    if ((id_x < 1000) && (id_y == 0))
    {
        //printf("\nid_x = %d, srcIdx = %u, dstIdx = %u", id_x, srcIdx, dstIdx);
        printf("\nsrc -> dst id_y = %d, id_x = %d || srcCoords[0] = %u, srcCoords[1] = %u, srcCoords[2] = %u || srcCoords[permTensor[0]] = %u, srcCoords[permTensor[1]] = %u, srcCoords[permTensor[2]] = %u || srcIdx = %u, dstIdx = %u", id_y, id_x, srcCoords[0], srcCoords[1], srcCoords[2], srcCoords[permTensor[0]], srcCoords[permTensor[1]], srcCoords[permTensor[2]], srcIdx, dstIdx);
    }

    dstPtr[dstIdx] = srcPtr[srcIdx];
}*/

// Unvectorized dst->src
/* template <typename T>
__global__ void transpose_generic_hip_tensor(T *srcPtr,
                                             uint *srcStrides,
                                             T *dstPtr,
                                             uint *dstStrides,
                                             uint *dstDims,
                                             uint dstNumDims,
                                             uint *permTensor,
                                             uint *roiTensor)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);
    uint dstCoords[RPPT_MAX_DIMS];

    for (int i = 0; i < dstNumDims; i++)
        dstCoords[i] = id_x / dstStrides[i] % dstDims[i];


    for (int i = 0; i < dstNumDims; i++)
    {
        srcIdx += (dstCoords[permTensor[i]] * srcStrides[permTensor[permTensor[i]]]);
        dstIdx += (dstCoords[i] * dstStrides[i]);
    }

    dstPtr[dstIdx] = srcPtr[srcIdx];
} */

// Vectorized src->dst
// template <typename T> // temporarily only float
/* __global__ void transpose_generic_hip_tensor(float *srcPtr, // T *srcPtr, // temporarily only float
                                             uint *srcStrides,
                                             uint *srcDims,
                                             uint srcNumDims,
                                             float *dstPtr, // T *dstPtr, // temporarily only float
                                             uint *dstStrides,
                                             uint *permTensor,
                                             uint *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);
    d_uint8 srcCoords[RPPT_MAX_DIMS], dstIdxs;
    uint4 idx0123 = make_uint4(id_x, id_x + 1, id_x + 2, id_x + 3);
    uint4 idx4567 = make_uint4(id_x + 4, id_x + 5, id_x + 6, id_x + 7);
    dstIdxs.ui4[0] = make_uint4(dstIdx, dstIdx, dstIdx, dstIdx);
    dstIdxs.ui4[1] = make_uint4(dstIdx, dstIdx, dstIdx, dstIdx);

    for (int i = 0; i < srcNumDims; i++)
    {
        srcCoords[i].ui4[0] = idx0123 / srcStrides[i] % srcDims[i];
        srcCoords[i].ui4[1] = idx4567 / srcStrides[i] % srcDims[i];
    }

    for (int i = 0; i < srcNumDims; i++)
    {
        for (int j = 0; j < 8; j++)
            dstIdxs.ui1[j] += (srcCoords[permTensor[i]].ui1[j] * dstStrides[i]);
        srcIdx += (srcCoords[i].ui1[0] * srcStrides[i]);
    }

    d_float8 src_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8); // temporarily only float
    dstPtr[dstIdxs.ui1[0]] = src_f8.f1[0];  // temporarily only float
    dstPtr[dstIdxs.ui1[1]] = src_f8.f1[1];  // temporarily only float
    dstPtr[dstIdxs.ui1[2]] = src_f8.f1[2];  // temporarily only float
    dstPtr[dstIdxs.ui1[3]] = src_f8.f1[3];  // temporarily only float
    dstPtr[dstIdxs.ui1[4]] = src_f8.f1[4];  // temporarily only float
    dstPtr[dstIdxs.ui1[5]] = src_f8.f1[5];  // temporarily only float
    dstPtr[dstIdxs.ui1[6]] = src_f8.f1[6];  // temporarily only float
    dstPtr[dstIdxs.ui1[7]] = src_f8.f1[7];  // temporarily only float
    // *(d_uint8_s *)&dstPtr[dstIdx] = *(d_uint8_s *)&srcPtr[srcIdx];
} */

// Vectorized dst->src
// template <typename T> // temporarily only float
__global__ void transpose_generic_hip_tensor(float *srcPtr,
                                             uint *srcStrides,
                                             float *dstPtr,
                                             uint *dstStrides,
                                             uint *dstDims,
                                             uint dstNumDims,
                                             uint *permTensor,
                                             uint *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    //if(id_x > (id_y * dstStrides[0]))
        //return;

    uint dstIdx = (id_y * *dstStrides++);
    uint srcIdx = (id_y * *srcStrides++);

    //if(id_x > dstIdx)
        //return;

    d_uint8 dstCoords[RPPT_MAX_DIMS], srcIdxs;
    uint4 idx0123 = make_uint4(id_x, id_x + 1, id_x + 2, id_x + 3);
    uint4 idx4567 = make_uint4(id_x + 4, id_x + 5, id_x + 6, id_x + 7);
    srcIdxs.ui4[0] = make_uint4(srcIdx, srcIdx, srcIdx, srcIdx);
    srcIdxs.ui4[1] = make_uint4(srcIdx, srcIdx, srcIdx, srcIdx);

    for (int i = 0; i < dstNumDims; i++)
    {
        dstCoords[i].ui4[0] = idx0123 / dstStrides[i] % dstDims[i];
        dstCoords[i].ui4[1] = idx4567 / dstStrides[i] % dstDims[i];
    }

    for (int i = 0; i < dstNumDims; i++)
    {
        for (int j = 0; j < 8; j++)
            srcIdxs.ui1[j] += (dstCoords[permTensor[i]].ui1[j] * srcStrides[permTensor[permTensor[i]]]);
        dstIdx += (dstCoords[i].ui1[0] * dstStrides[i]);
    }

    if ((id_x > 15624) && (id_x < 46900) && (id_y == 0))
    {
        printf("\nid_x = %d, dstIdx = %u | srcIdxs = %u, %u, %u, %u, %u, %u, %u, %u", id_x, dstIdx, srcIdxs.ui1[0], srcIdxs.ui1[1], srcIdxs.ui1[2], srcIdxs.ui1[3], srcIdxs.ui1[4], srcIdxs.ui1[5], srcIdxs.ui1[6], srcIdxs.ui1[7]);
    }

    d_float8 dst_f8;
    dst_f8.f1[0] = srcPtr[srcIdxs.ui1[0]]; // load 8 src pixels
    dst_f8.f1[1] = srcPtr[srcIdxs.ui1[1]];
    dst_f8.f1[2] = srcPtr[srcIdxs.ui1[2]];
    dst_f8.f1[3] = srcPtr[srcIdxs.ui1[3]];
    dst_f8.f1[4] = srcPtr[srcIdxs.ui1[4]];
    dst_f8.f1[5] = srcPtr[srcIdxs.ui1[5]];
    dst_f8.f1[6] = srcPtr[srcIdxs.ui1[6]];
    dst_f8.f1[7] = srcPtr[srcIdxs.ui1[7]];

    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
RppStatus hip_exec_transpose_generic_tensor(T *srcPtr,
                                            RpptGenericDescPtr srcGenericDescPtr,
                                            T *dstPtr,
                                            RpptGenericDescPtr dstGenericDescPtr,
                                            Rpp32u *permTensor,
                                            Rpp32u *roiTensor,
                                            rpp::Handle& handle)
{
    //int globalThreads_x = dstGenericDescPtr->strides[0];
    int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
    int globalThreads_y = handle.GetBatchSize();
    int globalThreads_z = 1;

    hipLaunchKernelGGL(transpose_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcGenericDescPtr->strides,
                       //srcGenericDescPtr->dims + 1,
                       //srcGenericDescPtr->numDims - 1,
                       dstPtr,
                       dstGenericDescPtr->strides,
                       dstGenericDescPtr->dims + 1,
                       dstGenericDescPtr->numDims - 1,
                       permTensor,
                       roiTensor);

    return RPP_SUCCESS;
}
