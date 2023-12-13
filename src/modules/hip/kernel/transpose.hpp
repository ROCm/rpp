#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// Unvectorized src->dst
template <typename T>
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
        printf("\nid_x = %d, srcIdx = %u, dstIdx = %u", id_x, srcIdx, dstIdx);
    }

    dstPtr[dstIdx] = srcPtr[srcIdx];
}

// Unvectorized dst->src
// template <typename T>
// __global__ void transpose_generic_hip_tensor(T *srcPtr,
//                                              uint *srcStrides,
//                                              T *dstPtr,
//                                              uint *dstStrides,
//                                              uint *dstDims,
//                                              uint dstNumDims,
//                                              uint *permTensor,
//                                              uint *roiTensor)
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

//     uint dstIdx = (id_y * *dstStrides++);
//     uint srcIdx = (id_y * *srcStrides++);
//     uint dstCoords[RPPT_MAX_DIMS];

//     for (int i = 0; i < dstNumDims; i++)
//         dstCoords[i] = id_x / dstStrides[i] % dstDims[i];

//     for (int i = 0; i < dstNumDims; i++)
//     {
//         srcIdx += (dstCoords[permTensor[i]] * srcStrides[permTensor[i]]);
//         dstIdx += (dstCoords[i] * dstStrides[i]);
//     }

//     if ((id_x < 500) && (id_y == 0))
//     {
//         printf("\nid_y = %d, id_x = %d || dstCoords[0] = %u, dstCoords[1] = %u, dstCoords[2] = %u || dstCoords[permTensor[0]] = %u, dstCoords[permTensor[1]] = %u, dstCoords[permTensor[2]] = %u || srcIdx = %u, dstIdx = %u", id_y, id_x, dstCoords[0], dstCoords[1], dstCoords[2], dstCoords[permTensor[0]], dstCoords[permTensor[1]], dstCoords[permTensor[2]], srcIdx, dstIdx);
//         // printf("\ndstCoords[0] = %u, dstCoords[1] = %u, dstCoords[2] = %u", dstCoords[0], dstCoords[1], dstCoords[2]);
//         // printf("\ndstCoords[permTensor[0]] = %u, dstCoords[permTensor[1]] = %u, dstCoords[permTensor[2]] = %u", dstCoords[permTensor[0]], dstCoords[permTensor[1]], dstCoords[permTensor[2]]);
//         // printf("\nsrcIdx = %u, dstIdx = %u", srcIdx, dstIdx);
//     }

//     dstPtr[dstIdx] = srcPtr[srcIdx];
// }

// Vectorized src->dst
// template <typename T> // temporarily only float
__global__ void transpose_generic_hip_tensor(float *srcPtr, // T *srcPtr, // temporarily only float
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
    uint4 idx0123 = make_uint4(0, 1, 2, 3);
    uint4 idx4567 = make_uint4(4, 5, 6, 7);
    dstIdxs.ui4[0] = make_uint4(dstIdx, dstIdx, dstIdx, dstIdx);
    dstIdxs.ui4[1] = make_uint4(dstIdx, dstIdx, dstIdx, dstIdx);

    for (int i = 0; i < srcNumDims; i++)
    {
        srcCoords[i].ui4[0] = idx0123 / srcStrides[i] % srcDims[i];
        srcCoords[i].ui4[1] = idx4567 / srcStrides[i] % srcDims[i];
    }

    // if ((id_x == 0) && (id_y == 0))
    // {
    //     printf("\nsrcCoords[0].ui1[0] = %u", srcCoords[0].ui1[0]);
    //     printf("\nsrcCoords[0].ui1[1] = %u", srcCoords[0].ui1[1]);
    //     printf("\nsrcCoords[0].ui1[2] = %u", srcCoords[0].ui1[2]);
    //     printf("\nsrcCoords[0].ui1[3] = %u", srcCoords[0].ui1[3]);
    //     printf("\nsrcCoords[0].ui1[4] = %u", srcCoords[0].ui1[4]);
    //     printf("\nsrcCoords[0].ui1[5] = %u", srcCoords[0].ui1[5]);
    //     printf("\nsrcCoords[0].ui1[6] = %u", srcCoords[0].ui1[6]);
    //     printf("\nsrcCoords[0].ui1[7] = %u", srcCoords[0].ui1[7]);
    //     printf("\n");
    //     printf("\nsrcCoords[1].ui1[0] = %u", srcCoords[1].ui1[0]);
    //     printf("\nsrcCoords[1].ui1[1] = %u", srcCoords[1].ui1[1]);
    //     printf("\nsrcCoords[1].ui1[2] = %u", srcCoords[1].ui1[2]);
    //     printf("\nsrcCoords[1].ui1[3] = %u", srcCoords[1].ui1[3]);
    //     printf("\nsrcCoords[1].ui1[4] = %u", srcCoords[1].ui1[4]);
    //     printf("\nsrcCoords[1].ui1[5] = %u", srcCoords[1].ui1[5]);
    //     printf("\nsrcCoords[1].ui1[6] = %u", srcCoords[1].ui1[6]);
    //     printf("\nsrcCoords[1].ui1[7] = %u", srcCoords[1].ui1[7]);
    // }


    for (int i = 0; i < srcNumDims; i++)
    {
        for (int j = 0; j < 8; j++)
            dstIdxs.ui1[j] += (srcCoords[permTensor[i]].ui1[j] * dstStrides[i]);
        srcIdx += (srcCoords[i].ui1[0] * srcStrides[i]);
    }

    if ((id_x < 1000) && (id_y == 0))
    {
        printf("\nid_x = %d, srcIdx = %u | dstIdxs = %u, %u, %u, %u, %u, %u, %u, %u", id_x, srcIdx, dstIdxs.ui1[0], dstIdxs.ui1[1], dstIdxs.ui1[2], dstIdxs.ui1[3], dstIdxs.ui1[4], dstIdxs.ui1[5], dstIdxs.ui1[6], dstIdxs.ui1[7]);
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
    int globalThreads_x = dstGenericDescPtr->strides[0];
    // int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3;
    int globalThreads_y = handle.GetBatchSize();
    int globalThreads_z = 1;

    hipLaunchKernelGGL(transpose_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcGenericDescPtr->strides,
                       srcGenericDescPtr->dims + 1,
                       srcGenericDescPtr->numDims,
                       dstPtr,
                       dstGenericDescPtr->strides,
                    //    dstGenericDescPtr->dims + 1,
                    //    dstGenericDescPtr->numDims,
                       permTensor,
                       roiTensor);

    return RPP_SUCCESS;
}
