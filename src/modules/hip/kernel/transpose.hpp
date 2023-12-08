#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

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

    dstPtr[dstIdx] = srcPtr[srcIdx];
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
                       srcGenericDescPtr->numDims - 1,
                       dstPtr,
                       dstGenericDescPtr->strides,
                       permTensor,
                       roiTensor);

    return RPP_SUCCESS;
}
