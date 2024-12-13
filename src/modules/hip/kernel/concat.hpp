#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void concat_generic_hip_tensor(T *srcPtr,
                                        T *srcPtr2,
                                        uint *srcStrides,
                                        uint *src2Strides,
                                        T *dstPtr,
                                        uint *dstStrides,
                                        uint *srcDims,
                                        uint axis,
                                        uint numDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int out_indices[10];
    int tmpIdx = id_x;
    if(id_x >= 8)
        return;
    for (int i = 0; i < numDims; i++)
    {
        out_indices[i] = tmpIdx / dstStrides[i];
        printf("\n strides %d %d ", i, dstStrides[i]);
        tmpIdx %= dstStrides[i];
    }

    if(id_x == 4)
        printf("\n out indixes %d %d %d", out_indices[0], out_indices[1], srcDims[axis]);

    // Determine if the index maps to input1 or input2
    bool in_input1 = (out_indices[axis] < srcDims[axis]);
    int in_idx = 0;
    if (in_input1)
    {
        for (int i = 0; i < numDims; i++)
        {
            in_idx += out_indices[i] * srcStrides[i];
        }
        dstPtr[id_x] = srcPtr[in_idx];
    }
    else
    {
        out_indices[axis] -= srcDims[axis];
        for (int i = 0; i < numDims; i++)
        {
            in_idx += out_indices[i] * src2Strides[i];
        }
        dstPtr[id_x] = srcPtr2[in_idx];
    }
}
template <typename T>
RppStatus hip_exec_concat_tensor(T *srcPtr,
                                 RpptGenericDescPtr srcGenericDescPtr,
                                 T *srcPtr2,
                                 RpptGenericDescPtr src2GenericDescPtr,
                                 T *dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32u *axis,
                                 Rpp32u *roiTensor,
                                 rpp::Handle& handle)
{
    int globalThreads_x = dstGenericDescPtr->strides[0];
    int globalThreads_y = dstGenericDescPtr->dims[0];
    int globalThreads_z = 1;

    hipLaunchKernelGGL(concat_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcPtr2,
                       srcGenericDescPtr->strides + 1,
                       src2GenericDescPtr->strides + 1,
                       dstPtr,
                       dstGenericDescPtr->strides + 1,
                       srcGenericDescPtr->dims + 1,
                       1,
                       dstGenericDescPtr->numDims - 1);

    return RPP_SUCCESS;
}