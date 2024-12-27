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
                                          uint numDims,
                                          Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    if(id_x >= dstStrides[0])
        return;

    uint *roi = roiTensor + id_z * numDims * 2;
    uint *begin = roi;
    uint *length = &roi[numDims];
    uint dstIdx = (id_z * *dstStrides++);
    uint srcIdx1 = (id_z * *srcStrides++);
    uint srcIdx2 = (id_z * *srcStrides);
    uint coords[RPPT_MAX_DIMS];

    uint temp = id_x;
    for (int i = 0; i < numDims; i++)
    {
        coords[i] = temp / dstStrides[i];
        temp %= dstStrides[i];
        if (i < axis)
        {
            dstIdx += coords[i] * dstStrides[i];
            srcIdx1 += (coords[i] + begin[i]) * srcStrides[i];
            srcIdx2 += (coords[i] + begin[i]) * srcStrides[i];
        }
        else if (i == axis)
        {
            if (coords[i] < length[i])
            {
                dstIdx += coords[i] * dstStrides[i];
                srcIdx1 += (coords[i] + begin[i]) * srcStrides[i];
            }
            else
            {
                uint shifted_coord = coords[i] - length[i];
                dstIdx += coords[i] * dstStrides[i];
                srcIdx2 += (shifted_coord + begin[i]) * srcStrides[i];
            }
        }
        else
        {
            dstIdx += coords[i] * dstStrides[i];
            srcIdx1 += coords[i] * srcStrides[i];
            srcIdx2 += coords[i] * srcStrides[i];
        }
    }

    // Write to output tensor
    if (coords[axis] < length[axis])
        dstPtr[dstIdx] = srcPtr[srcIdx1];
    else
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
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
    int globalThreads_y = 1;
    int globalThreads_z = dstGenericDescPtr->dims[0];

    
        hipLaunchKernelGGL(concat_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcPtr2,
                       srcGenericDescPtr->strides,
                       src2GenericDescPtr->strides,
                       dstPtr,
                       dstGenericDescPtr->strides,
                       srcGenericDescPtr->dims + 1,
                       0,
                       dstGenericDescPtr->numDims - 1,
                       roiTensor);

    return RPP_SUCCESS;
}