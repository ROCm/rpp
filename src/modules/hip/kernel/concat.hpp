#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void concat_2d_hip_tensor(T *srcPtr,
                                     T *srcPtr2,
                                     uint *srcStrides,
                                     uint *src2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *dstDims,
                                     uint axis,
                                     Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= dstDims[2] || id_y >= dstDims[1] || id_z >= dstDims[0])
        return;

    uint *roi = &roiTensor[id_z * 4];
    uint begin[2] = {roi[0], roi[1]};
    uint length[2] = {roi[2], roi[3]};

    uint dstIdx = id_z * dstStrides[0] + id_y * dstStrides[1] + id_x;
    int numElements = (id_x < length[1]) ? length[1] - id_x : dstDims[2] - id_x;

    d_float8 src_f8;

    if (axis == 0) // Concatenate along rows
    {
        if (id_y < length[0]) // Within the first source tensor
        {
            uint srcIdx1 = id_z * srcStrides[0] + (id_y + begin[0]) * srcStrides[1] + id_x + begin[1];

            if (numElements >= 8)
                rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx1, &src_f8);
            else
                for (int i = 0; i < numElements; i++)
                    dstPtr[dstIdx + i] = srcPtr[srcIdx1 + i];
        }
        else // Within the second source tensor
        {
            uint srcIdx2 = id_z * src2Strides[0] + (id_y - length[0] + begin[0]) * src2Strides[1] + id_x + begin[1];

            if (numElements >= 8)
                rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
            else
                for (int i = 0; i < numElements; i++)
                    dstPtr[dstIdx + i] = srcPtr2[srcIdx2 + i];
        }
    }
    else if (axis == 1) // Concatenate along columns
    {
        if ((id_x + 8) <= length[1]) // Fully within the first source tensor
        {
            uint srcIdx1 = id_z * srcStrides[0] + id_y * srcStrides[1] + (id_x + begin[1]);
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx1, &src_f8);
        }
        else if (id_x < length[1]) // Spanning both tensors
        {
            uint srcIdx1 = id_z * srcStrides[0] + id_y * srcStrides[1] + (id_x + begin[1]);
            uint srcIdx2 = id_z * src2Strides[0] + id_y * src2Strides[1] + (id_x - length[1] + begin[1]);

            for (int i = 0; i < 8; i++)
            {
                if ((id_x + i) < length[1])
                    dstPtr[dstIdx + i] = srcPtr[srcIdx1 + i];
                else
                    dstPtr[dstIdx + i] = srcPtr2[srcIdx2 + i];
            }
        }
        else // Fully within the second source tensor
        {
            uint srcIdx2 = id_z * src2Strides[0] + id_y * src2Strides[1] + (id_x - length[1] + begin[1]);

            if (numElements >= 8)
                rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
            else
                for (int i = 0; i < numElements; i++)
                    dstPtr[dstIdx + i] = srcPtr2[srcIdx2 + i];
        }
    }

    // Store the results back to the destination tensor
    if (numElements >= 8)
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
}

template <typename T>
__global__ void concat_3d_hip_tensor(T *srcPtr,
                                     T *srcPtr2,
                                     uint *srcStrides,
                                     uint *src2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *dims,
                                     uint axis,
                                     Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_x >= dims[2] || id_y >= dims[1] || id_z >= dims[0])
        return;

    uint dstIdx = id_z * dstStrides[1] + id_y * dstStrides[2] + id_x;
    uint srcIdx1 = id_z * srcStrides[1] + id_y * srcStrides[2] + id_x;
    uint srcIdx2 = id_z * src2Strides[1] + id_y * src2Strides[2] + id_x;

    d_float8 src_f8, src2_f8;
    if((dims[2] - id_x) >= 8)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx1, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx + srcStrides[2], &src_f8);
    }
    else
    {
        for(int i = 0; i < (dstStrides[2] - id_x); i++)
        {
            dstPtr[dstIdx + i] = srcPtr[srcIdx1 + i];
            dstPtr[dstIdx + srcStrides[2] + i] = srcPtr2[srcIdx2 + i];
        }
    }
}

template <typename T>
__global__ void concat_generic_hip_tensor(T *srcPtr,
                                          T *srcPtr2,
                                          uint *srcStrides,
                                          uint *src2Strides,
                                          T *dstPtr,
                                          uint *dstStrides,
                                          uint axis,
                                          uint numDims,
                                          Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_x >= dstStrides[0])
        return;

    uint *roi = roiTensor + id_z * numDims * 2;
    uint *begin = roi;
    uint *length = &roi[numDims];
    uint dstIdx = (id_z * *dstStrides++);
    uint srcIdx1 = (id_z * *srcStrides++);
    uint srcIdx2 = (id_z * *src2Strides++);
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
            srcIdx2 += (coords[i] + begin[i]) * src2Strides[i];
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
                srcIdx2 += (shifted_coord + begin[i]) * src2Strides[i];
            }
        }
        else
        {
            dstIdx += coords[i] * dstStrides[i];
            srcIdx1 += coords[i] * srcStrides[i];
            srcIdx2 += coords[i] * src2Strides[i];
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
                                 Rpp32u axis,
                                 Rpp32u *roiTensor,
                                 rpp::Handle& handle)
{
    int globalThreads_x = dstGenericDescPtr->strides[0];
    int globalThreads_y = 1;
    int globalThreads_z = dstGenericDescPtr->dims[0];

    int numDims = dstGenericDescPtr->numDims - 1;

    if (numDims == 2)
    {
        // NHW
        globalThreads_x = dstGenericDescPtr->dims[2];
        globalThreads_y = dstGenericDescPtr->dims[1];
        globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(concat_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcPtr2,
                           srcGenericDescPtr->strides,
                           src2GenericDescPtr->strides,
                           dstPtr,
                           dstGenericDescPtr->strides,
                           dstGenericDescPtr->dims,
                           axis,
                           roiTensor);
    }
    else if (numDims == 3)
    {
        if(axis == 0)
        {
            srcGenericDescPtr->strides[2] = srcGenericDescPtr->strides[0];
            srcGenericDescPtr->strides[0] = srcGenericDescPtr->strides[1] = 1;
            src2GenericDescPtr->strides[2] = src2GenericDescPtr->strides[0];
            src2GenericDescPtr->strides[0] = src2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        else if(axis == 1)
        {
            srcGenericDescPtr->strides[2] = srcGenericDescPtr->strides[1];
            srcGenericDescPtr->strides[0] = srcGenericDescPtr->strides[1] = 1;
            src2GenericDescPtr->strides[2] = src2GenericDescPtr->strides[1];
            src2GenericDescPtr->strides[0] = src2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[1];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32u *begin = roi;
            Rpp32u *length = &roi[numDims];
            Rpp32u *dims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
            if(axis == 0)
            {
                dims[0] = dims[1] = 1;
                dims[2] = length[0] * length[1] * length[2];
            }
            else if(axis == 1)
            {
                dims[0] = 1;
                dims[1] = length[0];
                dims[2] = length[1] * length[2];
            }
            else if(axis == 2)
            {
                dims[0] = length[0];
                dims[1] = length[1];
                dims[2] = length[2];
            }
            globalThreads_x = dims[2];
            globalThreads_y = dims[1];
            globalThreads_z = dims[0];
            hipLaunchKernelGGL(concat_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * dims[0] * dims[1] * dims[2]),
                               srcPtr2 + (batchCount * dims[0] * dims[1] * dims[2]),
                               srcGenericDescPtr->strides,
                               src2GenericDescPtr->strides,
                               dstPtr + (batchCount * dims[0] * dims[1] * dims[2] * 2),
                               dstGenericDescPtr->strides,
                               dims,
                               axis,
                               &roiTensor[batchCount * 6]);
        }
    }
    else
    {
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
                       axis,
                       dstGenericDescPtr->numDims - 1,
                       roiTensor);
    }

    return RPP_SUCCESS;
}