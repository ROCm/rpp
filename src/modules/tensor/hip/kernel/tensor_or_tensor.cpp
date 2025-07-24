#include "hip_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_hip_math.hpp"

template <typename T>
__global__ void tensor_or_tensor_1d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint2 srcStrides1,
                                               uint2 srcStrides2,
                                               uint *srcDims1,
                                               uint *srcDims2,
                                               T *dstPtr,
                                               uint2 dstStrides,
                                               uint *dstDims,
                                               uint *src1RoiTensor,
                                               uint *src2RoiTensor)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    if (id_x >= dstDims[0])
        return;

    uint srcIdx1 = (id_z * srcStrides1.x) + (id_x * srcStrides1.y);
    uint srcIdx2 = (id_z * srcStrides2.x) + (id_x * srcStrides1.y);
    uint dstIdx = (id_z * dstStrides.x) + (id_x * srcStrides1.y);

    dstPtr[dstIdx] = srcPtr1[srcIdx1] | srcPtr2[srcIdx2];
}

template <typename T>
__global__ void tensor_or_tensor_2d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint3 srcStrides1NH,
                                               uint3 srcStrides2NH,
                                               uint *srcDims1,
                                               uint *srcDims2,
                                               T *dstPtr,
                                               uint3 dstStridesNH,
                                               uint *dstDims,
                                               uint *src1RoiTensor,
                                               uint *src2RoiTensor)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    if (id_x >= dstDims[1] || id_y >= dstDims[0])
        return;

    uint srcIdx1 = (id_z * srcStrides1NH.x) + ((id_y) * srcStrides1NH.y) + (id_x * srcStrides1NH.z);
    uint srcIdx2 = (id_z * srcStrides2NH.x) + ((id_y) * srcStrides2NH.y) + (id_x * srcStrides2NH.z);

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * dstStridesNH.z);

    dstPtr[dstIdx] = srcPtr1[srcIdx1] | srcPtr2[srcIdx2];
}

template <typename T>
__global__ void tensor_or_tensor_3d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint3 srcStrides1DH,
                                               uint3 srcStrides2DH,
                                               uint *srcDims1,
                                               uint *srcDims2,
                                               T *dstPtr,
                                               uint3 dstStridesDH,
                                               uint *dstDims,
                                               uint *src1RoiTensor,
                                               uint *src2RoiTensor)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // lengthZ

    if (id_x >= dstDims[2] || id_y >= dstDims[1] || id_z >= dstDims[0])
        return;

    uint srcIdx1 = ((id_z) * srcStrides1DH.x) + ((id_y) * srcStrides1DH.y) + (id_x * srcStrides1DH.z);
    uint srcIdx2 = ((id_z) * srcStrides2DH.x) + ((id_y) * srcStrides2DH.y) + (id_x * srcStrides2DH.z);

    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + (id_x * dstStridesDH.z);

    dstPtr[dstIdx] = srcPtr1[srcIdx1] | srcPtr2[srcIdx2];
}

template <typename T>
__global__ void tensor_or_tensor_nd_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint *srcStrides1,
                                               uint *srcStrides2,
                                               uint *srcDims1,
                                               uint *srcDims2,
                                               uint numDims,
                                               T *dstPtr,
                                               uint *dstStrides,
                                               uint *dstDims,
                                               Rpp32u *roiTensor)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    if(id_x >= dstStrides[0])
        return;

    uint dstIdx = (id_z * *dstStrides++) + id_x;
    uint srcIdx1 = (id_z * *srcStrides1++) + id_x;
    uint srcIdx2 = (id_z * *srcStrides2++) + id_x;

    dstPtr[dstIdx] = srcPtr1[srcIdx1] | srcPtr2[srcIdx2];
}

template <typename T>
RppStatus hip_exec_tensor_or_tensor_generic_tensor(T *srcPtr1,
                                                   T *srcPtr2,
                                                   RpptGenericDescPtr srcGenericDescPtr1,
                                                   RpptGenericDescPtr srcGenericDescPtr2,
                                                   T *dstPtr,
                                                   RpptGenericDescPtr dstGenericDescPtr,
                                                   uint *roiTensor1,
                                                   uint *roiTensor2,
                                                   rpp::Handle& handle)
{
    checkEqualBatchSize(srcGenericDescPtr1, srcGenericDescPtr2);
    BroadcastDstShape(srcGenericDescPtr1, srcGenericDescPtr2, dstGenericDescPtr);
    RpptGenericDescPtr src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr;
    CHECK_RETURN_STATUS(hipHostMalloc(&src1BroadcastDescPtr, sizeof(RpptGenericDesc)));
    CHECK_RETURN_STATUS(hipHostMalloc(&src2BroadcastDescPtr, sizeof(RpptGenericDesc)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dstBroadcastDescPtr, sizeof(RpptGenericDesc)));
    src1BroadcastDescPtr = srcGenericDescPtr1;
    src2BroadcastDescPtr = srcGenericDescPtr2;
    dstBroadcastDescPtr = dstGenericDescPtr;
    GroupShapes(src1BroadcastDescPtr, src2BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src1BroadcastDescPtr, dstBroadcastDescPtr);
    StridesForBroadcasting(src2BroadcastDescPtr, dstBroadcastDescPtr);

    Rpp32u numDims = dstBroadcastDescPtr->numDims - 1; // exclude batchsize from input dims

    printf("NumDims are %d\n", numDims);
    if (numDims == 1)
    {
        // NW
        int globalThreads_x = dstBroadcastDescPtr->dims[1];
        int globalThreads_y = 1;
        int globalThreads_z = dstBroadcastDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_1d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint2(src1BroadcastDescPtr->strides[0], src1BroadcastDescPtr->strides[1]),
                           make_uint2(src2BroadcastDescPtr->strides[0], src2BroadcastDescPtr->strides[1]),
                           src1BroadcastDescPtr->dims + 1,
                           src2BroadcastDescPtr->dims + 1,
                           dstPtr,
                           make_uint2(dstBroadcastDescPtr->strides[0], dstBroadcastDescPtr->strides[1]),
                           dstBroadcastDescPtr->dims + 1,
                           roiTensor1,
                           roiTensor2);
    }
    else if (numDims == 2)
    {
        printf("NumDims are %d\n", numDims);
        // NHW
        int globalThreads_x = dstBroadcastDescPtr->dims[2];
        int globalThreads_y = dstBroadcastDescPtr->dims[1];
        int globalThreads_z = dstBroadcastDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           make_uint3(src1BroadcastDescPtr->strides[0], src1BroadcastDescPtr->strides[1], src1BroadcastDescPtr->strides[2]),
                           make_uint3(src2BroadcastDescPtr->strides[0], src2BroadcastDescPtr->strides[1], src2BroadcastDescPtr->strides[2]),
                           src1BroadcastDescPtr->dims + 1,
                           src2BroadcastDescPtr->dims + 1,
                           dstPtr,
                           make_uint3(dstBroadcastDescPtr->strides[0], dstBroadcastDescPtr->strides[1],  dstBroadcastDescPtr->strides[2]),
                           dstBroadcastDescPtr->dims + 1,
                           roiTensor1,
                           roiTensor2);
    }
    else if (numDims == 3)
    {
        printf("NumDims are %d\n", numDims);
        // NDHW
        int globalThreads_x = dstBroadcastDescPtr->dims[3];
        int globalThreads_y = dstBroadcastDescPtr->dims[2];
        int globalThreads_z = dstBroadcastDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstBroadcastDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(tensor_or_tensor_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * src1BroadcastDescPtr->strides[0]),
                               srcPtr2 + (batchCount * src2BroadcastDescPtr->strides[0]),
                               make_uint3(src1BroadcastDescPtr->strides[1], src1BroadcastDescPtr->strides[2], src1BroadcastDescPtr->strides[3]),
                               make_uint3(src2BroadcastDescPtr->strides[1], src2BroadcastDescPtr->strides[2], src2BroadcastDescPtr->strides[3]),
                               src1BroadcastDescPtr->dims + 1,
                               src2BroadcastDescPtr->dims + 1,
                               dstPtr + (batchCount * dstBroadcastDescPtr->strides[0]),
                               make_uint3(dstBroadcastDescPtr->strides[1], dstBroadcastDescPtr->strides[2], dstBroadcastDescPtr->strides[3]),
                               dstBroadcastDescPtr->dims + 1,
                               &roiTensor1[batchCount * 6],
                               &roiTensor2[batchCount * 6]);
        }
    }
    else
    {
        printf("NumDims are %d\n", numDims);
        // interpret the input as 1D tensor
        int globalThreads_x = dstBroadcastDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstBroadcastDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_nd_hip_tensor,
                        dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                        dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                        0,
                        handle.GetStream(),
                        srcPtr1,
                        srcPtr2,
                        src1BroadcastDescPtr->strides,
                        src2BroadcastDescPtr->strides,
                        src1BroadcastDescPtr->dims + 1,
                        src2BroadcastDescPtr->dims + 1,
                        dstBroadcastDescPtr->numDims - 1,
                        dstPtr,
                        dstBroadcastDescPtr->strides,
                        dstBroadcastDescPtr->dims + 1,
                        roiTensor1);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_tensor_or_tensor_generic_tensor<Rpp8u>(Rpp8u*,
                                                                   Rpp8u*,
                                                                   RpptGenericDescPtr,
                                                                   RpptGenericDescPtr,
                                                                   Rpp8u*,
                                                                   RpptGenericDescPtr,
                                                                   Rpp32u*,
                                                                   Rpp32u*,
                                                                   rpp::Handle&);
