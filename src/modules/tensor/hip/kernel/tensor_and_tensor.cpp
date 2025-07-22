#include "hip_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_hip_math.hpp"

template <typename T>
__global__ void tensor_and_tensor_1d_hip_tensor(T *srcPtr1,
                                                T *srcPtr2,
                                                uint srcStrides1,
                                                uint srcStrides2,
                                                uint *srcDims1,
                                                uint *srcDims2,
                                                T *dstPtr,
                                                uint dstStrides,
                                                uint *dstDims,
                                                uint *src1RoiTensor,
                                                uint *src2RoiTensor)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;       // batchsize

    if (id_x >= dstDims[0])
        return;

    uint srcIdx1 = (id_z * srcStrides1) + id_x;
    uint srcIdx2 = (id_z * srcStrides2) + id_x;
    uint dstIdx = (id_z * dstStrides) + id_x;

    d_uchar8 src1_uc8, src2_uc8, dst_uc8;
    uchar* src1Ptr_uc8 = (uchar*)&src1_uc8;
    uchar* src2Ptr_uc8 = (uchar*)&src2_uc8;

    if(srcDims1[0] == 1)
        src1_uc8.uc4[0] = src1_uc8.uc4[1] = (uchar4)srcPtr1[srcIdx1];
    else
        rpp_hip_load8_to_uchar8(srcPtr1 + srcIdx1, src1Ptr_uc8);
    if(srcDims2[0] == 1)
        src2_uc8.uc4[0] = src2_uc8.uc4[1] = (uchar4)srcPtr2[srcIdx2];
    else
        rpp_hip_load8_to_uchar8(srcPtr2 + srcIdx2, src2Ptr_uc8);
    rpp_hip_math_bitwiseAnd8(&src1_uc8, &src2_uc8, &dst_uc8);
    rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);
}

template <typename T>
__global__ void tensor_and_tensor_2d_hip_tensor(T *srcPtr1,
                                                T *srcPtr2,
                                                uint2 srcStrides1NH,
                                                uint2 srcStrides2NH,
                                                uint *srcDims1,
                                                uint *srcDims2,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint *dstDims,
                                                uint *src1RoiTensor,
                                                uint *src2RoiTensor)
{
    uint id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;       // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;       // batchsize

    if (id_x >= dstDims[1] || id_y >= dstDims[0])
        return;

    uint srcIdx1 = (id_z * srcStrides1NH.x) + ((id_y) * srcStrides1NH.y) + id_x;
    uint srcIdx2 = (id_z * srcStrides2NH.x) + ((id_y) * srcStrides2NH.y) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    d_uchar8 src1_uc8, src2_uc8, dst_uc8;
    uchar* src1Ptr_uc8 = (uchar*)&src1_uc8;
    uchar* src2Ptr_uc8 = (uchar*)&src2_uc8;

    if(srcDims1[1] == 1)
        src1_uc8.uc4[0] = src1_uc8.uc4[1] = (uchar4)srcPtr1[srcIdx1];
    else
        rpp_hip_load8_to_uchar8(srcPtr1 + srcIdx1, src1Ptr_uc8);
    if(srcDims2[1] == 1)
        src2_uc8.uc4[0] = src2_uc8.uc4[1] = (uchar4)srcPtr2[srcIdx2];
    else
        rpp_hip_load8_to_uchar8(srcPtr2 + srcIdx2, src2Ptr_uc8);
    rpp_hip_math_bitwiseAnd8(&src1_uc8, &src2_uc8, &dst_uc8);
    rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);
}

template <typename T>
__global__ void tensor_and_tensor_3d_hip_tensor(T *srcPtr1,
                                                T *srcPtr2,
                                                uint2 srcStrides1DH,
                                                uint2 srcStrides2DH,
                                                uint *srcDims1,
                                                uint *srcDims2,
                                                T *dstPtr,
                                                uint2 dstStridesDH,
                                                uint *dstDims,
                                                uint *src1RoiTensor,
                                                uint *src2RoiTensor)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // lengthZ

    if (id_x >= dstDims[2] || id_y >= dstDims[1] || id_z >= dstDims[0])
        return;

    uint srcIdx1 = ((id_z) * srcStrides1DH.x) + ((id_y) * srcStrides1DH.y) + id_x;
    uint srcIdx2 = ((id_z) * srcStrides2DH.x) + ((id_y) * srcStrides2DH.y) + id_x;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x;

    d_uchar8 src1_uc8, src2_uc8, dst_uc8;
    uchar* src1Ptr_uc8 = (uchar*)&src1_uc8;
    uchar* src2Ptr_uc8 = (uchar*)&src2_uc8;

    if(srcDims1[2] == 1)
        src1_uc8.uc4[0] = src1_uc8.uc4[1] = (uchar4)srcPtr1[srcIdx1];
    else
        rpp_hip_load8_to_uchar8(srcPtr1 + srcIdx1, src1Ptr_uc8);
    if(srcDims2[2] == 1)
        src2_uc8.uc4[0] = src2_uc8.uc4[1] = (uchar4)srcPtr2[srcIdx2];
    else
        rpp_hip_load8_to_uchar8(srcPtr2 + srcIdx2, src2Ptr_uc8);
    rpp_hip_math_bitwiseAnd8(&src1_uc8, &src2_uc8, &dst_uc8);
    rpp_hip_pack_uchar8_and_store8(dstPtr + dstIdx, &dst_uc8);
}

template <typename T>
__global__ void tensor_and_tensor_nd_hip_tensor(T *srcPtr1,
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

    dstPtr[dstIdx] = srcPtr1[srcIdx1] & srcPtr2[srcIdx2];
}

template <typename T>
RppStatus hip_exec_tensor_and_tensor_generic_tensor(T *srcPtr1,
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

    if (numDims == 1)
    {
        // NW
        int globalThreads_x = dstBroadcastDescPtr->dims[1];
        int globalThreads_y = 1;
        int globalThreads_z = dstBroadcastDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_and_tensor_1d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           src1BroadcastDescPtr->strides[0],
                           src2BroadcastDescPtr->strides[0],
                           src1BroadcastDescPtr->dims + 1,
                           src2BroadcastDescPtr->dims + 1,
                           dstPtr,
                           dstBroadcastDescPtr->strides[0],
                           dstBroadcastDescPtr->dims + 1,
                           roiTensor1,
                           roiTensor2);
    }
    else if (numDims == 2)
    {
        printf("Inside two dimension\n");
        // NHW
        int globalThreads_x = dstBroadcastDescPtr->dims[2];
        int globalThreads_y = dstBroadcastDescPtr->dims[1];
        int globalThreads_z = dstBroadcastDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_and_tensor_2d_hip_tensor,
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
    else if (numDims == 3)
    {
        // NDHW
        int globalThreads_x = dstBroadcastDescPtr->dims[3];
        int globalThreads_y = dstBroadcastDescPtr->dims[2];
        int globalThreads_z = dstBroadcastDescPtr->dims[1];

        for(int batchCount = 0; batchCount < dstBroadcastDescPtr->dims[0]; batchCount++)
        {
            hipLaunchKernelGGL(tensor_and_tensor_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * src1BroadcastDescPtr->strides[0]),
                               srcPtr2 + (batchCount * src2BroadcastDescPtr->strides[0]),
                               make_uint2(src1BroadcastDescPtr->strides[1], src1BroadcastDescPtr->strides[2]),
                               make_uint2(src2BroadcastDescPtr->strides[1], src2BroadcastDescPtr->strides[2]),
                               src1BroadcastDescPtr->dims + 1,
                               src2BroadcastDescPtr->dims + 1,
                               dstPtr + (batchCount * dstBroadcastDescPtr->strides[0]),
                               make_uint2(dstBroadcastDescPtr->strides[1], dstBroadcastDescPtr->strides[2]),
                               dstBroadcastDescPtr->dims + 1,
                               &roiTensor1[batchCount * 6],
                               &roiTensor2[batchCount * 6]);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstBroadcastDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstBroadcastDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_and_tensor_nd_hip_tensor,
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

template RppStatus hip_exec_tensor_and_tensor_generic_tensor<Rpp8u>(Rpp8u*,
                                                                    Rpp8u*,
                                                                    RpptGenericDescPtr,
                                                                    RpptGenericDescPtr,
                                                                    Rpp8u*,
                                                                    RpptGenericDescPtr,
                                                                    Rpp32u*,
                                                                    Rpp32u*,
                                                                    rpp::Handle&);
