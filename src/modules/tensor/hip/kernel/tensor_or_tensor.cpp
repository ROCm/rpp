#include "hip_tensor_executors.hpp"
#include "broadcast.hpp"
#include "rpp_hip_math.hpp"
#include <omp.h>

template <typename T>
struct BitwiseAnd {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a & b;
    }
};

template <typename T>
struct BitwiseOr {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a | b;
    }
};

template <typename T>
struct BitwiseXor {
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a ^ b;
    }
};

template <typename T, typename Operation>
__global__ void tensor_or_tensor_1d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint* srcStrides1,
                                               uint* srcStrides2,
                                               uint *src1BeginOffsets,
                                               uint *src2BeginOffsets,
                                               T *dstPtr,
                                               uint* dstStrides,
                                               uint *dstDims,
                                                Operation op)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint* dstSampleDims = dstDims + id_z * RPPT_MAX_DIMS;
    uint* src1SampleStrides = srcStrides1 + id_z * RPPT_MAX_DIMS;
    uint* src2SampleStrides = srcStrides2 + id_z * RPPT_MAX_DIMS;
    uint* dstSampleStrides = dstStrides + id_z * RPPT_MAX_DIMS;

    if (id_x >= dstSampleDims[0])
        return;

    uint srcIdx1 = (id_z * src1SampleStrides[0]) + (id_x * src1SampleStrides[1]) + src1BeginOffsets[id_z];
    uint srcIdx2 = (id_z * src2SampleStrides[0]) + (id_x * src2SampleStrides[1]) + src2BeginOffsets[id_z];
    uint dstIdx = (id_z * dstSampleStrides[0]) + (id_x * dstSampleStrides[1]);

    dstPtr[dstIdx] = op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_2d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint* srcStrides1,
                                               uint* srcStrides2,
                                               uint *src1BeginOffsets,
                                               uint *src2BeginOffsets,
                                               T *dstPtr,
                                               uint *dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // width
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // height
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint* dstSampleDims = dstDims + id_z * RPPT_MAX_DIMS;
    uint* src1SampleStrides = srcStrides1 + id_z * RPPT_MAX_DIMS;
    uint* src2SampleStrides = srcStrides2 + id_z * RPPT_MAX_DIMS;
    uint* dstSampleStrides = dstStrides + id_z * RPPT_MAX_DIMS;

    if (id_x >= dstSampleDims[1] || id_y >= dstSampleDims[0])
        return;

    uint srcIdx1 = (id_z * src1SampleStrides[0]) + ((id_y) * src1SampleStrides[1]) + ((id_x) * src1SampleStrides[2]) + src1BeginOffsets[id_z];
    uint srcIdx2 = (id_z * src2SampleStrides[0]) + ((id_y) * src2SampleStrides[1]) + (id_x * src2SampleStrides[2]) + src2BeginOffsets[id_z];

    uint dstIdx = (id_z * dstSampleStrides[0]) + (id_y * dstSampleStrides[1]) + (id_x * dstSampleStrides[2]);

    dstPtr[dstIdx] = op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_3d_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint* srcStrides1,
                                               uint* srcStrides2,
                                               uint src1BeginOffset,
                                               uint src2BeginOffset,
                                               T *dstPtr,
                                               uint* dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    uint id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // lengthX
    uint id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y; // lengthY
    uint id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // lengthZ

    if (id_x >= dstDims[2] || id_y >= dstDims[1] || id_z >= dstDims[0])
        return;

    uint srcIdx1 = ((id_z) * srcStrides1[1]) + ((id_y) * srcStrides1[2]) + (id_x * srcStrides1[3]) + src1BeginOffset;
    uint srcIdx2 = ((id_z) * srcStrides2[1]) + ((id_y) * srcStrides2[2]) + (id_x * srcStrides2[3]) + src2BeginOffset;

    uint dstIdx = (id_z * dstStrides[1]) + (id_y * dstStrides[2]) + (id_x * dstStrides[3]);

    dstPtr[dstIdx] = op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
}

template <typename T, typename Operation>
__global__ void tensor_or_tensor_nd_hip_tensor(T *srcPtr1,
                                               T *srcPtr2,
                                               uint *srcStrides1,
                                               uint *srcStrides2,
                                               uint *src1BeginOffsets,
                                               uint *src2BeginOffsets,
                                               uint numDims,
                                               T *dstPtr,
                                               uint *dstStrides,
                                               uint *dstDims,
                                               Operation op)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z; // batchsize

    uint* dstSampleDims = dstDims + id_z * RPPT_MAX_DIMS;
    uint* src1SampleStrides = srcStrides1 + id_z * RPPT_MAX_DIMS;
    uint* src2SampleStrides = srcStrides2 + id_z * RPPT_MAX_DIMS;
    uint* dstSampleStrides = dstStrides + id_z * RPPT_MAX_DIMS;
    if(id_x >= dstSampleStrides[0])
        return;

    uint dstIdx = id_x + id_z * dstSampleStrides[0];
    uint srcIdx1 = id_z * src1SampleStrides[0];
    uint srcIdx2 = id_z * src2SampleStrides[0];

    for(int i = numDims - 1; i >= 0; i--)
    {
        int index = id_x % dstSampleDims[i];
        srcIdx1 = srcIdx1 + (index * src1SampleStrides[i + 1]);
        srcIdx2 = srcIdx2 + (index * src2SampleStrides[i + 1]);
        id_x = id_x / dstSampleDims[i];
    }

    srcIdx1 += src1BeginOffsets[id_z];
    srcIdx2 += src2BeginOffsets[id_z];

    dstPtr[dstIdx] = op(srcPtr1[srcIdx1], srcPtr2[srcIdx2]);
}

template <typename T, typename Operation>
RppStatus hip_exec_tensor_binary_bitwise_generic_tensor(T *srcPtr1,
                                                        T *srcPtr2,
                                                        RpptGenericDescPtr srcGenericDescPtr1,
                                                        RpptGenericDescPtr srcGenericDescPtr2,
                                                        T *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        Operation op,
                                                        uint *roiTensor1,
                                                        uint *roiTensor2,
                                                        rpp::Handle& handle)
{
    Rpp32u batchSize = dstGenericDescPtr->dims[0];
    Rpp32u src1NDim = srcGenericDescPtr1->numDims - 1;
    Rpp32u src2NDim = srcGenericDescPtr2->numDims - 1;
    Rpp32u dstDim = src1NDim > src2NDim ? src1NDim : src2NDim;
    Rpp32u minDim = src1NDim < src2NDim ? src1NDim : src2NDim;
    Rpp32u *src1Strides = srcGenericDescPtr1->strides;
    Rpp32u *src2Strides = srcGenericDescPtr2->strides;
    Rpp32u *dstStrides = dstGenericDescPtr->strides;

    for(int i = 0; i < minDim; i++)
    {
        if(srcGenericDescPtr1->dims[src1NDim - i] != srcGenericDescPtr2->dims[src2NDim - i])
        {
            if((srcGenericDescPtr1->dims[src1NDim - i] != 1) && (srcGenericDescPtr2->dims[src2NDim - i] != 1))
            {
                printf("Incompatible dimensions for the batch\n");
                return RPP_SUCCESS;
            }
        }
    }

    // Allocate host-side buffers for broadcast dims/strides
    Rpp32u *src1BroadcastDims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    Rpp32u *src2BroadcastDims = src1BroadcastDims + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *dstBroadcastDims = src2BroadcastDims + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *src1BeginOffsets = dstBroadcastDims + (batchSize * RPPT_MAX_DIMS);
    Rpp32u *src2BeginOffsets = src1BeginOffsets + batchSize;

    Rpp32u *src1BroadcastStrides = src2BeginOffsets + batchSize;//(Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *src2BroadcastStrides = src1BroadcastStrides + (batchSize * RPPT_MAX_DIMS);//(Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *dstBroadcastStrides = src2BroadcastStrides + (batchSize * RPPT_MAX_DIMS);//(Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(batchSize)
    for (int i = 0; i < batchSize; i++)
    {
        bool incompatibleDims = false;

        Rpp32u *src1roi = roiTensor1 + i * src1NDim * 2;
        Rpp32u *src1begin = src1roi;
        Rpp32u *src1dims = src1begin + src1NDim;

        Rpp32u *src2roi = roiTensor2 + i * src2NDim * 2;
        Rpp32u *src2begin = src2roi;
        Rpp32u *src2dims = src2begin + src2NDim;

        Rpp32u *src1BDims = src1BroadcastDims + i * RPPT_MAX_DIMS;
        Rpp32u *src2BDims = src2BroadcastDims + i * RPPT_MAX_DIMS;
        Rpp32u *dstBDims  = dstBroadcastDims  + i * RPPT_MAX_DIMS;

        Rpp32u *src1BStrides = src1BroadcastStrides + i * RPPT_MAX_DIMS;
        Rpp32u *src2BStrides = src2BroadcastStrides + i * RPPT_MAX_DIMS;
        Rpp32u *dstBStrides  = dstBroadcastStrides  + i * RPPT_MAX_DIMS;

        // Step 0: copy the first stride
        src1BStrides[0] = src1Strides[0];
        src2BStrides[0] = src2Strides[0];
        dstBStrides[0]  = dstStrides[0];

        src1BeginOffsets[i] = 0;
        src2BeginOffsets[i] = 0;

        // Step 1: copy minDim dimensions & strides
        memcpy(src1BDims, src1dims, minDim * sizeof(Rpp32u));
        memcpy(src2BDims, src2dims, minDim * sizeof(Rpp32u));
        memcpy(src1BStrides + 1, src1Strides + 1, minDim * sizeof(Rpp32u));
        memcpy(src2BStrides + 1, src2Strides + 1, minDim * sizeof(Rpp32u));
        memcpy(dstBStrides + 1, dstStrides + 1, minDim * sizeof(Rpp32u));

        // Compute offsets & check incompatibility
        for (int j = 0; j < minDim; j++)
        {
            if ((src1BDims[j] != src2BDims[j]) && (src1BDims[j] != 1) && (src2BDims[j] != 1))
                incompatibleDims = true;

            dstBDims[j] = std::max(src1BDims[j], src2BDims[j]);
            src1BeginOffsets[i] += src1begin[j] * src1Strides[j + 1];
            src2BeginOffsets[i] += src2begin[j] * src2Strides[j + 1];
        }

        // Step 2: handle extra dims beyond minDim
        if (src1NDim < src2NDim)
        {
            int extraDims = dstDim - minDim;
            memset(src1BDims + minDim, 1, extraDims * sizeof(Rpp32u));
            memcpy(src2BDims + minDim, src2dims + minDim, extraDims * sizeof(Rpp32u));
            memcpy(dstBDims  + minDim, src2dims + minDim, extraDims * sizeof(Rpp32u));

            memset(src1BStrides + minDim + 1, 0, extraDims * sizeof(Rpp32u));
            memcpy(src2BStrides + minDim + 1, src2Strides + minDim + 1, extraDims * sizeof(Rpp32u));
            memcpy(dstBStrides  + minDim + 1, dstStrides  + minDim + 1, extraDims * sizeof(Rpp32u));
        }
        else if (src1NDim > src2NDim)
        {
            int extraDims = dstDim - minDim;
            memcpy(src1BDims + minDim, src1dims + minDim, extraDims * sizeof(Rpp32u));
            memset(src2BDims + minDim, 1, extraDims * sizeof(Rpp32u));
            memcpy(dstBDims  + minDim, src1dims + minDim, extraDims * sizeof(Rpp32u));

            memcpy(src1BStrides + minDim + 1, src1Strides + minDim + 1, extraDims * sizeof(Rpp32u));
            memset(src2BStrides + minDim + 1, 0, extraDims * sizeof(Rpp32u));
            memcpy(dstBStrides + minDim + 1, dstStrides + minDim + 1, extraDims * sizeof(Rpp32u));
        }

        // Step 3: zero-out strides for broadcast dims
        for (int j = 0; j < minDim; j++) {
            if ((src1BDims[j] != dstBDims[j]) && (src1BDims[j] == 1))
                src1BStrides[j + 1] = 0;
            if ((src2BDims[j] != dstBDims[j]) && (src2BDims[j] == 1))
                src2BStrides[j + 1] = 0;
        }
    }

    // Allocate device memory for HIP kernel inputs
    Rpp32u *d_dstBroadcastDims, *d_src1BeginOffsets, *d_src2BeginOffsets;
    Rpp32u *d_src1BroadcastStrides, *d_src2BroadcastStrides, *d_dstBroadcastStrides;

    d_dstBroadcastDims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    d_src1BeginOffsets = d_dstBroadcastDims + (batchSize * RPPT_MAX_DIMS);
    d_src2BeginOffsets = d_src1BeginOffsets + batchSize;
    d_src1BroadcastStrides = d_src2BeginOffsets + batchSize;//(Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    d_src2BroadcastStrides = d_src1BroadcastStrides + (batchSize * RPPT_MAX_DIMS);//(Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    d_dstBroadcastStrides = d_src2BroadcastStrides + (batchSize * RPPT_MAX_DIMS);//(Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));

    // Copy to device
    hipMemcpyAsync(d_dstBroadcastDims, dstBroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream());
    hipMemcpyAsync(d_src1BeginOffsets, src1BeginOffsets, batchSize * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream());
    hipMemcpyAsync(d_src2BeginOffsets, src2BeginOffsets, batchSize * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream());
    hipMemcpyAsync(d_src1BroadcastStrides, src1BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream());
    hipMemcpyAsync(d_src2BroadcastStrides, src2BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream());
    hipMemcpyAsync(d_dstBroadcastStrides, dstBroadcastStrides,  batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream());

    if(dstDim == 1)
    {
        // NW
        int globalThreads_x = dstGenericDescPtr->dims[1];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];
        hipLaunchKernelGGL(tensor_or_tensor_1d_hip_tensor,
                        dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                        dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                        0,
                        handle.GetStream(),
                        srcPtr1,
                        srcPtr2,
                        d_src1BroadcastStrides,
                        d_src2BroadcastStrides,
                        d_src1BeginOffsets,
                        d_src2BeginOffsets,
                        dstPtr,
                        d_dstBroadcastStrides,
                        d_dstBroadcastDims,
                        op);
    }
    else if(dstDim == 2)
    {
        // NHW
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           d_src1BroadcastStrides,
                           d_src2BroadcastStrides,
                           d_src1BeginOffsets,
                           d_src2BeginOffsets,
                           dstPtr,
                           d_dstBroadcastStrides,
                           d_dstBroadcastDims,
                           op);
    }
    else if(dstDim == 3)
    {
        // NDHW
        int globalThreads_x = dstGenericDescPtr->dims[3];
        int globalThreads_y = dstGenericDescPtr->dims[2];
        int globalThreads_z = dstGenericDescPtr->dims[1];

        for(int batchCount = 0; batchCount < batchSize; batchCount++)
        {
            hipLaunchKernelGGL(tensor_or_tensor_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * srcGenericDescPtr1->strides[0]),
                               srcPtr2 + (batchCount * srcGenericDescPtr2->strides[0]),
                               d_src1BroadcastStrides + batchCount * RPPT_MAX_DIMS,
                               d_src2BroadcastStrides + batchCount * RPPT_MAX_DIMS,
                               *(d_src1BeginOffsets + batchCount),
                               *(d_src2BeginOffsets + batchCount),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               d_dstBroadcastStrides + batchCount * RPPT_MAX_DIMS,
                               d_dstBroadcastDims + batchCount * RPPT_MAX_DIMS,
                               op);
        }
    }
    else
    {
        // interpret the input as 1D tensor
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(tensor_or_tensor_nd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           0,
                           handle.GetStream(),
                           srcPtr1,
                           srcPtr2,
                           d_src1BroadcastStrides,
                           d_src2BroadcastStrides,
                           d_src1BeginOffsets,
                           d_src2BeginOffsets,
                           dstGenericDescPtr->numDims - 1,
                           dstPtr,
                           d_dstBroadcastStrides,
                           d_dstBroadcastDims,
                           op);
    }

    return RPP_SUCCESS;
}

template<typename T>
RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor(T *srcPtr1,
                                                       T *srcPtr2,
                                                       RpptGenericDescPtr srcPtr1GenericDescPtr,
                                                       RpptGenericDescPtr srcPtr2GenericDescPtr,
                                                       T *dstPtr,
                                                       RpptGenericDescPtr dstGenericDescPtr,
                                                       RpptBitwiseOp tensorOp,
                                                       Rpp32u *srcPtr1roiTensor,
                                                       Rpp32u *srcPtr2roiTensor,
                                                       rpp::Handle& handle)
{
    switch(tensorOp)
    {
        case RPP_TENSOR_OP_AND:
            hip_exec_tensor_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseAnd<T>(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_OR:
            hip_exec_tensor_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseOr<T>(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
        case RPP_TENSOR_OP_XOR:
            hip_exec_tensor_binary_bitwise_generic_tensor(srcPtr1, srcPtr2, srcPtr1GenericDescPtr, srcPtr2GenericDescPtr, dstPtr, dstGenericDescPtr, BitwiseXor<T>(), srcPtr1roiTensor, srcPtr2roiTensor, handle);
            break;
    }

    return RPP_SUCCESS;
}

template RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor<Rpp8u>(Rpp8u*,
                                                                   Rpp8u*,
                                                                   RpptGenericDescPtr,
                                                                   RpptGenericDescPtr,
                                                                   Rpp8u*,
                                                                   RpptGenericDescPtr,
                                                                   RpptBitwiseOp,
                                                                   Rpp32u*,
                                                                   Rpp32u*,
                                                                   rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor<Rpp16u>(Rpp16u*,
                                                                    Rpp16u*,
                                                                    RpptGenericDescPtr,
                                                                    RpptGenericDescPtr,
                                                                    Rpp16u*,
                                                                    RpptGenericDescPtr,
                                                                    RpptBitwiseOp,
                                                                    Rpp32u*,
                                                                    Rpp32u*,
                                                                    rpp::Handle&);

template RppStatus tensor_binary_bitwise_op_dispatch_gpu_tensor<Rpp32u>(Rpp32u*,
                                                                    Rpp32u*,
                                                                    RpptGenericDescPtr,
                                                                    RpptGenericDescPtr,
                                                                    Rpp32u*,
                                                                    RpptGenericDescPtr,
                                                                    RpptBitwiseOp,
                                                                    Rpp32u*,
                                                                    Rpp32u*,
                                                                    rpp::Handle&);
