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

    uint dstIdx = id_x + id_z * dstSampleStrides[0];
    uint srcIdx1 = id_z * src1SampleStrides[0];
    uint srcIdx2 = id_z * src2SampleStrides[0];

    for(int i = numDims - 1; i >= 0; i--)
    {
        int index = id_x % dstSampleDims[i];
        if(index >= dstSampleDims[i])
            return;
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

    // Allocate host-side buffers for broadcast dims/strides
    Rpp32u *src1BroadcastDims = (Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *src2BroadcastDims = (Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *dstBroadcastDims = (Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *src1BeginOffsets = (Rpp32u*)calloc(batchSize, sizeof(Rpp32u));
    Rpp32u *src2BeginOffsets = (Rpp32u*)calloc(batchSize, sizeof(Rpp32u));

    Rpp32u *src1BroadcastStrides = (Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *src2BroadcastStrides = (Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    Rpp32u *dstBroadcastStrides = (Rpp32u*)malloc(batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(handle.GetNumThreads())
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
        Rpp32u *dstBDims = dstBroadcastDims + i * RPPT_MAX_DIMS;

        Rpp32u *src1BStrides = src1BroadcastStrides + i * RPPT_MAX_DIMS;
        Rpp32u *src2BStrides = src2BroadcastStrides + i * RPPT_MAX_DIMS;
        Rpp32u *dstBStrides = dstBroadcastStrides  + i * RPPT_MAX_DIMS;

        src1BStrides[0] = src1Strides[0];
        src2BStrides[0] = src2Strides[0];
        dstBStrides[0] = dstStrides[0];

        for (int j = 0; j < minDim; j++)
        {
            src1BDims[j] = src1dims[j];
            src2BDims[j] = src2dims[j];
            src1BStrides[j + 1] = src1Strides[j + 1];
            src2BStrides[j + 1] = src2Strides[j + 1];
            dstBStrides[j + 1] = dstStrides[j + 1];

            if ((src1BDims[j] != src2BDims[j]) && (src1BDims[j] != 1) && (src2BDims[j] != 1))
                incompatibleDims = true;
            dstBDims[j] = std::max(src1BDims[j], src2BDims[j]);
            src1BeginOffsets[i] += src1begin[j] * src1Strides[j + 1];
            src2BeginOffsets[i] += src2begin[j] * src2Strides[j + 1];
        }

        if (src1NDim < src2NDim)
        {
            for (int j = minDim; j < dstDim; j++)
            {
                src1BDims[j] = 1;
                src2BDims[j] = src2dims[j];
                dstBDims[j] = src2dims[j];
                src1BStrides[j + 1] = 0;
                src2BStrides[j + 1] = src2Strides[j + 1];
                dstBStrides[j + 1]  = dstStrides[j + 1];
            }
        }

        else if (src1NDim > src2NDim)
        {
            for (int j = minDim; j < dstDim; j++)
            {
                src2BDims[j] = 1;
                src1BDims[j] = src1dims[j];
                dstBDims[j] = src1dims[j];
                src1BStrides[j + 1] = src1Strides[j + 1];
                src2BStrides[j + 1] = 0;
                dstBStrides[j + 1] = dstStrides[j + 1];
            }
        }

        // Step 3: Zero-out strides for size-1 broadcast dims
        for (int j = 0; j < minDim; j++) {
            if ((src1BDims[j] != dstBDims[j]) && (src1BDims[j] == 1))
                src1BStrides[j + 1] = 0;
            if ((src2BDims[j] != dstBDims[j]) && (src2BDims[j] == 1))
                src2BStrides[j + 1] = 0;
        }

        std::cout<<"\n batchSize "<< i;
        for(int j = 0; j < 5; j++)
            printf("\n B srcStrides1[%d] : %d", j, src1BStrides[j]);
        for(int j = 0; j < 5; j++)
            printf("\n B srcStrides2[%d] : %d", j, src2BStrides[j]);
        for(int j = 0; j < 5; j++)
            printf("\n B dstStrides2[%d] : %d", j, dstBStrides[j]);
    }
    
    // Allocate device memory for HIP kernel inputs
    Rpp32u *d_src1BroadcastDims, *d_src2BroadcastDims, *d_dstBroadcastDims, *d_src1BeginOffsets, *d_src2BeginOffsets;
    Rpp32u *d_src1BroadcastStrides, *d_src2BroadcastStrides, *d_dstBroadcastStrides;

    hipMalloc(&d_src1BroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    hipMalloc(&d_src2BroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    hipMalloc(&d_dstBroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    hipMalloc(&d_src1BeginOffsets, batchSize * sizeof(Rpp32u));
    hipMalloc(&d_src2BeginOffsets, batchSize * sizeof(Rpp32u));
    hipMalloc(&d_src1BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    hipMalloc(&d_src2BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));
    hipMalloc(&d_dstBroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u));

    // Copy to device
    hipMemcpy(d_src1BroadcastDims, src1BroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_src2BroadcastDims, src2BroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_dstBroadcastDims, dstBroadcastDims, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_src1BeginOffsets, src1BeginOffsets, batchSize * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_src2BeginOffsets, src2BeginOffsets, batchSize * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_src1BroadcastStrides, src1BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_src2BroadcastStrides, src2BroadcastStrides, batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice);
    hipMemcpy(d_dstBroadcastStrides, dstBroadcastStrides,  batchSize * RPPT_MAX_DIMS * sizeof(Rpp32u), hipMemcpyHostToDevice);

    if(dstDim == 1)
    {
        // NW
        int globalThreads_x = dstGenericDescPtr->dims[1];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];
        //printf("Broadcast Test case 2\n");
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
