#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void down_mixing_hip_tensor(float *srcPtr,
                                       uint srcStride,
                                       float *dstPtr,
                                       uint dstStride,
                                       int *srcDimsTensor)

{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int srcLength = srcDimsTensor[id_z * 2];
    int channels = srcDimsTensor[id_z * 2 + 1];

    if (id_x >= srcLength)
        return;

    // multi channel
    if(channels > 1)
    {
        float nomalizedWeight = 1.f / channels;
        float outVal = 0.0f;
        uint srcIdx = id_z * srcStride + id_x * channels;
        for(int i = 0; i < channels; i++, srcIdx++)
            outVal += srcPtr[srcIdx] * nomalizedWeight;

        uint dstIdx = id_z * dstStride + id_x;
        dstPtr[dstIdx] = outVal;
    }
    // single channel - copy input to output
    else
    {
        uint srcIdx = id_z * srcStride + id_x;
        uint dstIdx = id_z * dstStride + id_x;
        dstPtr[dstIdx] = srcPtr[srcIdx];
    }
}

RppStatus hip_exec_down_mixing_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      bool normalizeWeights,
                                      rpp::Handle& handle)
{
    int globalThreads_x = dstDescPtr->strides.nStride;
    int globalThreads_y = 1;
    int globalThreads_z = dstDescPtr->n;

    hipLaunchKernelGGL(down_mixing_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       dstPtr,
                       dstDescPtr->strides.nStride,
                       srcDimsTensor);

    return RPP_SUCCESS;
}