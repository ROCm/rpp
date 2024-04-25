#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void pre_emphasis_filter_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *coeff_f4)
{
    dst_f8->f4[0] = src1_f8->f4[0]  - *coeff_f4 * src2_f8->f4[0];
    dst_f8->f4[1] = src1_f8->f4[1]  - *coeff_f4 * src2_f8->f4[1];
}

__global__ void pre_emphasis_filter_tensor(float *srcPtr,
                                           uint2 srcStridesNH,
                                           float *dstPtr,
                                           uint2 dstStridesNH,
                                           RpptImagePatchPtr srcDims,
                                           float *coeffTensor,
                                           RpptAudioBorderType borderType)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8 + 1;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_x >= srcDims[id_z].width) || (id_y >= srcDims[id_z].height))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    float4 coeff_f4 = (float4)coeffTensor[id_z];
    d_float8 src1_f8, src2_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx - 1, &src2_f8);
    pre_emphasis_filter_hip_compute(&src1_f8, &src2_f8, &dst_f8, &coeff_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

RppStatus hip_exec_pre_emphasis_filter_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr srcDims,
                                              RpptAudioBorderType borderType,
                                              rpp::Handle& handle)
{
    int globalThreads_x = (dstDescPtr->w + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    float *coeff = handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem;

    for(int i = 0; i < srcDescPtr->n; i++)
    {
        int id_x = i * srcDescPtr->strides.nStride;
        if(borderType == RpptAudioBorderType::ZERO)
            dstPtr[id_x] = srcPtr[id_x];
        else
        {
            float border = (borderType == RpptAudioBorderType::CLAMP) ? srcPtr[id_x] : srcPtr[id_x + 1];
            dstPtr[id_x] = srcPtr[id_x] - coeff[id_x] * border;
        }
    }

    hipLaunchKernelGGL(pre_emphasis_filter_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       srcDims,
                       coeff,
                       borderType);

    return RPP_SUCCESS;
}