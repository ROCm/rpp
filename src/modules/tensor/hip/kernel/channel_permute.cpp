/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_tensor_executors.hpp"

__device__ void channel_permute_hip_compute(d_float24 *pix_f24, uint3  permutationTensor)
{
    // Temporary structure to hold swapped values
    d_float24 pixSwap_f24;

    // Reorder channels based on permutationTensor
    pixSwap_f24.f8[0] = pix_f24->f8[permutationTensor.x]; // Map R
    pixSwap_f24.f8[1] = pix_f24->f8[permutationTensor.y]; // Map G
    pixSwap_f24.f8[2] = pix_f24->f8[permutationTensor.z]; // Map B

    // Write back the swapped values to pix_f24
    pix_f24->f8[0] = pixSwap_f24.f8[0]; // Write R
    pix_f24->f8[1] = pixSwap_f24.f8[1]; // Write G
    pix_f24->f8[2] = pixSwap_f24.f8[2]; // Write B
}

template <typename T>
__global__ void channel_permute_pkd_hip_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               uint2 maxDim,
                                               uint *permutationTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    d_float24 pix_f24;

    // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in the batch
    uint3 permutationOrder = ((uint3*)permutationTensor)[id_z];

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    channel_permute_hip_compute(&pix_f24, permutationOrder);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void channel_permute_pln_hip_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               uint2 maxDim,
                                               uint *permutationTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in the batch
    uint3 permutationOrder = ((uint3*)permutationTensor)[id_z];

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    channel_permute_hip_compute(&pix_f24, permutationOrder);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void channel_permute_pkd3_pln3_hip_tensor(T *srcPtr,
                                                     uint2 srcStridesNH,
                                                     T *dstPtr,
                                                     uint3 dstStridesNCH,
                                                     uint2 maxDim,
                                                     uint *permutationTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in the batch
    uint3 permutationOrder = ((uint3*)permutationTensor)[id_z];

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    channel_permute_hip_compute(&pix_f24, permutationOrder);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void channel_permute_pln3_pkd3_hip_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     uint2 maxDim,
                                                     uint *permutationTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + (id_y * srcStridesNCH.z) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);

    d_float24 pix_f24;

    // Load the 3-channel permutation order (e.g., {0, 2, 1} for R-B-G) for the current image in the batch
    uint3 permutationOrder = ((uint3*)permutationTensor)[id_z];

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    channel_permute_hip_compute(&pix_f24, permutationOrder);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_channel_permute_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32u *permutationTensor,
                                          rpp::Handle& handle)
{
    if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
            hipLaunchKernelGGL(channel_permute_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               permutationTensor);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(channel_permute_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               permutationTensor);
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(channel_permute_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               permutationTensor);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(channel_permute_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               make_uint2(srcDescPtr->w, srcDescPtr->h),
                               permutationTensor);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_channel_permute_tensor<Rpp8u>(Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp32u*,
                                                          rpp::Handle&);

template RppStatus hip_exec_channel_permute_tensor<half>(half*,
                                                         RpptDescPtr,
                                                         half*,
                                                         RpptDescPtr,
                                                         Rpp32u*,
                                                         rpp::Handle&);

template RppStatus hip_exec_channel_permute_tensor<Rpp32f>(Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32u*,
                                                           rpp::Handle&);

template RppStatus hip_exec_channel_permute_tensor<Rpp8s>(Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp32u*,
                                                          rpp::Handle&);
