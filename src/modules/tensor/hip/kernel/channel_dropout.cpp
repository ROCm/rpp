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
#include <random>

// -------------------- Set 0 - Dropout main kernels --------------------

template<typename T>
__device__ __forceinline__ void compute_dropout_f8(d_float8 &pix_f8, uint8_t *maskTensor, T *srcPtr)
{
    // Convert mask value (0 or 1) to float
    float4 mask_f4 = MAKE_FLOAT4(static_cast<float>(*maskTensor));

    // Multiply pixel data by mask
    pix_f8.f4[0] = pix_f8.f4[0] * mask_f4;
    pix_f8.f4[1] = pix_f8.f4[1] * mask_f4;
}

template<typename T>
__device__ __forceinline__ void compute_dropout_f24(d_float24 &pix_f24, uint8_t *maskTensor, T *srcPtr)
{
    // Convert mask values (0 or 1) for R, G, B
    uchar4 mask_uc4 = *(uchar4 *) maskTensor;
    float4 maskR_f4 = MAKE_FLOAT4(static_cast<float>(mask_uc4.x));
    float4 maskG_f4 = MAKE_FLOAT4(static_cast<float>(mask_uc4.y));
    float4 maskB_f4 = MAKE_FLOAT4(static_cast<float>(mask_uc4.z));

    // Multiply each channel’s pixels by its mask
    pix_f24.f4[0] = pix_f24.f4[0] * maskR_f4; // Red
    pix_f24.f4[1] = pix_f24.f4[1] * maskR_f4;
    
    pix_f24.f4[2] = pix_f24.f4[2] * maskG_f4; // Green
    pix_f24.f4[3] = pix_f24.f4[3] * maskG_f4;
    
    pix_f24.f4[4] = pix_f24.f4[4] * maskB_f4; // Blue
    pix_f24.f4[5] = pix_f24.f4[5] * maskB_f4;
}

__device__ __forceinline__ void compute_dropout_f8(d_float8 &pix_f8, uint8_t *maskTensor, schar *srcPtr)
{
    // Convert mask value (0 or 1) to float
    float mask = static_cast<float>(*maskTensor);
    float4 mask_f4 = FLOAT4_I8_MIN_VALUE;

    pix_f8.f4[0] = mask ? pix_f8.f4[0] : mask_f4;
    pix_f8.f4[1] = mask ? pix_f8.f4[1] : mask_f4;
}

__device__ __forceinline__ void compute_dropout_f24(d_float24 &pix_f24, uint8_t *maskTensor, schar *srcPtr)
{
    // Convert mask values (0 or 1) for R, G, B
    uchar4 mask_uc4 = *(uchar4 *) maskTensor;
    float maskR = static_cast<float>(mask_uc4.x);
    float maskG = static_cast<float>(mask_uc4.y);
    float maskB = static_cast<float>(mask_uc4.z);
    
    float4 mask_f4 = FLOAT4_I8_MIN_VALUE;

    pix_f24.f4[0] = maskR ? pix_f24.f4[0] : mask_f4; // Red
    pix_f24.f4[1] = maskR ? pix_f24.f4[1] : mask_f4;
    
    pix_f24.f4[2] = maskG ? pix_f24.f4[2] : mask_f4; // Green
    pix_f24.f4[3] = maskG ? pix_f24.f4[3] : mask_f4;
    
    pix_f24.f4[4] = maskB ? pix_f24.f4[4] : mask_f4; // Blue
    pix_f24.f4[5] = maskB ? pix_f24.f4[5] : mask_f4;
}

template <typename T>
__global__ void channel_dropout_pkd_hip_tensor(T *srcPtr,
                                               uint2 srcStridesNH,
                                               T *dstPtr,
                                               uint2 dstStridesNH,
                                               uint8_t *channelMask,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 dst_f24;
    uint8_t *maskTensor = channelMask + id_z * 3;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &dst_f24);
    compute_dropout_f24(dst_f24, maskTensor, srcPtr);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// PLN kernel
template <typename T>
__global__ void channel_dropout_pln_hip_tensor(T *srcPtr,
                                               uint3 srcStridesNCH,
                                               T *dstPtr,
                                               uint3 dstStridesNCH,
                                               int channelsDst,
                                               uint8_t *channelMask,
                                               RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float8 dst_f8;
    uint8_t *maskTensor = channelMask + id_z * channelsDst;
    
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &dst_f8);
    compute_dropout_f8(dst_f8, maskTensor, srcPtr);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &dst_f8);
        compute_dropout_f8(dst_f8, maskTensor + 1, srcPtr);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &dst_f8);
        compute_dropout_f8(dst_f8, maskTensor + 2, srcPtr);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void channel_dropout_pkd3_pln3_hip_tensor(T *srcPtr,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    uint8_t *channelMask,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 dst_f24;
    uint8_t *maskTensor = channelMask + id_z * 3;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &dst_f24);
    compute_dropout_f24(dst_f24, maskTensor, srcPtr);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}


template <typename T>
__global__ void channel_dropout_pln3_pkd3_hip_tensor(T *srcPtr,
                                                     uint3 srcStridesNCH,
                                                     T *dstPtr,
                                                     uint2 dstStridesNH,
                                                     uint8_t *channelMask,
                                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        return;

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 dst_f24;
    uint8_t *maskTensor = channelMask + id_z * 3;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &dst_f24);
    compute_dropout_f24(dst_f24, maskTensor, srcPtr);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_channel_dropout_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32f *dropoutProbability,
                                          bool randomSeed,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle &handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    // Generate channel mask on host
    uint8_t *channelMaskHost = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    int seed = randomSeed ? std::random_device{}() : DROPOUT_FIXED_SEED; // Use a true random seed if requested, otherwise use the fixed seed for deterministic QA
    Rpp32u numThreads = handle.GetNumThreads();

#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        std::mt19937 gen(seed + batchCount);
        std::bernoulli_distribution keepDist(1.0f - dropoutProbability[batchCount]); // Distribution for the probability of keeping or dropping a channel
        bool atLeastOneChannelKept = false; // Flag to track if all channels were dropped, to ensure at least one is kept
        int base = batchCount * srcDescPtr->c;
        for (int c = 0; c < srcDescPtr->c; c++)
        {
            channelMaskHost[base + c] = keepDist(gen);
            atLeastOneChannelKept |= channelMaskHost[base + c];
        }
        // Ensure at least one channel is kept
        if (!atLeastOneChannelKept)
            channelMaskHost[base + (gen() % srcDescPtr->c)] = 1;
    }

    uint8_t *d_channelMask = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mgpu.scratchBufferHip.floatmem);
    CHECK_RETURN_STATUS(hipMemcpyAsync(d_channelMask, channelMaskHost, dstDescPtr->n * srcDescPtr->c * sizeof(uint8_t), hipMemcpyHostToDevice, handle.GetStream()));

    if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NHWC && srcDescPtr->c == 3)
    {
        hipLaunchKernelGGL(channel_dropout_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           d_channelMask,
                           roiTensorPtrSrc);
    }
    else if (srcDescPtr->layout == RpptLayout::NHWC && dstDescPtr->layout == RpptLayout::NCHW && srcDescPtr->c == 3)
    {
        hipLaunchKernelGGL(channel_dropout_pkd3_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), globalThreads_z),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           d_channelMask,
                           roiTensorPtrSrc);
    }
    else if (srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NHWC && srcDescPtr->c == 3)
    {
        globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
        hipLaunchKernelGGL(channel_dropout_pln3_pkd3_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           d_channelMask,
                           roiTensorPtrSrc);
    }
    else
    {
        hipLaunchKernelGGL(channel_dropout_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           d_channelMask,
                           roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}

template RppStatus hip_exec_channel_dropout_tensor<Rpp8u>(Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp8u*,
                                                          RpptDescPtr,
                                                          Rpp32f*,
                                                          bool,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);

template RppStatus hip_exec_channel_dropout_tensor<Rpp8s>(Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp8s*,
                                                          RpptDescPtr,
                                                          Rpp32f*,
                                                          bool,
                                                          RpptROIPtr,
                                                          RpptRoiType,
                                                          rpp::Handle&);

template RppStatus hip_exec_channel_dropout_tensor<Rpp32f>(Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32f*,
                                                           RpptDescPtr,
                                                           Rpp32f*,
                                                           bool,
                                                           RpptROIPtr,
                                                           RpptRoiType,
                                                           rpp::Handle&);

template RppStatus hip_exec_channel_dropout_tensor<half>(half*,
                                                         RpptDescPtr,
                                                         half*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         bool,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);
