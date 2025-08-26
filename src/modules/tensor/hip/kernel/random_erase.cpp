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

__device__ inline uint generate_seed(uint x, uint y, uint z, uint *seed)
{
    return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) * (*seed);
}

__device__ inline float generate_random_float(uint seed)
{
    seed = (1103515245u * seed + 12345u);
    return static_cast<float>(seed & 0xFFFFFF) / static_cast<float>(0x1000000);
}

__device__ inline uint generate_random_int(uint seed)
{
    seed = (1103515245 * seed + 12345);
    return (seed >> 16) & 0xFF;
}

// -------------------- Set 0 - random_erase main kernels --------------------
template <typename T>
__global__ void random_erase_pkd_hip_tensor(T *dstPtr,
                                            uint2 dstStridesNH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            Rpp32u *numBoxesTensor,
                                            uint *rd_seed,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    auto &roi = roiTensorPtrSrc[id_z].xywhROI;
    if (id_x >= roi.roiWidth || id_y >= roi.roiHeight)
        return;

    uint seed = generate_seed(id_x, id_y, id_z, rd_seed);
    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = id_z * dstStridesNH.x + id_y * dstStridesNH.y + id_x * 3;

    for (int i = 0; i < numBoxes; i++)
    {
        int temp = id_z * numBoxes + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x &&
            id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            if constexpr (std::is_same<T, Rpp8u>::value)
            {
                dstPtr[dstIdx]     = static_cast<Rpp8u>(generate_random_int(seed + 0));
                dstPtr[dstIdx + 1] = static_cast<Rpp8u>(generate_random_int(seed + 1));
                dstPtr[dstIdx + 2] = static_cast<Rpp8u>(generate_random_int(seed + 2));
            }
            else if constexpr (std::is_same<T, Rpp8s>::value)
            {
                dstPtr[dstIdx]     = static_cast<Rpp8s>(generate_random_int(seed + 0) - 128);
                dstPtr[dstIdx + 1] = static_cast<Rpp8s>(generate_random_int(seed + 1) - 128);
                dstPtr[dstIdx + 2] = static_cast<Rpp8s>(generate_random_int(seed + 2) - 128);
            }
            else if constexpr (std::is_same<T, Rpp32f>::value)
            {
                dstPtr[dstIdx]     = generate_random_float(seed + 0);
                dstPtr[dstIdx + 1] = generate_random_float(seed + 1);
                dstPtr[dstIdx + 2] = generate_random_float(seed + 2);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                dstPtr[dstIdx]     = __float2half(generate_random_float(seed + 0));
                dstPtr[dstIdx + 1] = __float2half(generate_random_float(seed + 1));
                dstPtr[dstIdx + 2] = __float2half(generate_random_float(seed + 2));
            }
            break;
        }
    }
}

template <typename T>
__global__ void random_erase_pln_hip_tensor(T *dstPtr,
                                            uint3 dstStridesNCH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            Rpp32u *numBoxesTensor,
                                            uint *rd_seed,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    auto &roi = roiTensorPtrSrc[id_z].xywhROI;
    if (id_x >= roi.roiWidth || id_y >= roi.roiHeight)
        return;

    uint seed = generate_seed(id_x, id_y, id_z, rd_seed);
    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = id_z * dstStridesNCH.x + id_y * dstStridesNCH.z + id_x;

    for (int i = 0; i < numBoxes; i++)
    {
        int temp = id_z * numBoxes + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x &&
            id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            if constexpr (std::is_same<T, Rpp8u>::value)
            {
                dstPtr[dstIdx] = static_cast<Rpp8u>(generate_random_int(seed));
            }
            else if constexpr (std::is_same<T, Rpp8s>::value)
            {
                dstPtr[dstIdx] = static_cast<Rpp8s>(generate_random_int(seed) - 128);  // Centered around 0
            }
            else if constexpr (std::is_same<T, Rpp32f>::value)
            {
                dstPtr[dstIdx] = generate_random_float(seed);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                dstPtr[dstIdx] = __float2half(generate_random_float(seed));
            }
            break;
        }
    }
}

template <typename T>
__global__ void random_erase_pln3_hip_tensor(T *dstPtr,
                                             uint3 dstStridesNCH,
                                             RpptRoiLtrb *anchorBoxInfoTensor,
                                             Rpp32u *numBoxesTensor,
                                             uint *rd_seed,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    auto &roi = roiTensorPtrSrc[id_z].xywhROI;
    if (id_x >= roi.roiWidth || id_y >= roi.roiHeight)
        return;

    uint seed = generate_seed(id_x, id_y, id_z, rd_seed);
    Rpp32u numBoxes = numBoxesTensor[id_z];
    uint dstIdx = id_z * dstStridesNCH.x + id_y * dstStridesNCH.z + id_x;

    for (int i = 0; i < numBoxes; i++)
    {
        int temp = id_z * numBoxes + i;
        if (id_x >= anchorBoxInfoTensor[temp].lt.x && id_x <= anchorBoxInfoTensor[temp].rb.x &&
            id_y >= anchorBoxInfoTensor[temp].lt.y && id_y <= anchorBoxInfoTensor[temp].rb.y)
        {
            if constexpr (std::is_same<T, Rpp8u>::value)
            {
                dstPtr[dstIdx]                      = static_cast<Rpp8u>(generate_random_int(seed + 0));
                dstPtr[dstIdx + dstStridesNCH.y]    = static_cast<Rpp8u>(generate_random_int(seed + 1));
                dstPtr[dstIdx + 2 * dstStridesNCH.y]= static_cast<Rpp8u>(generate_random_int(seed + 2));
            }
            else if constexpr (std::is_same<T, Rpp8s>::value)
            {
                dstPtr[dstIdx]                      = static_cast<Rpp8s>(generate_random_int(seed + 0) - 128);
                dstPtr[dstIdx + dstStridesNCH.y]    = static_cast<Rpp8s>(generate_random_int(seed + 1) - 128);
                dstPtr[dstIdx + 2 * dstStridesNCH.y]= static_cast<Rpp8s>(generate_random_int(seed + 2) - 128);
            }
            else if constexpr (std::is_same<T, Rpp32f>::value)
            {
                dstPtr[dstIdx]                      = generate_random_float(seed + 0);
                dstPtr[dstIdx + dstStridesNCH.y]    = generate_random_float(seed + 1);
                dstPtr[dstIdx + 2 * dstStridesNCH.y]= generate_random_float(seed + 2);
            }
            else if constexpr (std::is_same<T, half>::value)
            {
                dstPtr[dstIdx]                      = __float2half(generate_random_float(seed + 0));
                dstPtr[dstIdx + dstStridesNCH.y]    = __float2half(generate_random_float(seed + 1));
                dstPtr[dstIdx + 2 * dstStridesNCH.y]= __float2half(generate_random_float(seed + 2));
            }
            break;
        }
    }
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_random_erase_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptRoiLtrb *anchorBoxInfoTensor,
                                       Rpp32u *numBoxesTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    std::random_device rd;
    std::mt19937 gen(rd());
    uint rd_seed = 42;
    uint *rd_seed_d;
    hipMalloc(&rd_seed_d, sizeof(uint));
    hipMemcpyAsync(rd_seed_d, &rd_seed, sizeof(uint), hipMemcpyHostToDevice, handle.GetStream());

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
            hipStreamSynchronize(handle.GetStream());
        }
        // if src layout is NCHW, convert src from NCHW to NHWC
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            hipLaunchKernelGGL(convert_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            globalThreads_x = dstDescPtr->w;
            hipStreamSynchronize(handle.GetStream());
        }

        if (srcDescPtr->dataType == RpptDataType::U8)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               numBoxesTensor,
                               rd_seed_d,
                               roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::F16)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               numBoxesTensor,
                               rd_seed_d,
                               roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::F32)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               numBoxesTensor,
                               rd_seed_d,
                               roiTensorPtrSrc);
        }
        else if (srcDescPtr->dataType == RpptDataType::I8)
        {
            hipLaunchKernelGGL(random_erase_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               numBoxesTensor,
                               rd_seed_d,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 1)
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(random_erase_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           numBoxesTensor,
                           rd_seed_d,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && dstDescPtr->c == 3)
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipStreamSynchronize(handle.GetStream());
        hipLaunchKernelGGL(random_erase_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           anchorBoxInfoTensor,
                           numBoxesTensor,
                           rd_seed_d,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            hipLaunchKernelGGL(convert_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
            hipStreamSynchronize(handle.GetStream());
            globalThreads_x = dstDescPtr->w;
            hipLaunchKernelGGL(random_erase_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               anchorBoxInfoTensor,
                               numBoxesTensor,
                               rd_seed_d,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_random_erase_tensor<Rpp8u>(Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp8u*,
                                                       RpptDescPtr,
                                                       RpptRoiLtrb*,
                                                       Rpp32u*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_random_erase_tensor<half>(half*,
                                                     RpptDescPtr,
                                                     half*,
                                                     RpptDescPtr,
                                                     RpptRoiLtrb*,
                                                     Rpp32u*,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_random_erase_tensor<Rpp32f>(Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         RpptDescPtr,
                                                         RpptRoiLtrb*,
                                                         Rpp32u*,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_random_erase_tensor<Rpp8s>(Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp8s*,
                                                       RpptDescPtr,
                                                       RpptRoiLtrb*,
                                                       Rpp32u*,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);