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
#include "api_helpers.hpp"

// -------------------- Set 0 - grid_dropout main kernels --------------------

template <typename T>
__global__ void grid_dropout_pkd_hip_tensor(T *dstPtr,
                                            uint2 dstStridesNH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            uint boxesInEachImage)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batch_idx = id_z / boxesInEachImage;

    if ((id_y >= (anchorBoxInfoTensor[id_z].rb.y - anchorBoxInfoTensor[id_z].lt.y + 1)) || (id_x >= (anchorBoxInfoTensor[id_z].rb.x - anchorBoxInfoTensor[id_z].lt.x + 1)))
        return;

    uint dstIdx = batch_idx * dstStridesNH.x;
    dstIdx += (id_y + anchorBoxInfoTensor[id_z].lt.y) * dstStridesNH.y + (id_x + anchorBoxInfoTensor[id_z].lt.x) * 3; 

    dstPtr[dstIdx] = 0.0f;
    dstPtr[dstIdx + 1] = 0.0f;
    dstPtr[dstIdx + 2] = 0.0f;
}

template <typename T>
__global__ void grid_dropout_pln_hip_tensor(T *dstPtr,
                                            uint3 dstStridesNCH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            uint boxesInEachImage)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batch_idx = id_z / boxesInEachImage;

    if ((id_y >= (anchorBoxInfoTensor[id_z].rb.y - anchorBoxInfoTensor[id_z].lt.y + 1)) || (id_x >= (anchorBoxInfoTensor[id_z].rb.x - anchorBoxInfoTensor[id_z].lt.x + 1)))
        return;

    uint dstIdx = batch_idx * dstStridesNCH.x;
    dstIdx += (id_y + anchorBoxInfoTensor[id_z].lt.y) * dstStridesNCH.z + (id_x + anchorBoxInfoTensor[id_z].lt.x); 

    dstPtr[dstIdx] = 0.0f;
    dstPtr[dstIdx + dstStridesNCH.y] = 0.0f;
    dstPtr[dstIdx + dstStridesNCH.y * 2] = 0.0f;
}

template <typename T>
__global__ void grid_dropout_pln1_hip_tensor(T *dstPtr,
                                            uint3 dstStridesNCH,
                                            RpptRoiLtrb *anchorBoxInfoTensor,
                                            uint boxesInEachImage)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batch_idx = id_z / boxesInEachImage;

    if ((id_y >= (anchorBoxInfoTensor[id_z].rb.y - anchorBoxInfoTensor[id_z].lt.y + 1)) || (id_x >= (anchorBoxInfoTensor[id_z].rb.x - anchorBoxInfoTensor[id_z].lt.x + 1)))
        return;

    uint dstIdx = batch_idx * dstStridesNCH.x;
    dstIdx += (id_y + anchorBoxInfoTensor[id_z].lt.y) * dstStridesNCH.z + (id_x + anchorBoxInfoTensor[id_z].lt.x); 

    dstPtr[dstIdx] = 0.0f;
}

// -------------------- Set 1 - Kernel Executors --------------------
template <typename T>
RppStatus hip_exec_grid_dropout_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32u gridW,
                                Rpp32u gridH,
                                Rpp32f holeRatio,
                                bool randomOffset,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    Rpp32u boxesInEachImage = gridH * gridW;
    Rpp32u totalBoxes = srcDescPtr->n * boxesInEachImage;

    RpptRoiLtrb *anchorBoxInfoTensor = new RpptRoiLtrb[totalBoxes];
    RpptRoiLtrb *d_anchorBoxInfoTensor;
    hipMalloc(&d_anchorBoxInfoTensor, totalBoxes * sizeof(RpptRoiLtrb));

    Rpp32u maxHoleW = 0, maxHoleH = 0;
    init_grid_dropout(srcDescPtr->n, anchorBoxInfoTensor, roiTensorPtrSrc, gridH, gridW, maxHoleW, maxHoleH, holeRatio, randomOffset);
    hipMemcpy(d_anchorBoxInfoTensor, anchorBoxInfoTensor, totalBoxes * sizeof(RpptRoiLtrb), hipMemcpyHostToDevice);

    int globalThreads_x = maxHoleW;
    int globalThreads_y = maxHoleH;
    int globalThreads_z = totalBoxes;
    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        }
        else if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();
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
            globalThreads_x = maxHoleW;
            globalThreads_y = maxHoleH;
            globalThreads_z = totalBoxes;
        }

        hipLaunchKernelGGL(grid_dropout_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           d_anchorBoxInfoTensor,
                           boxesInEachImage);
    }
    else if((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->c == 1))
    {
        hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        hipLaunchKernelGGL(grid_dropout_pln1_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           d_anchorBoxInfoTensor,
                           boxesInEachImage);
    }
    else if((dstDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->c == 3))
    {
        // if src layout is NHWC, copy src to dst
        if (srcDescPtr->layout == RpptLayout::NCHW)
        {
            hipMemcpyAsync(dstPtr, srcPtr, static_cast<size_t>(srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(T)), hipMemcpyDeviceToDevice, handle.GetStream());
        }
        else if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            globalThreads_x = (dstDescPtr->w + 7) >> 3;
            globalThreads_y = dstDescPtr->h;
            globalThreads_z = handle.GetBatchSize();
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
            globalThreads_x = maxHoleW;
            globalThreads_y = maxHoleH;
            globalThreads_z = totalBoxes;
        }

        hipLaunchKernelGGL(grid_dropout_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x / LOCAL_THREADS_X), ceil((float)globalThreads_y / LOCAL_THREADS_Y), ceil((float)globalThreads_z / LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride,dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           d_anchorBoxInfoTensor,
                           boxesInEachImage);
    }

    hipFree(d_anchorBoxInfoTensor);
    delete[] anchorBoxInfoTensor;
    return RPP_SUCCESS;
}

template RppStatus hip_exec_grid_dropout_tensor<Rpp8u>(Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp8u*,
                                                       RpptDescPtr,
                                                       Rpp32u,
                                                       Rpp32u,
                                                       Rpp32f,
                                                       bool,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_grid_dropout_tensor<half>(half*,
                                                     RpptDescPtr,
                                                     half*,
                                                     RpptDescPtr,
                                                     Rpp32u,
                                                     Rpp32u,
                                                     Rpp32f,
                                                     bool,
                                                     RpptROIPtr,
                                                     RpptRoiType,
                                                     rpp::Handle&);

template RppStatus hip_exec_grid_dropout_tensor<Rpp32f>(Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32u,
                                                         Rpp32u,
                                                         Rpp32f,
                                                         bool,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_grid_dropout_tensor<Rpp8s>(Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp8s*,
                                                       RpptDescPtr,
                                                       Rpp32u,
                                                       Rpp32u,
                                                       Rpp32f,
                                                       bool,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);
