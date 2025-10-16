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

template <typename T>
__global__ void concat_generic_hip_tensor(T *srcPtr1,
                                          T *srcPtr2,
                                          uint *srcTensor1Strides,
                                          uint *srcTensor2Strides,
                                          T *dstPtr,
                                          uint *dstStrides,
                                          uint axis,
                                          uint numDims,
                                          Rpp32u *roiTensor,
                                          Rpp32u *srcOffsets,
                                          Rpp32u *dstOffsets)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Get ROI information for both source tensors
    uint *roi1 = roiTensor + id_z * numDims * 4;  // First tensor ROI
    uint *roi2 = roi1 + numDims * 2;              // Second tensor ROI
    uint *begin1 = roi1;
    uint *length1 = &roi1[numDims];
    uint *begin2 = roi2;
    uint *length2 = &roi2[numDims];
    
    // Calculate total elements for this batch
    uint totalElements = 1;
    for (int i = 0; i < numDims; i++)
    {
        if (i == axis)
            totalElements *= (length1[i] + length2[i]);
        else
            totalElements *= length1[i];
    }
    
    if(id_x >= totalElements)
        return;

    // Get batch-specific offsets
    uint srcOffset1 = srcOffsets[id_z * 2];
    uint srcOffset2 = srcOffsets[id_z * 2 + 1];
    uint dstOffset = dstOffsets[id_z];
    
    uint dstIdx = dstOffset;
    uint srcIdx1 = srcOffset1;
    uint srcIdx2 = srcOffset2;
    uint coords[RPPT_MAX_DIMS];

    // Calculate strides for this specific batch
    uint src1Strides[RPPT_MAX_DIMS];
    uint src2Strides[RPPT_MAX_DIMS];
    uint dstStridesLocal[RPPT_MAX_DIMS];
    
    for (int i = numDims - 1; i >= 0; i--)
    {
        if (i == numDims - 1)
        {
            src1Strides[i] = 1;
            src2Strides[i] = 1;
            dstStridesLocal[i] = 1;
        }
        else
        {
            src1Strides[i] = src1Strides[i + 1] * length1[i + 1];
            src2Strides[i] = src2Strides[i + 1] * length2[i + 1];
            if (i + 1 == axis)
                dstStridesLocal[i] = dstStridesLocal[i + 1] * (length1[i + 1] + length2[i + 1]);
            else
                dstStridesLocal[i] = dstStridesLocal[i + 1] * length1[i + 1];
        }
    }

    uint temp = id_x;
    for (int i = 0; i < numDims; i++)
    {
        coords[i] = temp / dstStridesLocal[i];
        temp %= dstStridesLocal[i];
        if(i < axis)
        {
            dstIdx += coords[i] * dstStridesLocal[i];
            srcIdx1 += (coords[i] + begin1[i]) * src1Strides[i];
            srcIdx2 += (coords[i] + begin2[i]) * src2Strides[i];
        }
        else if(i == axis)
        {
            if(coords[i] < length1[i])
            {
                dstIdx += coords[i] * dstStridesLocal[i];
                srcIdx1 += (coords[i] + begin1[i]) * src1Strides[i];
            }
            else
            {
                uint shifted_coord = coords[i] - length1[i];
                dstIdx += coords[i] * dstStridesLocal[i];
                srcIdx2 += (shifted_coord + begin2[i]) * src2Strides[i];
            }
        }
        else
        {
            dstIdx += coords[i] * dstStridesLocal[i];
            srcIdx1 += coords[i] * src1Strides[i];
            srcIdx2 += coords[i] * src2Strides[i];
        }
    }

    // Write to output tensor
    if(coords[axis] < length1[axis])
        dstPtr[dstIdx] = srcPtr1[srcIdx1];
    else
        dstPtr[dstIdx] = srcPtr2[srcIdx2];
}

template <typename T>
__global__ void concat_2d_hip_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     uint *srcTensor1Strides,
                                     uint *srcTensor2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *dims,
                                     Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(id_x >= dims[1] || id_y >= dims[0])
        return;

    uint dstIdx = id_y * dstStrides[1] + id_x;
    uint srcIdx1 = id_y * srcTensor1Strides[1] + id_x;
    uint srcIdx2 = id_y * srcTensor2Strides[1] + id_x;

    d_float8 src_f8, src2_f8;
    if((dims[1] - id_x) >= 8)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx + srcTensor1Strides[1], &src_f8);
    }
    else
    {
        for(int i = 0; i < (dims[1] - id_x); i++)
        {
            dstPtr[dstIdx + i] = srcPtr1[srcIdx1 + i];
            dstPtr[dstIdx + srcTensor1Strides[1] + i] = srcPtr2[srcIdx2 + i];
        }
    }
}

template <typename T>
__global__ void concat_3d_hip_tensor(T *srcPtr1,
                                     T *srcPtr2,
                                     uint *srcTensor1Strides,
                                     uint *srcTensor2Strides,
                                     T *dstPtr,
                                     uint *dstStrides,
                                     uint *dims,
                                     Rpp32u *roiTensor)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if(id_x >= dims[2] || id_y >= dims[1] || id_z >= dims[0])
        return;

    uint dstIdx = id_z * dstStrides[1] + id_y * dstStrides[2] + id_x;
    uint srcIdx1 = id_z * srcTensor1Strides[1] + id_y * srcTensor1Strides[2] + id_x;
    uint srcIdx2 = id_z * srcTensor2Strides[1] + id_y * srcTensor2Strides[2] + id_x;

    d_float8 src_f8, src2_f8;
    if((dims[2] - id_x) >= 8)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr1 + srcIdx1, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &src_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr2 + srcIdx2, &src_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx + srcTensor1Strides[2], &src_f8);
    }
    else
    {
        for(int i = 0; i < (dstStrides[2] - id_x); i++)
        {
            dstPtr[dstIdx + i] = srcPtr1[srcIdx1 + i];
            dstPtr[dstIdx + srcTensor1Strides[2] + i] = srcPtr2[srcIdx2 + i];
        }
    }
}

template <typename T>
RppStatus hip_exec_concat_tensor(T *srcPtr1,
                                 RpptGenericDescPtr srcPtr1GenericDescPtr,
                                 T *srcPtr2,
                                 RpptGenericDescPtr srcPtr2GenericDescPtr,
                                 T *dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32u axis,
                                 Rpp32u *roiTensor,
                                 rpp::Handle& handle)
{
    int globalThreads_x = dstGenericDescPtr->strides[0];
    int globalThreads_y = 1;
    int globalThreads_z = dstGenericDescPtr->dims[0];

    int numDims = dstGenericDescPtr->numDims - 1;

    if(numDims == 2)
    {
        if(axis == 0)
        {
            srcPtr1GenericDescPtr->strides[1] = srcPtr1GenericDescPtr->strides[0];
            srcPtr1GenericDescPtr->strides[0] = 1;
            srcPtr2GenericDescPtr->strides[1] = srcPtr2GenericDescPtr->strides[0];
            srcPtr2GenericDescPtr->strides[0] = 1;
            dstGenericDescPtr->strides[1] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = 1;
        }
        else if(axis == 1)
        {
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[2];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[2];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[2];
        }
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32u *begin = roi;
            Rpp32u *length = &roi[numDims];
            Rpp32u *dims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
            if(axis == 0)
            {
                dims[0] = 1;
                dims[1] = length[0] * length[1];
            }
            else if(axis == 1)
            {
                dims[0] = length[0];
                dims[1] = length[1];
            }
            globalThreads_x = dims[1];
            globalThreads_y = dims[0];
            hipLaunchKernelGGL(concat_2d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * dims[0] * dims[1]),
                               srcPtr2 + (batchCount * dims[0] * dims[1]),
                               srcPtr1GenericDescPtr->strides,
                               srcPtr2GenericDescPtr->strides,
                               dstPtr + (batchCount * dims[0] * dims[1] * 2),
                               dstGenericDescPtr->strides,
                               dims,
                               roi);
        }
    }
    else if(numDims == 3)
    {
        if(axis == 0)
        {
            srcPtr1GenericDescPtr->strides[2] = srcPtr1GenericDescPtr->strides[0];
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[1] = 1;
            srcPtr2GenericDescPtr->strides[2] = srcPtr2GenericDescPtr->strides[0];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[0];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        else if(axis == 1)
        {
            srcPtr1GenericDescPtr->strides[2] = srcPtr1GenericDescPtr->strides[1];
            srcPtr1GenericDescPtr->strides[0] = srcPtr1GenericDescPtr->strides[1] = 1;
            srcPtr2GenericDescPtr->strides[2] = srcPtr2GenericDescPtr->strides[1];
            srcPtr2GenericDescPtr->strides[0] = srcPtr2GenericDescPtr->strides[1] = 1;
            dstGenericDescPtr->strides[2] = dstGenericDescPtr->strides[1];
            dstGenericDescPtr->strides[0] = dstGenericDescPtr->strides[1] = 1;
        }
        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32u *begin = roi;
            Rpp32u *length = &roi[numDims];
            Rpp32u *dims = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
            if(axis == 0)
            {
                dims[0] = dims[1] = 1;
                dims[2] = length[0] * length[1] * length[2];
            }
            else if(axis == 1)
            {
                dims[0] = 1;
                dims[1] = length[0];
                dims[2] = length[1] * length[2];
            }
            else if(axis == 2)
            {
                dims[0] = length[0];
                dims[1] = length[1];
                dims[2] = length[2];
            }
            globalThreads_x = dims[2];
            globalThreads_y = dims[1];
            globalThreads_z = dims[0];
            hipLaunchKernelGGL(concat_3d_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr1 + (batchCount * dims[0] * dims[1] * dims[2]),
                               srcPtr2 + (batchCount * dims[0] * dims[1] * dims[2]),
                               srcPtr1GenericDescPtr->strides,
                               srcPtr2GenericDescPtr->strides,
                               dstPtr + (batchCount * dims[0] * dims[1] * dims[2] * 2),
                               dstGenericDescPtr->strides,
                               dims,
                               &roiTensor[batchCount * 6]);
        }
    }
    else
    {
        // Calculate offsets for each batch element
        Rpp32u batchSize = dstGenericDescPtr->dims[0];
        Rpp32u *srcOffsets = reinterpret_cast<Rpp32u *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
        Rpp32u *dstOffsets = srcOffsets + batchSize * 2;
        
        Rpp32u srcOffset1 = 0, srcOffset2 = 0, dstOffset = 0;
        for (int i = 0; i < batchSize; i++)
        {
            srcOffsets[i * 2] = srcOffset1;
            srcOffsets[i * 2 + 1] = srcOffset2;
            dstOffsets[i] = dstOffset;
            
            Rpp32u *roi1 = roiTensor + i * numDims * 4;
            Rpp32u *roi2 = roi1 + numDims * 2;
            Rpp32u *length1 = &roi1[numDims];
            Rpp32u *length2 = &roi2[numDims];
            
            Rpp32u src1Size = 1, src2Size = 1, dstSize = 1;
            for (int j = 0; j < numDims; j++)
            {
                src1Size *= length1[j];
                src2Size *= length2[j];
                if (j == axis)
                    dstSize *= (length1[j] + length2[j]);
                else
                    dstSize *= length1[j];
            }
            srcOffset1 += src1Size;
            srcOffset2 += src2Size;
            dstOffset += dstSize;
        }
        
        // Find max elements in any batch
        Rpp32u maxElements = 0;
        for (int i = 0; i < batchSize; i++)
        {
            Rpp32u *roi1 = roiTensor + i * numDims * 4;
            Rpp32u *roi2 = roi1 + numDims * 2;
            Rpp32u *length1 = &roi1[numDims];
            Rpp32u *length2 = &roi2[numDims];
            
            Rpp32u elements = 1;
            for (int j = 0; j < numDims; j++)
            {
                if (j == axis)
                    elements *= (length1[j] + length2[j]);
                else
                    elements *= length1[j];
            }
            maxElements = (elements > maxElements) ? elements : maxElements;
        }
        
        globalThreads_x = maxElements;
        
        hipLaunchKernelGGL(concat_generic_hip_tensor,
                       dim3(ceil((float)globalThreads_x/1024), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(1024, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr1,
                       srcPtr2,
                       srcPtr1GenericDescPtr->strides,
                       srcPtr2GenericDescPtr->strides,
                       dstPtr,
                       dstGenericDescPtr->strides,
                       axis,
                       dstGenericDescPtr->numDims - 1,
                       roiTensor,
                       srcOffsets,
                       dstOffsets);
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_concat_tensor<Rpp8u>(Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp8u*,
                                                 RpptGenericDescPtr,
                                                 Rpp32u,
                                                 Rpp32u*,
                                                 rpp::Handle&);

template RppStatus hip_exec_concat_tensor<half>(half*,
                                                RpptGenericDescPtr,
                                                half*,
                                                RpptGenericDescPtr,
                                                half*,
                                                RpptGenericDescPtr,
                                                Rpp32u,
                                                Rpp32u*,
                                                rpp::Handle&);

template RppStatus hip_exec_concat_tensor<Rpp32f>(Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32f*,
                                                  RpptGenericDescPtr,
                                                  Rpp32u,
                                                  Rpp32u*,
                                                  rpp::Handle&);

template RppStatus hip_exec_concat_tensor<Rpp8s>(Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp8s*,
                                                 RpptGenericDescPtr,
                                                 Rpp32u,
                                                 Rpp32u*,
                                                 rpp::Handle&);
