/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

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

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

template<typename T>
void transpose_generic_nd_recursive(T *dst, Rpp32u *dstStrides, T *src, Rpp32u *srcStrides, Rpp32u *dstShape, Rpp32u tensorDims)
{
    // exit case for recursion
    if (tensorDims == 0)
    {
        *dst = *src;
    }
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            transpose_generic_nd_recursive(dst, dstStrides + 1, src, srcStrides + 1, dstShape + 1, tensorDims - 1);
            dst += *dstStrides;
            src += *srcStrides;
        }
    }
}

template<typename T>
void transpose_generic_setup_and_run(T *srcPtrTemp, T *dstPtrTemp, Rpp32u *length, Rpp32u *perm, Rpp32u tensorDims)
{
    Rpp32u dstShape[RPPT_MAX_DIMS];
    Rpp32u srcStrides[RPPT_MAX_DIMS];
    Rpp32u dstStrides[RPPT_MAX_DIMS];

    // compute output shape
    for(Rpp32u i = 0; i < tensorDims; i++)
        dstShape[i] = length[perm[i]];

    // compute output strides
    compute_strides(dstStrides, dstShape, tensorDims);

    // compute input strides and update as per the permute order
    Rpp32u tempStrides[RPPT_MAX_DIMS];
    compute_strides(tempStrides, length, tensorDims);
    for(int i = 0; i < tensorDims; i++)
        srcStrides[i] = tempStrides[perm[i]];

    // perform transpose as per the permute order
    transpose_generic_nd_recursive(dstPtrTemp, dstStrides, srcPtrTemp, srcStrides, dstShape, tensorDims);
}

template<typename T>
RppStatus transpose_generic_host_tensor(T *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = dstGenericDescPtr->numDims - 1;  // exclude batchsize from input dims
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // get the starting address of begin and length values from roiTensor
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *perm = permTensor;

        bool copyInput = true;
        for(int i = 0; i < tensorDims; i++)
            copyInput *= (perm[i] == i);

        // do memcpy of input to output since output order is same as input order
        if(copyInput)
        {
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcGenericDescPtr->strides[0] * sizeof(T)));
        }
        else
        {
            for(int i = 1; i < tensorDims; i++)
                srcPtrTemp += begin[i - 1] * srcGenericDescPtr->strides[i];
            transpose_generic_setup_and_run(srcPtrTemp, dstPtrTemp, length, perm, tensorDims);
        }
    }

    return RPP_SUCCESS;
}

RppStatus transpose_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle);
