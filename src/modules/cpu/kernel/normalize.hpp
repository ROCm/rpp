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
#ifndef NORMALIZE_HPP
#define NORMALIZE_HPP
#include "rppdefs.h"
#include "rpp_cpu_common.hpp"

template<typename T1, typename T2>
RppStatus normalize_generic_host_tensor(T1 *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T2 *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32f *meanTensorPtr,
                                        Rpp32f *stdDevTensorPtr,
                                        Rpp8u computeMeanStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    for(int batch = 0; batch < batchSize; batch++)
    {
        Rpp32u size = 1; // length of input tensors differ based on axisMask and tensorDims
        for(int i = 0; i < tensorDims; i++)
            size *= ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : roiTensor[(tensorDims * 2 * batch) + tensorDims + i];
        maxSize = std::max(maxSize, size);
    }
    if(!computeMeanStddev)
    {
        for(Rpp32u i = 0; i < maxSize; i++)
            stdDevTensorPtr[i] = (!stdDevTensorPtr[i])? 1.0f : scale / stdDevTensorPtr[i];
        maxSize = 0;
    }

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        int size = 1;
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];

        for(int i = 0; i < tensorDims; i++)
            size *= ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : length[i];

        T1 *srcPtrTemp;
        T2 *dstPtrTemp;
        Rpp32f *meanTensor, *stdDevTensor;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];
        meanTensor = meanTensorPtr + batchCount * maxSize;
        stdDevTensor = stdDevTensorPtr + batchCount * maxSize;

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        T1 *srcPtrChannel = srcPtrTemp;

        int totalElements = 1;
        Rpp32u lastNormAxis = 0;
        Rpp32u axis[tensorDims], newAxis[tensorDims], newDims[tensorDims];
        // Initialize newAxis and newDims used to store final Axis and Dims after removing redundant axis
        memset(newAxis, 0, sizeof(newAxis));
        memset(newDims, 0, sizeof(newDims));

        for(int i = 0; i < tensorDims; i++)
        {
            axis[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;
            totalElements *= axis[i] ? length[i] : 1;
            srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i + 1];
        }

        Rpp32u paramStride[tensorDims], srcStride[tensorDims];
        Rpp32u newTensorDims = tensorDims;
        collapse_axis(&newTensorDims, axis, length, newAxis, newDims, &lastNormAxis);
        compute_strides(srcStride, newDims, newTensorDims);

        if(computeMeanStddev & 1) // Check if mean is to be computed internally
        {
            compute_ND_mean(srcPtrChannel, meanTensor, newDims, srcStride, newAxis, newTensorDims, 0, 0, size, 0, lastNormAxis);
            Rpp32f normFactor = 1.0 / totalElements;
            for(int i = 0; i < size; i++)
                meanTensor[i] *= normFactor;
        }
        if(computeMeanStddev & 2) // Check if stddev is to be computed internally
        {
            compute_ND_stddev(srcPtrChannel, meanTensor, stdDevTensor, newDims, srcStride, newAxis, newTensorDims, 0, 0, size, 0, lastNormAxis);
            Rpp32f normFactor = (Rpp32f)(1.0 / totalElements);
            rpp_rsqrt_avx(stdDevTensor, (Rpp32s)size, 0, normFactor, scale);
        }

        for(int i = 0; i < newTensorDims; i++)
            paramStride[i] = !newAxis[i];

        Rpp32u idx = 0;
        normalize_ND_tensor_nontoggle(srcPtrChannel, srcStride, dstPtrTemp, meanTensor, stdDevTensor, shift, paramStride, newDims, newTensorDims, 0, idx);
    }

    return RPP_SUCCESS;
}

RppStatus normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32f *meanTensorPtr,
                                        Rpp32f *stdDevTensorPtr,
                                        Rpp8u computeMeanStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);
#endif