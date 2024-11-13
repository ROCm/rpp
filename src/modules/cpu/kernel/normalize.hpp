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

// Computes normalize for ND non toggle variants for i8 dataype
inline void normalize_ND_tensor_nontoggle(Rpp8s *srcPtr, Rpp32u *srcStride, Rpp32f *dstPtr, Rpp32f *meanPtr, Rpp32f *multiplierPtr,
                                   Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u tensorDim, Rpp32u level, Rpp32u& idx)
{
    Rpp32u idx1 = 0;
    if(tensorDim == 1)
    {
        for(Rpp32u k = 0; k < length[level]; k++)
        {
            *dstPtr++ = (((Rpp32f)(*srcPtr + 128) - meanPtr[idx]) * multiplierPtr[idx]) + shift;
            if(k < length[level] - 1)
                idx += paramStride[level];
            srcPtr++;
        }
    }
    else
    {
        idx1 = idx;
        for (Rpp32u i = 0; i < length[level]; i++)
        {
            normalize_ND_tensor_nontoggle(srcPtr, srcStride, dstPtr, meanPtr, multiplierPtr, shift, paramStride, length, tensorDim - 1, level + 1, idx);
            if(i < length[level] - 1)
                idx = (!paramStride[level]) ? idx1 : idx + paramStride[level];
            dstPtr += srcStride[level];
            srcPtr += srcStride[level];
        }
    }
}

// Computes normalize for ND non toggle variants
template<typename T1, typename T2>
inline void normalize_ND_tensor_nontoggle(T1 *srcPtr, Rpp32u *srcStride, T2 *dstPtr, Rpp32f *meanPtr, Rpp32f *multiplierPtr,
                                   Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u tensorDim, Rpp32u level, Rpp32u& idx)
{
    Rpp32u idx1 = 0;
    if(tensorDim == 1)
    {
        T1 *srcPtrTemp = srcPtr;
        T2 *dstPtrTemp = dstPtr;

        for(Rpp32u k = 0; k < length[level]; k++)
        {
            *dstPtrTemp = (((T2)*srcPtrTemp - meanPtr[idx]) * multiplierPtr[idx]) + shift;
            if(k < length[level] - 1)
                idx += paramStride[level];
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }
    else
    {
        idx1 = idx;
        for (Rpp32u i = 0; i < length[level]; i++)
        {
            normalize_ND_tensor_nontoggle(srcPtr, srcStride, dstPtr, meanPtr, multiplierPtr, shift, paramStride, length, tensorDim - 1, level + 1, idx);
            if(i < length[level] - 1)
                idx = (!paramStride[level]) ? idx1 : idx + paramStride[level];
            dstPtr += srcStride[level];
            srcPtr += srcStride[level];
        }
    }
}

// Recursive reduction helper function to compute difference of input with mean and squares them up
template<typename T>
void compute_diff_square_sum(Rpp32f &output, T *input, Rpp32s inputStride, Rpp32s numElements, Rpp32f mean)
{
    if (numElements > 32)
    {
        Rpp32s currElements = numElements >> 1;
        Rpp32f tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_diff_square_sum(tmp1, input, inputStride, currElements, mean);

        // reduce second half and accumulate
        compute_diff_square_sum(tmp2, input + currElements * inputStride, inputStride, numElements - currElements, mean);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        Rpp32f tmp = 0;
        for (Rpp32s i = 0; i < numElements; i++)
        {
            Rpp32f curr = (input[i * inputStride] - mean);
            auto curSq = curr * curr;
            tmp += curSq;
        }

        // accumulate in target value
        output += tmp;
    }
}

// Computes inverse stddev for ND inputs
template<typename T>
void compute_ND_stddev(T *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride, Rpp32u *axis, Rpp32u tensorDim, Rpp32u level, Rpp32u index, Rpp32u size, Rpp32u norm, Rpp32u lastNormAxis)
{
    if((level == (tensorDim - 1)) && axis[tensorDim - 1]) // Calls computeDiffSumSquare when last dimension is to be normalized
        compute_diff_square_sum(stdDevPtr[index], srcPtr, stride[level], dims[level], meanPtr[index]);
    else if(level == tensorDim) // Calls computeDiffSumSquare when only 1st axis need to be normalized
        compute_diff_square_sum(stdDevPtr[index], srcPtr, stride[norm], dims[norm], meanPtr[index]);
    else if (!axis[level]) // When that axis at present level isn't normalized, split srcPtr and modify index to store stddev
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_stddev(srcPtr + (i * stride[level]), meanPtr, stdDevPtr, dims, stride, axis, tensorDim, level + 1, index + (i * (size / dims[level])), size / dims[level], norm, lastNormAxis);
    }
    else if(axis[level] && (level == lastNormAxis)) // Increment level alone if its last axis to be normalized
        compute_ND_stddev(srcPtr, meanPtr, stdDevPtr, dims, stride, axis, tensorDim, level + 1, index, size, level, lastNormAxis);
    else if(axis[level]) // Called when axis at present level needs to be normalized
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_stddev(srcPtr + (i * stride[level]), meanPtr, stdDevPtr, dims, stride, axis, tensorDim, level + 1, index, size, level, lastNormAxis);
    }
}

// Recursive reduction helper function to sum up input values
template<typename T>
void compute_sum(Rpp32f &output, T *input, Rpp32s inputStride, Rpp32s numElements)
{
    if (numElements > 32)
    {
        Rpp32s currElements = numElements >> 1;
        Rpp32f tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_sum(tmp1, input, inputStride, currElements);

        // reduce second half and accumulate
        compute_sum(tmp2, input + currElements * inputStride, inputStride, numElements - currElements);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        Rpp32f tmp = 0;
        for (Rpp32s i = 0; i < numElements; i++)
            tmp += input[i * inputStride];

        // accumulate in target value
        output += tmp;
    }
}

// Computes mean for ND inputs
template<typename T>
void compute_ND_mean(T *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride, Rpp32u *axis, Rpp32u tensorDim, Rpp32u level, Rpp32u index, Rpp32u size, Rpp32u norm, Rpp32u lastNormAxis)
{
    if((level == (tensorDim - 1)) && axis[tensorDim - 1]) // Calls computeSum when last dimension is to be normalized
        compute_sum(meanPtr[index], srcPtr, stride[level], dims[level]);
    else if(level == tensorDim) // Calls computeSum when only 1st axis need to be normalized
        compute_sum(meanPtr[index], srcPtr, stride[norm], dims[norm]);
    else if (!axis[level]) // When that axis at present level isn't normalized, split srcPtr and modify index to store mean
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_mean(srcPtr + (i * stride[level]), meanPtr, dims, stride, axis, tensorDim, level + 1, index + (i * (size / dims[level])), size / dims[level], norm, lastNormAxis);
    }
    else if(axis[level] && (level == lastNormAxis)) // Increment level alone if its last axis to be normalized
        compute_ND_mean(srcPtr, meanPtr, dims, stride, axis, tensorDim, level + 1, index, size, level, lastNormAxis);
    else if(axis[level]) // Called when axis at present level needs to be normalized
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_mean(srcPtr + (i * stride[level]), meanPtr, dims, stride, axis, tensorDim, level + 1, index, size, level, lastNormAxis);
    }
}

// Performs collapse axis operation wherein continuous axis that require normalization are combined together
inline void collapse_axis(Rpp32u *tensorDim, Rpp32u *axis, Rpp32u *length, Rpp32u *newAxis, Rpp32u *newDims, Rpp32u *lastNormAxis)
{
    int skipped = 0, prev = -2, k = 0;
    for(Rpp32u i = 0; i < *tensorDim; i++)
    {
        if(axis[i])
        {
            int temp = i - skipped;
            if(temp != prev + 1)
            {
                newAxis[k] = 1;
                newDims[k] = length[i];
                prev = i;
                k++;
            }
            else if(prev >= 0)
            {
                newDims[prev] *= length[i];
                skipped++;
            }
        }
        else
        {
            newDims[k] = length[i];
            k++;
        }
    }
    *tensorDim -= skipped;
    for(Rpp32u i = 0; i < *tensorDim; i++)
    {
        if(newAxis[i])
            *lastNormAxis = i;
    }
}

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