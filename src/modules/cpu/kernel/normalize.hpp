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

// Computes strides
void compute_strides(Rpp32u *strides, Rpp32u *shape, Rpp32u nDim)
{
    if (nDim > 0)
    {
        Rpp32u v = 1;
        for (Rpp32u i = nDim - 1; i > 0; i--)
        {
            strides[i] = v;
            v *= shape[i];
        }
        strides[0] = v;
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
            auto curnew = curr * curr;
            tmp += curnew;
        }

        // accumulate in target value
        output += tmp;
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

// Computes mean for 2D inputs
void compute_2D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = 1.0 / dims[1];
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        meanPtr[i] = 0;
        compute_sum(meanPtr[i], srcPtrTemp, stride[0], dims[1]);
        srcPtrTemp += stride[1];
        meanPtr[i] *= normFactor;
    }
}

// Computes inverse stddev for 2D inputs
void compute_2D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride, Rpp32f scale)
{

    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = (Rpp32f)(1.0 / dims[1]);
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        stdDevPtr[i] = 0;
        compute_diff_square_sum(stdDevPtr[i], srcPtrTemp, stride[0], dims[1], meanPtr[i]);
        srcPtrTemp += stride[1];
    }
    rpp_rsqrt_avx(stdDevPtr, (Rpp32s)dims[0], 0, normFactor, scale);
}

// Computes mean for 3D inputs
void compute_3D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride, bool isConsecutive = true)
{
    Rpp32f *srcPtrTemp = srcPtr;
    if(isConsecutive)
    {
        Rpp32f normFactor = 1.0 / dims[2];
        for(Rpp32u i = 0; i < dims[0]; i++)
        {
            float *srcPtrRow = srcPtrTemp;
            for(Rpp32u j = 0; j < dims[1]; j++)
            {
                Rpp32u index = i * dims[1] + j;
                meanPtr[index] = 0;
                compute_sum(meanPtr[index], srcPtrRow, stride[0], dims[2]);
                srcPtrRow += stride[1];
                meanPtr[index] *= normFactor;
            }
            srcPtrTemp += stride[2];
        }
    }
    else
    {
        Rpp32f normFactor = 1.0 / (dims[1] * dims[2]);
        for(Rpp32u i = 0; i < dims[0]; i++)
        {
            meanPtr[i] = 0;
            Rpp32f *srcPtrRow = srcPtrTemp;
            for(Rpp32u j = 0; j < dims[1]; j++)
            {
                compute_sum(meanPtr[i], srcPtrRow, stride[0], dims[2]);
                srcPtrRow += stride[1];
            }
            meanPtr[i] *= normFactor;
            srcPtrTemp += stride[2];
        }
    }
}

// Computes inverse stddev for 3D inputs
void compute_3D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride, Rpp32f scale, bool isConsecutive = true)
{
    Rpp32f *srcPtrTemp = srcPtr;
    if(isConsecutive)
    {
        Rpp32f normFactor = (Rpp32f)(1.0 / dims[2]);
        for(Rpp32u i = 0; i < dims[0]; i++)
        {
            float *srcPtrRow = srcPtrTemp;
            for(Rpp32u j = 0; j < dims[1]; j++)
            {
                Rpp32u index = i * dims[1] + j;
                stdDevPtr[index] = 0;
                compute_diff_square_sum(stdDevPtr[index], srcPtrRow, stride[0], dims[2], meanPtr[index]);
                srcPtrRow += stride[1];
            }
            srcPtrTemp += stride[2];
        }
        rpp_rsqrt_avx(stdDevPtr, (Rpp32s)(dims[0] * dims[1]), 0, normFactor, scale);
    }
    else
    {
        Rpp32f normFactor = (Rpp32f)(1.0 / (dims[1] * dims[2]));
        for(Rpp32u i = 0; i < dims[0]; i++)
        {
            stdDevPtr[i] = 0;
            Rpp32f *srcPtrRow = srcPtrTemp;
            for(Rpp32u j = 0; j < dims[1]; j++)
            {
                compute_diff_square_sum(stdDevPtr[i], srcPtrRow, stride[0], dims[2], meanPtr[i]);
                srcPtrRow += stride[1];
            }
            srcPtrTemp += stride[2];
        }
        rpp_rsqrt_avx(stdDevPtr, (Rpp32s)(dims[0]), 0, normFactor, scale);
    }
}

// Computes mean for ND inputs
template<typename T>
void compute_ND_mean(T *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride, Rpp32u *axis, Rpp32u nDim, Rpp32u level, Rpp32u index, Rpp32u size, Rpp32u norm, Rpp32u lastNormAxis)
{
    if((level == (nDim - 1)) && axis[nDim - 1]) // Calls computeSum when last dimension is to be normalized
        compute_sum(meanPtr[index], srcPtr, stride[level], dims[level]);
    else if(level == nDim) // Calls computeSum when only 1st axis need to be normalized
        compute_sum(meanPtr[index], srcPtr, stride[norm], dims[norm]);
    else if (!axis[level]) // By default split srcPtr and modify index to store mean
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_mean(srcPtr + (i * stride[level]), meanPtr, dims, stride, axis, nDim, level + 1, index + (i * (size / dims[level])), size / dims[level], norm, lastNormAxis);
    }
    else if(axis[level] && (level == lastNormAxis)) // Increment level alone if its last axis to be normalized
        compute_ND_mean(srcPtr, meanPtr, dims, stride, axis, nDim, level + 1, index, size, level, lastNormAxis);
    else if(axis[level]) // Called when that particular axis need to be normalized
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_mean(srcPtr + (i * stride[level]), meanPtr, dims, stride, axis, nDim, level + 1, index, size, level, lastNormAxis);
    }
}

// Computes inverse stddev for ND inputs
template<typename T>
void compute_ND_stddev(T *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride, Rpp32u *axis, Rpp32u nDim, Rpp32u level, Rpp32u index, Rpp32u size, Rpp32u norm, Rpp32u lastNormAxis)
{
    if((level == (nDim - 1)) && axis[nDim - 1]) // Calls computeDiffSumSquare when last dimension is to be normalized
        compute_diff_square_sum(stdDevPtr[index], srcPtr, stride[level], dims[level], meanPtr[index]);
    else if(level == nDim) // Calls computeDiffSumSquare when only 1st axis need to be normalized
        compute_diff_square_sum(stdDevPtr[index], srcPtr, stride[norm], dims[norm], meanPtr[index]);
    else if (!axis[level]) // By default split srcPtr and modify index to store stddev
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_stddev(srcPtr + (i * stride[level]), meanPtr, stdDevPtr, dims, stride, axis, nDim, level + 1, index + (i * (size / dims[level])), size / dims[level], norm, lastNormAxis);
    }
    else if(axis[level] && (level == lastNormAxis)) // Increment level alone if its last axis to be normalized
        compute_ND_stddev(srcPtr, meanPtr, stdDevPtr, dims, stride, axis, nDim, level + 1, index, size, level, lastNormAxis);
    else if(axis[level]) // Called when that particular axis need to be normalized
    {
        for(Rpp32u i = 0; i < dims[level]; i++)
            compute_ND_stddev(srcPtr + (i * stride[level]), meanPtr, stdDevPtr, dims, stride, axis, nDim, level + 1, index, size, level, lastNormAxis);
    }
}

// Computes normalize for 3D non toggle variants
void normalize_3D_tensor_nontoggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u *length)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    Rpp32s idx1 = 0;

    for(Rpp32u i = 0; i < length[0]; i++)
    {
        Rpp32f *srcPtrRow = srcPtrTemp;
        Rpp32f *dstPtrRow = dstPtrTemp;
        for(Rpp32u j = 0; j < length[1]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *dstPtrRowTemp = dstPtrRow;
            idx1 = paramIdx;
            for(Rpp32u k = 0; k < length[2]; k++)
            {
                *dstPtrRowTemp = ((*srcPtrRowTemp - meanPtr[paramIdx]) * multiplierPtr[paramIdx]) + shift;
                if(k < length[2] - 1)
                    paramIdx += paramStride[2];
                srcPtrRowTemp++;
                dstPtrRowTemp++;
            }
            if(j < length[1] - 1)
                paramIdx = (!paramStride[1]) ? idx1 : paramIdx + paramStride[1];
            srcPtrRow += srcGenericDescPtr->strides[2];
            dstPtrRow += dstGenericDescPtr->strides[2];
        }
        if(i < length[0] - 1)
            paramIdx = (!paramStride[0]) ? 0 : paramIdx + paramStride[0];
        srcPtrTemp += srcGenericDescPtr->strides[1];
        dstPtrTemp += dstGenericDescPtr->strides[1];
    }
}

// Computes normalize for 3D toggle variants when axis mask is set to 3
void normalize_3D_tensor_axis3_toggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u *length)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp[length[2]];
    dstPtrTemp[0] = dstPtr;
    for(Rpp32u i = 1; i < length[2]; i++)
        dstPtrTemp[i] = dstPtrTemp[i-1] + dstGenericDescPtr->strides[1];

    Rpp32s paramIdx = 0;

    for(Rpp32u i = 0; i < length[0]; i++)
    {
        Rpp32f *srcPtrRow = srcPtrTemp;
        Rpp32f *dstPtrRow[length[2]];
        for(Rpp32u l = 0; l < length[2]; l++)
            dstPtrRow[l] = dstPtrTemp[l];
        for(Rpp32u j = 0; j < length[1]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *dstPtrRowTemp[length[2]];
            for(Rpp32u l = 0; l < length[2]; l++)
                dstPtrRowTemp[l] = dstPtrRow[l];
            for(Rpp32u k = 0; k < length[2]; k++)
            {
                *dstPtrRowTemp[k]++ = ((*srcPtrRowTemp++ - meanPtr[paramIdx]) * multiplierPtr[paramIdx]) + shift;
                paramIdx += paramStride[2];
            }
            paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
            srcPtrRow += srcGenericDescPtr->strides[2];
            for(Rpp32u l = 0; l < length[2]; l++)
                dstPtrRow[l] += dstGenericDescPtr->strides[3];
        }
        srcPtrTemp += srcGenericDescPtr->strides[1];
        for(Rpp32u l = 0; l < length[2]; l++)
            dstPtrTemp[l] += dstGenericDescPtr->strides[2];
    }
}

// Computes normalize for 3D non toggle variants its optimized with AVX when axis mask set to 3 and 16 channel normalize
void normalize_3D_tensor_avx_axis3(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u bufferLength, Rpp32u *length)
{
    Rpp32u alignedLength = (bufferLength / 16) * 16;
    Rpp32u outerDim = length[0];

    // set shift, mean and stddev
    __m256 pShift = _mm256_set1_ps(shift);
    __m256 pMean1 = _mm256_loadu_ps(meanPtr);
    __m256 pMean2 = _mm256_loadu_ps(meanPtr + 8);
    __m256 pMultiplier1 =  _mm256_loadu_ps(multiplierPtr);
    __m256 pMultiplier2 = _mm256_loadu_ps(multiplierPtr + 8);

    for(Rpp32u i = 0; i < outerDim; i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + i * srcGenericDescPtr->strides[1];
        Rpp32f *dstPtrTemp = dstPtr + i * dstGenericDescPtr->strides[1];

        Rpp32u vectorLoopCount = 0;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += 16)
        {
            __m256 pSrc1 = _mm256_loadu_ps(srcPtrTemp);
            __m256 pSrc2 = _mm256_loadu_ps(srcPtrTemp + 8);
            __m256 pDst1 = _mm256_fmadd_ps(_mm256_sub_ps(pSrc1, pMean1), pMultiplier1, pShift);
            __m256 pDst2 = _mm256_fmadd_ps(_mm256_sub_ps(pSrc2, pMean2), pMultiplier2, pShift);
            _mm256_storeu_ps(dstPtrTemp, pDst1);
            _mm256_storeu_ps(dstPtrTemp + 8, pDst2);
            srcPtrTemp += 16;
            dstPtrTemp += 16;
        }
    }
}

// Computes normalize for ND non toggle variants for i8 dataype
void normalize_ND_tensor_nontoggle(Rpp32s *srcPtr, Rpp32u *srcStride, Rpp32f *dstPtr, Rpp32f *meanPtr, Rpp32f *multiplierPtr,
                                   Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u nDim, Rpp32u level, Rpp32u& idx)
{
    Rpp32u idx1 = 0;
    if(nDim == 1)
    {
        Rpp32s *srcPtrTemp = srcPtr;
        Rpp32f *dstPtrTemp = dstPtr;

        for(Rpp32u k = 0; k < length[level]; k++)
        {
            *dstPtrTemp++ = (((Rpp32f)(*srcPtrTemp + 128) - meanPtr[idx]) * multiplierPtr[idx]) + shift;
            if(k < length[level] - 1)
                idx += paramStride[level];
            srcPtrTemp++;
        }
    }
    else
    {
        idx1 = idx;
        for (Rpp32u i = 0; i < length[level]; i++)
        {
            normalize_ND_tensor_nontoggle(srcPtr, srcStride, dstPtr, meanPtr, multiplierPtr, shift, paramStride, length + 1, nDim - 1, level + 1, idx);
            if(i < length[level] - 1)
                idx = (!paramStride[level]) ? idx1 : idx + paramStride[level];
            dstPtr += srcStride[level];
            srcPtr += srcStride[level];
        }
    }
}

// Computes normalize for ND non toggle variants
template<typename T1, typename T2>
void normalize_ND_tensor_nontoggle(T1 *srcPtr, Rpp32u *srcStride, T2 *dstPtr, Rpp32f *meanPtr, Rpp32f *multiplierPtr,
                                   Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u nDim, Rpp32u level, Rpp32u& idx)
{
    Rpp32u idx1 = 0;
    if(nDim == 1)
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
            normalize_ND_tensor_nontoggle(srcPtr, srcStride, dstPtr, meanPtr, multiplierPtr, shift, paramStride, length, nDim - 1, level + 1, idx);
            if(i < length[level] - 1)
                idx = (!paramStride[level]) ? idx1 : idx + paramStride[level];
            dstPtr += srcStride[level];
            srcPtr += srcStride[level];
        }
    }
}

// Computes normalize for 2D
void normalize_2D_tensor(Rpp32f *srcPtr, RpptGenericDescPtr srcDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            *dstPtrTempRow++ = (*srcPtrTempRow++ - meanPtr[paramIdx]) * invStdDevPtr[paramIdx] + shift;
            paramIdx += paramStride[0];
        }
        paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
        srcPtrTemp += srcDescPtr->strides[1];
        dstPtrTemp += dstDescPtr->strides[1];
    }
}

// Computes normalize for 2D, optimized with AVX when axis mask set to 2
void normalize_2D_tensor_avx_axis2(Rpp32f *srcPtr, RpptGenericDescPtr srcDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstDescPtr,
                                   Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = dims[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    Rpp32u numRows = dims[0];

    __m256 pShift = _mm256_set1_ps(shift);
    for(Rpp32u i = 0; i < numRows; i++)
    {
        Rpp32f *srcPtrTempRow = srcPtrTemp + i * srcDescPtr->strides[1];
        Rpp32f *dstPtrTempRow = dstPtrTemp + i * dstDescPtr->strides[1];

        // set mean and stddev
        Rpp32f mean = meanPtr[i];
        Rpp32f invStdDev = invStdDevPtr[i];
        __m256 pMean, pInvStdDev;
        pMean = _mm256_set1_ps(mean);
        pInvStdDev = _mm256_set1_ps(invStdDev);

        Rpp32u vectorLoopCount = 0;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += 8)
        {
            __m256 pSrc = _mm256_loadu_ps(srcPtrTempRow);
            __m256 pDst = _mm256_fmadd_ps(_mm256_sub_ps(pSrc, pMean), pInvStdDev, pShift);
            _mm256_storeu_ps(dstPtrTempRow, pDst);
            srcPtrTempRow += 8;
            dstPtrTempRow += 8;
        }
        for(; vectorLoopCount < dims[1] ; vectorLoopCount ++)
             *dstPtrTempRow++ = (*srcPtrTempRow++ - mean) * invStdDev + shift;
    }
}

// Performs collapse axis operation wherein continuous axis that require normalization are combined together
void collapse_axis(Rpp32u *nDim, Rpp32u *axis, Rpp32u *length, Rpp32u *newAxis, Rpp32u *newDims, Rpp32u *lastNormAxis)
{
    int skipped = 0, prev = -2, k = 0;
    for(Rpp32u i = 0; i < *nDim; i++)
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
    *nDim -= skipped;
    for(Rpp32u i = 0; i < *nDim; i++)
    {
        if(newAxis[i])
            *lastNormAxis = i;
    }
}

RppStatus normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32f *meanTensorPtr,
                                        Rpp32f *stdDevTensorPtr,
                                        Rpp32u computeMean,
                                        Rpp32u computeStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u numDims = srcGenericDescPtr->numDims - 1;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    for(int batch = 0; batch < batchSize; batch++)
    {
        Rpp32u size = 1; // length of input tensors differ based on axisMask and nDim
        for(int i = 0; i < numDims; i++)
            size *= ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : roiTensor[(numDims * 2 * batch) + numDims + i];
        maxSize = std::max(maxSize, size);
    }

    if(!computeMean && !computeStddev)
    {
        for(Rpp32u i = 0; i < maxSize; i++)
            stdDevTensorPtr[i] = (!stdDevTensorPtr[i])? 1.0f : scale / stdDevTensorPtr[i];
        maxSize = 0;
    }

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u nDim = numDims;
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp32f *srcPtrTemp, *dstPtrTemp, *meanTensor, *stdDevTensor;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];
        meanTensor = meanTensorPtr + batchCount * maxSize;
        stdDevTensor = stdDevTensorPtr + batchCount * maxSize;

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        Rpp32f *srcPtrChannel = srcPtrTemp;

        if(nDim == 2) // Called for audio testcase and for anyother 2D case
        {
            Rpp32u paramStride[2];
            Rpp32u srcReductionDims[2], srcStride[2];
            if (axisMask == 3)
            {
                srcStride[0] = srcStride[1] = srcGenericDescPtr->strides[2];
                srcReductionDims[0] = 1;
                srcReductionDims[1] = length[0] * length[1];
                paramStride[0] = paramStride[1] = 0;
            }
            else if (axisMask == 1)
            {
                srcStride[0] = srcGenericDescPtr->strides[1];
                srcStride[1] = srcGenericDescPtr->strides[2];
                srcReductionDims[0] = length[1];
                srcReductionDims[1] = length[0];
                paramStride[0] = 1;
                paramStride[1] = 0;
            }
            else if (axisMask == 2)
            {
                srcStride[0] = srcGenericDescPtr->strides[2];
                srcStride[1] = srcGenericDescPtr->strides[1];
                srcReductionDims[0] = length[0];
                srcReductionDims[1] = length[1];
                paramStride[0] = 0;
                paramStride[1] = 1;
            }

            if(computeMean)
                compute_2D_mean(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
            if(computeStddev)
                compute_2D_inv_std_dev(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride, scale);

            if(axisMask == 2)
                normalize_2D_tensor_avx_axis2(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, length, paramStride);
            else
                normalize_2D_tensor(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, length, paramStride);
        }
        else if(nDim == 3) // Called when a 3D tensor is passed to kernel
        {
            Rpp32u paramStride[3];
            Rpp32u srcReductionDims[3], srcStride[3];
            Rpp32u reductionDims;
            bool isConsecutive = true;
            switch(axisMask)
            {
                case 1: // Normalize axes 0
                {
                    reductionDims = length[1] * length[2];
                    paramStride[0] = 0;
                    paramStride[1] = paramStride[2] = 1;
                    srcReductionDims[0] = length[1];
                    srcReductionDims[1] = length[2];
                    srcReductionDims[2] = length[0];
                    srcStride[0] = srcGenericDescPtr->strides[1];
                    srcStride[1] = srcGenericDescPtr->strides[3];
                    srcStride[2] = srcGenericDescPtr->strides[2];
                    break;
                }
                case 2: // Normalize axes 1
                {
                    reductionDims = length[0] * length[2];
                    paramStride[1] = 0;
                    paramStride[0] = paramStride[2] = 1;
                    srcReductionDims[0] = length[0];
                    srcReductionDims[1] = length[2];
                    srcReductionDims[2] = length[1];
                    srcStride[0] = srcGenericDescPtr->strides[2];
                    srcStride[1] = srcGenericDescPtr->strides[3];
                    srcStride[2] = srcGenericDescPtr->strides[1];
                    break;
                }
                case 3: // Normalize axes 0,1
                {
                    reductionDims = length[2];
                    paramStride[0] = paramStride[1] = 0;
                    paramStride[2] = 1;
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[2];
                    srcReductionDims[2] = length[0] * length[1];
                    srcStride[0] = srcGenericDescPtr->strides[2];
                    srcStride[1] = srcGenericDescPtr->strides[3];
                    srcStride[2] = srcGenericDescPtr->strides[3];
                    break;
                }
                case 4: // Normalize across 2
                {
                    reductionDims = length[0] * length[1];
                    paramStride[2] = 0;
                    paramStride[0] = paramStride[1] = 1;
                    srcReductionDims[0] = length[0];
                    srcReductionDims[1] = length[1];
                    srcReductionDims[2] = length[2];
                    srcStride[0] = srcGenericDescPtr->strides[3];
                    srcStride[1] = srcGenericDescPtr->strides[2];
                    srcStride[2] = srcGenericDescPtr->strides[1];
                    break;
                }
                case 5: // Normalize across 0,2
                {
                    reductionDims = length[1];
                    paramStride[0] = paramStride[2] = 0;
                    paramStride[1] = 1;
                    srcReductionDims[0] = length[1];
                    srcReductionDims[1] = length[0];
                    srcReductionDims[2] = length[2];
                    srcStride[0] = srcGenericDescPtr->strides[3];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[2];
                    isConsecutive = false;
                    break;
                }
                case 6: // Normalize across 1,2
                {
                    reductionDims = length[0];
                    paramStride[1] = paramStride[2] = 0;
                    paramStride[0] = 1;
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[0];
                    srcReductionDims[2] = length[1] * length[2];
                    srcStride[0] = srcGenericDescPtr->strides[3];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[3];
                    break;
                }
                case 7: // Normalize across 0,1,2
                {
                    reductionDims = 1;
                    paramStride[0] = paramStride[1] = paramStride[2] = 0;
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = 1;
                    srcReductionDims[2] = length[0] * length[1] * length[2];
                    srcStride[0] = srcStride[1] = srcStride[2] = srcGenericDescPtr->strides[3];
                    break;
                }
                default:
                {
                    std::cout<<"Invalid Axis mask"<<std::endl;
                }
            }

            for(Rpp32u i = 1; i < nDim; i++)
                srcPtrChannel += begin[i - 1] * srcGenericDescPtr->strides[i];

            if(computeMean)
                compute_3D_mean(srcPtrChannel, meanTensor, srcReductionDims, srcStride, isConsecutive);
            if(computeStddev)
                compute_3D_inv_std_dev(srcPtrChannel, meanTensor, stdDevTensor, srcReductionDims, srcStride, scale, isConsecutive);

            if((axisMask == 3) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC) && (srcGenericDescPtr->dims[3] == 16))
                normalize_3D_tensor_avx_axis3(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, paramStride, length[1] * layoutParams.bufferMultiplier, length);
            else if((srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC))
                normalize_3D_tensor_nontoggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, paramStride, length);
            else if((axisMask == 3) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NCHW))
                normalize_3D_tensor_axis3_toggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, paramStride, length);
        }
        else // Called when any other ND tensor is passed to kernel
        {
            int size = 1; // length of input tensors differ based on axisMask and nDim
            for(int i = 0; i < nDim; i++)
                size *= ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : length[i];

            Rpp32u totalElements = 1;
            Rpp32u lastNormAxis = 0;
            Rpp32u axis[nDim], newAxis[nDim], newDims[nDim];
            memset(newAxis, 0, nDim * sizeof(Rpp32u));
            memset(newDims, 0, nDim * sizeof(Rpp32u));

            for(Rpp32u i = 0; i < nDim; i++)
            {
                axis[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;
                totalElements *= axis[i] ? length[i] : 1;
                srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i + 1];
            }

            Rpp32u paramStride[nDim], srcStride[nDim];
            collapse_axis(&nDim, axis, length, newAxis, newDims, &lastNormAxis);
            compute_strides(srcStride, newDims, nDim);

            if(computeMean)
            {
                compute_ND_mean(srcPtrChannel, meanTensor, newDims, srcStride, newAxis, nDim, 0, 0, size, 0, lastNormAxis);
                Rpp32f normFactor = 1.0 / totalElements;
                for(int i = 0; i < size; i++)
                    meanTensor[i] *= normFactor;
            }
            if(computeStddev)
            {
                compute_ND_stddev(srcPtrChannel, meanTensor, stdDevTensor, newDims, srcStride, newAxis, nDim, 0, 0, size, 0, lastNormAxis);
                Rpp32f normFactor = (Rpp32f)(1.0 / totalElements);
                rpp_rsqrt_avx(stdDevTensor, (Rpp32s)size, 0, normFactor, scale);
            }

            for(Rpp32u i = 0; i < nDim; i++)
                paramStride[i] = !newAxis[i];

            Rpp32u idx = 0;
            normalize_ND_tensor_nontoggle(srcPtrChannel, srcStride, dstPtrTemp, meanTensor, stdDevTensor, shift, paramStride, newDims, nDim, 0, idx);
        }
    }

    return RPP_SUCCESS;
}
template<typename T1, typename T2>
RppStatus normalize_generic_host_tensor(T1 *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T2 *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32f *meanTensorPtr,
                                        Rpp32f *stdDevTensorPtr,
                                        Rpp32u computeMean,
                                        Rpp32u computeStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    for(int batch = 0; batch < batchSize; batch++)
    {
        Rpp32u size = 1; // length of input tensors differ based on axisMask and nDim
        for(int i = 0; i < nDim; i++)
            size *= ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
        maxSize = std::max(maxSize, size);
    }
    if(!computeMean && !computeStddev)
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
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        for(int i = 0; i < nDim; i++)
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
        Rpp32u axis[nDim], newAxis[nDim], newDims[nDim];
        memset(newAxis, 0, sizeof(newAxis));
        memset(newDims, 0, sizeof(newDims));

        for(int i = 0; i < nDim; i++)
        {
            axis[i] = ((axisMask & (int)(pow(2, i))) >= 1) ? 1 : 0;
            totalElements *= axis[i] ? length[i] : 1;
            srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i + 1];
        }

        Rpp32u paramStride[nDim], srcStride[nDim];
        collapse_axis(&nDim, axis, length, newAxis, newDims, &lastNormAxis);
        compute_strides(srcStride, newDims, nDim);

        if(computeMean)
        {
            compute_ND_mean(srcPtrChannel, meanTensor, newDims, srcStride, newAxis, nDim, 0, 0, size, 0, lastNormAxis);
            Rpp32f normFactor = 1.0 / totalElements;
            for(int i = 0; i < size; i++)
                meanTensor[i] *= normFactor;
        }
        if(computeStddev)
        {
            compute_ND_stddev(srcPtrChannel, meanTensor, stdDevTensor, newDims, srcStride, newAxis, nDim, 0, 0, size, 0, lastNormAxis);
            Rpp32f normFactor = (Rpp32f)(1.0 / totalElements);
            rpp_rsqrt_avx(stdDevTensor, (Rpp32s)size, 0, normFactor, scale);
        }

        for(int i = 0; i < nDim; i++)
            paramStride[i] = !newAxis[i];

        Rpp32u idx = 0;
        normalize_ND_tensor_nontoggle(srcPtrChannel, srcStride, dstPtrTemp, meanTensor, stdDevTensor, shift, paramStride, newDims, nDim, 0, idx);
    }

    return RPP_SUCCESS;
}