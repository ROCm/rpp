/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void compute_strides(Rpp32u *strides, Rpp32u *shape, Rpp32u nDim)
{
    if (nDim > 0)
    {
        uint64_t v = 1;
        for (int i = nDim - 1; i > 0; i--)
        {
            strides[i] = v;
            v *= shape[i];
        }
        strides[0] = v;
    }
}

void compute_diff_square_sum(Rpp32f &output, Rpp32f *input, Rpp32s inputStride, Rpp32s numElements, Rpp32f mean)
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

void compute_sum(Rpp32f &output, Rpp32f *input, Rpp32s inputStride, Rpp32s numElements)
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

Rpp32f rpp_rsqrt(Rpp32f x)
{
    // Use SSE intrinsic and one Newton-Raphson refinement step
    // - faster and less hacky than the hack below.
    __m128 X = _mm_set_ss(x);
    __m128 tmp = _mm_rsqrt_ss(X);
    Rpp32f y = _mm_cvtss_f32(tmp);
    return y * (1.5f - x * 0.5f * y * y);
}

static void rpp_rsqrt_avx(Rpp32f *input, Rpp32s numElements, Rpp32f eps, Rpp32f rdiv, Rpp32f mul)
{
    Rpp32s i = 0;
    __m256 rdivx8 = _mm256_set1_ps(rdiv);
    __m256 mulx8 = _mm256_set1_ps(mul * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m256 epsx8 = _mm256_set1_ps(eps);
        for (; i + 8 <= numElements; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&input[i]);
            x = _mm256_mul_ps(x, rdivx8);
            x = _mm256_add_ps(x, epsx8);
            __m256 y = _mm256_rsqrt_ps(x);
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 xy2 = _mm256_mul_ps(x, y2);
            __m256 three_minus_xy2 = _mm256_sub_ps(avx_p3, xy2);
            y = _mm256_mul_ps(y, three_minus_xy2);
            y = _mm256_mul_ps(y, mulx8);
            _mm256_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 8 <= numElements; i += 8)
        {
            __m256 x = _mm256_loadu_ps(&input[i]);
            x = _mm256_mul_ps(x, rdivx8);
            __m256 mask = _mm256_cmp_ps(x, avx_p0, _CMP_NEQ_OQ);
            __m256 y = _mm256_rsqrt_ps(x);
            y = _mm256_and_ps(y, mask);
            __m256 y2 = _mm256_mul_ps(y, y);
            __m256 xy2 = _mm256_mul_ps(x, y2);
            __m256 three_minus_xy2 = _mm256_sub_ps(avx_p3, xy2);
            y = _mm256_mul_ps(y, three_minus_xy2);
            y = _mm256_mul_ps(y, mulx8);
            _mm256_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt(input[i] * rdiv + eps) * mul;
    }
    else
    {
        for (; i < numElements; i++)
        {
            Rpp32f x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt(x) * mul : 0;
        }
    }
}

void compute_2D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = 1.0 / dims[1];
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        meanPtr[i] = 0;
        compute_sum(meanPtr[i], srcPtrTemp, 1, dims[1]);
        srcPtrTemp += stride[1];
        meanPtr[i] *= normFactor;
    }
}

void compute_2D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {

    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = (Rpp32f)(1.0 / dims[1]);
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        stdDevPtr[i] = 0;
        compute_diff_square_sum(stdDevPtr[i], srcPtrTemp, 1, dims[1], meanPtr[i]);
        srcPtrTemp += stride[1];
    }
    rpp_rsqrt_avx(stdDevPtr, (Rpp32s)dims[0], 0, normFactor, 1);
}

void compute_3D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = 1.0 / dims[2];
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        float *srcPtrRow = srcPtrTemp;
        for(unsigned int j = 0; j < dims[1]; j++)
        {
            int index = i * dims[1] + j;
            meanPtr[index] = 0;
            compute_sum(meanPtr[index], srcPtrRow, stride[0], dims[2]);
            srcPtrRow += stride[1];
            meanPtr[index] = meanPtr[index] * normFactor;
        }
        srcPtrTemp += stride[2];
    }
}

void compute_3D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {

    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = 1.0 / dims[2];
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        float *srcPtrRow = srcPtrTemp;
        for(unsigned int j = 0; j < dims[1]; j++)
        {
            int index = i * dims[1] + j;
            stdDevPtr[index] = 0;
            compute_diff_square_sum(stdDevPtr[index], srcPtrRow, stride[0], dims[2], meanPtr[index]);
            srcPtrRow += stride[1];
        }
        srcPtrTemp += stride[2];
    }
    rpp_rsqrt_avx(stdDevPtr, (Rpp32s)(dims[0] * dims[1]), 0, normFactor, 1);
}

void compute_ND_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride, Rpp32u *axis, Rpp32u nDim, Rpp32u level, Rpp32u index, Rpp32u size, Rpp32u norm, Rpp32u numAxisNorm)
{
    if(level == nDim-1 && axis[nDim-1]) // Calls computeSum when last dimension is to be normalized
    {
        std::cout<<"level == nDim-1 and axis set at "<< level <<std::endl;
        std::cout<<"src: "<<srcPtr[0]<<"stride: "<<stride[level]<<"dims: "<<dims[level]<<std::endl;
        compute_sum(meanPtr[index], srcPtr, stride[level], dims[level]);
        std::cout<<"meanPtr at"<<index <<": "<< meanPtr[index];
    }
    else if(level == nDim) // Calls computeSum when only 1st axis need to be normalized
    {
        std::cout<<"level == nDim and axis not set at "<< level <<std::endl;
        std::cout<<"src: "<<srcPtr[0]<<"stride: "<<stride[norm]<<"dims: "<<dims[norm]<<std::endl;
        compute_sum(meanPtr[index], srcPtr, stride[norm], dims[norm]);
        std::cout<<"meanPtr at"<<index <<": "<< meanPtr[index];
    }
    else if (!axis[level]) // By default split srcPtr and modify meanPtr index
    {
        std::cout<<"axis not set at"<< level <<std::endl;
        for(int i = 0; i < dims[level]; i++)
            compute_ND_mean(srcPtr + (i * stride[level]), meanPtr, dims, stride, axis, nDim, level + 1, index + (i * (size / dims[level])), size / dims[level], norm, numAxisNorm);
    }
    else if(!level && axis[level] && (numAxisNorm == 1))
    {
        std::cout<<"!level and axis set at"<< level <<std::endl;
        compute_ND_mean(srcPtr, meanPtr, dims, stride, axis, nDim, level + 1, index, size, level, numAxisNorm);
    }
    else if(axis[level]) // Called when that particular axis need to be normalized
    {
        std::cout<<"level && axis set at"<< level <<std::endl;
        for(int i = 0; i < dims[level]; i++)
            compute_ND_mean(srcPtr + (i * stride[level]), meanPtr, dims, stride, axis, nDim, level + 1, index, size, level, numAxisNorm);
    }
}

void compute_ND_stddev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride, Rpp32u *axis, Rpp32u nDim, Rpp32u level, Rpp32u index, Rpp32u size, Rpp32u norm, Rpp32u numAxisNorm)
{
    if(level == nDim-1 && axis[nDim-1]) // Calls computeSum when last dimension is to be normalized
        compute_diff_square_sum(stdDevPtr[index], srcPtr, stride[level], dims[level], meanPtr[index]);
    else if(level == nDim) // Calls computeSum when only 1st axis need to be normalized
        compute_diff_square_sum(stdDevPtr[index], srcPtr, stride[norm], dims[norm], meanPtr[index]);
    else if (!axis[level]) // By default split srcPtr and modify meanPtr index
    {
        for(int i = 0; i < dims[level]; i++)
            compute_ND_stddev(srcPtr + (i * stride[level]), meanPtr, stdDevPtr, dims, stride, axis, nDim, level + 1, index + (i * (size / dims[level])), size / dims[level], norm, numAxisNorm);
    }
    else if(!level && axis[level] && (numAxisNorm == 1))
        compute_ND_stddev(srcPtr, meanPtr, stdDevPtr, dims, stride, axis, nDim, level + 1, index, size, level, numAxisNorm);
    else if(axis[level]) // Called when that particular axis need to be normalized
    {
        for(int i = 0; i < dims[level]; i++)
            compute_ND_stddev(srcPtr + (i * stride[level]), meanPtr, stdDevPtr, dims, stride, axis, nDim, level + 1, index, size, level, numAxisNorm);
    }
}

void normalize_3D_tensor_nontoggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    int size = sizeof(stdDevPtr[0]) / sizeof(Rpp32f);
    Rpp32f *multiplier = (Rpp32f *) calloc(size, sizeof(Rpp32f));

    for(int i = 0; i < size; i++)
        multiplier[i] = scale / stdDevPtr[i];

    for(Rpp32u i = 0; i < srcGenericDescPtr->dims[1]; i++)
    {
        Rpp32f *srcPtrRow = srcPtrTemp;
        Rpp32f *dstPtrRow = dstPtrTemp;
        for(Rpp32u j = 0; j < srcGenericDescPtr->dims[2]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *dstPtrRowTemp = dstPtrRow;
            for(Rpp32u k = 0; k < srcGenericDescPtr->dims[3]; k++)
            {
                *dstPtrRowTemp++ = ((*srcPtrRowTemp++ - meanPtr[paramIdx]) * multiplier[paramIdx]) + shift;
                paramIdx += paramStride[2];
            }
            paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
            srcPtrRow += srcGenericDescPtr->strides[2];
            dstPtrRow += dstGenericDescPtr->strides[2];
        }
        paramIdx = (!paramStride[0]) ? 0 : paramIdx + paramStride[0];
        srcPtrTemp += srcGenericDescPtr->strides[1];
        dstPtrTemp += dstGenericDescPtr->strides[1];
    }

    free(multiplier);
}

void normalize_3D_tensor_axis3_toggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp[srcGenericDescPtr->dims[3]];
    dstPtrTemp[0] = dstPtr;
    for(int i = 1; i < srcGenericDescPtr->dims[3]; i++)
        dstPtrTemp[i] = dstPtrTemp[i-1] + dstGenericDescPtr->strides[1];

    Rpp32s paramIdx = 0;
    int size = sizeof(stdDevPtr[0]) / sizeof(Rpp32f);
    Rpp32f *multiplier = (Rpp32f *) calloc(size, sizeof(Rpp32f));

    for(int i = 0; i < size; i++)
        multiplier[i] = scale / stdDevPtr[i];

    for(Rpp32u i = 0; i < srcGenericDescPtr->dims[1]; i++)
    {
        Rpp32f *srcPtrRow = srcPtrTemp;
        Rpp32f *dstPtrRow[srcGenericDescPtr->dims[3]];
        for(int l = 0; l < srcGenericDescPtr->dims[3]; l++)
            dstPtrRow[l] = dstPtrTemp[l];
        for(Rpp32u j = 0; j < srcGenericDescPtr->dims[2]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *dstPtrRowTemp[srcGenericDescPtr->dims[3]];
            for(int l = 0; l < srcGenericDescPtr->dims[3]; l++)
                dstPtrRowTemp[l] = dstPtrRow[l];
            for(Rpp32u k = 0; k < srcGenericDescPtr->dims[3]; k++)
            {
                *dstPtrRowTemp[k]++ = ((*srcPtrRowTemp++ - meanPtr[paramIdx]) * multiplier[paramIdx]) + shift;
                paramIdx += paramStride[2];
            }
            paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
            srcPtrRow += srcGenericDescPtr->strides[2];
            for(int l = 0; l < srcGenericDescPtr->dims[3]; l++)
                dstPtrRow[l] += dstGenericDescPtr->strides[3];
        }
        srcPtrTemp += srcGenericDescPtr->strides[1];
        for(int l = 0; l < srcGenericDescPtr->dims[3]; l++)
            dstPtrTemp[l] += dstGenericDescPtr->strides[2];
    }
    free(multiplier);
}

void normalize_3D_tensor_avx_axis3(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *paramStride, Rpp32u bufferLength)
{
    Rpp32u alignedLength = (bufferLength / 16) * 16;
    Rpp32u outerDim = srcGenericDescPtr->dims[1];

    // set shift, mean and stddev
    __m256 pShift = _mm256_set1_ps(shift);
    __m256 pScale = _mm256_set1_ps(scale);
    __m256 pMean1 = _mm256_loadu_ps(meanPtr);
    __m256 pMean2 = _mm256_loadu_ps(meanPtr + 8);
    __m256 pStdDev1 = _mm256_loadu_ps(stdDevPtr);
    __m256 pStdDev2 = _mm256_loadu_ps(stdDevPtr + 8);
    __m256 pMultiplier1 = _mm256_div_ps(pScale, pStdDev1); // Using division operation as stddev is a vector and scale is a scalar thus can't be pre-divided
    __m256 pMultiplier2 = _mm256_div_ps(pScale, pStdDev2);

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

template<typename T1, typename T2>
void normalize_ND_tensor_nontoggle(T1 *srcPtr, RpptGenericDescPtr srcGenericDescPtr, T2 *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u nDim, Rpp32u level, Rpp32u paramIdx)
{
    if(nDim == 1)
    {
        T1 *srcPtrTemp = srcPtr;
        T2 *dstPtrTemp = dstPtr;

        for(Rpp32u k = 0; k < srcGenericDescPtr->dims[level]; k++)
        {
            *dstPtrTemp++ = (((T2)*srcPtrTemp++ - meanPtr[paramIdx]) * multiplierPtr[paramIdx]) + shift;
            paramIdx += paramStride[level - 1];
        }
    }
    else
    {
        for (int i = 0; i < *length; i++)
        {
            normalize_ND_tensor_nontoggle(srcPtr, srcGenericDescPtr, dstPtr, dstGenericDescPtr, meanPtr, multiplierPtr, shift, paramStride, length + 1, nDim - 1, level + 1, paramIdx);
            paramIdx = (!paramStride[level - 1]) ? 0 : paramIdx + paramStride[level - 1];
            dstPtr += dstGenericDescPtr->strides[level];
            srcPtr += srcGenericDescPtr->strides[level];
        }
    }
}

void normalize_ND_tensor_nontoggle(Rpp32s *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u nDim, Rpp32u level, Rpp32u paramIdx)
{
    if(nDim == 1)
    {
        Rpp32s *srcPtrTemp = srcPtr;
        Rpp32f *dstPtrTemp = dstPtr;

        for(Rpp32u k = 0; k < srcGenericDescPtr->dims[level]; k++)
        {
            *dstPtrTemp++ = (((Rpp32f)(*srcPtrTemp + 128) - meanPtr[paramIdx]) * multiplierPtr[paramIdx]) + shift;
            paramIdx += paramStride[level - 1];
            srcPtrTemp++;
        }
    }
    else
    {
        for (int i = 0; i < *length; i++)
        {
            normalize_ND_tensor_nontoggle(srcPtr, srcGenericDescPtr, dstPtr, dstGenericDescPtr, meanPtr, multiplierPtr, shift, paramStride, length + 1, nDim - 1, level + 1, paramIdx);
            paramIdx = (!paramStride[level - 1]) ? 0 : paramIdx + paramStride[level - 1];
            dstPtr += dstGenericDescPtr->strides[level];
            srcPtr += srcGenericDescPtr->strides[level];
        }
    }
}

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
            __m256 pDst = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc, pMean), pInvStdDev), pShift);
            _mm256_storeu_ps(dstPtrTempRow, pDst);
            srcPtrTempRow += 8;
            dstPtrTempRow += 8;
        }
        for(; vectorLoopCount < dims[1] ; vectorLoopCount += 8)
             *dstPtrTempRow++ = (*srcPtrTempRow++ - mean) * invStdDev + shift;
    }
}

RppStatus normalize_generic_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                RpptGenericDescPtr srcGenericDescPtr,
                                                Rpp32f *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u axis_mask,
                                                Rpp32f *meanTensor,
                                                Rpp32f *stdDevTensor,
                                                Rpp32u computeMean,
                                                Rpp32u computeStddev,
                                                Rpp32f scale,
                                                Rpp32f shift,
                                                Rpp32u *roiTensor,
                                                RppLayoutParams layoutParams,
                                                rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32f *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp32f *srcPtrChannel;

        if(nDim == 2)
        {
            Rpp32u paramStride[2];
            Rpp32u srcAudioDims[2], srcReductionDims[2], srcStride[2];
            srcAudioDims[0] = length[0];
            srcAudioDims[1] = length[1];
            Rpp32u reductionDims;
            if (axis_mask == 3)
            {
                reductionDims = 1;
                srcStride[0] = srcStride[1] = srcGenericDescPtr->strides[1];
                srcReductionDims[0] = 1;
                srcReductionDims[1] = srcAudioDims[0] * srcAudioDims[1];
                paramStride[0] = paramStride[1] = 0;
            }
            else if (axis_mask == 1)
            {
                reductionDims = 1;
                srcStride[0] = srcGenericDescPtr->strides[1];
                srcStride[1] = srcGenericDescPtr->strides[0];
                srcReductionDims[0] = srcAudioDims[1];
                srcReductionDims[1] = srcAudioDims[0];
                paramStride[0] = 1;
                paramStride[1] = 0;
            }
            else if (axis_mask == 2)
            {
                reductionDims = 0;
                srcStride[0] = srcGenericDescPtr->strides[0];
                srcStride[1] = srcGenericDescPtr->strides[1];
                srcReductionDims[0] = srcAudioDims[0];
                srcReductionDims[1] = srcAudioDims[1];
                paramStride[0] = 0;
                paramStride[1] = 1;
            }
            Rpp32f *invStdDevTensor, *meanInternalTensor;
            meanInternalTensor = (Rpp32f *)calloc(length[reductionDims], sizeof(Rpp32f));
            invStdDevTensor = (Rpp32f *)calloc(length[reductionDims], sizeof(Rpp32f));

            if(computeMean)
                compute_2D_mean(srcPtrTemp, meanInternalTensor, srcReductionDims, srcStride);
            if(computeStddev)
                compute_2D_inv_std_dev(srcPtrTemp, meanInternalTensor, invStdDevTensor, srcReductionDims, srcStride);

            // Inv std dev calculations missing
            if(axis_mask == 2)
                normalize_2D_tensor_avx_axis2(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanInternalTensor, invStdDevTensor, shift, srcAudioDims, paramStride);
            else
                normalize_2D_tensor(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanInternalTensor, invStdDevTensor, shift, srcAudioDims, paramStride);

            free(meanInternalTensor);
            free(invStdDevTensor);
        }
        else if(nDim == 3)
        {
            Rpp32u paramStride[3];
            Rpp32u srcReductionDims[3], srcStride[3];
            Rpp32u reductionDims;
            switch(axis_mask)
            {
                case 1: // Normalize axes 0
                {
                    reductionDims = length[1] * length[2];
                    paramStride[0] = 0;
                    paramStride[1] = paramStride[2] = 1;
                    srcReductionDims[0] = length[1];
                    srcReductionDims[1] = length[2];
                    srcReductionDims[2] = length[0];
                    srcStride[0] = srcGenericDescPtr->strides[0];
                    srcStride[1] = srcGenericDescPtr->strides[2];
                    srcStride[2] = srcGenericDescPtr->strides[1];
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
                    srcStride[0] = srcGenericDescPtr->strides[1];
                    srcStride[1] = srcGenericDescPtr->strides[2];
                    srcStride[2] = srcGenericDescPtr->strides[0];
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
                    srcStride[0] = srcGenericDescPtr->strides[1];
                    srcStride[1] = srcGenericDescPtr->strides[2];
                    srcStride[2] = srcGenericDescPtr->strides[2];
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
                    srcStride[0] = srcGenericDescPtr->strides[2];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[0];
                    break;
                }
                case 5: // Normalize across 0,2
                {
                    reductionDims = length[1];
                    paramStride[0] = paramStride[2] = 0;
                    paramStride[1] = 1;
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[1];
                    srcReductionDims[2] = length[0] * length[2];
                    srcStride[0] = srcGenericDescPtr->strides[0];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[2];
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
                    srcStride[0] = srcGenericDescPtr->strides[2];
                    srcStride[1] = srcGenericDescPtr->strides[0];
                    srcStride[2] = srcGenericDescPtr->strides[2];
                    break;
                }
                case 7: // Normalize across 0,1,2
                {
                    reductionDims = 1;
                    paramStride[0] = paramStride[1] = paramStride[2] = 0;
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = 1;
                    srcReductionDims[2] = length[0] * length[1] * length[2];
                    srcStride[0] = srcStride[1] = srcStride[2] = srcGenericDescPtr->strides[2];
                    break;
                }
                default:
                {
                    std::cout<<"Invalid Axis mask"<<std::endl;
                }
            }

            Rpp32f *invStdDevTensor, *meanInternalTensor;
            meanInternalTensor = (Rpp32f *)calloc(reductionDims, sizeof(Rpp32f));
            invStdDevTensor = (Rpp32f *)calloc(reductionDims, sizeof(Rpp32f));

            if(computeMean)
            {
                compute_3D_mean(srcPtrTemp, meanInternalTensor, srcReductionDims, srcStride);
                memcpy(&meanTensor, &meanInternalTensor, sizeof(meanInternalTensor));
            }
            if(computeStddev)
            {
                compute_3D_inv_std_dev(srcPtrTemp, meanInternalTensor, invStdDevTensor, srcReductionDims, srcStride);
                memcpy(&stdDevTensor, &invStdDevTensor, sizeof(invStdDevTensor));
            }

            srcPtrChannel = srcPtrTemp + (begin[0] * layoutParams.bufferMultiplier);
            for(int i = 1; i < nDim; i++)
                srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i];

            if((axis_mask == 3) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC) && (srcGenericDescPtr->dims[3] == 16))
                normalize_3D_tensor_avx_axis3(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, scale, shift, paramStride, length[1] * layoutParams.bufferMultiplier);
            else if((srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC))
                normalize_3D_tensor_nontoggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, scale, shift, paramStride);
            else if((axis_mask == 3) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NCHW))
                normalize_3D_tensor_axis3_toggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, scale, shift, paramStride);

            free(meanInternalTensor);
            free(invStdDevTensor);
        }
        else
        {
            Rpp32u axis[nDim];
            srcPtrChannel = srcPtrTemp + (begin[0] * length[nDim]);
            Rpp32u paramIdx = 0;
            int skipped = 0, prev = -2, size = 1, totalElements = 1;
            int k = 0, numAxisNorm = 0;
            Rpp32u newAxis[nDim], newDims[nDim];
            memset(newAxis, 0, sizeof(newAxis));
            memset(newDims, 0, sizeof(newDims));
            std::cout<<"\n axis_mask: "<<axis_mask<<std::endl;
            for(int i = 0; i < nDim; i++)
            {
                axis[i] = ((axis_mask & (int)(pow(2,i))) >= 1) ? 1 : 0;
                size *= !axis[i] ? length[i] : 1;
                totalElements *= axis[i] ? length[i] : 1;

                srcPtrChannel += begin[i + 1] * srcGenericDescPtr->strides[i + 1];
            }
            for(int i = 0; i < nDim; i++)
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
            nDim -= skipped;
            Rpp32u paramStride[nDim];
            for(int i=0;i<nDim;i++)
            {
                std::cout<<"newAxis: "<<newAxis[i]<<"newDim: "<<newDims[i]<<std::endl;
                if(newAxis[i])
                    numAxisNorm++;
            }

            std::cout<<"numAxisNorm: "<<numAxisNorm<<std::endl;

            Rpp32u srcStride[nDim];
            compute_strides(srcStride, newDims, nDim);
            for(int i=0;i<nDim;i++)
            {
                std::cout<<"stride: "<<srcStride[i]<<std::endl;
            }
            Rpp32f *meanInternalTensor;
            if(computeMean)
            {
                meanInternalTensor = (Rpp32f *)calloc(size, sizeof(Rpp32f));
                std::cout<<"size: "<<size<<std::endl;
                compute_ND_mean(srcPtrTemp, meanInternalTensor, newDims, srcStride, newAxis, nDim, 0, 0, size, 0, numAxisNorm);
                Rpp32f normFactor = 1.0 / totalElements;
                for(int i = 0; i < size; i++)
                {
                    meanInternalTensor[i] *= normFactor;
                    std::cout<<"mean: "<<meanInternalTensor[i]<<std::endl;
                }

                memcpy(&meanTensor, &meanInternalTensor, sizeof(meanInternalTensor));
            }
            if(computeStddev)
            {
                Rpp32f *invStdDevTensor;
                invStdDevTensor = (Rpp32f *)calloc(size, sizeof(Rpp32f));
                compute_ND_stddev(srcPtrTemp, meanInternalTensor, invStdDevTensor, newDims, srcStride, newAxis, nDim, 0, 0, size, 0, numAxisNorm);
                Rpp32f normFactor = 1.0 / totalElements;
                rpp_rsqrt_avx(invStdDevTensor, (Rpp32s)totalElements, 0, normFactor, 1);
                memcpy(&stdDevTensor, &invStdDevTensor, sizeof(invStdDevTensor));
            }

            //int size = sizeof(stdDevTensor[0]) / sizeof(Rpp32f);
            Rpp32f *multiplier = (Rpp32f *) calloc(size, sizeof(Rpp32f));
            for(int i = 0; i < size; i++)
                multiplier[i] = scale / stdDevTensor[i];
            for(int i = 0; i < nDim; i++)
                paramStride[i] = !newAxis[i];
            normalize_ND_tensor_nontoggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, multiplier, shift, paramStride, length, nDim, 1, paramIdx);
            free(multiplier);

            if(meanTensor != NULL)
                free(meanTensor);
            if(stdDevTensor != NULL)
                free(stdDevTensor);
        }
    }

    return RPP_SUCCESS;
}
template<typename T1, typename T2>
RppStatus normalize_generic_host_tensor(T1 *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T2 *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axis_mask,
                                        Rpp32f *meanTensor,
                                        Rpp32f *stdDevTensor,
                                        Rpp32u computeMean,
                                        Rpp32u computeStddev,
                                        Rpp32f scale,
                                        Rpp32f shift,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        T1 *srcPtrTemp;
        T2 *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        Rpp32u *paramStride = (Rpp32u *) malloc(nDim * sizeof(Rpp32u));
        T1 *srcPtrChannel;

        srcPtrChannel = srcPtrTemp + (begin[0] * length[nDim]);
        Rpp32u paramIdx = 0;
        for(int i = 0; i < nDim; i++)
        {
            paramStride[i] = ((axis_mask & (int)(pow(2,i))) >= 1) ? 0 : 1;
            srcPtrChannel += begin[i + 1] * srcGenericDescPtr->strides[i + 1];
        }

        int size = sizeof(stdDevTensor[0]) / sizeof(Rpp32f);
        Rpp32f *multiplier = (Rpp32f *) calloc(size, sizeof(Rpp32f));
        for(int i = 0; i < size; i++)
            multiplier[i] = scale / stdDevTensor[i];
        normalize_ND_tensor_nontoggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, multiplier, shift, paramStride, newDims, nDim, 1, paramIdx);
        free(multiplier);
        free(paramStride);
    }

    return RPP_SUCCESS;
}