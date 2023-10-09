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

void compute_diff_square_sum(Rpp32f &output, Rpp32f *input, Rpp32s inputStride, Rpp32s numElements, Rpp32f mean)
{
    const Rpp32s stride = 1;
    if (numElements > 32)
    {
        Rpp32s currElements = numElements >> 1;
        Rpp32f tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_diff_square_sum(tmp1, input, stride, currElements, mean);

        // reduce second half and accumulate
        compute_diff_square_sum(tmp2, input + currElements * stride, stride, numElements - currElements, mean);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        Rpp32f tmp = 0;
        for (Rpp32s i = 0; i < numElements; i++)
        {
            Rpp32f curr = (input[i * stride] - mean);
            auto curnew = curr * curr;
            tmp += curnew;
        }

        // accumulate in target value
        output += tmp;
    }
}

void compute_sum(Rpp32f &output, Rpp32f *input, Rpp32s inputStride, Rpp32s numElements)
{
    const Rpp32s stride = 1;
    if (numElements > 32)
    {
        Rpp32s currElements = numElements >> 1;
        Rpp32f tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_sum(tmp1, input, stride, currElements);

        // reduce second half and accumulate
        compute_sum(tmp2, input + currElements * stride, stride, numElements - currElements);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        Rpp32f tmp = 0;
        for (Rpp32s i = 0; i < numElements; i++)
            tmp += input[i * stride];

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

void normalize_3D_tensor_axis1_nontoggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    Rpp32f multiplier[srcGenericDescPtr->dims[3]];

    for(int i = 0; i < srcGenericDescPtr->dims[3]; i++)
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
                paramIdx += paramStride[1];
            }
            paramIdx = (!paramStride[2]) ? 0 : paramIdx + paramStride[2];
            srcPtrRow += srcGenericDescPtr->strides[2];
            dstPtrRow += dstGenericDescPtr->strides[2];
        }
        srcPtrTemp += srcGenericDescPtr->strides[1];
        dstPtrTemp += dstGenericDescPtr->strides[1];
    }
}

void normalize_3D_tensor_axis1_toggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp[srcGenericDescPtr->dims[3]];
    dstPtrTemp[0] = dstPtr;
    for(int i = 1; i < srcGenericDescPtr->dims[3]; i++)
        dstPtrTemp[i] = dstPtrTemp[i-1] + dstGenericDescPtr->strides[1];

    Rpp32s paramIdx = 0;
    Rpp32f multiplier[srcGenericDescPtr->dims[3]];

    for(int i = 0; i < srcGenericDescPtr->dims[3]; i++)
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
                paramIdx += paramStride[1];
            }
            paramIdx = (!paramStride[2]) ? 0 : paramIdx + paramStride[2];
            srcPtrRow += srcGenericDescPtr->strides[2];
            for(int l = 0; l < srcGenericDescPtr->dims[3]; l++)
                dstPtrRow[l] += dstGenericDescPtr->strides[3];
        }
        srcPtrTemp += srcGenericDescPtr->strides[1];
        for(int l = 0; l < srcGenericDescPtr->dims[3]; l++)
            dstPtrTemp[l] += dstGenericDescPtr->strides[2];
    }
}

void normalize_3D_tensor_avx_axis1(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *paramStride, Rpp32u bufferLength)
{
    Rpp32s paramIdx = 0;
    Rpp32u alignedLength = (bufferLength / 16) * 16;
    Rpp32u OuterDim = srcGenericDescPtr->dims[1];

    // set shift, mean and stddev
    __m256 pShift = _mm256_set1_ps(shift);
    __m256 pScale = _mm256_set1_ps(scale);
    __m256 pMean1 = _mm256_loadu_ps(meanPtr);
    __m256 pMean2 = _mm256_loadu_ps(meanPtr + 8);
    __m256 pStdDev1 = _mm256_loadu_ps(stdDevPtr);
    __m256 pStdDev2 = _mm256_loadu_ps(stdDevPtr + 8);
    __m256 pMultiplier1 = _mm256_div_ps(pScale, pStdDev1); // Using division operation as stddev is a vector and scale is a scalar thus can't be pre-divided
    __m256 pMultiplier2 = _mm256_div_ps(pScale, pStdDev2);

    for(Rpp32u i = 0; i < OuterDim; i++)
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

void normalize_ND_tensor_axis1_nontoggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u nDim, Rpp32u level)
{
    if(nDim == 1)
    {
        Rpp32f *srcPtrTemp = srcPtr;
        Rpp32f *dstPtrTemp = dstPtr;
        Rpp32s paramIdx = 0;

        for(Rpp32u k = 0; k < srcGenericDescPtr->dims[level]; k++)
        {
            *dstPtrTemp++ = ((*srcPtrTemp++ - meanPtr[paramIdx]) * multiplierPtr[paramIdx]) + shift;
            paramIdx += paramStride[1];
        }
        paramIdx = (!paramStride[2]) ? 0 : paramIdx + paramStride[2];
    }
    else
    {
        for (int i = 0; i < *length; i++)
        {
            normalize_ND_tensor_axis1_nontoggle(srcPtr, srcGenericDescPtr, dstPtr, dstGenericDescPtr, meanPtr, multiplierPtr, shift, paramStride, length + 1, nDim - 1, level + 1);
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

        Rpp32u paramStride[nDim];
        Rpp32f *srcPtrChannel;

        if(nDim == 2)
        {
            Rpp32u srcAudioDims[2], srcReductionDims[2], srcStride[2];
            srcAudioDims[0] = length[0];
            srcAudioDims[1] = length[1];
            Rpp32u reductionDims;
            if (axis_mask == 3)
            {
                reductionDims = 1;
                srcStride[0] = srcStride[1] = srcGenericDescPtr->strides[2];
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
            meanTensor = (Rpp32f *)calloc(length[reductionDims], sizeof(Rpp32f));
            stdDevTensor = (Rpp32f *)calloc(length[reductionDims], sizeof(Rpp32f));

            if(computeMean)
                compute_2D_mean(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
            if(computeStddev)
                compute_2D_inv_std_dev(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);

            // Inv std dev calculations missing
        if(axis_mask == 2)
            normalize_2D_tensor_avx_axis2(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);
        else
            normalize_2D_tensor(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);
        }
        else if(nDim == 3)
        {
            if(axis_mask == 1) // Normalize across Channels
            {
                paramStride[0] = paramStride[1] = 1;
                paramStride[2] = 0;
            }
            else if(axis_mask == 2) // Normalize across Width
            {
                //TODO
            }
            else if(axis_mask == 3) // Normalize across Height
            {
                //TODO
            }
            srcPtrChannel = srcPtrTemp + (begin[0] * layoutParams.bufferMultiplier);
            for(int i = 1; i < nDim; i++)
                srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i];

            if((axis_mask == 1) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC) && (srcGenericDescPtr->dims[3] == 16))
                normalize_3D_tensor_avx_axis1(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, scale, shift, paramStride, length[1] * layoutParams.bufferMultiplier);
            else if((axis_mask == 1) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NHWC))
                normalize_3D_tensor_axis1_nontoggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, scale, shift, paramStride);
            else if((axis_mask == 1) && (srcGenericDescPtr->layout == RpptLayout::NHWC) && (dstGenericDescPtr->layout == RpptLayout::NCHW))
                normalize_3D_tensor_axis1_toggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, stdDevTensor, scale, shift, paramStride);
        }
        else
        {
            srcPtrChannel = srcPtrTemp + (begin[0] * layoutParams.bufferMultiplier);
            if(axis_mask == 1) // Normalize across Channels
            {
                for(int i = 0; i < nDim; i++)
                {
                    paramStride[i] = (i == nDim - 1)? 0 : 1;
                    srcPtrChannel += begin[i + 1] * srcGenericDescPtr->strides[i + 1];
                }
            }

            Rpp32f multiplier[srcGenericDescPtr->dims[nDim]];
            for(int i = 0; i < srcGenericDescPtr->dims[nDim]; i++)
                multiplier[i] = scale / stdDevTensor[i];
            if(axis_mask == 1)
                normalize_ND_tensor_axis1_nontoggle(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTensor, multiplier, shift, paramStride, length, nDim, 1);
        }
    }

    return RPP_SUCCESS;
}