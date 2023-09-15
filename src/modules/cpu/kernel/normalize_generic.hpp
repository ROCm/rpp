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

void normalize_3D_tensor_axis3(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                         Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    Rpp32f invStdDev[dims[2]];

    for(int i = 0; i < dims[2]; i++)
    {
        invStdDev[i] = 1 / stdDevPtr[i];
    }

    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32f *srcPtrRow = srcPtrTemp;
        Rpp32f *dstPtrRow = dstPtrTemp;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *dstPtrRowTemp = dstPtrRow;
            for(Rpp32u k = 0; k < dims[2]; k++)
            {
                *dstPtrRowTemp++ = (scale * (*srcPtrRowTemp++ - meanPtr[paramIdx])) * invStdDev[paramIdx] + shift;
                paramIdx += paramStride[1];
            }
            paramIdx = (!paramStride[2]) ? 0 : paramIdx + paramStride[2];
            srcPtrRow += srcGenericDescPtr->strides[1];
            dstPtrRow += srcGenericDescPtr->strides[1];
        }
        srcPtrTemp += srcGenericDescPtr->strides[0];
        dstPtrTemp += srcGenericDescPtr->strides[0];
    }
}

void normalize_3D_tensor_avx_axis3(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
                                   Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32f scale, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = dims[2];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    Rpp32u OuterDim = dims[0];
    Rpp32u numRows = dims[1];

    // set shift, mean and stddev
    __m256 pShift = _mm256_set1_ps(shift);
    __m256 pScale = _mm256_set1_ps(scale);
    __m256 pMean1 = _mm256_loadu_ps(meanPtr);
    __m256 pMean2 = _mm256_loadu_ps(meanPtr + 8);
    __m256 pStdDev1 = _mm256_loadu_ps(stdDevPtr);
    __m256 pStdDev2 = _mm256_loadu_ps(stdDevPtr + 8);
    __m256 pInvStdDev1 = _mm256_mul_ps(pScale, _mm256_div_ps(avx_p1, pStdDev1));
    __m256 pInvStdDev2 = _mm256_mul_ps(pScale, _mm256_div_ps(avx_p1, pStdDev2));

    for(Rpp32u i = 0; i < OuterDim; i++)
    {
        Rpp32f *srcPtrTempOuterDim = srcPtrTemp + i * srcGenericDescPtr->strides[0];
        Rpp32f *dstPtrTempOuterDim = dstPtrTemp + i * dstGenericDescPtr->strides[0];

        for(Rpp32u j = 0; j < numRows; j++)
        {
            Rpp32f *srcPtrTempRow = srcPtrTempOuterDim + i * srcGenericDescPtr->strides[1];
            Rpp32f *dstPtrTempRow = dstPtrTempOuterDim + i * dstGenericDescPtr->strides[1];

            Rpp32u vectorLoopCount = 0;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += 16)
            {
                __m256 pSrc1 = _mm256_loadu_ps(srcPtrTempRow);
                __m256 pSrc2 = _mm256_loadu_ps(srcPtrTempRow + 8);
                __m256 pDst1 = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc1, pMean1), pInvStdDev1), pShift);
                __m256 pDst2 = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc2, pMean2), pInvStdDev2), pShift);
                _mm256_storeu_ps(dstPtrTempRow, pDst1);
                _mm256_storeu_ps(dstPtrTempRow + 8, pDst2);
                srcPtrTempRow += 16;
                dstPtrTempRow += 16;
            }
            int k = 0;
            for(; vectorLoopCount < dims[2] ; vectorLoopCount += 16)
            {
                *dstPtrTempRow++ = (scale * ((*srcPtrTempRow++ - meanPtr[k]) / stdDevPtr[k])) + shift;
                k++;
            }
            k = 0;
        }
    }
}

RppStatus normalize_generic_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                RpptGenericDescPtr srcGenericDescPtr,
                                                Rpp32f *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u axis_mask,
                                                Rpp32f *meanTensor,
                                                Rpp32f *stdDevTensor,
                                                Rpp32f scale,
                                                Rpp32f shift,
                                                Rpp32u *roiTensor,
                                                RppLayoutParams layoutParams,
                                                rpp::Handle& handle)
{
	Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u *srcStridesTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
	for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32f *srcPtrTemp, *dstPtrTemp, *meanTemp, *stdDevTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];
        meanTemp = meanTensor + batchCount * dstGenericDescPtr->strides[0];
        stdDevTemp = stdDevTensor + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u *roi = roiTensor + batchCount * nDim;
        Rpp32u *srcStrides = srcStridesTensor + batchCount * nDim;

        // compute output strides
        compute_strides(srcStrides, roi, nDim);

        Rpp32u paramStride[nDim];
        if(nDim == 3)
        {
            if (axis_mask == 3) // Normalize across Channels
            {
                paramStride[0] = paramStride[1] = 1;
                paramStride[2] = 0;
            }
            else if (axis_mask == 2) // Normalize across Width
            {
                //TODO
            }
            else if (axis_mask == 1) // Normalize across Height
            {
                //TODO
            }

            //if((axis_mask == 3) && (roi[2] == 16))
                //normalize_3D_tensor_avx_axis3(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTemp, stdDevTemp, scale, shift, roi, paramStride);
            //else if(axis_mask == 3)
                normalize_3D_tensor_axis3(srcPtrTemp, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, meanTemp, stdDevTemp, scale, shift, roi, paramStride);

        }
    }
    free(srcStridesTensor);

    return RPP_SUCCESS;
}