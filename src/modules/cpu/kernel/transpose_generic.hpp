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
using namespace std;

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

void increment_ndim_ptr(Rpp32f **dstPtr, Rpp32u nDim, Rpp32u increment)
{
    for(int i = 0; i < nDim; i++)
        dstPtr[i] += increment;
}

void rpp_store16_f32_f32_channelwise(Rpp32f **dstPtr, __m128 *p)
{
    _mm_storeu_ps(dstPtr[0], p[0]);
    _mm_storeu_ps(dstPtr[1], p[1]);
    _mm_storeu_ps(dstPtr[2], p[2]);
    _mm_storeu_ps(dstPtr[3], p[3]);
}

void rpp_transpose4x8_avx(__m256 *pSrc, __m128 *pDst)
{
  __m256 tmp0, tmp1, tmp2, tmp3;
  tmp0 = _mm256_shuffle_ps(pSrc[0], pSrc[1], 0x44);
  tmp2 = _mm256_shuffle_ps(pSrc[0], pSrc[1], 0xEE);
  tmp1 = _mm256_shuffle_ps(pSrc[2], pSrc[3], 0x44);
  tmp3 = _mm256_shuffle_ps(pSrc[2], pSrc[3], 0xEE);
  pSrc[0] = _mm256_shuffle_ps(tmp0, tmp1, 0x88);
  pSrc[1] = _mm256_shuffle_ps(tmp0, tmp1, 0xDD);
  pSrc[2] = _mm256_shuffle_ps(tmp2, tmp3, 0x88);
  pSrc[3] = _mm256_shuffle_ps(tmp2, tmp3, 0xDD);

  pDst[0] = _mm256_castps256_ps128(pSrc[0]);
  pDst[1] = _mm256_castps256_ps128(pSrc[1]);
  pDst[2] = _mm256_castps256_ps128(pSrc[2]);
  pDst[3] = _mm256_castps256_ps128(pSrc[3]);
  pDst[4] = _mm256_extractf128_ps(pSrc[0], 1);
  pDst[5] = _mm256_extractf128_ps(pSrc[1], 1);
  pDst[6] = _mm256_extractf128_ps(pSrc[2], 1);
  pDst[7] = _mm256_extractf128_ps(pSrc[3], 1);
}

void compute_2d_transpose(Rpp32f *srcPtrTemp, Rpp32f *dstPtrTemp, Rpp32u height, Rpp32u width, Rpp32u srcRowStride, Rpp32u dstRowStride)
{
    Rpp32u alignedRows = (height / 4) * 4;
    Rpp32u alignedCols = (width / 8) * 8;
    Rpp32u vectorIncrement = 8;
    Rpp32u dstRowVectorStride = vectorIncrement * dstRowStride;

    int i = 0;
    for(; i < alignedRows; i += 4)
    {
        Rpp32s k = (i / 4);
        Rpp32f *srcPtrRow[4], *dstPtrRow[8];

        for(int j = 0; j < 4; j++)
            srcPtrRow[j] = srcPtrTemp + (i + j) * srcRowStride;
        for(int j = 0; j < 8; j++)
            dstPtrRow[j] = dstPtrTemp + j * dstRowStride + k * 4;

        Rpp32u vectorLoopCount = 0;
        for(; vectorLoopCount < alignedCols; vectorLoopCount += vectorIncrement)
        {
            __m256 pSrc[4];
            pSrc[0] = _mm256_loadu_ps(srcPtrRow[0]);
            pSrc[1] = _mm256_loadu_ps(srcPtrRow[1]);
            pSrc[2] = _mm256_loadu_ps(srcPtrRow[2]);
            pSrc[3] = _mm256_loadu_ps(srcPtrRow[3]);

            __m128 pDst[8];
            rpp_transpose4x8_avx(pSrc, pDst);
            _mm_storeu_ps(dstPtrRow[0], pDst[0]);
            _mm_storeu_ps(dstPtrRow[1], pDst[1]);
            _mm_storeu_ps(dstPtrRow[2], pDst[2]);
            _mm_storeu_ps(dstPtrRow[3], pDst[3]);
            _mm_storeu_ps(dstPtrRow[4], pDst[4]);
            _mm_storeu_ps(dstPtrRow[5], pDst[5]);
            _mm_storeu_ps(dstPtrRow[6], pDst[6]);
            _mm_storeu_ps(dstPtrRow[7], pDst[7]);

            srcPtrRow[0] += vectorIncrement;
            srcPtrRow[1] += vectorIncrement;
            srcPtrRow[2] += vectorIncrement;
            srcPtrRow[3] += vectorIncrement;
            dstPtrRow[0] += dstRowVectorStride;
            dstPtrRow[1] += dstRowVectorStride;
            dstPtrRow[2] += dstRowVectorStride;
            dstPtrRow[3] += dstRowVectorStride;
            dstPtrRow[4] += dstRowVectorStride;
            dstPtrRow[5] += dstRowVectorStride;
            dstPtrRow[6] += dstRowVectorStride;
            dstPtrRow[7] += dstRowVectorStride;
        }
    }

    // Handle remaining cols
    for(int k = 0; k < alignedRows; k++)
    {
        Rpp32f *srcPtrRowTemp = srcPtrTemp + k * srcRowStride + alignedCols;
        Rpp32f *dstPtrRowTemp = dstPtrTemp + alignedCols * dstRowStride + k;
        for(int j = alignedCols; j < width; j++)
        {
            *dstPtrRowTemp = *srcPtrRowTemp;
            srcPtrRowTemp++;
            dstPtrRowTemp += dstRowStride;
        }
    }

    // Handle remaining rows
    for( ; i < height; i++)
    {
        Rpp32f *srcPtrRowTemp = srcPtrTemp + i * srcRowStride;
        Rpp32f *dstPtrRowTemp = dstPtrTemp + i;
        for(int j = 0; j < width; j++)
        {
            *dstPtrRowTemp = *srcPtrRowTemp;
            srcPtrRowTemp++;
            dstPtrRowTemp += dstRowStride;
        }
    }
}

template<typename T>
void transpose(T *dst, Rpp32u *dstStrides, T *src, Rpp32u *srcStrides, Rpp32u *dstShape, Rpp32u nDim)
{
    if (nDim == 0)
    {
        *dst = *src;
    }
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            transpose(dst, dstStrides + 1, src, srcStrides + 1, dstShape + 1, nDim - 1);
            dst += *dstStrides;
            src += *srcStrides;
        }
    }
}

RppStatus transpose_generic_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                RpptGenericDescPtr srcGenericDescPtr,
                                                Rpp32f *dstPtr,
                                                RpptGenericDescPtr dstGenericDescPtr,
                                                Rpp32u *permTensor,
                                                Rpp32u *roiTensor,
                                                RppLayoutParams layoutParams,
                                                rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = dstGenericDescPtr->numDims;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    // allocate buffer for input strides, output strides, output shapes
    Rpp32u *srcStridesTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));
    Rpp32u *dstStridesTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));
    Rpp32u *dstShapeTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32f *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u *dstShape = dstShapeTensor + batchCount * nDim;
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];
        Rpp32u *perm = permTensor;
        Rpp32u *srcStrides = srcStridesTensor + batchCount * nDim;
        Rpp32u *dstStrides = dstStridesTensor + batchCount * nDim;

        bool copyInput = true;
        for(int i = 0; i < nDim; i++)
            copyInput *= (perm[i] == i);

        // do memcpy of input to output since output order is same as input order
        if(copyInput)
        {
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcGenericDescPtr->strides[0] * sizeof(Rpp32f)));
        }
        else
        {
            for(int i = 1; i < nDim; i++)
                srcPtrTemp += begin[i] * srcGenericDescPtr->strides[i];

            if (nDim == 2 && perm[0] == 1 && perm[1] == 0)
            {
                compute_2d_transpose(srcPtrTemp, dstPtrTemp, length[0], length[1], srcGenericDescPtr->strides[1], dstGenericDescPtr->strides[1]);
            }
            else if (nDim == 3)
            {
                if(perm[0] == 2 && perm[1] == 0 && perm[2] == 1 && length[2] == 16)
                {
                    Rpp32u height = length[0];
                    Rpp32u width = length[1];
                    Rpp32u channels = 16;
                    Rpp32u bufferLength = width * channels;
                    Rpp32u alignedLength = (bufferLength / 64) * 64;
                    Rpp32u vectorIncrement = 64;
                    Rpp32u vectorIncrementPerChannel = 4;

                    // initialize pointers for 16 channel
                    Rpp32f *dstPtrChannel[16];
                    for(int i = 0; i < 16; i++)
                        dstPtrChannel[i] = dstPtrTemp + i * dstGenericDescPtr->strides[1];

                    // loop over rows
                    for(int i = 0; i < height; i++)
                    {
                        Rpp32f *srcPtrRow = srcPtrTemp;

                        // update temporary pointers for 16 channel
                        Rpp32f *dstPtrTempChannel[16];
                        for(int k = 0; k < 16; k++)
                            dstPtrTempChannel[k] = dstPtrChannel[k];

                        Rpp32u vectorLoopCount = 0;
                        for( ; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                        {
                            __m256 pSrc[8];
                            // Load 64 values for source
                            pSrc[0] = _mm256_loadu_ps(srcPtrRow);
                            pSrc[1] = _mm256_loadu_ps(srcPtrRow + 16);
                            pSrc[2] = _mm256_loadu_ps(srcPtrRow + 32);
                            pSrc[3] = _mm256_loadu_ps(srcPtrRow + 48);

                            pSrc[4] = _mm256_loadu_ps(srcPtrRow + 8);
                            pSrc[5] = _mm256_loadu_ps(srcPtrRow + 24);
                            pSrc[6] = _mm256_loadu_ps(srcPtrRow + 40);
                            pSrc[7] = _mm256_loadu_ps(srcPtrRow + 56);

                            __m128 pDst[16];
                            rpp_transpose4x8_avx(&pSrc[0], &pDst[0]);
                            rpp_transpose4x8_avx(&pSrc[4], &pDst[8]);

                            // Store 4 values per in output per channel
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[0], &pDst[0]);
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[4], &pDst[4]);
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[8], &pDst[8]);
                            rpp_store16_f32_f32_channelwise(&dstPtrTempChannel[12], &pDst[12]);

                            srcPtrRow += vectorIncrement;
                            increment_ndim_ptr(dstPtrTempChannel, 16, vectorIncrementPerChannel);
                        }
                        for( ; vectorLoopCount < bufferLength; vectorLoopCount += 16)
                        {
                            for(int k = 0; k < 16; k++)
                                *dstPtrTempChannel[k] = srcPtrRow[k];

                            srcPtrRow += 16;
                            increment_ndim_ptr(dstPtrTempChannel, 16, 1);
                        }
                        srcPtrTemp += srcGenericDescPtr->strides[1];
                        increment_ndim_ptr(dstPtrChannel, 16, dstGenericDescPtr->dims[3]);
                    }
                }
                else if(perm[0] == 1 && perm[1] == 0 && perm[2] == 2)
                {
                    Rpp32f *srcPtrRow = srcPtrTemp;
                    Rpp32f *dstPtrRow = dstPtrTemp;
                    Rpp32u height = length[0];
                    Rpp32u width = length[1];
                    Rpp32u channels = length[2];
                    for(int i = 0; i < height; i++)
                    {
                        Rpp32f *srcPtrRowTemp = srcPtrRow;
                        Rpp32f *dstPtrRowTemp = dstPtrRow;
                        for(int j = 0; j < width; j++)
                        {
                            memcpy(dstPtrRowTemp, srcPtrRowTemp, channels * sizeof(Rpp32f));
                            srcPtrRowTemp += srcGenericDescPtr->strides[2];
                            dstPtrRowTemp += dstGenericDescPtr->strides[1];
                        }
                        srcPtrRow += srcGenericDescPtr->strides[1];
                        dstPtrRow += dstGenericDescPtr->strides[2];
                    }
                }
                else if(perm[0] == 0 && perm[1] == 2 && perm[2] == 1)
                {
                    Rpp32f *srcPtrRow = srcPtrTemp;
                    Rpp32f *dstPtrRow = dstPtrTemp;
                    for(int i = 0; i < length[0]; i++)
                    {
                        compute_2d_transpose(srcPtrTemp, dstPtrTemp, length[1], length[2], srcGenericDescPtr->strides[2], dstGenericDescPtr->strides[2]);

                        // increment src and dst pointers
                        srcPtrTemp += srcGenericDescPtr->strides[1];
                        dstPtrTemp += dstGenericDescPtr->strides[1];
                    }
                }
            }
            else if (nDim == 4)
            {
                Rpp32u vectorIncrement = 8;
                if(perm[0] == 1 && perm[1] == 2 && perm[2] == 3 && perm[3] == 0)
                {
                    Rpp32u bufferLength = length[perm[3]];
                    Rpp32u alignedLength = (bufferLength / vectorIncrement) * vectorIncrement;
                    Rpp32f *srcPtr0 = srcPtrTemp;
                    Rpp32f *dstPtr0 = dstPtrTemp;
                    for(int i = 0; i < length[perm[0]]; i++)
                    {
                        Rpp32f *srcPtr1 = srcPtr0;
                        Rpp32f *dstPtr1 = dstPtr0;
                        for(int j = 0; j < length[perm[1]]; j++)
                        {
                            Rpp32f *srcPtr2 = srcPtr1;
                            Rpp32f *dstPtr2 = dstPtr1;
                            for(int k = 0; k < length[perm[2]]; k++)
                            {
                                Rpp32f *srcPtr3 = srcPtr2;
                                Rpp32f *dstPtr3 = dstPtr2;

                                Rpp32u vectorLoopCount = 0;
                                for( ; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                                {
                                    __m256 pSrc = _mm256_setr_ps(*srcPtr3,
                                                                 *(srcPtr3 + srcGenericDescPtr->strides[1]),
                                                                 *(srcPtr3 + 2 * srcGenericDescPtr->strides[1]),
                                                                 *(srcPtr3 + 3 * srcGenericDescPtr->strides[1]),
                                                                 *(srcPtr3 + 4 * srcGenericDescPtr->strides[1]),
                                                                 *(srcPtr3 + 5 * srcGenericDescPtr->strides[1]),
                                                                 *(srcPtr3 + 6 * srcGenericDescPtr->strides[1]),
                                                                 *(srcPtr3 + 7 * srcGenericDescPtr->strides[1]));
                                    _mm256_storeu_ps(dstPtr3, pSrc);
                                    srcPtr3 += vectorIncrement * srcGenericDescPtr->strides[1];
                                    dstPtr3 += vectorIncrement;
                                }
                                for( ; vectorLoopCount < bufferLength; vectorLoopCount++)
                                {
                                    *dstPtr3 = *srcPtr3;
                                    srcPtr3 += srcGenericDescPtr->strides[1];
                                    dstPtr3++;
                                }
                                srcPtr2 += 1;
                                dstPtr2 += dstGenericDescPtr->strides[3];
                            }
                            srcPtr1 += srcGenericDescPtr->strides[3];
                            dstPtr1 += dstGenericDescPtr->strides[2];
                        }
                        srcPtr0 +=  srcGenericDescPtr->strides[2];
                        dstPtr0 += dstGenericDescPtr->strides[1];
                    }
                }
            }
            else
            {
                // compute output shape
                for(Rpp32u i = 0; i < nDim; i++)
                    dstShape[i] = length[perm[i]];

                // compute output strides
                compute_strides(dstStrides, dstShape, nDim);

                // compute input strides and update as per the permute order
                Rpp32u tempStrides[RPPT_MAX_DIMS];
                compute_strides(tempStrides, roi, nDim);
                for(int i = 0; i < nDim; i++)
                    srcStrides[i] = tempStrides[perm[i]];

                // perform transpose as per the permute order
                transpose(dstPtrTemp, dstStrides, srcPtrTemp, srcStrides, dstShape, nDim);
            }
        }
    }
    free(srcStridesTensor);
    free(dstStridesTensor);
    free(dstShapeTensor);

    return RPP_SUCCESS;
}

template<typename T>
RppStatus transpose_generic_host_tensor(T *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = dstGenericDescPtr->numDims;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    // allocate buffer for input strides, output strides, output shapes
    Rpp32u *srcStridesTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));
    Rpp32u *dstStridesTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));
    Rpp32u *dstShapeTensor = (Rpp32u *)calloc(nDim * batchSize, sizeof(int));

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u *dstShape = dstShapeTensor + batchCount * nDim;
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];
        Rpp32u *perm = permTensor;
        Rpp32u *srcStrides = srcStridesTensor + batchCount * nDim;
        Rpp32u *dstStrides = dstStridesTensor + batchCount * nDim;

        bool copyInput = true;
        for(int i = 0; i < nDim; i++)
            copyInput *= (perm[i] == i);

        // do memcpy of input to output since output order is same as input order
        if(copyInput)
        {
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcGenericDescPtr->strides[0] * sizeof(T)));
        }
        else
        {
            for(int i = 1; i < nDim; i++)
                srcPtrTemp += begin[i] * srcGenericDescPtr->strides[i];

            // compute output shape
            for(Rpp32u i = 0; i < nDim; i++)
                dstShape[i] = length[perm[i]];

            // compute output strides
            compute_strides(dstStrides, dstShape, nDim);

            // compute input strides and update as per the permute order
            Rpp32u tempStrides[RPPT_MAX_DIMS];
            compute_strides(tempStrides, roi, nDim);
            for(int i = 0; i < nDim; i++)
                srcStrides[i] = tempStrides[perm[i]];

            // perform transpose as per the permute order
            transpose(dstPtrTemp, dstStrides, srcPtrTemp, srcStrides, dstShape, nDim);
        }
    }
    free(srcStridesTensor);
    free(dstStridesTensor);
    free(dstShapeTensor);

    return RPP_SUCCESS;
}