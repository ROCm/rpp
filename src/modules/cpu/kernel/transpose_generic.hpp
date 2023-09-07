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

void transpose(Rpp32f *dst, Rpp32u *dstStrides, Rpp32f *src, Rpp32u *srcStrides, Rpp32u *dstShape, Rpp32u nDim) 
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
        Rpp32u *roi = roiTensor + batchCount * nDim;
        Rpp32u *perm = permTensor;
        Rpp32u *srcStrides = srcStridesTensor + batchCount * nDim;
        Rpp32u *dstStrides = dstStridesTensor + batchCount * nDim;
        
        // compute output shape
        for(Rpp32u i = 0; i < nDim; i++)
            dstShape[i] = roi[perm[i]];
        
        // compute output strides
        compute_strides(dstStrides, dstShape, nDim);
        
        // compute input strides and update as per the permute order
        Rpp32u tempStrides[RPPT_MAX_DIMS];
        compute_strides(tempStrides, roi, nDim);
        for(int i = 0; i < nDim; i++)
            srcStrides[i] = tempStrides[perm[i]];
        
        if (nDim == 2)
        {
            if(perm[0] == 0 && perm[1] == 1)
            {
                // Do memcpy since output order is same as input order
                memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcGenericDescPtr->strides[0] * sizeof(Rpp32f)));
            }
            else if(perm[0] == 1 && perm[1] == 0)
            {
               
                Rpp32u height = roi[0];
                Rpp32u width = roi[1];
                Rpp32u alignedRows = (roi[0] / 4) * 4;
                Rpp32u alignedCols = (roi[1] / 4) * 4;
                Rpp32u vectorIncrement = 4;
                int i = 0;
                for(; i < alignedRows; i += vectorIncrement)
                {
                    Rpp32s k = (i / 4);
                    Rpp32f *srcPtrRow[4] = {srcPtrTemp + i * width, srcPtrTemp + (i + 1) * width, srcPtrTemp + (i + 2) * width, srcPtrTemp + (i + 3) * width};
                    Rpp32f *dstPtrRow[4] = {dstPtrTemp + k * vectorIncrement, dstPtrTemp + height + k * vectorIncrement, dstPtrTemp + 2 * height + k * vectorIncrement, dstPtrTemp + 3 * height + k * vectorIncrement};
                    Rpp32u vectorLoopCount = 0;
                    for(; vectorLoopCount < alignedCols; vectorLoopCount += vectorIncrement)
                    {
                        __m128 pSrc[4];
                        pSrc[0] = _mm_loadu_ps(srcPtrRow[0]);
                        pSrc[1] = _mm_loadu_ps(srcPtrRow[1]);
                        pSrc[2] = _mm_loadu_ps(srcPtrRow[2]);
                        pSrc[3] = _mm_loadu_ps(srcPtrRow[3]);
                        _MM_TRANSPOSE4_PS(pSrc[0], pSrc[1], pSrc[2], pSrc[3]);
                        _mm_storeu_ps(dstPtrRow[0], pSrc[0]);
                        _mm_storeu_ps(dstPtrRow[1], pSrc[1]);
                        _mm_storeu_ps(dstPtrRow[2], pSrc[2]);
                        _mm_storeu_ps(dstPtrRow[3], pSrc[3]);
                        
                        srcPtrRow[0] += vectorIncrement;
                        srcPtrRow[1] += vectorIncrement;
                        srcPtrRow[2] += vectorIncrement;
                        srcPtrRow[3] += vectorIncrement;
                        dstPtrRow[0] += vectorIncrement * height;
                        dstPtrRow[1] += vectorIncrement * height;
                        dstPtrRow[2] += vectorIncrement * height;
                        dstPtrRow[3] += vectorIncrement * height;
                    }
                }
                
                // Handle remaining cols
                for(int k = 0; k < alignedRows; k++)
                {
                    Rpp32f *srcPtrRow = srcPtrTemp + k * width + alignedCols;
                    Rpp32f *dstPtrRow = dstPtrTemp + alignedCols * height + k;
                    for(int j = alignedCols; j < width; j++)
                    {
                        *dstPtrRow = *srcPtrRow;
                        srcPtrRow++;
                        dstPtrRow += height;
                    }
                }
                
                // Handle remaining rows
                for( ; i < height; i++)
                {
                    Rpp32f *srcPtrRow = srcPtrTemp + i * width;
                    Rpp32f *dstPtrRow = dstPtrTemp + i;
                    for(int j = 0; j < width; j++)
                    {
                        *dstPtrRow = *srcPtrRow;
                        srcPtrRow++;
                        dstPtrRow += height;
                    }  
                } 
            }
        }
        else
        {
            // perform transpose as per the permute order
            transpose(dstPtrTemp, dstStrides, srcPtrTemp, srcStrides, dstShape, nDim);   
        }
    }
    free(srcStridesTensor);
    free(dstStridesTensor);
    free(dstShapeTensor);

    return RPP_SUCCESS;
}
