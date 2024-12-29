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

// Computes concat for 3D variants
void concat_3D_tensor(Rpp32f *srcPtr, Rpp32f *srcPtr1, RpptGenericDescPtr srcGenericDescPtr, RpptGenericDescPtr srcGenericDescPtr1, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *strides, Rpp32u *dims1, Rpp32u *strides1, Rpp32u *dstStrides, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims[2] < dims1[2]) ? dims[2] : dims1[2];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32f *srcPtrRow = srcPtr;
        Rpp32f *srcPtrRow1 = srcPtr1;
        Rpp32f *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *srcPtrRowTemp1 = srcPtrRow1;
            Rpp32f *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRowTemp, &pDst); 
                rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrRowTemp, &pDst);
                rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrRowTemp1, &pDst); 
                rpp_simd_store(rpp_store8_f32_to_f32_avx, (dstPtrRowTemp + strides[2]), &pDst);
                srcPtrRowTemp += vectorIncrement;
                srcPtrRowTemp1 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(Rpp32u k = vectorLoopCount ; k < dims[2] ; k++)
            {
                *(dstPtrRowTemp + k - vectorLoopCount) = *srcPtrRowTemp++;
            }
            for(Rpp32u k = vectorLoopCount ; k < dims1[2] ; k++)
            {
                *(dstPtrRowTemp + strides[2] + k - vectorLoopCount) = *srcPtrRowTemp1++;
            }
            srcPtrRow += strides[2];
            srcPtrRow1 += strides1[2];
            dstPtrRow += dstStrides[2];
        }
        srcPtr += strides[1];
        srcPtr1 += strides1[1];
        dstPtr += dstStrides[1];
    }
   
}

// Computes concat for 3D variants
void concat_3D_tensor(Rpp8u *srcPtr, Rpp8u *srcPtr1, RpptGenericDescPtr srcGenericDescPtr, RpptGenericDescPtr srcGenericDescPtr1, Rpp8u *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *strides,  Rpp32u *dims1, Rpp32u *strides1,Rpp32u *dstStrides, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims[2] < dims1[2]) ? dims[2] : dims1[2];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp8u *srcPtrRow = srcPtr;
        Rpp8u *srcPtrRow1 = srcPtr1;
        Rpp8u *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
             for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst); 
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, pDst);
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp1, &pDst); 
                rpp_simd_store(rpp_store8_f32_to_u8_avx, (dstPtrRowTemp + strides[2]), pDst);
                srcPtrRowTemp += vectorIncrement;
                srcPtrRowTemp1 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(Rpp32u k = vectorLoopCount; k < dims[2] ; k ++)
            {
                *(dstPtrRowTemp + k - vectorLoopCount) = *srcPtrRowTemp++;
            }
            for(Rpp32u k = vectorLoopCount; k < dims1[2] ; k ++)
            {
                *(dstPtrRowTemp + strides[2] + k - vectorLoopCount) = *srcPtrRowTemp1++;
            }
            srcPtrRow += strides[2];
            srcPtrRow1 += strides1[2];
            dstPtrRow += dstStrides[2];
        }
        srcPtr += strides[1];
        srcPtr1 += strides1[1];
        dstPtr += dstStrides[1];
    }
   
}

// Computes concat for 3D variants
void concat_3D_axismask0_tensor(Rpp8u *srcPtr, Rpp8u *srcPtr1, RpptGenericDescPtr srcGenericDescPtr, RpptGenericDescPtr srcGenericDescPtr1, Rpp8u *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *strides, Rpp32u *dims1, Rpp32u *strides1, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32u bufferLength = dims[2];
        Rpp32u alignedLength = (bufferLength / 8) * 8;
        Rpp32u bufferLength1 = dims1[2];
        Rpp32u alignedLength1 = (bufferLength1 / 8) * 8;
        Rpp8u *srcPtrRow = srcPtr;
        Rpp8u *srcPtrRow1 = srcPtr1;
        Rpp8u *dstPtrRow = dstPtr;
        // for(Rpp32u j = 0; j < dims[1]; j++)
        // {
        //     Rpp8u *srcPtrRowTemp = srcPtrRow;
        //     Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
        //     Rpp8u *dstPtrRowTemp = dstPtrRow;
        //     Rpp32u vectorLoopCount = 0;
        //     __m256 pDst ;
        //     for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
        //     {
        //         rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst); 
        //         rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, pDst);
        //         rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp1, &pDst); 
        //         rpp_simd_store(rpp_store8_f32_to_u8_avx, (dstPtrRowTemp + strides[1]) , pDst);
        //         srcPtrRowTemp += vectorIncrement;
        //         srcPtrRowTemp1 += vectorIncrement;
        //         dstPtrRowTemp += vectorIncrement;
        //     }
        //     for(; vectorLoopCount < dims[2] ; vectorLoopCount ++)
        //     {
        //         *(dstPtrRowTemp + strides[1]) = *srcPtrRowTemp1++;
        //         *dstPtrRowTemp++ = *srcPtrRowTemp++;
        //     }

        //     srcPtrRow += strides[2];
        //     srcPtrRow1 += strides[2];
        //     dstPtrRow += (strides[2]);
        // }
        for(Rpp32u j = 0; j < dims[1]; j++)
        {
            Rpp8u *srcPtrRowTemp = srcPtrRow;
            Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
            for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp, &pDst); 
                rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrRowTemp, pDst);
                srcPtrRowTemp += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims[2] ; vectorLoopCount ++)
            {
                *dstPtrRowTemp++ = *srcPtrRowTemp++;
            }
            srcPtrRow += strides[2];
            dstPtrRow += strides[2];
        }
        dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < dims1[1]; j++)
        {
            Rpp8u *srcPtrRowTemp1 = srcPtrRow1;
            Rpp8u *dstPtrRowTemp = dstPtrRow;
            Rpp32u vectorLoopCount = 0;
            __m256 pDst ;
            for(; vectorLoopCount < alignedLength1; vectorLoopCount += vectorIncrement)
            {
                rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrRowTemp1, &pDst); 
                rpp_simd_store(rpp_store8_f32_to_u8_avx, (dstPtrRowTemp + strides[1]) , pDst);
                srcPtrRowTemp1 += vectorIncrement;
                dstPtrRowTemp += vectorIncrement;
            }
            for(; vectorLoopCount < dims1[2] ; vectorLoopCount ++)
            {
                *(dstPtrRowTemp + strides[1]) = *srcPtrRowTemp1++;
                dstPtrRowTemp++;
            }
            srcPtrRow1 += strides1[2];
            dstPtrRow += strides1[2];
        }
        srcPtr += strides[1];
        srcPtr1 += strides1[1];
        dstPtr += strides[1];
    }
}

// Computes concat for ND variants
template<typename T1, typename T2>
void concat_NDD_tensor(T1 *srcPtr, Rpp32u *srcStride, T2 *dstPtr, Rpp32u *dims, Rpp32u tensorDim, Rpp32u level, Rpp32u axisMask, Rpp32u maxDims)
{
    if(level == (tensorDim - 1))
    {
        for (Rpp32u i = 0; i < dims[level]; i++)
        {
            *(dstPtr + i) = *srcPtr;
            srcPtr++;
        }
    }
    else
    {
        int size = 1;
        for(int i = level + 1; i < tensorDim; i++)
        {
            size = size * dims[i];
        }
        for (Rpp32u i = 0; i < dims[level]; i++)
        {
            concat_NDD_tensor(srcPtr, srcStride, dstPtr, dims, tensorDim, level + 1, axisMask, maxDims);
            dstPtr += srcStride[level + 1]; 
            srcPtr += srcStride[level + 1];
        }
    }
}

// Computes concat for ND variants
template<typename T1, typename T2>
void concat_ND_tensor(T1 *srcPtr, T1 *srcPtr1, Rpp32u *srcStride, Rpp32u *srcStride1, Rpp32u *dstStride, T2 *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *dims, Rpp32u *dims1, Rpp32u tensorDim, Rpp32u level, Rpp32u axisMask, Rpp32u maxDims)
{

    if(level >= axisMask)
    {
        concat_NDD_tensor(srcPtr, srcStride, dstPtr, dims, tensorDim, level, axisMask, maxDims);
        dstPtr += srcStride[level];
        concat_NDD_tensor(srcPtr1, srcStride1, dstPtr, dims1, tensorDim, level, axisMask, maxDims);
    }
    else
    {
        for (Rpp32u i = 0; i < dims[level]; i++)
        {
            concat_ND_tensor(srcPtr, srcPtr1, srcStride, srcStride1, dstStride,  dstPtr, dstGenericDescPtr, dims, dims1, tensorDim, level + 1, axisMask, maxDims);
            dstPtr += dstStride[level + 1]; 
            srcPtr += srcStride[level + 1];
            srcPtr1 += srcStride[level + 1];
        }
    }
}

// Computes concat for 2D
void concat_2D_tensor(Rpp32f *srcPtr, Rpp32f *srcPtr1, RpptGenericDescPtr srcDescPtr, RpptGenericDescPtr srcDescPtr1, Rpp32f *dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32u *dims, Rpp32u *strides,Rpp32u *dims1, Rpp32u *strides1, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims[1] < dims1[1]) ? dims[1] : dims1[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + i * strides[1];
        Rpp32f *srcPtrTemp1 = srcPtr1 + i * strides[1];
        Rpp32f *dstPtrTemp = dstPtr + (i * strides[1] * 2);

        Rpp32u vectorLoopCount = 0;
        __m256 pDst;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
        {
            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &pDst); 
            rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &pDst);
            rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp1, &pDst); 
            rpp_simd_store(rpp_store8_f32_to_f32_avx, (dstPtrTemp + strides[1]), &pDst);
            srcPtrTemp += vectorIncrement;
            srcPtrTemp1 += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        dstPtrTemp = dstPtr + (i * strides[1] * 2);
        for(int j = vectorLoopCount; j < dims[1] ; j ++)
            *(dstPtrTemp + j) = *srcPtrTemp++;
        for(int j = vectorLoopCount; j < dims1[1] ; j ++)
            *(dstPtrTemp + strides[1] + j)  = *srcPtrTemp1++;
    }
}
// Computes concat for 2D
void concat_2D_tensor(Rpp8u *srcPtr, Rpp8u *srcPtr1, RpptGenericDescPtr srcDescPtr, RpptGenericDescPtr srcDescPtr1, Rpp8u *dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32u *dims, Rpp32u *strides, Rpp32u *dims1, Rpp32u *strides1, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = (dims[1] < dims1[1]) ? dims[1] : dims1[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp8u *srcPtrTemp = srcPtr + i * strides[1];
        Rpp8u *srcPtrTemp1 = srcPtr1 + i * strides[1];
        Rpp8u *dstPtrTemp = dstPtr + (i * strides[1] * 2);

        Rpp32u vectorLoopCount = 0;
        __m256 pDst ;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
        {
            rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrTemp, &pDst); 
            rpp_simd_store(rpp_store8_f32_to_u8_avx, dstPtrTemp, pDst);
            rpp_simd_load(rpp_load8_u8_to_f32_avx, srcPtrTemp1, &pDst); 
            rpp_simd_store(rpp_store8_f32_to_u8_avx, (dstPtrTemp + strides[1]), pDst);
            srcPtrTemp += vectorIncrement;
            srcPtrTemp1 += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        dstPtrTemp = dstPtr + (i * strides[1] * 2);
        for(int j = vectorLoopCount; j < dims[1] ; j ++)
            *(dstPtrTemp + j) = *srcPtrTemp++;
        for(int j = vectorLoopCount; j < dims1[1] ; j ++)
            *(dstPtrTemp + strides[1] + j)  = *srcPtrTemp1++;
    }
}

RppStatus concat_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     Rpp32f *srcPtr1,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     RpptGenericDescPtr srcGenericDescPtr1,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     Rpp32u axisMask,
                                     Rpp32u *roiTensor,
                                     Rpp32u *roiTensor1,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcGenericDescPtr->numDims - 1;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *roi1 = roiTensor1 + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *length1 = &roi1[tensorDims];

        Rpp32f *srcPtrTemp, *srcPtrTemp1, *dstPtrTemp, *meanTensor, *stdDevTensor;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        srcPtrTemp1 = srcPtr1 + batchCount * srcGenericDescPtr1->strides[0];
        dstPtrTemp = dstPtr + batchCount * (dstGenericDescPtr->strides[0]);

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        if(tensorDims == 2) // Called for audio testcase and for any other 2D case
        {
            Rpp32u srcReductionDims[2], srcReductionDims1[2], srcStride[2], srcStride1[2], dstStride[2];
            if (axisMask == 0)
            {
                srcStride[0] = 1;
                srcStride[1] = srcGenericDescPtr->strides[0];
                srcReductionDims[0] = 1;
                srcReductionDims[1] = length[0] * length[1];
                srcStride1[0] = 1;
                srcStride1[1] = srcGenericDescPtr1->strides[0];
                srcReductionDims1[0] = 1;
                srcReductionDims1[1] = length1[0] * length1[1];
                dstStride[0] = 1;
                dstStride[1] = dstGenericDescPtr->strides[0];
            }
            else if (axisMask == 1)
            {
                srcStride[0] = srcGenericDescPtr->strides[2];
                srcStride[1] = srcGenericDescPtr->strides[1];
                srcReductionDims[0] = length[0];
                srcReductionDims[1] = length[1];
                srcStride1[0] = srcGenericDescPtr1->strides[2];
                srcStride1[1] = srcGenericDescPtr1->strides[1];
                srcReductionDims1[0] = length1[0];
                srcReductionDims1[1] = length1[1];
                dstStride[0] = dstGenericDescPtr->strides[2];
                dstStride[1] = dstGenericDescPtr->strides[1];
            }
            concat_2D_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr, srcGenericDescPtr1, dstPtrTemp, dstGenericDescPtr, srcReductionDims, srcStride, srcReductionDims1, srcStride1, axisMask);
        }
        else if(tensorDims == 3) // Called when a 3D tensor is passed to kernel
        {
            Rpp32u srcReductionDims[3], srcStride[3], srcReductionDims1[3], srcStride1[3], dstStride[3];
            switch(axisMask)
            {
                case 0: // concat axes 0
                {
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = 1;
                    srcReductionDims[2] = length[0] * length[1] * length[2];
                    srcStride[0] = 1;
                    srcStride[1] = 1;
                    srcStride[2] = srcGenericDescPtr->strides[0];
                    srcReductionDims1[0] = 1;
                    srcReductionDims1[1] = 1;
                    srcReductionDims1[2] = length1[0] * length1[1] * length1[2];
                    srcStride1[0] = 1;
                    srcStride1[1] = 1;
                    srcStride1[2] = srcGenericDescPtr1->strides[0];
                    dstStride[0] = 1;
                    dstStride[1] = 1;
                    dstStride[2] = dstGenericDescPtr->strides[0];
                    break;
                }
                case 1: // concat axes 1
                {
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[0];
                    srcReductionDims[2] = length[1] * length[2];
                    srcStride[0] = 1;
                    srcStride[1] = 1;
                    srcStride[2] = srcGenericDescPtr->strides[1];
                    srcReductionDims1[0] = 1;
                    srcReductionDims1[1] = length1[0];
                    srcReductionDims1[2] = length1[1] * length1[2];
                    srcStride1[0] = 1;
                    srcStride1[1] = 1;
                    srcStride1[2] = srcGenericDescPtr1->strides[1];
                    dstStride[0] = 1;
                    dstStride[1] = 1;
                    dstStride[2] = dstGenericDescPtr->strides[1];
                    break;
                }
                case 2: // concat axes 2
                {
                    srcReductionDims[0] = length[0];
                    srcReductionDims[1] = length[1];
                    srcReductionDims[2] = length[2];
                    srcStride[0] = srcGenericDescPtr->strides[0];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[2];
                    srcReductionDims1[0] = length1[0];
                    srcReductionDims1[1] = length1[1];
                    srcReductionDims1[2] = length1[2];
                    srcStride1[0] = srcGenericDescPtr1->strides[0];
                    srcStride1[1] = srcGenericDescPtr1->strides[1];
                    srcStride1[2] = srcGenericDescPtr1->strides[2];
                    dstStride[0] = dstGenericDescPtr->strides[0];
                    dstStride[1] = dstGenericDescPtr->strides[1];
                    dstStride[2] = dstGenericDescPtr->strides[2];
                    break;
                }
                default:
                {
                    std::cout<<"Invalid Axis mask"<<std::endl;
                }
            }
            concat_3D_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr, srcGenericDescPtr1, dstPtrTemp, dstGenericDescPtr, srcReductionDims, srcStride, srcReductionDims1, srcStride1, dstStride, axisMask);
        }
        else // Handle any other ND tensor is passed to kernel
        {
            concat_ND_tensor(srcPtrTemp, srcPtrTemp1,srcGenericDescPtr->strides, srcGenericDescPtr1->strides, dstGenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr, length, length1, tensorDims, 0, axisMask, tensorDims);
        }
    }

    return RPP_SUCCESS;
}
RppStatus concat_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     Rpp8u *srcPtr1,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        RpptGenericDescPtr srcGenericDescPtr1,
                                        Rpp8u *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32u *roiTensor,
                                        Rpp32u *roiTensor1,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcGenericDescPtr->numDims - 1;
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    Rpp32u maxSize = 1;
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *roi1 = roiTensor1 + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *length1 = &roi1[tensorDims];

        Rpp8u *srcPtrTemp, *srcPtrTemp1, *dstPtrTemp, *meanTensor, *stdDevTensor;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        srcPtrTemp1 = srcPtr1 + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        Rpp8u *srcPtrChannel = srcPtrTemp;

        if(tensorDims == 2) // Called for audio testcase and for any other 2D case
        {
            Rpp32u srcReductionDims[2], srcStride[2], srcReductionDims1[2], srcStride1[2];
            if (axisMask == 0)
            {
                srcStride[0] = 1;
                srcStride[1] = srcGenericDescPtr->strides[0];
                srcReductionDims[0] = 1;
                srcReductionDims[1] = length[0] * length[1];
                srcStride1[0] = 1;
                srcStride1[1] = srcGenericDescPtr1->strides[0];
                srcReductionDims1[0] = 1;
                srcReductionDims1[1] = length1[0] * length1[1];
            }
            else if (axisMask == 1)
            {
                srcStride[0] = srcGenericDescPtr->strides[2];
                srcStride[1] = srcGenericDescPtr->strides[1];
                srcReductionDims[0] = length[0];
                srcReductionDims[1] = length[1];
                srcStride1[0] = srcGenericDescPtr1->strides[2];
                srcStride1[1] = srcGenericDescPtr1->strides[1];
                srcReductionDims1[0] = length1[0];
                srcReductionDims1[1] = length1[1];
            }
            concat_2D_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr, srcGenericDescPtr1, dstPtrTemp, dstGenericDescPtr, srcReductionDims, srcStride, srcReductionDims1, srcStride1,  axisMask);
        }
        else if(tensorDims == 3) // Called when a 3D tensor is passed to kernel
        {
            Rpp32u srcReductionDims[3], srcStride[3], srcReductionDims1[3], srcStride1[3], dstStride[3];
            switch(axisMask)
            {
                case 0: // concat axes 0
                {
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[0];
                    srcReductionDims[2] = length[1] * length[2];
                    srcStride[0] = 1;
                    srcStride[1] = srcGenericDescPtr->strides[0];
                    srcStride[2] = srcGenericDescPtr->strides[1];
                    srcReductionDims1[0] = 1;
                    srcReductionDims1[1] = length1[0];
                    srcReductionDims1[2] = length1[1] * length1[2];
                    srcStride1[0] = 1;
                    srcStride1[1] = srcGenericDescPtr1->strides[0];
                    srcStride1[2] = srcGenericDescPtr1->strides[1];
                    dstStride[0] = 1;
                    dstStride[1] = dstGenericDescPtr->strides[0];
                    dstStride[2] = dstGenericDescPtr->strides[1];
                    break;
                }
                case 1: // concat axes 1
                {
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[0];
                    srcReductionDims[2] = length[1] * length[2];
                    srcStride[0] = 1;
                    srcStride[1] = 1;
                    srcStride[2] = srcGenericDescPtr->strides[1];
                    srcReductionDims1[0] = 1;
                    srcReductionDims1[1] = length1[0];
                    srcReductionDims1[2] = length1[1] * length1[2];
                    srcStride1[0] = 1;
                    srcStride1[1] = 1;
                    srcStride1[2] = srcGenericDescPtr1->strides[1];
                    dstStride[0] = 1;
                    dstStride[1] = 1;
                    dstStride[2] = dstGenericDescPtr->strides[1];
                    break;
                }
                case 2: // concat axes 2
                {
                    srcReductionDims[0] = length[0];
                    srcReductionDims[1] = length[1];
                    srcReductionDims[2] = length[2];
                    srcStride[0] = srcGenericDescPtr->strides[0];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[2];
                    srcReductionDims1[0] = length1[0];
                    srcReductionDims1[1] = length1[1];
                    srcReductionDims1[2] = length1[2];
                    srcStride1[0] = srcGenericDescPtr1->strides[0];
                    srcStride1[1] = srcGenericDescPtr1->strides[1];
                    srcStride1[2] = srcGenericDescPtr1->strides[2];
                    dstStride[0] = dstGenericDescPtr->strides[0];
                    dstStride[1] = dstGenericDescPtr->strides[1];
                    dstStride[2] = dstGenericDescPtr->strides[2];
                    break;
                }
                default:
                {
                    std::cout<<"Invalid Axis mask"<<std::endl;
                }
            }
            if(axisMask == 0)
                concat_3D_axismask0_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr, srcGenericDescPtr1, dstPtrTemp, dstGenericDescPtr, srcReductionDims, srcStride, srcReductionDims1, srcStride1, axisMask);
            else
                concat_3D_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr, srcGenericDescPtr1, dstPtrTemp, dstGenericDescPtr, srcReductionDims, srcStride, srcReductionDims1, srcStride1, dstStride, axisMask);
        }
        else // Handle any other ND tensor is passed to kernel
        {
            concat_ND_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr->strides, srcGenericDescPtr1->strides, dstGenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr, length, length1, tensorDims, 0, axisMask, tensorDims);
        }
    }

    return RPP_SUCCESS;
}

template<typename T1, typename T2>
RppStatus concat_generic_host_tensor(T1 *srcPtr,
                                     T1 *srcPtr1,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        RpptGenericDescPtr srcGenericDescPtr1,
                                        T2 *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32u *roiTensor,
                                        Rpp32u *roiTensor1,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        T1 *srcPtrTemp;
        T1 *srcPtrTemp1;
        T2 *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        srcPtrTemp1 = srcPtr1 + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * (dstGenericDescPtr->strides[0]);
        int size = 1;
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *roi1 = roiTensor1 + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *begin1 = roi1;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u *length1 = &roi1[tensorDims];
        Rpp32u level = 0;
        T1 *srcPtrChannel = srcPtrTemp;
        T1 *srcPtrChannel1 = srcPtrTemp1;
         for(int i = 0; i < tensorDims; i++)
        {
            srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i + 1];
            srcPtrChannel1 += begin1[i] * srcGenericDescPtr1->strides[i + 1];
        }
        
        concat_ND_tensor(srcPtrChannel, srcPtrChannel1, srcGenericDescPtr->strides, srcGenericDescPtr1->strides, dstGenericDescPtr->strides, dstPtrTemp, dstGenericDescPtr, length, length1, tensorDims, level, axisMask, tensorDims);
    }

    return RPP_SUCCESS;
}