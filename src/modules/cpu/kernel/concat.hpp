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

// Computes concat for 3D non toggle variants
void concat_3D_tensor(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *length, Rpp32u axisMask)
{

    for(Rpp32u i = 0; i < length[0]; i++)
    {
        Rpp32f *srcPtrRow = srcPtr;
        Rpp32f *dstPtrRow = dstPtr;
        for(Rpp32u j = 0; j < length[1]; j++)
        {
            Rpp32f *srcPtrRowTemp = srcPtrRow;
            Rpp32f *dstPtrRowTemp = dstPtrRow;
            for(Rpp32u k = 0; k < length[2]; k++)
            {
                *dstPtrRowTemp = *srcPtrRowTemp;
                srcPtrRowTemp++;
                dstPtrRowTemp++;
            }
            if(axisMask == 2)
            {
                for(Rpp32u k = 0; k < length[2]; k++)
                {
                    *dstPtrRowTemp = *srcPtrRowTemp;
                    srcPtrRowTemp++;
                    dstPtrRowTemp++;
                }
            }
            srcPtrRow += srcGenericDescPtr->strides[2];
            dstPtrRow += dstGenericDescPtr->strides[2];
        }
        if(axisMask == 1)
        {
            for(Rpp32u j = 0; j < length[1]; j++)
            {
                Rpp32f *srcPtrRowTemp = srcPtrRow;
                Rpp32f *dstPtrRowTemp = dstPtrRow;
                for(Rpp32u k = 0; k < length[2]; k++)
                {
                    *dstPtrRowTemp = *srcPtrRowTemp;
                    srcPtrRowTemp++;
                    dstPtrRowTemp++;
                }
                srcPtrRow += srcGenericDescPtr->strides[2];
                dstPtrRow += dstGenericDescPtr->strides[2];
            }
            srcPtr += srcGenericDescPtr->strides[1];
            dstPtr += dstGenericDescPtr->strides[1];

        }
    }
    if(axisMask == 0)
    {
        for(Rpp32u i = 0; i < length[0]; i++)
        {
            Rpp32f *srcPtrRow = srcPtr;
            Rpp32f *dstPtrRow = dstPtr;
            for(Rpp32u j = 0; j < length[1]; j++)
            {
                Rpp32f *srcPtrRowTemp = srcPtrRow;
                Rpp32f *dstPtrRowTemp = dstPtrRow;
                for(Rpp32u k = 0; k < length[2]; k++)
                {
                    *dstPtrRowTemp = *srcPtrRowTemp;
                    srcPtrRowTemp++;
                    dstPtrRowTemp++;
                }
                srcPtrRow += srcGenericDescPtr->strides[2];
                dstPtrRow += dstGenericDescPtr->strides[2];
            }
            for(Rpp32u j = 0; j < length[1]; j++)
            {
                Rpp32f *srcPtrRowTemp = srcPtrRow;
                Rpp32f *dstPtrRowTemp = dstPtrRow;
                for(Rpp32u k = 0; k < length[2]; k++)
                {
                    *dstPtrRowTemp = *srcPtrRowTemp;
                    srcPtrRowTemp++;
                    dstPtrRowTemp++;
                }
                srcPtrRow += srcGenericDescPtr->strides[2];
                dstPtrRow += dstGenericDescPtr->strides[2];
            }
            srcPtr += srcGenericDescPtr->strides[1];
            dstPtr += dstGenericDescPtr->strides[1];
        }
    }
}

// // Computes normalize for 3D toggle variants when axis mask is set to 3
// void normalize_3D_tensor_axis3_toggle(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
//                          Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u *length)
// {
//     Rpp32f *srcPtrTemp = srcPtr;
//     Rpp32f *dstPtrTemp[length[2]];
//     dstPtrTemp[0] = dstPtr;
//     for(Rpp32u i = 1; i < length[2]; i++)
//         dstPtrTemp[i] = dstPtrTemp[i-1] + dstGenericDescPtr->strides[1];
//     Rpp32s paramIdx = 0;

//     for(Rpp32u i = 0; i < length[0]; i++)
//     {
//         Rpp32f *srcPtrRow = srcPtrTemp;
//         Rpp32f *dstPtrRow[length[2]];
//         for(Rpp32u l = 0; l < length[2]; l++)
//             dstPtrRow[l] = dstPtrTemp[l];
//         for(Rpp32u j = 0; j < length[1]; j++)
//         {
//             Rpp32f *srcPtrRowTemp = srcPtrRow;
//             Rpp32f *dstPtrRowTemp[length[2]];
//             for(Rpp32u l = 0; l < length[2]; l++)
//                 dstPtrRowTemp[l] = dstPtrRow[l];
//             for(Rpp32u k = 0; k < length[2]; k++)
//             {
//                 *dstPtrRowTemp[k]++ = ((*srcPtrRowTemp++ - meanPtr[paramIdx]) * multiplierPtr[paramIdx]) + shift;
//                 paramIdx += paramStride[2];
//             }
//             paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
//             srcPtrRow += srcGenericDescPtr->strides[2];
//             for(Rpp32u l = 0; l < length[2]; l++)
//                 dstPtrRow[l] += dstGenericDescPtr->strides[3];
//         }
//         srcPtrTemp += srcGenericDescPtr->strides[1];
//         for(Rpp32u l = 0; l < length[2]; l++)
//             dstPtrTemp[l] += dstGenericDescPtr->strides[2];
//     }
// }

// // Computes normalize for 3D non toggle variants, optimized with AVX when axis mask set to 3 and 16 channel normalize
// void normalize_3D_tensor_avx_axis3(Rpp32f *srcPtr, RpptGenericDescPtr srcGenericDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstGenericDescPtr,
//                                    Rpp32f *meanPtr, Rpp32f *multiplierPtr, Rpp32f shift, Rpp32u *paramStride, Rpp32u bufferLength, Rpp32u *length)
// {
//     Rpp32u vectorIncrement = 16;
//     Rpp32u alignedLength = (bufferLength / 16) * 16;
//     Rpp32u outerDim = length[0];

//     // set shift, mean and stddev
//     __m256 pShift = _mm256_set1_ps(shift);
//     __m256 pMean1 = _mm256_loadu_ps(meanPtr);
//     __m256 pMean2 = _mm256_loadu_ps(meanPtr + 8);
//     __m256 pMultiplier1 = _mm256_loadu_ps(multiplierPtr);
//     __m256 pMultiplier2 = _mm256_loadu_ps(multiplierPtr + 8);

//     for(Rpp32u i = 0; i < outerDim; i++)
//     {
//         Rpp32f *srcPtrTemp = srcPtr + i * srcGenericDescPtr->strides[1];
//         Rpp32f *dstPtrTemp = dstPtr + i * dstGenericDescPtr->strides[1];

//         Rpp32u vectorLoopCount = 0;
//         for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
//         {
//             __m256 pSrc1 = _mm256_loadu_ps(srcPtrTemp);
//             __m256 pSrc2 = _mm256_loadu_ps(srcPtrTemp + 8);
//             __m256 pDst1 = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc1, pMean1), pMultiplier1), pShift);
//             __m256 pDst2 = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc2, pMean2), pMultiplier2), pShift);
//             _mm256_storeu_ps(dstPtrTemp, pDst1);
//             _mm256_storeu_ps(dstPtrTemp + 8, pDst2);
//             srcPtrTemp += vectorIncrement;
//             dstPtrTemp += vectorIncrement;
//         }
//     }
// }

// Computes normalize for ND non toggle variants for i8 dataype
template<typename T1, typename T2>
void concat_ND_tensor(T1 *srcPtr, T1 *srcPtr1, Rpp32u *srcStride, T2 *dstPtr, Rpp32u *length, Rpp32u tensorDim, Rpp32u level, Rpp32u axisMask, Rpp32u maxDims)
{
    if(level >= axisMask)
    {
        int size = 1;
        for(int i = level; i < tensorDim; i++)
        {
            size = size * length[i];
        }
        printf("\n Size : %d",size);
        T1 *srcPtr1 = srcPtr;
        for(int j=0; j < size; j++)
        {
            *(dstPtr + j) = *srcPtr;
            *(dstPtr + size + j) = *srcPtr1;
            srcPtr++;
            srcPtr1++;
        }

    }
    else
    {
        for (Rpp32u i = 0; i < length[level]; i++)
        {
            concat_ND_tensor(srcPtr, srcPtr1, srcStride, dstPtr, length, tensorDim, level + 1, axisMask, maxDims);
            dstPtr += srcStride[level + 1] * 2; 
            srcPtr += srcStride[level + 1];
        }
    }
}

// // Computes normalize for ND non toggle variants
// template<typename T1, typename T2>
// void normalize_ND_tensor_nontoggle(T1 *srcPtr, Rpp32u *srcStride, T2 *dstPtr, Rpp32f *meanPtr, Rpp32f *multiplierPtr,
//                                    Rpp32f shift, Rpp32u *paramStride, Rpp32u *length, Rpp32u tensorDim, Rpp32u level, Rpp32u& idx)
// {
//     Rpp32u idx1 = 0;
//     if(tensorDim == 1)
//     {
//         T1 *srcPtrTemp = srcPtr;
//         T2 *dstPtrTemp = dstPtr;

//         for(Rpp32u k = 0; k < length[level]; k++)
//         {
//             *dstPtrTemp = (((T2)*srcPtrTemp - meanPtr[idx]) * multiplierPtr[idx]) + shift;
//             if(k < length[level] - 1)
//                 idx += paramStride[level];
//             srcPtrTemp++;
//             dstPtrTemp++;
//         }
//     }
//     else
//     {
//         idx1 = idx;
//         for (Rpp32u i = 0; i < length[level]; i++)
//         {
//             normalize_ND_tensor_nontoggle(srcPtr, srcStride, dstPtr, meanPtr, multiplierPtr, shift, paramStride, length, tensorDim - 1, level + 1, idx);
//             if(i < length[level] - 1)
//                 idx = (!paramStride[level]) ? idx1 : idx + paramStride[level];
//             dstPtr += srcStride[level];
//             srcPtr += srcStride[level];
//         }
//     }
// }

// Computes normalize for 2D
void concat_2D_tensor(Rpp32f *srcPtr, Rpp32f *srcPtr1, RpptGenericDescPtr srcDescPtr, Rpp32f *dstPtr, RpptGenericDescPtr dstDescPtr, Rpp32u *dims, Rpp32u axisMask)
{
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = dims[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;

    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        Rpp32f *srcPtrTemp = srcPtr + i * srcDescPtr->strides[1];
        Rpp32f *srcPtrTemp1 = srcPtr1 + i * srcDescPtr->strides[1];
        Rpp32f *dstPtrTemp = dstPtr + (i * dstDescPtr->strides[1] * 2);

        Rpp32u vectorLoopCount = 0;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
        {
            __m256 pDst = _mm256_loadu_ps(srcPtrTemp);
            _mm256_storeu_ps(dstPtrTemp, pDst);
            srcPtrTemp += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        for(; vectorLoopCount < dims[1] ; vectorLoopCount ++)
            *dstPtrTemp++ = *srcPtrTemp++;
        vectorLoopCount = 0;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += vectorIncrement)
        {
            __m256 pDst = _mm256_loadu_ps(srcPtrTemp1);
            _mm256_storeu_ps(dstPtrTemp, pDst);
            srcPtrTemp1 += vectorIncrement;
            dstPtrTemp += vectorIncrement;
        }
        for(; vectorLoopCount < dims[1] ; vectorLoopCount ++)
            *dstPtrTemp++ = *srcPtrTemp1++;
    }
}

// // Performs collapse axis operation wherein continuous axis that require normalization are combined together
// void collapse_axis(Rpp32u *tensorDim, Rpp32u *axis, Rpp32u *length, Rpp32u *newAxis, Rpp32u *newDims, Rpp32u *lastNormAxis)
// {
//     int skipped = 0, prev = -2, k = 0;
//     for(Rpp32u i = 0; i < *tensorDim; i++)
//     {
//         if(axis[i])
//         {
//             int temp = i - skipped;
//             if(temp != prev + 1)
//             {
//                 newAxis[k] = 1;
//                 newDims[k] = length[i];
//                 prev = i;
//                 k++;
//             }
//             else if(prev >= 0)
//             {
//                 newDims[prev] *= length[i];
//                 skipped++;
//             }
//         }
//         else
//         {
//             newDims[k] = length[i];
//             k++;
//         }
//     }
//     *tensorDim -= skipped;
//     for(Rpp32u i = 0; i < *tensorDim; i++)
//     {
//         if(newAxis[i])
//             *lastNormAxis = i;
//     }
// }

RppStatus concat_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     Rpp32f *srcPtr1,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32u *roiTensor,
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
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];

        Rpp32f *srcPtrTemp, *srcPtrTemp1, *dstPtrTemp, *meanTensor, *stdDevTensor;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        srcPtrTemp1 = srcPtr1 + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstGenericDescPtr->strides[0]; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        Rpp32f *srcPtrChannel = srcPtrTemp;

        if(tensorDims == 2) // Called for audio testcase and for any other 2D case
        {
            Rpp32u srcReductionDims[2], srcStride[2];
            if (axisMask == 0)
            {
                srcStride[0] = srcGenericDescPtr->strides[2];
                srcStride[1] = 1;
                srcReductionDims[0] = 1;
                srcReductionDims[1] = length[0] * length[1];
            }
            else if (axisMask == 1)
            {
                srcStride[0] = srcGenericDescPtr->strides[1];
                srcStride[1] = srcGenericDescPtr->strides[2];
                srcReductionDims[0] = length[0];
                srcReductionDims[1] = length[1];
            }
            concat_2D_tensor(srcPtrTemp, srcPtrTemp1, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, length, axisMask);
        }
        else if(tensorDims == 3) // Called when a 3D tensor is passed to kernel
        {
            Rpp32u srcReductionDims[3], srcStride[3];
            Rpp32u reductionDims;
            bool isConsecutive = true;
            switch(axisMask)
            {
                case 0: // Normalize axes 0
                {
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = 1;
                    srcReductionDims[2] = length[0] * length[1] * length[2];
                    srcStride[0] = 1;
                    srcStride[1] = 1;
                    srcStride[2] = srcGenericDescPtr->strides[3];
                    break;
                }
                case 1: // Normalize axes 1
                {
                    srcReductionDims[0] = 1;
                    srcReductionDims[1] = length[0];
                    srcReductionDims[2] = length[1] * length[2];
                    srcStride[0] = srcGenericDescPtr->strides[3];
                    srcStride[1] = srcGenericDescPtr->strides[1];
                    srcStride[2] = srcGenericDescPtr->strides[3];
                    break;
                }
                case 2: // Normalize axes 0, 1
                {
                    srcReductionDims[0] = length[0];
                    srcReductionDims[1] = length[1];
                    srcReductionDims[2] = length[2];
                    srcStride[0] = srcGenericDescPtr->strides[1];
                    srcStride[1] = srcGenericDescPtr->strides[2];
                    srcStride[2] = srcGenericDescPtr->strides[3];
                    break;
                }
                default:
                {
                    std::cout<<"Invalid Axis mask"<<std::endl;
                }
            }

            for(Rpp32u i = 1; i < tensorDims; i++)
                srcPtrChannel += begin[i - 1] * srcGenericDescPtr->strides[i];

            concat_3D_tensor(srcPtrChannel, srcGenericDescPtr, dstPtrTemp, dstGenericDescPtr, length, axisMask);
        }
        // else // Handle any other ND tensor is passed to kernel
        // {
        //     concat_ND_tensor(srcPtrChannel, srcStride, dstPtrTemp, length, axisMask);
        // }
    }

    return RPP_SUCCESS;
}
template<typename T1, typename T2>
RppStatus concat_generic_host_tensor(T1 *srcPtr,
                                     T1 *srcPtr1,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T2 *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u axisMask,
                                        Rpp32u *roiTensor,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u tensorDims = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];
//     omp_set_dynamic(0);
// #pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
	{
        T1 *srcPtrTemp;
        T1 *srcPtrTemp1;
        T2 *dstPtrTemp;
        srcPtrTemp = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        srcPtrTemp1 = srcPtr1 + batchCount * srcGenericDescPtr->strides[0];
        dstPtrTemp = dstPtr + batchCount * dstGenericDescPtr->strides[0] * 2;
        int size = 1;
        Rpp32u *roi = roiTensor + batchCount * tensorDims * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[tensorDims];
        Rpp32u level = 0;
        T1 *srcPtrChannel = srcPtrTemp;
        T1 *srcPtrChannel1 = srcPtrTemp1;
         for(int i = 0; i < tensorDims; i++)
        {
            srcPtrChannel += begin[i] * srcGenericDescPtr->strides[i + 1];
            srcPtrChannel1 += begin[i] * srcGenericDescPtr->strides[i + 1];
        }
        concat_ND_tensor(srcPtrChannel, srcPtrChannel1, dstGenericDescPtr->strides, dstPtrTemp, length, tensorDims, level, axisMask, tensorDims);
    }

    return RPP_SUCCESS;
}