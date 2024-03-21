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
#include "rpp_cpu_common.hpp"

// 1 pixel log helper functions
inline void compute_log(Rpp8u *src, Rpp32f *dst) { *dst = (!*src) ? std::log(std::nextafter(0.0f, 1.0f)) : std::log(*src); }
inline void compute_log(Rpp8s *src, Rpp32f *dst) { *dst = (!*src) ? std::log(std::nextafter(0.0f, 1.0f)) : std::log(abs(*src + 128)); }
inline void compute_log(Rpp16f *src, Rpp16f *dst) { *dst = (!*src) ? log(std::nextafter(0.0f, 1.0f)) : log(abs(*src)); }
inline void compute_log(Rpp32f *src, Rpp32f *dst) { *dst = (!*src) ? std::log(std::nextafter(0.0f, 1.0f)) : std::log(abs(*src)); }

// Computes ND log recursively
template<typename T1, typename T2>
void log_recursive(T1 *src, Rpp32u *srcStrides, T2 *dst, Rpp32u *dstStrides, Rpp32u *dstShape, Rpp32u nDim)
{
    if (!nDim)
    {
        compute_log(src, dst);
    }
    else
    {
        for (int i = 0; i < *dstShape; i++)
        {
            log_recursive(src, srcStrides + 1, dst, dstStrides + 1, dstShape + 1, nDim - 1);
            dst += *dstStrides;
            src += *srcStrides;
        }
    }
}

template<typename T1, typename T2>
RppStatus log_generic_host_tensor(T1 *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  T2 *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u nDim = srcGenericDescPtr->numDims - 1; // Omitting batchSize here to get tensor dimension.
    Rpp32u batchSize = dstGenericDescPtr->dims[0];

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < batchSize; batchCount++)
    {
        Rpp32u *roi = roiTensor + batchCount * nDim * 2;
        Rpp32u *begin = roi;
        Rpp32u *length = &roi[nDim];

        T1 *srcPtr1 = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        T2 *dstPtr1 = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        for(int i = 0; i < nDim; i++)
            srcPtr1 += begin[i] * srcGenericDescPtr->strides[i + 1];
        if(nDim == 2)
        {
            T1 *srcPtrRow = srcPtr1;
            T2 *dstPtrRow = dstPtr1;

            for(int i = 0; i < length[0]; i++)
            {
                T1 *srcPtrTemp = srcPtrRow;
                T2 *dstPtrTemp = dstPtrRow;

                for (int vectorLoopCount = 0; vectorLoopCount < length[1]; vectorLoopCount++)
                {
                    compute_log(srcPtrTemp, dstPtrTemp);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrRow += srcGenericDescPtr->strides[1];
                dstPtrRow += dstGenericDescPtr->strides[1];
            }
        }
        else if(nDim == 3)
        {
            T1 *srcPtrDepth = srcPtr1;
            T2 *dstPtrDepth = dstPtr1;

            for(int i = 0; i < length[0]; i++)
            {
                T1 *srcPtrRow = srcPtrDepth;
                T2 *dstPtrRow = dstPtrDepth;

                for(int j = 0; j < length[1]; j++)
                {
                    T1 *srcPtrTemp = srcPtrRow;
                    T2 *dstPtrTemp = dstPtrRow;

                    for (int vectorLoopCount = 0; vectorLoopCount < length[2]; vectorLoopCount++)
                    {
                        compute_log(srcPtrTemp, dstPtrTemp);
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtrDepth += srcGenericDescPtr->strides[1];
                dstPtrDepth += dstGenericDescPtr->strides[1];
            }
        }
        else
            log_recursive(srcPtr1, srcGenericDescPtr->strides, dstPtr1, dstGenericDescPtr->strides, length, nDim);
    }

    return RPP_SUCCESS;
}