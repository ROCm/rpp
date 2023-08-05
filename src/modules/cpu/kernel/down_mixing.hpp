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
#include <omp.h>

Rpp32f rpp_hsum_ps(__m128 x)
{
    __m128 shuf = _mm_movehdup_ps(x);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);        // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  Rpp32s *channelsTensor,
                                  bool normalizeWeights,
                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s channels = channelsTensor[batchCount];
        Rpp32s samples = srcLengthTensor[batchCount];

        if(channels == 1)
        {
            // No need of downmixing, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(samples * sizeof(Rpp32f)));
        }
        else
        {
            std::vector<Rpp32f> weights;
            weights.resize(channels, 1.f / channels);
            std::vector<Rpp32f> normalizedWeights;

            if(normalizeWeights)
            {
                normalizedWeights.resize(channels);

                // Compute sum of the weights
                Rpp32f sum = 0.0;
                for(int i = 0; i < channels; i++)
                    sum += weights[i];

                // Normalize the weights
                Rpp32f invSum = 1.0 / sum;
                for(int i = 0; i < channels; i++)
                    normalizedWeights[i] = weights[i] * invSum;

                weights = normalizedWeights;
            }

            Rpp32s channelIncrement = 4;
            Rpp32s alignedChannels = (channels / 4) * 4;

            // use weights to downmix to mono
            for(int64_t dstIdx = 0; dstIdx < samples; dstIdx++)
            {
                __m128 pDst = _mm_setzero_ps();
                Rpp32s channelLoopCount = 0;
                for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
                {
                    __m128 pSrc, pWeights;
                    pWeights = _mm_setr_ps(weights[channelLoopCount], weights[channelLoopCount + 1], weights[channelLoopCount + 2], weights[channelLoopCount + 3]);
                    pSrc = _mm_loadu_ps(srcPtrTemp);
                    pSrc = _mm_mul_ps(pSrc, pWeights);
                    pDst = _mm_add_ps(pDst, pSrc);
                    srcPtrTemp += channelIncrement;
                }

                dstPtrTemp[dstIdx] = rpp_hsum_ps(pDst);
                for(; channelLoopCount < channels; channelLoopCount++)
                {
                    dstPtrTemp[dstIdx] += ((*srcPtrTemp) * weights[channelLoopCount]);
                    srcPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}
