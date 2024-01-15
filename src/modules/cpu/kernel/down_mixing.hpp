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

Rpp32f rpp_hsum256_ps(__m256 x)
{
    __m128 p0 = _mm256_extractf128_ps(x, 1); // Contains x7, x6, x5, x4
    __m128 p1 = _mm256_castps256_ps128(x);   // Contains x3, x2, x1, x0
    __m128 sum = _mm_add_ps(p0, p1);         // Contains x3 + x7, x2 + x6, x1 + x5, x0 + x4
    p0 = sum;                                // Contains -, -, x1 + x5, x0 + x4
    p1 = _mm_movehl_ps(sum, sum);            // Contains -, -, x3 + x7, x2 + x6
    sum = _mm_add_ps(p0, p1);                // Contains -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6
    p0 = sum;                                // Contains -, -, -, x0 + x2 + x4 + x6
    p1 = _mm_shuffle_ps(sum, sum, 0x1);      // Contains -, -, -, x1 + x3 + x5 + x7
    sum = _mm_add_ss(p0, p1);                // Contains -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
    return _mm_cvtss_f32(sum);
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
        bool flag_avx = 0;

        if(channels == 1)
        {
            // No need of downmixing, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(samples * sizeof(Rpp32f)));
        }
        else
        {
            Rpp32f *weights = handle.GetInitHandle()->mem.mcpu.tempFloatmem + batchCount * channels;
            std::fill(weights, weights + channels, 1.f / channels);

            if(normalizeWeights)
            {
                // Compute sum of the weights
                Rpp32f sum = 0.0;
                for(int i = 0; i < channels; i++)
                    sum += weights[i];

                // Normalize the weights
                Rpp32f invSum = 1.0 / sum;
                for(int i = 0; i < channels; i++)
                    weights[i] *= invSum;
            }

            Rpp32s channelIncrement = 4;
            Rpp32s alignedChannels = (channels / 4) * 4;

            if(channels > 6)
            {
                flag_avx = 1;
                channelIncrement = 8;
                alignedChannels = (channels / 8) * 8;
            }

            // use weights to downmix to mono
            for(int64_t dstIdx = 0; dstIdx < samples; dstIdx++)
            {
                Rpp32s channelLoopCount = 0;
                if(flag_avx)
                {
                    __m256 pDst = avx_p0;
                    for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
                    {
                        __m256 pSrc, pWeights;
                        pWeights = _mm256_setr_ps(weights[channelLoopCount], weights[channelLoopCount + 1], weights[channelLoopCount + 2], weights[channelLoopCount + 3],
                                weights[channelLoopCount + 4], weights[channelLoopCount + 5], weights[channelLoopCount + 6], weights[channelLoopCount + 7]);
                        pSrc = _mm256_loadu_ps(srcPtrTemp);
                        pSrc = _mm256_mul_ps(pSrc, pWeights);
                        pDst = _mm256_add_ps(pDst, pSrc);
                        srcPtrTemp += channelIncrement;
                    }
                    dstPtrTemp[dstIdx] = rpp_hsum256_ps(pDst);
                    for(; channelLoopCount < channels; channelLoopCount++)
                        dstPtrTemp[dstIdx] += ((*srcPtrTemp++) * weights[channelLoopCount]);
                }
                else
                {
                    __m128 pDst = xmm_p0;
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
                        dstPtrTemp[dstIdx] += ((*srcPtrTemp++) * weights[channelLoopCount]);
                }
            }
        }
    }

    return RPP_SUCCESS;
}
