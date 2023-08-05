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

inline Rpp64f Hann(Rpp64f x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

struct ResamplingWindow {
    inline void input_range(Rpp32f x, Rpp32s *loc0, Rpp32s *loc1) {
        Rpp32s xc = ceilf(x);
        *loc0 = xc - lobes;
        *loc1 = xc + lobes;
    }

    inline Rpp32f operator()(Rpp32f x) {
        Rpp32f locRaw = x * scale + center;
        Rpp32s locFloor = floorf(locRaw);
        Rpp32f weight = locRaw - locFloor;
        locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
        Rpp32f current = lookup[locFloor];
        Rpp32f next = lookup[locFloor + 1];
        return current + weight * (next - current);
    }

    inline __m128 operator()(__m128 x) {
        __m128 pLocRaw = _mm_add_ps(_mm_mul_ps(x, pScale), pCenter);
        __m128i pxLocFloor = _mm_cvttps_epi32(pLocRaw);
        __m128 pLocFloor = _mm_cvtepi32_ps(pxLocFloor);
        __m128 pWeight = _mm_sub_ps(pLocRaw, pLocFloor);
        Rpp32s idx[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), pxLocFloor);
        __m128 pCurrent = _mm_setr_ps(lookup[idx[0]], lookup[idx[1]], lookup[idx[2]], lookup[idx[3]]);
        __m128 pNext = _mm_setr_ps(lookup[idx[0] + 1], lookup[idx[1] + 1], lookup[idx[2] + 1], lookup[idx[3] + 1]);

        return _mm_add_ps(pCurrent, _mm_mul_ps(pWeight, _mm_sub_ps(pNext, pCurrent)));
    }

    Rpp32f scale = 1, center = 1;
    Rpp32s lobes = 0, coeffs = 0;
    Rpp32s lookupSize = 0;
    std::vector<Rpp32f> lookup;
    __m128 pCenter, pScale;
};

inline void windowed_sinc(ResamplingWindow &window,
        Rpp32s coeffs, Rpp32s lobes) {
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.clear();
    window.lookup.resize(coeffs + 5);
    window.lookupSize = window.lookup.size();
    Rpp32s center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
        Rpp32f x = (i - center) * scale;
        Rpp32f y = (i - center) * scale_envelope;
        Rpp32f w = sinc(x) * Hann(y);
        window.lookup[i + 1] = w;
    }
    window.center = center + 1;
    window.scale = 1 / scale;
}

RppStatus resample_host_tensor(Rpp32f *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *inRateTensor,
                               Rpp32f *outRateTensor,
                               Rpp32s *srcLengthTensor,
                               Rpp32s *channelTensor,
                               Rpp32f quality,
                               rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    ResamplingWindow window;
    Rpp32s lobes = std::round(0.007 * quality * quality - 0.09 * quality + 3);
    Rpp32s lookupSize = lobes * 64 + 1;
    windowed_sinc(window, lookupSize, lobes);
    window.pCenter = _mm_set1_ps(window.center);
    window.pScale = _mm_set1_ps(window.scale);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f inRate = inRateTensor[batchCount];
        Rpp32f outRate = outRateTensor[batchCount];
        Rpp32s srcLength = srcLengthTensor[batchCount];
        Rpp32s numChannels = channelTensor[batchCount];

        if(outRate == inRate) {
            // No need of Resampling, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcLength * numChannels * sizeof(Rpp32f)));
        } else {
            Rpp32s outBegin = 0;
            Rpp32s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp32s inPos = 0;
            Rpp32s block = 1 << 8;
            Rpp64f scale = (Rpp64f)inRate / outRate;
            Rpp32f fscale = scale;

            if(numChannels == 1) {
                for (int outBlock = outBegin; outBlock < outEnd; outBlock += block) {
                    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp32s inBlockRounded = std::floor(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f * __restrict__ inBlockPtr = srcPtrTemp + inBlockRounded;

                    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                        Rpp32s loc0, loc1;
                        window.input_range(inPos, &loc0, &loc1);
                        if (loc0 + inBlockRounded < 0)
                            loc0 = -inBlockRounded;
                        if (loc1 + inBlockRounded > srcLength)
                            loc1 = srcLength - inBlockRounded;
                        Rpp32f accum = 0.0f;
                        Rpp32s locInWindow = loc0;

                        __m128 pAccum = xmm_p0;
                        __m128 pLocInWindow = _mm_setr_ps(locInWindow - inPos, locInWindow + 1 - inPos, locInWindow + 2 - inPos, locInWindow + 3 - inPos);
                        for (; locInWindow + 3 < loc1; locInWindow += 4) {
                            __m128 w4 = window(pLocInWindow);

                            pAccum = _mm_add_ps(pAccum, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + locInWindow), w4));
                            pLocInWindow = _mm_add_ps(pLocInWindow, xmm_p4);
                        }

                        pAccum = _mm_add_ps(pAccum, _mm_shuffle_ps(pAccum, pAccum, _MM_SHUFFLE(1, 0, 3, 2)));
                        pAccum = _mm_add_ps(pAccum, _mm_shuffle_ps(pAccum, pAccum, _MM_SHUFFLE(0, 1, 0, 1)));
                        accum = _mm_cvtss_f32(pAccum);

                        Rpp32f x = locInWindow - inPos;
                        for (; locInWindow < loc1; locInWindow++, x++) {
                            Rpp32f w = window(x);
                            accum += inBlockPtr[locInWindow] * w;
                        }

                        dstPtrTemp[outPos] = accum;
                    }
                }
            }
            else {
                std::vector<Rpp32f> tmp;
                tmp.resize(numChannels);
                for (int outBlock = outBegin; outBlock < outEnd; outBlock += block) {
                    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp32s inBlockRounded = std::floor(inBlockRaw);

                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f * __restrict__ inBlockPtr = srcPtrTemp + inBlockRounded * numChannels;
                    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                        Rpp32s loc0, loc1;
                        window.input_range(inPos, &loc0, &loc1);
                        if (loc0 + inBlockRounded < 0)
                            loc0 = -inBlockRounded;
                        if (loc1 + inBlockRounded > srcLength)
                            loc1 = srcLength - inBlockRounded;

                        for (int c = 0; c < numChannels; c++)
                            tmp[c] = 0;

                        Rpp32f locInWindow = loc0 - inPos;
                        Rpp32s ofs0 = loc0 * numChannels;
                        Rpp32s ofs1 = loc1 * numChannels;

                        for (int inOfs = ofs0; inOfs < ofs1; inOfs += numChannels, locInWindow++) {
                            Rpp32f w = window(locInWindow);
                            for (int c = 0; c < numChannels; c++)
                                tmp[c] += inBlockPtr[inOfs + c] * w;
                        }

                        Rpp32s dstLoc = outPos * numChannels;
                        for (int c = 0; c < numChannels; c++)
                            dstPtrTemp[dstLoc + c] = tmp[c];
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}