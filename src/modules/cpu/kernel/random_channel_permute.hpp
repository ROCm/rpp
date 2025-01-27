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

RppStatus random_channel_permute_u8_u8_host_tensor(Rpp8u *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp8u *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp8u axisMask,
                                          RppLayoutParams layoutParams,
                                          rpp::Handle& handle)
{
    Rpp8u mapping[][3] = {
        {0, 1, 2}, // axisMask 0 → R, G, B
        {0, 2, 1}, // axisMask 1 → R, B, G
        {1, 0, 2}, // axisMask 2 → G, R, B
        {1, 2, 0}, // axisMask 3 → G, B, R
        {2, 0, 1}, // axisMask 4 → B, R, G
        {2, 1, 0}  // axisMask 5 → B, G, R
    };
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8u *dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < 48; vectorLoopCount+=48)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp, px);    // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTemp[mapping[axisMask][0]], dstPtrTemp[mapping[axisMask][1]], dstPtrTemp[mapping[axisMask][2]], px);    // simd stores with channel swap
                    srcPtrTemp += 48;
                    dstPtrTemp[mapping[axisMask][0]] += 16;
                    dstPtrTemp[mapping[axisMask][1]] += 16;
                    dstPtrTemp[mapping[axisMask][2]] += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTemp[mapping[axisMask][0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[mapping[axisMask][1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[mapping[axisMask][2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp8u *srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], px);    // simd loads with channel swap
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += 16;
                    srcPtrTemp[mapping[axisMask][1]] += 16;
                    srcPtrTemp[mapping[axisMask][2]] += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTemp[mapping[axisMask][0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[mapping[axisMask][1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[mapping[axisMask][2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128i px[3], pxSwap[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp, px);    // simd loads
                    pxSwap[mapping[axisMask][0]] = px[0];     // channel swap
                    pxSwap[mapping[axisMask][1]] = px[1];     // channel swap
                    pxSwap[mapping[axisMask][2]] = px[2];     // channel swap
                    px[0] = pxSwap[0];     // channel swap
                    px[1] = pxSwap[1];     // channel swap
                    px[2] = pxSwap[2];     // channel swap
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    dstPtrTemp[mapping[axisMask][0]] = srcPtrTemp[0];
                    dstPtrTemp[mapping[axisMask][1]] = srcPtrTemp[1];
                    dstPtrTemp[mapping[axisMask][2]] = srcPtrTemp[2];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8u* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], px);    // simd loads with channel swap
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += 16;
                    srcPtrTemp[mapping[axisMask][1]] += 16;
                    srcPtrTemp[mapping[axisMask][2]] += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = *srcPtrTemp[mapping[axisMask][0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[mapping[axisMask][1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[mapping[axisMask][2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus random_channel_permute_f32_f32_host_tensor(Rpp32f *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp32f *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            Rpp8u axisMask,
                                            RppLayoutParams layoutParams,
                                            rpp::Handle& handle)
{
    Rpp8u mapping[][3] = {
        {0, 1, 2}, // axisMask 0 → R, G, B
        {0, 2, 1}, // axisMask 1 → R, B, G
        {1, 0, 2}, // axisMask 2 → G, R, B
        {1, 2, 0}, // axisMask 3 → G, B, R
        {2, 0, 1}, // axisMask 4 → B, R, G
        {2, 1, 0}  // axisMask 5 → B, G, R
    };
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32f *dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTemp[mapping[axisMask][0]], dstPtrTemp[mapping[axisMask][1]], dstPtrTemp[mapping[axisMask][2]], p);    // simd stores with channel swap
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp[mapping[axisMask][0]] += vectorIncrementPerChannel;
                    dstPtrTemp[mapping[axisMask][1]] += vectorIncrementPerChannel;
                    dstPtrTemp[mapping[axisMask][2]] += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp[mapping[axisMask][0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[mapping[axisMask][1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[mapping[axisMask][2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp32f *srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], p);    // simd loads with channel swap
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][1]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][2]] += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTemp[mapping[axisMask][0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[mapping[axisMask][1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[mapping[axisMask][2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3], pSwap[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    pSwap[mapping[axisMask][0]] = p[0];     // channel swap
                    pSwap[mapping[axisMask][1]] = p[1];     // channel swap
                    pSwap[mapping[axisMask][2]] = p[2];     // channel swap
                    p[0] = pSwap[0];     // channel swap
                    p[1] = pSwap[1];     // channel swap
                    p[2] = pSwap[2];     // channel swap
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[mapping[axisMask][0]] = srcPtrTemp[0];
                    dstPtrTemp[mapping[axisMask][1]] = srcPtrTemp[1];
                    dstPtrTemp[mapping[axisMask][2]] = srcPtrTemp[2];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32f* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], p);    // simd loads with channel swap
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][1]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][2]] += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = *srcPtrTemp[mapping[axisMask][0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[mapping[axisMask][1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[mapping[axisMask][2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus random_channel_permute_f16_f16_host_tensor(Rpp16f *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp16f *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            Rpp8u axisMask,
                                            RppLayoutParams layoutParams,
                                            rpp::Handle& handle)
{
    Rpp8u mapping[][3] = {
        {0, 1, 2}, // axisMask 0 → R, G, B
        {0, 2, 1}, // axisMask 1 → R, B, G
        {1, 0, 2}, // axisMask 2 → G, R, B
        {1, 2, 0}, // axisMask 3 → G, B, R
        {2, 0, 1}, // axisMask 4 → B, R, G
        {2, 1, 0}  // axisMask 5 → B, G, R
    };
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp16f *dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTemp[mapping[axisMask][0]], dstPtrTemp[mapping[axisMask][1]], dstPtrTemp[mapping[axisMask][2]], p);    // simd stores with channel swap
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp[mapping[axisMask][0]] += vectorIncrementPerChannel;
                    dstPtrTemp[mapping[axisMask][1]] += vectorIncrementPerChannel;
                    dstPtrTemp[mapping[axisMask][2]] += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTemp[mapping[axisMask][0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[mapping[axisMask][1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[mapping[axisMask][2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp16f *srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], p);    // simd loads with channel swap
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][1]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][2]] += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTemp[mapping[axisMask][0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[mapping[axisMask][1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[mapping[axisMask][2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[3], pSwap[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    pSwap[mapping[axisMask][0]] = p[0];     // channel swap
                    pSwap[mapping[axisMask][1]] = p[1];     // channel swap
                    pSwap[mapping[axisMask][2]] = p[2];     // channel swap
                    p[0] = pSwap[0];     // channel swap
                    p[1] = pSwap[1];     // channel swap
                    p[2] = pSwap[2];     // channel swap
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[mapping[axisMask][0]] = srcPtrTemp[0];
                    dstPtrTemp[mapping[axisMask][1]] = srcPtrTemp[1];
                    dstPtrTemp[mapping[axisMask][2]] = srcPtrTemp[2];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp16f* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], p);    // simd loads with channel swap
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
                    
                    srcPtrTemp[mapping[axisMask][0]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][1]] += vectorIncrementPerChannel;
                    srcPtrTemp[mapping[axisMask][2]] += vectorIncrementPerChannel;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = *srcPtrTemp[mapping[axisMask][0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[mapping[axisMask][1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[mapping[axisMask][2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus random_channel_permute_i8_i8_host_tensor(Rpp8s *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp8s *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp8u axisMask,
                                          RppLayoutParams layoutParams,
                                          rpp::Handle& handle)
{
    Rpp8u mapping[][3] = {
        {0, 1, 2}, // axisMask 0 → R, G, B
        {0, 2, 1}, // axisMask 1 → R, B, G
        {1, 0, 2}, // axisMask 2 → G, R, B
        {1, 2, 0}, // axisMask 3 → G, B, R
        {2, 0, 1}, // axisMask 4 → B, R, G
        {2, 1, 0}  // axisMask 5 → B, G, R
    };
    Rpp32u numThreads = handle.GetNumThreads();
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = srcDescPtr->w * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrImage;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8s *dstPtrTemp[3] = {dstPtrTempR, dstPtrTempG, dstPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp, px);    // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTemp[mapping[axisMask][0]], dstPtrTemp[mapping[axisMask][1]], dstPtrTemp[mapping[axisMask][2]], px);    // simd stores with channel swap
                    srcPtrTemp += 48;
                    dstPtrTemp[mapping[axisMask][0]] += 16;
                    dstPtrTemp[mapping[axisMask][1]] += 16;
                    dstPtrTemp[mapping[axisMask][2]] += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    *dstPtrTemp[mapping[axisMask][0]]++ = srcPtrTemp[0];
                    *dstPtrTemp[mapping[axisMask][1]]++ = srcPtrTemp[1];
                    *dstPtrTemp[mapping[axisMask][2]]++ = srcPtrTemp[2];
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                Rpp8s *srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], px);    // simd loads with channel swap
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += 16;
                    srcPtrTemp[mapping[axisMask][1]] += 16;
                    srcPtrTemp[mapping[axisMask][2]] += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTemp[mapping[axisMask][0]]++;
                    dstPtrTemp[1] = *srcPtrTemp[mapping[axisMask][1]]++;
                    dstPtrTemp[2] = *srcPtrTemp[mapping[axisMask][2]]++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrImage;
            dstPtrRow = dstPtrImage;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=48)
                {
                    __m128i px[3], pxSwap[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp, px);    // simd loads
                    pxSwap[mapping[axisMask][0]] = px[0];     // channel swap
                    pxSwap[mapping[axisMask][1]] = px[1];     // channel swap
                    pxSwap[mapping[axisMask][2]] = px[2];     // channel swap
                    px[0] = pxSwap[0];     // channel swap
                    px[1] = pxSwap[1];     // channel swap
                    px[2] = pxSwap[2];     // channel swap
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTemp += 48;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount+=3)
                {
                    dstPtrTemp[mapping[axisMask][0]] = srcPtrTemp[0];
                    dstPtrTemp[mapping[axisMask][1]] = srcPtrTemp[1];
                    dstPtrTemp[mapping[axisMask][2]] = srcPtrTemp[2];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Swap Channels (RGB<->BGR) without fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRowR = srcPtrImage;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRowR = dstPtrImage;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < srcDescPtr->h; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp8s* srcPtrTemp[3] = {srcPtrTempR, srcPtrTempG, srcPtrTempB};

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTemp[mapping[axisMask][0]], srcPtrTemp[mapping[axisMask][1]], srcPtrTemp[mapping[axisMask][2]], px);    // simd loads with channel swap
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);    // simd stores
                    srcPtrTemp[mapping[axisMask][0]] += 16;
                    srcPtrTemp[mapping[axisMask][1]] += 16;
                    srcPtrTemp[mapping[axisMask][2]] += 16;
                    dstPtrTempR += 16;
                    dstPtrTempG += 16;
                    dstPtrTempB += 16;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = *srcPtrTemp[mapping[axisMask][0]]++;
                    *dstPtrTempG++ = *srcPtrTemp[mapping[axisMask][1]]++;
                    *dstPtrTempB++ = *srcPtrTemp[mapping[axisMask][2]]++;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
