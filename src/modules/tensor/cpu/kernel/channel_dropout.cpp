/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "host_tensor_executors.hpp"
#include <random>

inline void dropout_mask_apply_avx(__m256 *p, uint8_t maskValR, uint8_t maskValG, uint8_t maskValB)
{
    __m256 maskR = _mm256_set1_ps((float)maskValR);
    __m256 maskG = _mm256_set1_ps((float)maskValG);
    __m256 maskB = _mm256_set1_ps((float)maskValB);

    // R channel → p[0], p[1]
    p[0] = _mm256_mul_ps(p[0], maskR);
    p[1] = _mm256_mul_ps(p[1], maskR);

    // G channel → p[2], p[3]
    p[2] = _mm256_mul_ps(p[2], maskG);
    p[3] = _mm256_mul_ps(p[3], maskG);

    // B channel → p[4], p[5]
    p[4] = _mm256_mul_ps(p[4], maskB);
    p[5] = _mm256_mul_ps(p[5], maskB);
}

inline void dropout_mask_apply_sse(__m128 *p, uint8_t maskValR, uint8_t maskValG, uint8_t maskValB)
{
    __m128 maskR = _mm_set1_ps((float)maskValR);
    __m128 maskG = _mm_set1_ps((float)maskValG);
    __m128 maskB = _mm_set1_ps((float)maskValB);

    // R channel
    p[0] = _mm_mul_ps(p[0], maskR);
    p[1] = _mm_mul_ps(p[1], maskR);
    p[2] = _mm_mul_ps(p[2], maskR);
    p[3] = _mm_mul_ps(p[3], maskR);

    // G channel
    p[4] = _mm_mul_ps(p[4], maskG);
    p[5] = _mm_mul_ps(p[5], maskG);
    p[6] = _mm_mul_ps(p[6], maskG);
    p[7] = _mm_mul_ps(p[7], maskG);

    // B channel
    p[8]  = _mm_mul_ps(p[8],  maskB);
    p[9]  = _mm_mul_ps(p[9],  maskB);
    p[10] = _mm_mul_ps(p[10], maskB);
    p[11] = _mm_mul_ps(p[11], maskB);
}

inline void generate_channel_masks(uint8_t *channelMasks,
                                   Rpp32f *dropProb,
                                   Rpp32u batchSize,
                                   Rpp32u numChannels)
{
    std::mt19937 rng(42); // fixed seed, or std::random_device{}()
    for (int b = 0; b < batchSize; b++)
    {
        std::bernoulli_distribution keepDist(1.0f - dropProb[b]);
        bool anyKept = false;
        int base = b * numChannels;
        for (Rpp32u c = 0; c < numChannels; c++)
        {
            channelMasks[base + c] = keepDist(rng);
            anyKept |= channelMasks[base + c];
        }

        if (!anyKept)
            channelMasks[base + (rng() % numChannels)] = 1;
    }
}

RppStatus channel_dropout_u8_u8_host_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            Rpp32f *dropProb,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams layoutParams,
                                            rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Generate channel mask for this batch
    uint8_t *channelMaskHost = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    generate_channel_masks(channelMaskHost, dropProb, dstDescPtr->n, srcDescPtr->c);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            uint8_t maskValR, maskValG, maskValB;
            maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    dropout_mask_apply_avx(p, maskValR, maskValG, maskValB);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    dropout_mask_apply_sse(p, maskValR, maskValG, maskValB);       // apply dropout masks
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = maskValR * srcPtrTemp[0];
                    *dstPtrTempG = maskValG * srcPtrTemp[1];
                    *dstPtrTempB = maskValB * srcPtrTemp[2];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR, maskValG, maskValB;
            maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    dropout_mask_apply_avx(p, maskValR, maskValG, maskValB);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    dropout_mask_apply_sse(p, maskValR, maskValG, maskValB);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR * *srcPtrTempR;
                    dstPtrTemp[1] = maskValG * *srcPtrTempG;
                    dstPtrTemp[2] = maskValB * *srcPtrTempB;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR, maskValG, maskValB;
            maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    dropout_mask_apply_avx(p, maskValR, maskValG, maskValB);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    dropout_mask_apply_sse(p, maskValR, maskValG, maskValB);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR * srcPtrTemp[0];
                    dstPtrTemp[1] = maskValG * srcPtrTemp[1];
                    dstPtrTemp[2] = maskValB * srcPtrTemp[2];

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                uint8_t maskVal = channelMaskHost[batchCount * srcDescPtr->c + c];
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
#if __AVX2__
                        __m256 p[2], mask;
                        mask = _mm256_set1_ps(maskVal);
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads                        
                        p[0] = _mm256_mul_ps(p[0], mask);
                        p[1] = _mm256_mul_ps(p[1], mask);
                        rpp_simd_store(rpp_store16_f32_to_u8_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[4], mask;
                        mask = _mm_set1_ps(maskVal);
                        rpp_simd_load(rpp_load16_u8_to_f32, srcPtrTemp, p);    // simd loads
                        p[0] = _mm_mul_ps(p[0], mask);
                        p[1] = _mm_mul_ps(p[1], mask);
                        p[2] = _mm_mul_ps(p[2], mask);
                        p[3] = _mm_mul_ps(p[3], mask);
                        rpp_simd_store(rpp_store16_f32_to_u8, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = maskVal * *srcPtrTemp;
                        
                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}


RppStatus channel_dropout_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32f *dropProb,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Generate channel mask for this batch
    uint8_t *channelMaskHost = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    generate_channel_masks(channelMaskHost, dropProb, dstDescPtr->n, srcDescPtr->c);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
#endif
        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if(!maskValR)
                        p[0] = avx_p0;
                    if(!maskValG)
                        p[1] = avx_p0;
                    if(!maskValB)
                        p[2] = avx_p0;
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if(!maskValR)
                        p[0] = xmm_p0;
                    if(!maskValG)
                        p[1] = xmm_p0;
                    if(!maskValB)
                        p[2] = xmm_p0;
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = maskValR ? srcPtrTemp[0] : 0.0f;
                    *dstPtrTempG = maskValG ? srcPtrTemp[1] : 0.0f;
                    *dstPtrTempB = maskValB ? srcPtrTemp[2] : 0.0f;

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if(!maskValR)
                        p[0] = avx_p0;
                    if(!maskValG)
                        p[1] = avx_p0;
                    if(!maskValB)
                        p[2] = avx_p0;
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if(!maskValR)
                        p[0] = xmm_p0;
                    if(!maskValG)
                        p[1] = xmm_p0;
                    if(!maskValB)
                        p[2] = xmm_p0;
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR ? *srcPtrTempR : 0.0f;
                    dstPtrTemp[1] = maskValG ? *srcPtrTempG : 0.0f;
                    dstPtrTemp[2] = maskValB ? *srcPtrTempB : 0.0f;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);    // simd loads
                    if(!maskValR)
                        p[0] = avx_p0;
                    if(!maskValG)
                        p[1] = avx_p0;
                    if(!maskValB)
                        p[2] = avx_p0;
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if(!maskValR)
                        p[0] = xmm_p0;
                    if(!maskValG)
                        p[1] = xmm_p0;
                    if(!maskValB)
                        p[2] = xmm_p0;
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR ? srcPtrTemp[0] : 0.0f;
                    dstPtrTemp[1] = maskValG ? srcPtrTemp[1] : 0.0f;
                    dstPtrTemp[2] = maskValB ? srcPtrTemp[2] : 0.0f;

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness without fused output-layout toggle (NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                uint8_t maskVal = channelMaskHost[batchCount * srcDescPtr->c + c];
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
#if __AVX2__
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        if(!maskVal)
                            p[0] = avx_p0;
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp, p);    // simd loads
                        if(!maskVal)
                            p[0] = xmm_p0;
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = maskVal ? *srcPtrTemp : 0.0f;

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus channel_dropout_f16_f16_host_tensor(Rpp16f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp16f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32f *dropProb,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Generate channel mask for this batch
    uint8_t *channelMaskHost = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    generate_channel_masks(channelMaskHost, dropProb, dstDescPtr->n, srcDescPtr->c);
    
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
#else
        Rpp32u alignedLength = (bufferLength / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;
#endif

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            
            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    Rpp32f dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];

                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];

#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    if(!maskValR)
                        p[0] = avx_p0;
                    if(!maskValG)
                        p[1] = avx_p0;
                    if(!maskValB)
                        p[2] = avx_p0;
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    if(!maskValR)
                        p[0] = xmm_p0;
                    if(!maskValG)
                        p[1] = xmm_p0;
                    if(!maskValB)
                        p[2] = xmm_p0;
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);    // simd stores
#endif

                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        dstPtrTempR[cnt] = (Rpp16f) dstPtrTempR_ps[cnt];
                        dstPtrTempG[cnt] = (Rpp16f) dstPtrTempG_ps[cnt];
                        dstPtrTempB[cnt] = (Rpp16f) dstPtrTempB_ps[cnt];
                    }

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = maskValR ? srcPtrTemp[0] : 0.0f;
                    *dstPtrTempG = maskValG ? srcPtrTemp[1] : 0.0f;
                    *dstPtrTempB = maskValB ? srcPtrTemp[2] : 0.0f;

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            
            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    if(!maskValR)
                        p[0] = avx_p0;
                    if(!maskValG)
                        p[1] = avx_p0;
                    if(!maskValB)
                        p[2] = avx_p0;
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);    // simd loads
                    if(!maskValR)
                        p[0] = xmm_p0;
                    if(!maskValG)
                        p[1] = xmm_p0;
                    if(!maskValB)
                        p[2] = xmm_p0;
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR ? *srcPtrTempR : 0.0f;
                    dstPtrTemp[1] = maskValG ? *srcPtrTempG : 0.0f;
                    dstPtrTemp[2] = maskValB ? *srcPtrTempB : 0.0f;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[25];
                    Rpp32f dstPtrTemp_ps[25];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);    // simd loads
                    if(!maskValR)
                        p[0] = avx_p0;
                    if(!maskValG)
                        p[1] = avx_p0;
                    if(!maskValB)
                        p[2] = avx_p0;
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);    // simd stores
#else
                    __m128 p[3];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    if(!maskValR)
                        p[0] = xmm_p0;
                    if(!maskValG)
                        p[1] = xmm_p0;
                    if(!maskValB)
                        p[2] = xmm_p0;
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores
#endif
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR ? srcPtrTemp[0] : 0.0f;
                    dstPtrTemp[1] = maskValG ? srcPtrTemp[1] : 0.0f;
                    dstPtrTemp[2] = maskValB ? srcPtrTemp[2] : 0.0f;

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        
        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                uint8_t maskVal = channelMaskHost[batchCount * srcDescPtr->c + c];
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            srcPtrTemp_ps[cnt] = (Rpp16f) srcPtrTemp[cnt];
#if __AVX2__
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp_ps, p);    // simd loads
                        if(!maskVal)
                            p[0] = avx_p0;
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp_ps, p);    // simd stores
#else
                        __m128 p[1];

                        rpp_simd_load(rpp_load4_f32_to_f32, srcPtrTemp_ps, p);    // simd loads
                        if(!maskVal)
                            p[0] = xmm_p0;
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTemp_ps, p);    // simd stores
#endif

                        for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                            dstPtrTemp[cnt] = (Rpp16f) dstPtrTemp_ps[cnt];
                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = maskVal ? *srcPtrTemp : 0.0f;

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus channel_dropout_i8_i8_host_tensor(Rpp8s *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8s *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            Rpp32f *dropProb,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams layoutParams,
                                            rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    // Generate channel mask for this batch
    uint8_t *channelMaskHost = reinterpret_cast<uint8_t *>(handle.GetInitHandle()->mem.mcpu.scratchBufferHost);
    generate_channel_masks(channelMaskHost, dropProb, dstDescPtr->n, srcDescPtr->c);
    
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Brightness with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    if(!maskValR)
                    {    
                        p[0] = avx_p0;
                        p[1] = avx_p0;
                    }
                    if(!maskValG)
                    {    
                        p[2] = avx_p0;
                        p[3] = avx_p0;
                    }
                    if(!maskValB)
                    {    
                        p[4] = avx_p0;
                        p[5] = avx_p0;
                    }
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if(!maskValR)
                    {    
                        p[0] = xmm_p0;
                        p[1] = xmm_p0;
                        p[2] = xmm_p0;
                        p[3] = xmm_p0;
                    }
                    if(!maskValG)
                    {    
                        p[4] = xmm_p0;
                        p[5] = xmm_p0;
                        p[6] = xmm_p0;
                        p[7] = xmm_p0;
                    }
                    if(!maskValB)
                    {    
                        p[8] = xmm_p0;
                        p[9] = xmm_p0;
                        p[10] = xmm_p0;
                        p[11] = xmm_p0;
                    }
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores
#endif

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = maskValR ? srcPtrTemp[0] : -128;
                    *dstPtrTempG = maskValG ? srcPtrTemp[1] : -128;
                    *dstPtrTempB = maskValB ? srcPtrTemp[2] : -128;

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];
            
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if(!maskValR)
                    {    
                        p[0] = avx_p0;
                        p[1] = avx_p0;
                    }
                    if(!maskValG)
                    {    
                        p[2] = avx_p0;
                        p[3] = avx_p0;
                    }
                    if(!maskValB)
                    {    
                        p[4] = avx_p0;
                        p[5] = avx_p0;
                    }
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pln3_to_f32pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    if(!maskValR)
                    {    
                        p[0] = xmm_p0;
                        p[1] = xmm_p0;
                        p[2] = xmm_p0;
                        p[3] = xmm_p0;
                    }
                    if(!maskValG)
                    {    
                        p[4] = xmm_p0;
                        p[5] = xmm_p0;
                        p[6] = xmm_p0;
                        p[7] = xmm_p0;
                    }
                    if(!maskValB)
                    {    
                        p[8] = xmm_p0;
                        p[9] = xmm_p0;
                        p[10] = xmm_p0;
                        p[11] = xmm_p0;
                    }
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR ? *srcPtrTempR : -128;
                    dstPtrTemp[1] = maskValG ? *srcPtrTempG : -128;
                    dstPtrTemp[2] = maskValB ? *srcPtrTempB : -128;

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Brightness with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            uint8_t maskValR = channelMaskHost[batchCount * srcDescPtr->c + 0];
            uint8_t maskValG = channelMaskHost[batchCount * srcDescPtr->c + 1];
            uint8_t maskValB = channelMaskHost[batchCount * srcDescPtr->c + 2];

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    if(!maskValR)
                    {    
                        p[0] = avx_p0;
                        p[1] = avx_p0;
                    }
                    if(!maskValG)
                    {    
                        p[2] = avx_p0;
                        p[3] = avx_p0;
                    }
                    if(!maskValB)
                    {    
                        p[4] = avx_p0;
                        p[5] = avx_p0;
                    }
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, dstPtrTemp, p);    // simd stores
#else
                    __m128 p[12];
                    rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3, srcPtrTemp, p);    // simd loads
                    if(!maskValR)
                    {    
                        p[0] = xmm_p0;
                        p[1] = xmm_p0;
                        p[2] = xmm_p0;
                        p[3] = xmm_p0;
                    }
                    if(!maskValG)
                    {    
                        p[4] = xmm_p0;
                        p[5] = xmm_p0;
                        p[6] = xmm_p0;
                        p[7] = xmm_p0;
                    }
                    if(!maskValB)
                    {    
                        p[8] = xmm_p0;
                        p[9] = xmm_p0;
                        p[10] = xmm_p0;
                        p[11] = xmm_p0;
                    }
                    rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3, dstPtrTemp, p);    // simd stores
#endif
                    srcPtrTemp += vectorIncrement;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = maskValR ? srcPtrTemp[0] : -128;
                    dstPtrTemp[1] = maskValG ? srcPtrTemp[1] : -128;
                    dstPtrTemp[2] = maskValB ? srcPtrTemp[2] : -128;

                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        
        // Brightness without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u alignedLength = bufferLength & ~15;

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                uint8_t maskVal = channelMaskHost[batchCount * srcDescPtr->c + c];
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
#if __AVX2__
                        __m256 p[2];

                        rpp_simd_load(rpp_load16_i8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        if(!maskVal)
                        {
                            p[0] = avx_p0;
                            p[1] = avx_p0;
                        }
                        rpp_simd_store(rpp_store16_f32_to_i8_avx, dstPtrTemp, p);    // simd stores
#else
                        __m128 p[4];

                        rpp_simd_load(rpp_load16_i8_to_f32, srcPtrTemp, p);    // simd loads
                        if(!maskVal)
                        {
                            p[0] = xmm_p0;
                            p[1] = xmm_p0;
                            p[2] = xmm_p0;
                            p[3] = xmm_p0;
                        }
                        rpp_simd_store(rpp_store16_f32_to_i8, dstPtrTemp, p);    // simd stores
#endif

                        srcPtrTemp +=16;
                        dstPtrTemp +=16;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = maskVal ? *srcPtrTemp : -128;

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}
