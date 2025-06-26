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

inline void compute_solarize_8_host(__m256 *p, __m256 pThresholdParam)
{
    __m256 pMask = _mm256_cmp_ps(p[0], pThresholdParam, _CMP_GE_OQ);
    p[0] = _mm256_blendv_ps(p[0], _mm256_sub_ps(avx_p1, p[0]), pMask);
}

inline void compute_solarize_24_host(__m256 *p, __m256 pThresholdParam)
{
    __m256 pMask[3];
    pMask[0] = _mm256_cmp_ps(p[0], pThresholdParam, _CMP_GE_OQ);
    pMask[1] = _mm256_cmp_ps(p[1], pThresholdParam, _CMP_GE_OQ);
    pMask[2] = _mm256_cmp_ps(p[2], pThresholdParam, _CMP_GE_OQ);
    p[0] = _mm256_blendv_ps(p[0], _mm256_sub_ps(avx_p1, p[0]), pMask[0]);
    p[1] = _mm256_blendv_ps(p[1], _mm256_sub_ps(avx_p1, p[1]), pMask[1]);
    p[2] = _mm256_blendv_ps(p[2], _mm256_sub_ps(avx_p1, p[2]), pMask[2]);
}

inline void compute_solarize_32_i8_host(__m256i *px, __m256i pxThresholdParam)
{
    __m256i pxMax = _mm256_set1_epi8((char)127);
    __m256i pxMaskTemp, pxMask;
    pxMaskTemp = _mm256_cmpgt_epi8(px[0], pxThresholdParam);
    pxMask = _mm256_or_si256(pxMaskTemp, _mm256_cmpeq_epi8(px[0], pxThresholdParam));
    px[0] = _mm256_blendv_epi8(px[0], _mm256_sub_epi8(pxMax, px[0]), pxMask);
}

inline void compute_solarize_96_i8_host(__m256i *px, __m256i pxThresholdParam)
{
    __m256i pxMaskTemp[3], pxMask[3];
    __m256i pxMax = _mm256_set1_epi8((char)127);

    pxMaskTemp[0] = _mm256_cmpgt_epi8(px[0], pxThresholdParam);
    pxMaskTemp[1] = _mm256_cmpgt_epi8(px[1], pxThresholdParam);
    pxMaskTemp[2] = _mm256_cmpgt_epi8(px[2], pxThresholdParam);

    pxMask[0] = _mm256_or_si256(pxMaskTemp[0], _mm256_cmpeq_epi8(px[0], pxThresholdParam));
    pxMask[1] = _mm256_or_si256(pxMaskTemp[1], _mm256_cmpeq_epi8(px[1], pxThresholdParam));
    pxMask[2] = _mm256_or_si256(pxMaskTemp[2], _mm256_cmpeq_epi8(px[2], pxThresholdParam));

    px[0] = _mm256_blendv_epi8(px[0], _mm256_sub_epi8(pxMax, px[0]), pxMask[0]);
    px[1] = _mm256_blendv_epi8(px[1], _mm256_sub_epi8(pxMax, px[1]), pxMask[1]);
    px[2] = _mm256_blendv_epi8(px[2], _mm256_sub_epi8(pxMax, px[2]), pxMask[2]);
}

inline void compute_solarize_32_host(__m256i *px, __m256i pxThresholdParam)
{
    __m256i pxMaskTemp, pxMask;
    __m256i pxConverted = _mm256_add_epi8(px[0], avx_pxConvertI8);
    pxMaskTemp = _mm256_cmpgt_epi8(pxConverted, pxThresholdParam);
    pxMask = _mm256_or_si256(pxMaskTemp, _mm256_cmpeq_epi8(pxConverted, pxThresholdParam));
    px[0] = _mm256_blendv_epi8(px[0], _mm256_sub_epi8(avx_pxChar255, px[0]), pxMask);
}

inline void compute_solarize_96_host(__m256i *p, __m256i pxThresholdParam)
{
    __m256i pxMaskTemp[3], pxMask[3], pxConverted[3];
    pxConverted[0] = _mm256_add_epi8(p[0], avx_pxConvertI8);
    pxConverted[1] = _mm256_add_epi8(p[1], avx_pxConvertI8);
    pxConverted[2] = _mm256_add_epi8(p[2], avx_pxConvertI8);

    pxMaskTemp[0] = _mm256_cmpgt_epi8(pxConverted[0], pxThresholdParam);
    pxMaskTemp[1] = _mm256_cmpgt_epi8(pxConverted[1], pxThresholdParam);
    pxMaskTemp[2] = _mm256_cmpgt_epi8(pxConverted[2], pxThresholdParam);

    pxMask[0] = _mm256_or_si256(pxMaskTemp[0], _mm256_cmpeq_epi8(pxConverted[0], pxThresholdParam));
    pxMask[1] = _mm256_or_si256(pxMaskTemp[1], _mm256_cmpeq_epi8(pxConverted[1], pxThresholdParam));
    pxMask[2] = _mm256_or_si256(pxMaskTemp[2], _mm256_cmpeq_epi8(pxConverted[2], pxThresholdParam));

    p[0] = _mm256_blendv_epi8(p[0], _mm256_sub_epi8(avx_pxChar255, p[0]), pxMask[0]);
    p[1] = _mm256_blendv_epi8(p[1], _mm256_sub_epi8(avx_pxChar255, p[1]), pxMask[1]);
    p[2] = _mm256_blendv_epi8(p[2], _mm256_sub_epi8(avx_pxChar255, p[2]), pxMask[2]);
}

RppStatus solarize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *thresholdTensor,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

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

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        Rpp8u thresholdParam = static_cast<Rpp8u>(std::round(thresholdTensor[batchCount] * 255));
#if __AVX2__
        // Create a 256-bit vector where each byte is (thresholdParam + 128), 
        // effectively shifting the unsigned [0–255] threshold into the signed range [–128–127] 
        // so that subsequent signed-byte comparisons (_mm256_cmpgt_epi8/_mm256_cmpeq_epi8) 
        // will correctly emulate an unsigned ≥ threshold test.
        __m256i pxThresholdParam = _mm256_set1_epi8(thresholdParam + 128);
#endif
        // Solarize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                // for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                // {
                //     __m256i p[3];
                //     rpp_simd_load(rpp_load96_u8pkd3_to_u8pln3, srcPtrTemp, p);                               // simd loads
                //     compute_solarize_96_host(p, pxThresholdParam);                                           // threshold adjustment
                //     rpp_simd_store(rpp_store96_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                //     srcPtrTemp += vectorIncrement;
                //     dstPtrTempR += vectorIncrementPerChannel;
                //     dstPtrTempG += vectorIncrementPerChannel;
                //     dstPtrTempB += vectorIncrementPerChannel;
                // }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (255 - *(srcPtrTemp)) : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam) ? (255 - *(srcPtrTemp + 1)) : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam) ? (255 - *(srcPtrTemp + 2)) : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
        // Threshold with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                // for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                // {
                //     __m256i p[3];
                //     rpp_simd_load(rpp_load96_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                //     compute_solarize_96_host(p, pxThresholdParam); 	                                        // threshold adjustment
                //     rpp_simd_store(rpp_store96_u8pln3_to_u8pkd3, dstPtrTemp, p);                           // simd stores
                //     srcPtrTempR += 32;
                //     srcPtrTempG += 32;
                //     srcPtrTempB += 32;
                //     dstPtrTemp += 96;
                // }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (*srcPtrTempR >= thresholdParam) ? (255 - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ = (*srcPtrTempG >= thresholdParam) ? (255 - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ = (*srcPtrTempB >= thresholdParam) ? (255 - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else
        {
            Rpp32u alignedLength = bufferLength & ~31;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
//                     for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
//                     {
// #if __AVX2__
//                         __m256i p;
//                         p = _mm256_loadu_si256((__m256i *)srcPtrTemp);
//                         compute_solarize_32_host(&p, pxThresholdParam);
//                         _mm256_storeu_si256((__m256i *)dstPtrTemp, p);
// #endif
//                         srcPtrTemp +=32;
//                         dstPtrTemp +=32;
//                     }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (255 - (*srcPtrTemp)) : *srcPtrTemp;
                        srcPtrTemp++;
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

RppStatus solarize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *thresholdTensor,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

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

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32f thresholdParam = thresholdTensor[batchCount];

#if __AVX2__
        __m256 pThresholdParam = _mm256_set1_ps(thresholdParam);
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p);                               // simd loads
                    compute_solarize_24_host(p, pThresholdParam); 	                                           // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                    srcPtrTemp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (1.0f - *(srcPtrTemp)) : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam) ? (1.0f - *(srcPtrTemp + 1)) : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam) ? (1.0f - *(srcPtrTemp + 2)) : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_solarize_24_host(p, pThresholdParam); 	                                         // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);                           // simd stores
                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (*srcPtrTempR >= thresholdParam) ? (1.0f - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ = (*srcPtrTempG >= thresholdParam) ? (1.0f - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ = (*srcPtrTempB >= thresholdParam) ? (1.0f - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        else
        {
            Rpp32u alignedLength = bufferLength & ~7;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
#if __AVX2__
                        __m256 p;
                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, &p);      // simd loads
                        compute_solarize_8_host(&p, pThresholdParam);              // threshold adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, &p);    // simd stores
#endif
                        srcPtrTemp += 8;
                        dstPtrTemp += 8;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (1.0f - (*srcPtrTemp)) : *srcPtrTemp;
                        srcPtrTemp++;
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

RppStatus solarize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *thresholdTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

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

        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;

        Rpp8s thresholdParam = static_cast<Rpp8s>(std::round(thresholdTensor[batchCount] * 255)) - 128;
#if __AVX2__
        // Create a 256-bit vector where each byte is (thresholdParam + 128), 
        // effectively shifting the unsigned [0–255] threshold into the signed range [–128–127] 
        // so that subsequent signed-byte comparisons (_mm256_cmpgt_epi8/_mm256_cmpeq_epi8) 
        // will correctly emulate an unsigned ≥ threshold test.
        __m256i pxThresholdParam = _mm256_set1_epi8(thresholdParam);
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                {
                    __m256i px[3];
                    rpp_simd_load(rpp_load96_i8pkd3_to_i8pln3, srcPtrTemp, px);                               // simd loads
                    compute_solarize_96_i8_host(px, pxThresholdParam);
                    rpp_simd_store(rpp_store96_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, px);  // simd stores
                    srcPtrTemp += 96;
                    dstPtrTempR += 32;
                    dstPtrTempG += 32;
                    dstPtrTempB += 32;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (255 - *(srcPtrTemp)) : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam) ? (255 - *(srcPtrTemp + 1)) : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam) ? (255 - *(srcPtrTemp + 2)) : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                {
                    __m256i px[3];
                    rpp_simd_load(rpp_load96_i8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, px);  // simd loads
                    compute_solarize_96_i8_host(px, pxThresholdParam); 	                                        // threshold adjustment
                    rpp_simd_store(rpp_store96_i8pln3_to_i8pkd3, dstPtrTemp, px);                           // simd stores
                    srcPtrTempR += 32;
                    srcPtrTempG += 32;
                    srcPtrTempB += 32;
                    dstPtrTemp += 96;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (*srcPtrTempR >= thresholdParam) ? (255 - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ = (*srcPtrTempG >= thresholdParam) ? (255 - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ = (*srcPtrTempB >= thresholdParam) ? (255 - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else
        {
            Rpp32u alignedLength = bufferLength & ~31;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                    {
#if __AVX2__
                        __m256i px;
                        px = _mm256_loadu_si256((__m256i *)srcPtrTemp);
                        compute_solarize_32_i8_host(&px, pxThresholdParam);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, px);
#endif
                        srcPtrTemp +=32;
                        dstPtrTemp +=32;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (255 - (*srcPtrTemp)) : *srcPtrTemp;
                        srcPtrTemp++;
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

RppStatus solarize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32f *thresholdTensor,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

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

        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        Rpp32f thresholdParam = thresholdTensor[batchCount];

#if __AVX2__
        __m256 pThresholdParam = _mm256_set1_ps(thresholdParam);
#endif
        // Threshold with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pkd3_to_f32pln3_avx, srcPtrTemp, p);                               // simd loads
                    compute_solarize_24_host(p, pThresholdParam); 	                                           // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);  // simd stores
                    srcPtrTemp += 24;
                    dstPtrTempR += 8;
                    dstPtrTempG += 8;
                    dstPtrTempB += 8;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTempR++ = (*(srcPtrTemp) >= thresholdParam) ? (1.0f - *(srcPtrTemp)) : *(srcPtrTemp);
                    *dstPtrTempG++ = (*(srcPtrTemp + 1) >= thresholdParam) ? (1.0f - *(srcPtrTemp + 1)) : *(srcPtrTemp + 1);
                    *dstPtrTempB++ = (*(srcPtrTemp + 2) >= thresholdParam) ? (1.0f - *(srcPtrTemp + 2)) : *(srcPtrTemp + 2);
                    srcPtrTemp += 3;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Threshold with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                {
                    __m256 p[3];
                    rpp_simd_load(rpp_load24_f16pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);  // simd loads
                    compute_solarize_24_host(p, pThresholdParam); 	                                         // threshold adjustment
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);                           // simd stores
                    srcPtrTempR += 8;
                    srcPtrTempG += 8;
                    srcPtrTempB += 8;
                    dstPtrTemp += 24;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    *dstPtrTemp++ = (*srcPtrTempR >= thresholdParam) ? (1.0f - (*srcPtrTempR)) : *srcPtrTempR;
                    *dstPtrTemp++ = (*srcPtrTempG >= thresholdParam) ? (1.0f - (*srcPtrTempG)) : *srcPtrTempG;
                    *dstPtrTemp++ = (*srcPtrTempB >= thresholdParam) ? (1.0f - (*srcPtrTempB)) : *srcPtrTempB;
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        else
        {
            Rpp32u alignedLength = bufferLength & ~7;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
#if __AVX2__
                        __m256 p[1];
                        rpp_simd_load(rpp_load8_f16_to_f32_avx, srcPtrTemp, p);      // simd loads
                        compute_solarize_8_host(p, pThresholdParam);              // threshold adjustment
                        rpp_simd_store(rpp_store8_f32_to_f16_avx, dstPtrTemp, p);    // simd stores
#endif
                        srcPtrTemp += 8;
                        dstPtrTemp += 8;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp++ = (*srcPtrTemp >= thresholdParam) ? (1.0f - (*srcPtrTemp)) : *srcPtrTemp;
                        srcPtrTemp++;
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
