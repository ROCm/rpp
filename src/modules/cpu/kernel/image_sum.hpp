#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include "reduction.hpp"

RppStatus image_sum_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *imageSumArr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Image Sum without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 16) * 16;
            Rpp32f sum = 0.0;
            Rpp32f sumAvx[8] = {0.0};

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256 psum = _mm256_setzero_ps();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p1[2];
                    rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p1);
                    compute_sum_16_host(p1, &psum);
#endif
                    srcPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sum += (Rpp32f)(*srcPtrTemp);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvx, &psum);
#endif
            for(int i = 0; i < 4; i++)
                sum += (sumAvx[i] + sumAvx[i + 4]);

            imageSumArr[batchCount] = sum;
        }

        // Image Sum without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp32f sumAvxR[8] = {0.0};
            Rpp32f sumAvxG[8] = {0.0};
            Rpp32f sumAvxB[8] = {0.0};

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256 psumR = _mm256_setzero_ps();
            __m256 psumG = _mm256_setzero_ps();
            __m256 psumB = _mm256_setzero_ps();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_sum_48_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += (Rpp32f)(*srcPtrTempR);
                    sumG += (Rpp32f)(*srcPtrTempG);
                    sumB += (Rpp32f)(*srcPtrTempB);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 4; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 4]);
                sumG += (sumAvxG[i] + sumAvxG[i + 4]);
                sumB += (sumAvxB[i] + sumAvxB[i + 4]);
            }

            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = sumR;
            imageSumArr[index + 1] = sumG;
            imageSumArr[index + 2] = sumB;
            imageSumArr[index + 3] = sum;
        }

        // Image Sum without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp32f sumAvxR[8] = {0.0};
            Rpp32f sumAvxG[8] = {0.0};
            Rpp32f sumAvxB[8] = {0.0};

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256 psumR = _mm256_setzero_ps();
            __m256 psumG = _mm256_setzero_ps();
            __m256 psumB = _mm256_setzero_ps();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    compute_sum_48_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += (Rpp32f)(srcPtrTemp[0]);
                    sumG += (Rpp32f)(srcPtrTemp[1]);
                    sumB += (Rpp32f)(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store8_f32_to_f32_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 4; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 4]);
                sumG += (sumAvxG[i] + sumAvxG[i + 4]);
                sumB += (sumAvxB[i] + sumAvxB[i + 4]);
            }

            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = sumR;
            imageSumArr[index + 1] = sumG;
            imageSumArr[index + 2] = sumB;
            imageSumArr[index + 3] = sum;
        }

    }

    return RPP_SUCCESS;
}

RppStatus image_sum_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *imageSumArr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Image Sum without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            vectorIncrement = 8;
            Rpp64f sum = 0.0;
            Rpp64f sumAvx[4] = {0.0};

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d psum = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_f32_to_f64_avx, srcPtrTemp, p1);
                    compute_sum_8_host(p1, &psum);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sum += (Rpp64f)(*srcPtrTemp);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvx, &psum);
#endif
            for(int i = 0; i < 2; i++)
                sum += (sumAvx[i] + sumAvx[i + 2]);

            imageSumArr[batchCount] = (Rpp32f)sum;
        }

        // Image Sum with fused output-layout toggle (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d psumR = _mm256_setzero_pd();
            __m256d psumG = _mm256_setzero_pd();
            __m256d psumB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += (Rpp64f)(*srcPtrTempR);
                    sumG += (Rpp64f)(*srcPtrTempG);
                    sumB += (Rpp64f)(*srcPtrTempB);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 2; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                sumB += (sumAvxB[i] + sumAvxB[i + 2]);
            }

            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = (Rpp32f)sumR;
            imageSumArr[index + 1] = (Rpp32f)sumG;
            imageSumArr[index + 2] = (Rpp32f)sumB;
            imageSumArr[index + 3] = (Rpp32f)sum;
        }

        // Image Sum with fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp32f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d psumR = _mm256_setzero_pd();
            __m256d psumG = _mm256_setzero_pd();
            __m256d psumB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f64pln3_avx, srcPtrTemp, p);
                    compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += (Rpp64f)(srcPtrTemp[0]);
                    sumG += (Rpp64f)(srcPtrTemp[1]);
                    sumB += (Rpp64f)(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 2; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                sumB += (sumAvxB[i] + sumAvxB[i + 2]);
            }

            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = (Rpp32f)sumR;
            imageSumArr[index + 1] = (Rpp32f)sumG;
            imageSumArr[index + 2] = (Rpp32f)sumB;
            imageSumArr[index + 3] = (Rpp32f)sum;
        }
    }

    return RPP_SUCCESS;
}

RppStatus image_sum_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *imageSumArr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp16f *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Image Sum without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = bufferLength & ~(vectorIncrementPerChannel-1);
            vectorIncrement = 8;
            Rpp64f sum = 0.0;
            Rpp64f sumAvx[4] = {0.0};

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d psum = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[8];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_f32_to_f64_avx, srcPtrTemp_ps, p1);
                    compute_sum_8_host(p1, &psum);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sum += (Rpp64f)(*srcPtrTemp);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvx, &psum);
#endif
            for(int i = 0; i < 2; i++)
                sum += (sumAvx[i] + sumAvx[i + 2]);

            imageSumArr[batchCount] = (Rpp32f)sum;
        }

        // Image Sum with fused output-layout toggle (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d psumR = _mm256_setzero_pd();
            __m256d psumG = _mm256_setzero_pd();
            __m256d psumB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8];
                    for(int cnt = 0; cnt < vectorIncrementPerChannel; cnt++)
                    {
                        srcPtrTempR_ps[cnt] = (Rpp32f) srcPtrTempR[cnt];
                        srcPtrTempG_ps[cnt] = (Rpp32f) srcPtrTempG[cnt];
                        srcPtrTempB_ps[cnt] = (Rpp32f) srcPtrTempB[cnt];
                    }
#if __AVX2__
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pln3_to_f64pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);
                    compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += (Rpp64f)(*srcPtrTempR);
                    sumG += (Rpp64f)(*srcPtrTempG);
                    sumB += (Rpp64f)(*srcPtrTempB);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 2; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                sumB += (sumAvxB[i] + sumAvxB[i + 2]);
            }

            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = (Rpp32f)sumR;
            imageSumArr[index + 1] = (Rpp32f)sumG;
            imageSumArr[index + 2] = (Rpp32f)sumB;
            imageSumArr[index + 3] = (Rpp32f)sum;
        }

        // Image Sum with fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp16f *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d psumR = _mm256_setzero_pd();
            __m256d psumG = _mm256_setzero_pd();
            __m256d psumB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[24];
                    for(int cnt = 0; cnt < vectorIncrement; cnt++)
                        srcPtrTemp_ps[cnt] = (Rpp32f) srcPtrTemp[cnt];
#if __AVX2__
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_f32pkd3_to_f64pln3_avx, srcPtrTemp_ps, p);
                    compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += (Rpp64f)(srcPtrTemp[0]);
                    sumG += (Rpp64f)(srcPtrTemp[1]);
                    sumB += (Rpp64f)(srcPtrTemp[2]);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 2; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                sumB += (sumAvxB[i] + sumAvxB[i + 2]);
            }

            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = (Rpp32f)sumR;
            imageSumArr[index + 1] = (Rpp32f)sumG;
            imageSumArr[index + 2] = (Rpp32f)sumB;
            imageSumArr[index + 3] = (Rpp32f)sum;
        }
    }

    return RPP_SUCCESS;
}

RppStatus image_sum_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *imageSumArr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8s *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32u totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;

        // Image Sum without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            vectorIncrement = 8;
            Rpp64f sum = 0.0;
            Rpp64f sumAvx[4] = {0.0};

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d psum = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_i8_to_f64_avx, srcPtrTemp, p1);
                    compute_sum_8_host(p1, &psum);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sum += (Rpp64f)(*srcPtrTemp + 128);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvx, &psum);
#endif
            for(int i = 0; i < 2; i++)
                sum += (sumAvx[i] + sumAvx[i + 2]);

            sum -= 128.0 * totalPixelsPerChannel;
            imageSumArr[batchCount] = (Rpp32f)sum;
        }

        // Image Sum with fused output-layout toggle (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d psumR = _mm256_setzero_pd();
            __m256d psumG = _mm256_setzero_pd();
            __m256d psumB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
#if __AVX2__
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_i8pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    sumR += (Rpp64f)(*srcPtrTempR + 128);
                    sumG += (Rpp64f)(*srcPtrTempG + 128);
                    sumB += (Rpp64f)(*srcPtrTempB + 128);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 2; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                sumB += (sumAvxB[i] + sumAvxB[i + 2]);
            }

            sumR -= 128.0 * totalPixelsPerChannel;
            sumG -= 128.0 * totalPixelsPerChannel;
            sumB -= 128.0 * totalPixelsPerChannel;
            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = (Rpp32f)sumR;
            imageSumArr[index + 1] = (Rpp32f)sumG;
            imageSumArr[index + 2] = (Rpp32f)sumB;
            imageSumArr[index + 3] = (Rpp32f)sum;
        }

        // Image Sum with fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            Rpp8s *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d psumR = _mm256_setzero_pd();
            __m256d psumG = _mm256_setzero_pd();
            __m256d psumB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
#if __AVX2__
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_i8pkd3_to_f64pln3_avx, srcPtrTemp, p);
                    compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    sumR += (Rpp64f)(srcPtrTemp[0] + 128);
                    sumG += (Rpp64f)(srcPtrTemp[1] + 128);
                    sumB += (Rpp64f)(srcPtrTemp[2] + 128);
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
            for(int i = 0; i < 2; i++)
            {
                sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                sumB += (sumAvxB[i] + sumAvxB[i + 2]);
            }

            sumR -= 128.0 * totalPixelsPerChannel;
            sumG -= 128.0 * totalPixelsPerChannel;
            sumB -= 128.0 * totalPixelsPerChannel;
            sum = sumR + sumG + sumB;
            int index = batchCount * 4;
            imageSumArr[index] = (Rpp32f)sumR;
            imageSumArr[index + 1] = (Rpp32f)sumG;
            imageSumArr[index + 2] = (Rpp32f)sumB;
            imageSumArr[index + 3] = (Rpp32f)sum;
        }
    }

    return RPP_SUCCESS;
}
