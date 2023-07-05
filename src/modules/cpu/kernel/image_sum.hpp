#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus image_sum_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *imageSumArr,
                                      Rpp32u imageSumArrLength,
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

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        // Image Sum without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            vectorIncrement = 8;
            Rpp64f sum = 0.0;
            Rpp64f sumAvx[4] = {0.0};

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256d psum = _mm256_setzero_pd();
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
#if __AVX2__
                        __m256d p1[2];
                        rpp_simd_load(rpp_load8_u8_to_f64_avx, srcPtrTemp, p1);
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
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
                rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvx, &psum);
#endif
                for(int i=0;i<2;i++)
                    sum += (sumAvx[i] + sumAvx[i + 2]);
            }
            imageSumArr[batchCount] = (Rpp32f)sum;
        }

        // Image Sum without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
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
                    Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                    srcPtrTempR = srcPtrRowR;
                    srcPtrTempG = srcPtrRowG;
                    srcPtrTempB = srcPtrRowB;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
#if __AVX2__
                        __m256d p[6];
                        rpp_simd_load(rpp_load24_u8pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                        compute_sum_24_host(p, &psumR, &psumG, &psumB);
#endif
                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
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
                for(int i=0;i<2;i++)
                {
                    sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                    sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                    sumB += (sumAvxB[i] + sumAvxB[i + 2]);
                }
            }
            sum = sumR + sumG + sumB;
            imageSumArr[batchCount * 4] = (Rpp32f)sumR;
            imageSumArr[(batchCount * 4) + 1] = (Rpp32f)sumG;
            imageSumArr[(batchCount * 4) + 2] = (Rpp32f)sumB;
            imageSumArr[(batchCount * 4) + 3] = (Rpp32f)sum;
        }

        // Image Sum without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f sum, sumR = 0.0, sumG = 0.0, sumB = 0.0;
            Rpp64f sumAvxR[4] = {0.0};
            Rpp64f sumAvxG[4] = {0.0};
            Rpp64f sumAvxB[4] = {0.0};

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256d psumR = _mm256_setzero_pd();
                __m256d psumG = _mm256_setzero_pd();
                __m256d psumB = _mm256_setzero_pd();
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
#if __AVX2__
                        __m256d p[6];
                        rpp_simd_load(rpp_load24_u8pkd3_to_f64pln3_avx, srcPtrTemp, p);
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
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
                rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxR, &psumR);
                rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxG, &psumG);
                rpp_simd_store(rpp_store4_f64_to_f64_avx, sumAvxB, &psumB);
#endif
                for(int i=0;i<2;i++)
                {
                    sumR += (sumAvxR[i] + sumAvxR[i + 2]);
                    sumG += (sumAvxG[i] + sumAvxG[i + 2]);
                    sumB += (sumAvxB[i] + sumAvxB[i + 2]);
                }
            }
            sum = sumR + sumG + sumB;
            imageSumArr[batchCount * 4] = (Rpp32f)sumR;
            imageSumArr[(batchCount * 4) + 1] = (Rpp32f)sumG;
            imageSumArr[(batchCount * 4) + 2] = (Rpp32f)sumB;
            imageSumArr[(batchCount * 4) + 3] = (Rpp32f)sum;
        }

    }

    return RPP_SUCCESS;
}