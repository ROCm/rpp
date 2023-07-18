#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include "reduction.hpp"

RppStatus image_stddev_u8_u8_host_tensor(Rpp8u *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *imageStddevArr,
                                         Rpp32f *meanTensor,
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

        Rpp32f mean = meanTensor[batchCount];

        Rpp8u *srcPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);

        Rpp32u alignedLength = (bufferLength / 24) * 24;
        Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;
        Rpp32f totalPixelsPerChannel = roi.xywhROI.roiWidth * roi.xywhROI.roiHeight;
        int idx = batchCount * 4;

        // Image Stdev without fused output-layout toggle (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / 8) * 8;
            Rpp64f var = 0.0;
            Rpp32f stddev = 0.0;
            Rpp64f varAvx[4] = {0.0};

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVar = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p1[2];
                    rpp_simd_load(rpp_load8_u8_to_f64_avx, srcPtrTemp, p1);
                    compute_var_8_host(p1, &pMean, &pVar);

                    srcPtrTemp += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    var += ((Rpp64f)mean - (Rpp64f)(*srcPtrTemp)) * ((Rpp64f)mean - (Rpp64f)(*srcPtrTemp));
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvx, &pVar);

            for(int i = 0; i < 2; i++)
                var += (varAvx[i] + varAvx[i + 2]);
#endif
            stddev = sqrt(var / totalPixelsPerChannel);
            imageStddevArr[batchCount] = (Rpp32f)stddev;
        }

        // Image Stddev without fused output-layout toggle 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp64f var, varR = 0.0, varG = 0.0, varB = 0.0;
            Rpp32f stddev, stddevR = 0.0, stddevG = 0.0, stddevB = 0.0;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVarR, pVarG, pVarB;
            pVarR = pVarG = pVarB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_u8pln3_to_f64pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_var_24_host(p, &pMean, &pVarR, &pVarG, &pVarB);
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    varR += (mean - (Rpp64f)(*srcPtrTempR)) * (mean - (Rpp64f)(*srcPtrTempR));
                    varG += (mean - (Rpp64f)(*srcPtrTempG)) * (mean - (Rpp64f)(*srcPtrTempG));
                    varB += (mean - (Rpp64f)(*srcPtrTempB)) * (mean - (Rpp64f)(*srcPtrTempB));
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvxR, &pVarR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvxG, &pVarG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvxB, &pVarB);

            for(int i = 0; i < 2; i++)
            {
                varR += (varAvxR[i] + varAvxR[i + 2]);
                varG += (varAvxG[i] + varAvxG[i + 2]);
                varB += (varAvxB[i] + varAvxB[i + 2]);
            }
#endif
            var = varR + varG + varB;
            stddev = (Rpp32f)sqrt(var / (totalPixelsPerChannel * 3));
            stddevR = (Rpp32f)sqrt(varR / totalPixelsPerChannel);
            stddevG = (Rpp32f)sqrt(varG / totalPixelsPerChannel);
            stddevB = (Rpp32f)sqrt(varB / totalPixelsPerChannel);
            imageStddevArr[idx] = stddevR;
            imageStddevArr[idx + 1] = stddevG;
            imageStddevArr[idx + 2] = stddevB;
            imageStddevArr[idx + 3] = stddev;
        }

        // Image Sum without fused output-layout toggle (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp64f var, varR = 0.0, varG = 0.0, varB = 0.0;
            Rpp32f stddev, stddevR = 0.0, stddevG = 0.0, stddevB = 0.0;
            Rpp64f varAvxR[4] = {0.0};
            Rpp64f varAvxG[4] = {0.0};
            Rpp64f varAvxB[4] = {0.0};

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256d pMean = _mm256_set1_pd(mean);
            __m256d pVarR, pVarG, pVarB;
            pVarR = pVarG = pVarB = _mm256_setzero_pd();
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256d p[6];
                    rpp_simd_load(rpp_load24_u8pkd3_to_f64pln3_avx, srcPtrTemp, p);
                    compute_var_24_host(p, &pMean, &pVarR, &pVarG, &pVarB);
                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    varR += (mean - (Rpp64f)(srcPtrTemp[0])) * (mean - (Rpp64f)(srcPtrTemp[0]));
                    varG += (mean - (Rpp64f)(srcPtrTemp[1])) * (mean - (Rpp64f)(srcPtrTemp[1]));
                    varB += (mean - (Rpp64f)(srcPtrTemp[2])) * (mean - (Rpp64f)(srcPtrTemp[2]));
                    srcPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvxR, &pVarR);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvxG, &pVarG);
            rpp_simd_store(rpp_store4_f64_to_f64_avx, varAvxB, &pVarB);

            for(int i = 0; i < 2; i++)
            {
                varR += (varAvxR[i] + varAvxR[i + 2]);
                varG += (varAvxG[i] + varAvxG[i + 2]);
                varB += (varAvxB[i] + varAvxB[i + 2]);
            }
#endif
            var = varR + varG + varB;
            stddev = (Rpp32f)sqrt(var / (totalPixelsPerChannel * 3));
            stddevR = (Rpp32f)sqrt(varR / totalPixelsPerChannel);
            stddevG = (Rpp32f)sqrt(varG / totalPixelsPerChannel);
            stddevB = (Rpp32f)sqrt(varB / totalPixelsPerChannel);
            imageStddevArr[idx] = stddevR;
            imageStddevArr[idx + 1] = stddevG;
            imageStddevArr[idx + 2] = stddevB;
            imageStddevArr[idx + 3] = stddev;
        }
    }

    return RPP_SUCCESS;
}
