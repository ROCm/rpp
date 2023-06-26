#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus image_min_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *imageMinArr,
                                      Rpp32u imageMinArrLength,
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

        Rpp32u alignedLength = (bufferLength / 96) * 96;
        Rpp32u vectorIncrement = 96;
        Rpp32u vectorIncrementPerChannel = 32;
        int flag_avx = 0;
        if(alignedLength)
            flag_avx = 1;

        // Image Min 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp8u min = 255;
            Rpp8u resultAvx[16];

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
            __m256i pMin = _mm256_set1_epi8((char)255);
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                srcPtrTemp = srcPtrRow;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p1;
                    rpp_simd_load(rpp_load32_u8_avx, srcPtrTemp, &p1);
                    compute_min_32_host(&p1, &pMin);

                    srcPtrTemp += vectorIncrement;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    min = std::min(*srcPtrTemp, min);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
            srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
            __m128i result;
            reduce_min_32_host(&pMin, &result);
            rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

            min = std::min(std::min(resultAvx[0], resultAvx[1]), min);
#endif
            imageMinArr[batchCount] = min;
        }

        // Image Min 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u minC = 255, minR = 255, minG = 255, minB = 255;
            Rpp8u resultAvx[16];

            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
            __m256i pMinR = _mm256_set1_epi8((char)255);
            __m256i pMinG = pMinR;
            __m256i pMinB = pMinR;
#endif
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;

                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256i p[3];
                    rpp_simd_load(rpp_load96_u8_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
                    compute_min_96_host(p, &pMinR, &pMinG, &pMinB);

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    minR = std::min(*srcPtrTempR, minR);
                    minG = std::min(*srcPtrTempG, minG);
                    minB = std::min(*srcPtrTempB, minB);
                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
            }
#if __AVX2__
            if(flag_avx)
            {
                __m128i result;
                reduce_min_96_host(&pMinR, &pMinG, &pMinB, &result);
                rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                minR = std::min(std::min(resultAvx[0], resultAvx[1]), minR);
                minG = std::min(std::min(resultAvx[2], resultAvx[3]), minG);
                minB = std::min(std::min(resultAvx[4], resultAvx[5]), minB);
            }
#endif
			minC = std::min(std::min(minR, minG), minB);
            imageMinArr[batchCount*4] = minR;
			imageMinArr[(batchCount*4) + 1] = minG;
			imageMinArr[(batchCount*4) + 2] = minB;
			imageMinArr[(batchCount*4) + 3] = minC;
        }

        // Image Min 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp8u minC = 255, minR = 255, minG = 255, minB = 255;
            Rpp8u resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;

#if __AVX__
                __m128i pMinR = _mm_set1_epi8((char)255);
                __m128i pMinG = pMinR;
                __m128i pMinB = pMinR;
#endif
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    srcPtrTemp = srcPtrRow;

                    int vectorLoopCount = 0;
#if __AVX__
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                    {
                        __m128i p[3];
                        rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp, p);
                        compute_min_48_host(p, &pMinR, &pMinG, &pMinB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        minR = std::min(srcPtrTemp[0], minR);
                        minG = std::min(srcPtrTemp[1], minG);
                        minB = std::min(srcPtrTemp[2], minB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX__
                if(flag_avx)
                {
                    __m128i result;
                    reduce_min_48_host(&pMinR, &pMinG, &pMinB, &result);
                    rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                    minR = std::min(std::min(resultAvx[0], resultAvx[1]), minR);
                    minG = std::min(std::min(resultAvx[2], resultAvx[3]), minG);
                    minB = std::min(std::min(resultAvx[4], resultAvx[5]), minB);
                }
#endif
            }
			minC = std::min(std::min(minR, minG), minB);
            imageMinArr[batchCount*4] = minR;
			imageMinArr[(batchCount*4) + 1] = minG;
			imageMinArr[(batchCount*4) + 2] = minB;
			imageMinArr[(batchCount*4) + 3] = minC;
        }
    }
    printf("\n min output\n");
    for(int i=0;i<imageMinArrLength;i++)
        printf("imageMinArr[%d]: %d\n", i, (int)imageMinArr[i]);

    return RPP_SUCCESS;
}