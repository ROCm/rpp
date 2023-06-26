#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus image_max_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *imageMaxArr,
                                      Rpp32u imageMaxArrLength,
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

        // Image Max 1 channel (NCHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            alignedLength = (bufferLength / vectorIncrementPerChannel) * vectorIncrementPerChannel;
            vectorIncrement = vectorIncrementPerChannel;
            Rpp8u max = 0;
            Rpp8u resultAvx[16];

            Rpp8u *srcPtrRow;
            srcPtrRow = srcPtrChannel;
#if __AVX2__
                __m256i pMax = _mm256_setzero_si256();
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
                        compute_max_32_host(&p1, &pMax);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        max = std::max(*srcPtrTemp, max);
                        srcPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX2__
                __m128i result;
                reduce_max_32_host(&pMax, &result);
                rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                max = std::max(std::max(resultAvx[0], resultAvx[1]), max);
#endif
            imageMaxArr[batchCount] = max;
        }
        // Image Max 3 channel (NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u maxC = 0, maxR = 0, maxG = 0, maxB = 0;
            Rpp8u resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
                srcPtrRowR = srcPtrChannel;
                srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
                srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
#if __AVX2__
                __m256i pMaxR = _mm256_setzero_si256();
                __m256i pMaxG = pMaxR;
                __m256i pMaxB = pMaxR;
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
                        compute_max_96_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTempR += vectorIncrementPerChannel;
                        srcPtrTempG += vectorIncrementPerChannel;
                        srcPtrTempB += vectorIncrementPerChannel;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {;
                        maxR = std::max(*srcPtrTempR, maxR);
                        maxG = std::max(*srcPtrTempG, maxG);
                        maxB = std::max(*srcPtrTempB, maxB);
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
                    reduce_max_96_host(&pMaxR, &pMaxG, &pMaxB, &result);
                    rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                    maxR = std::max(std::max(resultAvx[0], resultAvx[1]), maxR);
                    maxG = std::max(std::max(resultAvx[2], resultAvx[3]), maxG);
                    maxB = std::max(std::max(resultAvx[4], resultAvx[5]), maxB);
    #endif
                    maxC = std::max(std::max(maxR, maxG), maxB);
                    imageMaxArr[batchCount*4] = maxR;
                    imageMaxArr[(batchCount*4) + 1] = maxG;
                    imageMaxArr[(batchCount*4) + 2] = maxB;
                    imageMaxArr[(batchCount*4) + 3] = maxC;
                }
            }
        }

        // Image Max 3 channel (NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp32u vectorIncrement = 48;
            Rpp8u maxC = 0, maxR = 0, maxG = 0, maxB = 0;
            Rpp8u resultAvx[16];

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                srcPtrRow = srcPtrChannel;

#if __AVX__
                __m128i pMaxR = _mm_setzero_si128();
                __m128i pMaxG = pMaxR;
                __m128i pMaxB = pMaxR;
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
                        compute_max_48_host(p, &pMaxR, &pMaxG, &pMaxB);

                        srcPtrTemp += vectorIncrement;
                    }
#endif
                    for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                    {
                        maxR = std::max(srcPtrTemp[0], maxR);
                        maxG = std::max(srcPtrTemp[1], maxG);
                        maxB = std::max(srcPtrTemp[2], maxB);
                        srcPtrTemp += 3;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
#if __AVX__
                if(flag_avx)
                {
                    __m128i result;
                    reduce_max_48_host(&pMaxR, &pMaxG, &pMaxB, &result);
                    rpp_simd_store(rpp_store16_u8_to_u8, resultAvx, &result);

                    maxR = std::max(std::max(resultAvx[0], resultAvx[1]), maxR);
                    maxG = std::max(std::max(resultAvx[2], resultAvx[3]), maxG);
                    maxB = std::max(std::max(resultAvx[4], resultAvx[5]), maxB);
                }
#endif
            }
			maxC = std::max(std::max(maxR, maxG), maxB);
            imageMaxArr[batchCount*4] = maxR;
			imageMaxArr[(batchCount*4) + 1] = maxG;
			imageMaxArr[(batchCount*4) + 2] = maxB;
			imageMaxArr[(batchCount*4) + 3] = maxC;
        }
    }
    printf("\n Max output\n");
    for(int i=0;i<imageMaxArrLength;i++)
        printf("imageMaxArr[%d]: %d\n", i, (int)imageMaxArr[i]);

    return RPP_SUCCESS;
}