#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>
#include <cstring>
#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

template <typename T> void convert_float_pln3_pkd3(T *srcPtrChannel, T *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int width)
{
    T *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
    srcPtrRowR = srcPtrChannel;
    srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
    srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
    dstPtrRow = dstPtrChannel;
    for(int i = 0; i < srcDescPtr->h ; i++)
    {
        T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
        srcPtrTempR = srcPtrRowR;
        srcPtrTempG = srcPtrRowG;
        srcPtrTempB = srcPtrRowB;
        dstPtrTemp = dstPtrRow;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < width; vectorLoopCount += 8)
        {
            if(typeid(T) == typeid(Rpp32f))
            {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, (Rpp32f *)srcPtrTempR, (Rpp32f *)srcPtrTempG, (Rpp32f *)srcPtrTempB, p);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, (Rpp32f *)dstPtrTemp, p);
            }
            else if(typeid(T) == typeid(Rpp16f))
            {
                __m256 p[3];
                Rpp32f srcPtrTempR_ps[8], srcPtrTempG_ps[8], srcPtrTempB_ps[8], dstPtrTemp_ps[25];
                for(int cnt = 0; cnt < 8; cnt++)
                {
                    srcPtrTempR_ps[cnt] = (Rpp32f)srcPtrTempR[cnt];
                    srcPtrTempG_ps[cnt] = (Rpp32f)srcPtrTempG[cnt];
                    srcPtrTempB_ps[cnt] = (Rpp32f)srcPtrTempB[cnt];
                }
                rpp_simd_load(rpp_load24_f32pln3_to_f32pln3_avx, srcPtrTempR_ps, srcPtrTempG_ps, srcPtrTempB_ps, p);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p);
                for(int cnt = 0; cnt < 24; cnt++){
                    dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
                }
            }
            srcPtrTempR += 8;
            srcPtrTempG += 8;
            srcPtrTempB += 8;
            dstPtrTemp += 24;
        }
        srcPtrRowR += srcDescPtr->strides.hStride;
        srcPtrRowG += srcDescPtr->strides.hStride;
        srcPtrRowB += srcDescPtr->strides.hStride;
        dstPtrRow += dstDescPtr->strides.hStride;
    }
}

template <typename T> void convert_float_pkd3_pln3(T *srcPtrChannel, T *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int height)
{
    T *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
    srcPtrRow = srcPtrChannel;
    dstPtrRowR = dstPtrChannel;
    dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
    dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

    for(int i = 0; i < srcDescPtr->h; i++)
    {
        T *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
        srcPtrTemp = srcPtrRow;
        dstPtrTempR = dstPtrRowR;
        dstPtrTempG = dstPtrRowG;
        dstPtrTempB = dstPtrRowB;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < height; vectorLoopCount += 24)
        {
            if(typeid(T) == typeid(Rpp32f))
            {
                __m256 p[3];
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, (Rpp32f *)srcPtrTemp, p);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, (Rpp32f *)dstPtrTempR, (Rpp32f *)dstPtrTempG, (Rpp32f *)dstPtrTempB, p);
            }
            else if(typeid(T) == typeid(Rpp16f)){
                __m256 p[3];
                Rpp32f srcPtrTemp_ps[24], dstPtrTempR_ps[8], dstPtrTempG_ps[8], dstPtrTempB_ps[8];
                for(int cnt = 0; cnt < 24; cnt++){
                    srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];
                }
                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p);
                rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR_ps, dstPtrTempG_ps, dstPtrTempB_ps, p);
                for(int cnt = 0; cnt < 8; cnt++){
                    dstPtrTempR[cnt] = (Rpp16f)dstPtrTempR_ps[cnt];
                    dstPtrTempG[cnt] = (Rpp16f)dstPtrTempG_ps[cnt];
                    dstPtrTempB[cnt] = (Rpp16f)dstPtrTempB_ps[cnt];
                }
            }
            srcPtrTemp += 24;
            dstPtrTempR += 8;
            dstPtrTempG += 8;
            dstPtrTempB += 8;
        }
        srcPtrRow += srcDescPtr->strides.hStride;
        dstPtrRowR += dstDescPtr->strides.hStride;
        dstPtrRowG += dstDescPtr->strides.hStride;
        dstPtrRowB += dstDescPtr->strides.hStride;
    }
}

template <typename T> void convert_pln3_pkd3( T *srcPtrChannel, T *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int width)
{
    T *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
    srcPtrRowR = srcPtrChannel;
    srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
    srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
    dstPtrRow = dstPtrChannel;

    for(int i = 0; i < srcDescPtr->h; i++)
    {
        T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
        srcPtrTempR = srcPtrRowR;
        srcPtrTempG = srcPtrRowG;
        srcPtrTempB = srcPtrRowB;
        dstPtrTemp = dstPtrRow;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < width; vectorLoopCount += 16)
        {
            if(typeid(T) == typeid(Rpp8u)){
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, (Rpp8u *)srcPtrTempR, (Rpp8u *)srcPtrTempG, (Rpp8u *)srcPtrTempB, p);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, (Rpp8u *)dstPtrTemp, p);
            }
            else if(typeid(T) == typeid(Rpp8s)){
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pln3_to_f32pln3_avx, (Rpp8s *)srcPtrTempR, (Rpp8s *)srcPtrTempG, (Rpp8s *)srcPtrTempB, p);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pkd3_avx, (Rpp8s *)dstPtrTemp, p);
            }
            srcPtrTempR += 16;
            srcPtrTempG += 16;
            srcPtrTempB += 16;
            dstPtrTemp += 48;
        }
        srcPtrRowR += srcDescPtr->strides.hStride;
        srcPtrRowG += srcDescPtr->strides.hStride;
        srcPtrRowB += srcDescPtr->strides.hStride;
        dstPtrRow += dstDescPtr->strides.hStride;
    }
}

template <typename T> void convert_pkd3_pln3( T *srcPtrChannel, T *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int width)
{
    T *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
    srcPtrRow = srcPtrChannel;
    dstPtrRowR = dstPtrChannel;
    dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
    dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

    for(int i = 0; i < srcDescPtr->h; i++)
    {
        T *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
        srcPtrTemp = srcPtrRow;
        dstPtrTempR = dstPtrRowR;
        dstPtrTempG = dstPtrRowG;
        dstPtrTempB = dstPtrRowB;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < width; vectorLoopCount += 48)
        {
            if(typeid(T) == typeid(Rpp8u))
            {
                __m256 p[6];
                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, (Rpp8u *)srcPtrTemp, p);
                rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, (Rpp8u *)dstPtrTempR, (Rpp8u *)dstPtrTempG, (Rpp8u *)dstPtrTempB, p);
            }
            else if(typeid(T) == typeid(Rpp8s)){
                __m256 p[6];
                rpp_simd_load(rpp_load48_i8pkd3_to_f32pln3_avx, (Rpp8s *)srcPtrTemp, p);
                rpp_simd_store(rpp_store48_f32pln3_to_i8pln3_avx, (Rpp8s *)dstPtrTempR, (Rpp8s *)dstPtrTempG, (Rpp8s *)dstPtrTempB, p);
            }
            srcPtrTemp += 48;
            dstPtrTempR += 16;
            dstPtrTempG += 16;
            dstPtrTempB += 16;
        }
        srcPtrRow += srcDescPtr->strides.hStride;
        dstPtrRowR += dstDescPtr->strides.hStride;
        dstPtrRowG += dstDescPtr->strides.hStride;
        dstPtrRowB += dstDescPtr->strides.hStride;
    }
}

RppStatus glitch_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *xOffsetR,
                                   Rpp32u *yOffsetR,
                                   Rpp32u *xOffsetG,
                                   Rpp32u *yOffsetG,
                                   Rpp32u *xOffsetB,
                                   Rpp32u *yOffsetB,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        __m128i maskR = _mm_setr_epi8(0, 0x80, 0x80, 3, 0x80, 0x80, 6, 0x80, 0x80, 9, 0x80, 0x80, 12, 0x80, 0x80, 15);
        __m128i maskGB = _mm_setr_epi8(0x80, 1, 2, 0x80, 4, 5, 0x80, 7, 8, 0x80, 10, 11, 0x80, 13, 14, 0x80);
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = xOffsetR[batchCount];
        Rpp32u yOffsetRchn = yOffsetR[batchCount];
        Rpp32u xOffsetGchn = xOffsetG[batchCount];
        Rpp32u yOffsetGchn = yOffsetG[batchCount];
        Rpp32u xOffsetBchn = xOffsetB[batchCount];
        Rpp32u yOffsetBchn = yOffsetB[batchCount];

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            convert_pkd3_pln3<Rpp8u>(srcPtrImage, dstPtrImage, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth * 3);
            Rpp32u vectorIncrement = 32;
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = dstPtrChannel + (c * dstDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8u *srcPtrTemp = srcPtrImageTemp;
                        Rpp8u *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < dstDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256i p;
                                p = _mm256_loadu_si256((__m256i *)srcPtrTemp);
                                _mm256_storeu_si256((__m256i *)dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += dstDescPtr->strides.hStride;
                        dstPtrImageTemp += dstDescPtr->strides.hStride;
                        offset += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u vectorIncrement = 32;
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8u *srcPtrTemp = srcPtrImageTemp;
                        Rpp8u *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8(srcPtrTemp);
                                _mm256_storeu_epi8(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
            convert_pln3_pkd3<Rpp8u>(srcPtrImage, dstPtrImage, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth);
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp32u vectorIncrement = 32;
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8u *srcPtrTemp = srcPtrImageTemp;
                        Rpp8u *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8(srcPtrTemp);
                                _mm256_storeu_epi8(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u vectorIncrement =  18;
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp8u *srcPtrRow, *dstPtrRow, *srcPtrRowOff;
                Rpp32u offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                srcPtrRow = dstPtrImage + c;
                srcPtrRowOff = srcPtrImage + offset;
                dstPtrRow = dstPtrImage + c;
                if(offset > 2)
                {
                    for(int i = yOffsets[c]; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp, *dstPtrTemp, *srcOffTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow; 
                        srcOffTemp = srcPtrRowOff;
                        int aligLen = ((roi.xywhROI.roiWidth * 3 - (xOffsetsLoc[c] * 3)) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth * 3 - (aligLen + xOffsetsLoc[c] * 3);
                        if(offset < srcDescPtr->strides.nStride)
                        {
                            for(int j = 0; j < aligLen; j += vectorIncrement)
                            {
                               __m128i p[2];
                               p[0] = _mm_loadu_si128((__m128i *)srcPtrTemp);
                               p[1] = _mm_loadu_si128((__m128i *)srcOffTemp);
                               p[1] = _mm_shuffle_epi8(p[1],maskR);
                               p[0] = _mm_shuffle_epi8(p[0],maskGB);
                               p[1] = _mm_or_si128(p[1],p[0]);
                               _mm_storeu_si128((__m128i *)dstPtrTemp,p[1]);
                               srcPtrTemp += vectorIncrement;
                               dstPtrTemp += vectorIncrement;
                               srcOffTemp += vectorIncrement;
                            }
                            for(int j = 0; j < remElemLen; j+=3)
                            {
                                *dstPtrTemp = *srcOffTemp;
                                dstPtrTemp += 3;
                                srcOffTemp += 3;
                            }
                            srcPtrRow += srcDescPtr->strides.hStride;
                            srcPtrRowOff += srcDescPtr->strides.hStride;
                            dstPtrRow += dstDescPtr->strides.hStride;
                            offset += srcDescPtr->strides.hStride;
                        }
                    }
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *xOffsetR,
                                   Rpp32u *yOffsetR,
                                   Rpp32u *xOffsetG,
                                   Rpp32u *yOffsetG,
                                   Rpp32u *xOffsetB,
                                   Rpp32u *yOffsetB,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        __m128i maskR = _mm_setr_epi8(0, 0x80, 0x80, 3, 0x80, 0x80, 6, 0x80, 0x80, 9, 0x80, 0x80, 12, 0x80, 0x80, 15);
        __m128i maskGB = _mm_setr_epi8(0x80, 1, 2, 0x80, 4, 5, 0x80, 7, 8, 0x80, 10, 11, 0x80, 13, 14, 0x80);
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = xOffsetR[batchCount];
        Rpp32u yOffsetRchn = yOffsetR[batchCount];
        Rpp32u xOffsetGchn = xOffsetG[batchCount];
        Rpp32u yOffsetGchn = yOffsetG[batchCount];
        Rpp32u xOffsetBchn = xOffsetB[batchCount];
        Rpp32u yOffsetBchn = yOffsetB[batchCount];

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp32u vectorIncrement = 32;
            convert_pkd3_pln3<Rpp8s>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth * 3);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8s *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8s *srcPtrTemp = srcPtrImageTemp;
                        Rpp8s *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < dstDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8(srcPtrTemp);
                                _mm256_storeu_epi8(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += dstDescPtr->strides.hStride;
                        dstPtrImageTemp += dstDescPtr->strides.hStride;
                        offset += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u vectorIncrement = 32;
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8s *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8s *srcPtrTemp = srcPtrImageTemp;
                        Rpp8s *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8(srcPtrTemp);
                                _mm256_storeu_epi8(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
            convert_pln3_pkd3<Rpp8s>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth);
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp32u vectorIncrement = 32;
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8s) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8s *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8s *srcPtrTemp = srcPtrImageTemp;
                        Rpp8s *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8(srcPtrTemp);
                                _mm256_storeu_epi8(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC ))
        {
            Rpp32u vectorIncrement = 18;
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8s) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp8s *srcPtrRow, *dstPtrRow, *srcPtrRowOff;
                Rpp32u offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                srcPtrRow = dstPtrImage + c;
                srcPtrRowOff = srcPtrImage + offset;
                dstPtrRow = dstPtrImage + c;
                if(offset > 2)
                {
                    for(int i = yOffsets[c]; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8s *srcPtrTemp, *dstPtrTemp, *srcOffTemp;
                        srcPtrTemp = srcPtrRow;
                        dstPtrTemp = dstPtrRow; 
                        srcOffTemp = srcPtrRowOff;
                        int aligLen = ((roi.xywhROI.roiWidth * 3 - (xOffsetsLoc[0] * 3)) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth * 3 - (aligLen + xOffsetsLoc[c] * 3);
                        if(offset < srcDescPtr->strides.nStride)
                        {
                            for(int j = 0; j < aligLen; j += vectorIncrement)
                            {
                               __m128i p[2];
                               p[0] = _mm_loadu_si128((__m128i *)srcPtrTemp);
                               p[1] = _mm_loadu_si128((__m128i *)srcOffTemp);
                               p[1] = _mm_shuffle_epi8(p[1],maskR);
                               p[0] = _mm_shuffle_epi8(p[0],maskGB);
                               p[1] = _mm_or_si128(p[1],p[0]);
                               _mm_storeu_si128((__m128i *)dstPtrTemp,p[1]);
                               srcPtrTemp += vectorIncrement;
                               dstPtrTemp += vectorIncrement;
                               srcOffTemp += vectorIncrement; 
                            }
                            for(int j = 0; j < remElemLen; j+=3)
                            {
                                *dstPtrTemp = *srcOffTemp;
                                dstPtrTemp += 3;
                                srcOffTemp += 3;
                            }
                            srcPtrRow += srcDescPtr->strides.hStride;
                            srcPtrRowOff += srcDescPtr->strides.hStride;
                            dstPtrRow += dstDescPtr->strides.hStride;
                            offset += srcDescPtr->strides.hStride;
                        }
                    }
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *xOffsetR,
                                   Rpp32u *yOffsetR,
                                   Rpp32u *xOffsetG,
                                   Rpp32u *yOffsetG,
                                   Rpp32u *xOffsetB,
                                   Rpp32u *yOffsetB,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = xOffsetR[batchCount];
        Rpp32u yOffsetRchn = yOffsetR[batchCount];
        Rpp32u xOffsetGchn = xOffsetG[batchCount];
        Rpp32u yOffsetGchn = yOffsetG[batchCount];
        Rpp32u xOffsetBchn = xOffsetB[batchCount];
        Rpp32u yOffsetBchn = yOffsetB[batchCount];

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u vectorIncrement = 8;

        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            convert_float_pkd3_pln3(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth * 3);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp32f *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp32f *srcPtrTemp = srcPtrImageTemp;
                        Rpp32f *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < dstDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256 p;
                                p = _mm256_loadu_ps(srcPtrTemp);
                                _mm256_storeu_ps(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += dstDescPtr->strides.hStride;
                        dstPtrImageTemp += dstDescPtr->strides.hStride;
                        offset += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp32f *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp32f *srcPtrTemp = srcPtrImageTemp;
                        Rpp32f *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256 p;
                                p = _mm256_loadu_ps(srcPtrTemp);
                                _mm256_storeu_ps(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
            convert_float_pln3_pkd3(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth);
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp32f) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp32f *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp32f *srcPtrTemp = srcPtrImageTemp;
                        Rpp32f *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256 p;
                                p = _mm256_loadu_ps(srcPtrTemp);
                                _mm256_storeu_ps(dstPtrTemp, p);
                                srcPtrTemp += vectorIncrement;
                                dstPtrTemp += vectorIncrement;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC ))
        {
            vectorIncrement = 24;
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp32f) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                Rpp32f *srcPtrRow, *srcPtrRowChn, *dstPtrRow;
                srcPtrRow = dstPtrImage + c;
                srcPtrRowChn = srcPtrImage + offset;
                dstPtrRow = dstPtrImage + c;
                if(offset > 2)
                {
                    for( int i = yOffsets[c]; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp32f *srcPtrTemp, *srcPtrTempChn, *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        srcPtrTempChn = srcPtrRowChn;
                        dstPtrTemp = dstPtrRow;
                        int aligLen = ((roi.xywhROI.roiWidth * 3 - (xOffsetsLoc[0] * 3)) / 24) * 24;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth * 3 - (aligLen + xOffsetsLoc[c] * 3);
                        if(offset < srcDescPtr->strides.nStride)
                        {
                            for( int j = 0; j < aligLen; j += vectorIncrement)
                            {
                                __m256 p1[3], p2[3];
                                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp, p1);
                                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTempChn, p2);
                                p1[0] = p2[0];
                                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, p1);
                                dstPtrTemp += vectorIncrement;
                                srcPtrTemp += vectorIncrement;
                                srcPtrTempChn += vectorIncrement;
                            }
                            for(int j = 0; j < remElemLen; j+=3)
                            {
                                *dstPtrTemp = *srcPtrTempChn;
                                dstPtrTemp += 3;
                                srcPtrTempChn += 3;
                            }
                        }
                        dstPtrRow += dstDescPtr->strides.hStride;
                        srcPtrRow += srcDescPtr->strides.hStride;
                        srcPtrRowChn += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
    }
    return RPP_SUCCESS;
}

RppStatus glitch_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *xOffsetR,
                                   Rpp32u *yOffsetR,
                                   Rpp32u *xOffsetG,
                                   Rpp32u *yOffsetG,
                                   Rpp32u *xOffsetB,
                                   Rpp32u *yOffsetB,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u xOffsetRchn = xOffsetR[batchCount];
        Rpp32u yOffsetRchn = yOffsetR[batchCount];
        Rpp32u xOffsetGchn = xOffsetG[batchCount];
        Rpp32u yOffsetGchn = yOffsetG[batchCount];
        Rpp32u xOffsetBchn = xOffsetB[batchCount];
        Rpp32u yOffsetBchn = yOffsetB[batchCount];

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsets[3] = {yOffsetRchn, yOffsetGchn, yOffsetBchn};
        Rpp32u xOffsetsLoc[3] = {xOffsetRchn, xOffsetGchn, xOffsetBchn};
        Rpp32u yOffsetsLoc[3] = {yOffsetRchn * elementsInRowMax, yOffsetGchn * elementsInRowMax, yOffsetBchn * elementsInRowMax};

        Rpp16f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp16f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u vectorIncrement = 8;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            convert_float_pkd3_pln3<Rpp16f>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth * 3);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp16f *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp16f *srcPtrTemp = srcPtrImageTemp;
                        Rpp16f *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < dstDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256 p;
                                Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                {
                                    srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];
                                }
                                p = _mm256_loadu_ps(srcPtrTemp_ps);
                                _mm256_storeu_ps(dstPtrTemp_ps, p);
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                {
                                    dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
                                }
                                srcPtrTemp += 8;
                                dstPtrTemp += 8;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += dstDescPtr->strides.hStride;
                        dstPtrImageTemp += dstDescPtr->strides.hStride;
                        offset += dstDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp16f *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp16f *srcPtrTemp = srcPtrImageTemp;
                        Rpp16f *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                __m256 p;
                                Rpp32f srcPtrTemp_ps[8], dstPtrTemp_ps[8];
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                {
                                    srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];
                                }
                                p = _mm256_loadu_ps(srcPtrTemp_ps);
                                _mm256_storeu_ps(dstPtrTemp_ps, p);
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                {
                                    dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
                                }
                                srcPtrTemp += 8;
                                dstPtrTemp += 8;
                             }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
            convert_float_pln3_pkd3<Rpp16f>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr, roi.xywhROI.roiWidth);
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp16f) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp16f *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + offset;
                dstPtrImageTemp = dstPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = yOffsets[c]; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp16f *srcPtrTemp = srcPtrImageTemp;
                        Rpp16f *dstPtrTemp = dstPtrImageTemp;
                        Rpp32u aligLen = ((roi.xywhROI.roiWidth - xOffsetsLoc[c]) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth - ( aligLen + xOffsetsLoc[c]);
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < aligLen; i += vectorIncrement)
                            {
                                Rpp32f srcPtrTemp_ps[8];
                                Rpp32f dstPtrTemp_ps[8];
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                    srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];
                                __m256 p;
                                p = _mm256_loadu_ps(srcPtrTemp_ps);
                                _mm256_storeu_ps(dstPtrTemp_ps, p);
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                    dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
                                srcPtrTemp += 8;
                                dstPtrTemp += 8;
                            }
                            for(int i = 0; i < remElemLen; i++)
                            {
                                *dstPtrTemp = *srcPtrTemp;
                                dstPtrTemp++;
                                srcPtrTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC ))
        {
            vectorIncrement = 24;
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp16f) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                Rpp16f *srcPtrRow, *srcPtrRowChn, *dstPtrRow;
                srcPtrRow = dstPtrImage + c;
                srcPtrRowChn = srcPtrImage + offset;
                dstPtrRow = dstPtrImage + c;
                if(offset > 0)
                {
                    for( int i = yOffsets[c]; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp16f *srcPtrTemp, *srcPtrTempChn, *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        srcPtrTempChn = srcPtrRowChn;
                        dstPtrTemp = dstPtrRow;
                        int aligLen = ((roi.xywhROI.roiWidth * 3 - (xOffsetsLoc[0] * 3)) / vectorIncrement) * vectorIncrement;
                        Rpp32u remElemLen = roi.xywhROI.roiWidth * 3 - (aligLen + xOffsetsLoc[c] * 3);
                        if(offset < srcDescPtr->strides.nStride)
                        {
                            for( int j = 0; j < aligLen; j += vectorIncrement)
                            {
                                __m256 p1[3], p2[3];
                                Rpp32f srcPtrTemp_ps[24], srcPtrTempChn_ps[24], dstPtrTemp_ps[25];
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                {
                                    srcPtrTemp_ps[cnt] = (Rpp32f)srcPtrTemp[cnt];
                                    srcPtrTempChn_ps[cnt] = (Rpp32f)srcPtrTempChn[cnt];
                                }
                                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTemp_ps, p1);
                                rpp_simd_load(rpp_load24_f32pkd3_to_f32pln3_avx, srcPtrTempChn_ps, p2);
                                p1[0] = p2[0];
                                rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp_ps, p1);
                                for(int cnt = 0; cnt < vectorIncrement; cnt++)
                                {
                                    dstPtrTemp[cnt] = (Rpp16f)dstPtrTemp_ps[cnt];
                                }
                                dstPtrTemp += vectorIncrement;
                                srcPtrTemp += vectorIncrement;
                                srcPtrTempChn += vectorIncrement;
                            }
                            for(int j = 0; j < remElemLen; j+=3)
                            {
                                *dstPtrTemp = *srcPtrTempChn;
                                dstPtrTemp += 3;
                                srcPtrTempChn += 3;
                            }
                        }
                        dstPtrRow += dstDescPtr->strides.hStride;
                        srcPtrRow += srcDescPtr->strides.hStride;
                        srcPtrRowChn += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
    }
    return RPP_SUCCESS;
}
