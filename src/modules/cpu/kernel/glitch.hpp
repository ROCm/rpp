#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>
#include <cstring>
#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void rpp_store_f32pln3_to_u8pln3_avx(Rpp8u *dstPtrR, Rpp8u *dstPtrG, Rpp8u *dstPtrB, __m256 *p, __m256 *p1, __m256 *p2)
{
    __m256i pxCvt;
    __m128i px[4];

    pxCvt = _mm256_cvtps_epi32(p[0]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for R */
    pxCvt = _mm256_cvtps_epi32(p[1]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for R */
    px[0] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for R */
    pxCvt = _mm256_cvtps_epi32(p1[2]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for G */
    pxCvt = _mm256_cvtps_epi32(p1[3]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for G */
    px[1] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for G */
    pxCvt = _mm256_cvtps_epi32(p2[4]);
    px[2] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 0-7 for B */
    pxCvt = _mm256_cvtps_epi32(p2[5]);
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt, 0), _mm256_extracti128_si256(pxCvt, 1));    /* pack pixels 8-15 for B */
    px[2] = _mm_packus_epi16(px[2], px[3]);    /* pack pixels 0-15 for B */
    _mm_storeu_si128((__m128i *)dstPtrR, px[0]);    /* store [R01|R02|R03|R04|R05|R06|R07|R08|R09|R10|R11|R12|R13|R14|R15|R16] */
    _mm_storeu_si128((__m128i *)dstPtrG, px[1]);    /* store [G01|G02|G03|G04|G05|G06|G07|G08|G09|G10|G11|G12|G13|G14|G15|G16] */
    _mm_storeu_si128((__m128i *)dstPtrB, px[2]);    /* store [B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B13|B14|B15|B16] */
}

RppStatus glitch_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *batch_x_offset_r,
                                   Rpp32u *batch_y_offset_r,
                                   Rpp32u *batch_x_offset_g,
                                   Rpp32u *batch_y_offset_g,
                                   Rpp32u *batch_x_offset_b,
                                   Rpp32u *batch_y_offset_b,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams)
{
    __m256i maskR = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,0x80);
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u x_offset_r = batch_x_offset_r[batchCount];
        Rpp32u y_offset_r = batch_y_offset_r[batchCount];
        Rpp32u x_offset_g = batch_x_offset_g[batchCount];
        Rpp32u y_offset_g = batch_y_offset_g[batchCount];
        Rpp32u x_offset_b = batch_x_offset_b[batchCount];
        Rpp32u y_offset_b = batch_y_offset_b[batchCount];

        Rpp32u elementsInRowMax = srcDescPtr->w;

        Rpp32u xOffsets[3] = {
            x_offset_r,
            x_offset_g,
            x_offset_b};

        Rpp32u yOffsets[3] = {
            y_offset_r,
            y_offset_g,
            y_offset_b};

        Rpp32u xOffsetsLoc[3] = {
            x_offset_r,
            x_offset_g,
            x_offset_b};

        Rpp32u yOffsetsLoc[3] = {
            y_offset_r * elementsInRowMax,
            y_offset_g * elementsInRowMax,
            y_offset_b * elementsInRowMax};

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32u alignedLength = (bufferLength / 32) * 32;
        Rpp32u alignedLengthPerChannel = (srcDescPtr->w / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            for(int c = 0; c < 3; c++)
            {
                Rpp8u *srcPtr;
                Rpp8u *dstPtr;
                Rpp32u offset;
                offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                srcPtr = srcPtrImage + c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                dstPtr = dstPtrImage + (c * srcDescPtr->strides.cStride);
                int alignedLen = ((alignedLength - (xOffsetsLoc[0] * 3))/33)*33;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp8u *dstPtrTemp;
                    srcPtrTemp = srcPtr;
                    dstPtrTemp = dstPtr;
                    if(offset < srcDescPtr->strides.nStride)
                    {
                        for(int j = 0; j < alignedLen; j += 32)
                        {
                            __m256i p = _mm256_loadu_epi8(srcPtrTemp);
                            __m256i px = _mm256_shuffle_epi8(p, maskR);
                            _mm256_storeu_epi8(dstPtrTemp, px); 
                            srcPtrTemp += 33;
                            dstPtrTemp += 11;
                            offset += 33;
                        }
                        for(int j = 0; j < (alignedLength - alignedLen); j+=3)
                        {
                            *dstPtrTemp = *srcPtrTemp;
                            dstPtrTemp++;
                            srcPtrTemp += 3;
                            offset += 3;
                        }
                    srcPtrTemp += 24;
                    }
                    if(offset < srcDescPtr->strides.nStride)
                    {
                        for(int j = 0; j < xOffsetsLoc[0] * 3; j += 3){
                            *dstPtrTemp = *srcPtrTemp;
                            dstPtrTemp++;
                            srcPtrTemp += 3;
                            offset += 3;
                        }
                    }
                    srcPtr += srcDescPtr->strides.hStride;
                    dstPtr += srcDescPtr->strides.hStride;
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel + yOffsetsLoc[0] + xOffsetsLoc[0];
            srcPtrRowG = srcPtrChannel + srcDescPtr->strides.cStride + yOffsetsLoc[1] + xOffsetsLoc[1];
            srcPtrRowB = srcPtrChannel + 2 * srcDescPtr->strides.cStride + yOffsetsLoc[2] + xOffsetsLoc[2];
            dstPtrRow = dstPtrChannel;

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
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);    // simd stores
                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for(int i = 0; i < (srcDescPtr->strides.hStride - alignedLength) / 3; i++)
                {
                    *dstPtrTemp = *srcPtrTempR;
                    *(dstPtrTemp + 1) = *srcPtrTempG;
                    *(dstPtrTemp + 2) = *srcPtrTempB;
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
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = dstPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = 0; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8u *srcTemp = srcPtrImageTemp;
                        Rpp8u *dstTemp = dstPtrImageTemp;
                        Rpp32u AligLen = ((alignedLengthPerChannel - xOffsetsLoc[c]) / 32) * 32;
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < AligLen; i += 32)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8(srcTemp);
                                _mm256_storeu_epi8(dstTemp, p);
                                srcTemp += 32;
                                dstTemp += 32;
                                offset += 32;
                            }
                            for(int i = 0; i < (alignedLengthPerChannel - ( AligLen + xOffsetsLoc[c])); i++)
                            {
                                *dstTemp = *srcTemp;
                                dstTemp++;
                                srcTemp++;
                                offset++;
                            }
                            srcTemp += 8;
                        }
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; xOffsetsLoc[c] / 32; i ++)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8((__m256i *)srcTemp);
                                _mm256_storeu_epi8((__m256i *)dstTemp, p);
                                srcTemp += 32;
                                dstTemp += 32;
                                offset += 32;
                            }
                            for(int i = 0; i < xOffsetsLoc[c] % 3; i++){
                                *dstTemp = *srcTemp;
                                dstTemp++;
                                srcTemp++;
                                offset++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                    }
                }
            }
        }
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC ))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                Rpp32u offsetR =  yOffsetsLoc[0] * 3 + xOffsetsLoc[0] * 3;
                Rpp32u offsetG = 1 + yOffsetsLoc[1] * 3 + xOffsetsLoc[1] * 3;
                Rpp32u offsetB = 2 + yOffsetsLoc[2] * 3 + xOffsetsLoc[2] * 3;
                srcPtrTempR = srcPtrRow + yOffsetsLoc[0] * 3 + xOffsetsLoc[0] * 3;
                srcPtrTempG = srcPtrRow + 1 + yOffsetsLoc[1] * 3 + xOffsetsLoc[1] * 3;
                srcPtrTempB = srcPtrRow + 2 + yOffsetsLoc[2] * 3 + xOffsetsLoc[2] * 3;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    if(offsetR < srcDescPtr->strides.nStride && offsetG < srcDescPtr->strides.nStride && offsetB < srcDescPtr->strides.nStride)
                    {
                        __m256 p[6], p1[6], p2[6];
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempR, p);
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempG, p1);
                        rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempB, p2);
                        rpp_store_f32pln3_to_u8pln3_avx(dstPtrTempR, dstPtrTempG, dstPtrTempB, p, p1, p2);
                        srcPtrTempR += vectorIncrement;
                        srcPtrTempG += vectorIncrement;
                        srcPtrTempB += vectorIncrement;
                        dstPtrTempR += vectorIncrementPerChannel;
                        dstPtrTempG += vectorIncrementPerChannel;
                        dstPtrTempB += vectorIncrementPerChannel;
                    }
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }
    return RPP_SUCCESS;
}