#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>
#include <cstring>
#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void rpp_store_f32pln3_to_u8pkd3_avx(Rpp8u *dstPtr, __m256 *p1, __m256 *p2)
{
    __m256i pxCvt[3];
    __m128i px[5];
    __m128i pxMask = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14, 15);

    pxCvt[0] = _mm256_cvtps_epi32(p2[0]);    /* convert to int32 for R01-08 */
    pxCvt[1] = _mm256_cvtps_epi32(p1[2]);    /* convert to int32 for G01-08 */
    pxCvt[2] = _mm256_cvtps_epi32(p1[4]);    /* convert to int32 for B01-08 */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 0), _mm256_extracti128_si256(pxCvt[1], 0));    /* pack pixels 0-7 as R01-04|G01-04 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 0), xmm_px0);    /* pack pixels 8-15 as B01-04|X01-04 */
    px[0] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R01|R02|R03|R04|G01|G02|G03|G04|B01|B02|B03|B04|00|00|00|00] */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 1), _mm256_extracti128_si256(pxCvt[1], 1));    /* pack pixels 0-7 as R05-08|G05-08 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 1), xmm_px0);    /* pack pixels 8-15 as B05-08|X05-08 */
    px[1] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R05|R06|R07|R08|G05|G06|G07|G08|B05|B06|B07|B08|00|00|00|00] */
    pxCvt[0] = _mm256_cvtps_epi32(p2[1]);    /* convert to int32 for R09-16 */
    pxCvt[1] = _mm256_cvtps_epi32(p1[3]);    /* convert to int32 for G09-16 */
    pxCvt[2] = _mm256_cvtps_epi32(p1[5]);    /* convert to int32 for B09-16 */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 0), _mm256_extracti128_si256(pxCvt[1], 0));    /* pack pixels 0-7 as R09-12|G09-12 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 0), xmm_px0);    /* pack pixels 8-15 as B09-12|X09-12 */
    px[2] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R09|R10|R11|R12|G09|G10|G11|G12|B09|B10|B11|B12|00|00|00|00] */
    px[3] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[0], 1), _mm256_extracti128_si256(pxCvt[1], 1));    /* pack pixels 0-7 as R13-16|G13-16 */
    px[4] = _mm_packus_epi32(_mm256_extracti128_si256(pxCvt[2], 1), xmm_px0);    /* pack pixels 8-15 as B13-16|X13-16 */
    px[3] = _mm_packus_epi16(px[3], px[4]);    /* pack pixels 0-15 as [R13|R14|R15|R16|G13|G14|G15|G16|B13|B14|B15|B16|00|00|00|00] */
    px[0] = _mm_shuffle_epi8(px[0], pxMask);    /* shuffle to get [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    px[1] = _mm_shuffle_epi8(px[1], pxMask);    /* shuffle to get [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    px[2] = _mm_shuffle_epi8(px[2], pxMask);    /* shuffle to get [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    px[3] = _mm_shuffle_epi8(px[3], pxMask);    /* shuffle to get [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
    _mm_storeu_si128((__m128i *)dstPtr, px[0]);           /* store [R01|G01|B01|R02|G02|B02|R03|G03|B03|R04|G04|B04|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 12), px[1]);    /* store [R05|G05|B05|R06|G06|B06|R07|G07|B07|R08|G08|B08|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 24), px[2]);    /* store [R09|G09|B09|R10|G10|B10|R11|G11|B11|R12|G12|B12|00|00|00|00] */
    _mm_storeu_si128((__m128i *)(dstPtr + 36), px[3]);    /* store [R13|G13|B13|R14|G14|B14|R15|G15|B15|R16|G16|B16|00|00|00|00] */
}

void convert_Pln3_Pkd3( Rpp8u *srcPtrChannel, Rpp8u *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
    srcPtrRowR = srcPtrChannel;
    srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
    srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
    dstPtrRow = dstPtrChannel;

    for(int i = 0; i < srcDescPtr->h; i++)
    {
        Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
        srcPtrTempR = srcPtrRowR;
        srcPtrTempG = srcPtrRowG;
        srcPtrTempB = srcPtrRowB;
        dstPtrTemp = dstPtrRow;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < 672; vectorLoopCount += 16)
        {
            __m256 p[6];
            rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);
            rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstPtrTemp, p);
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

void convert_Pkd3_Pln3( Rpp8u *srcPtrChannel, Rpp8u *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
    srcPtrRow = srcPtrChannel;
    dstPtrRowR = dstPtrChannel;
    dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
    dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

    for(int i = 0; i < srcDescPtr->h; i++)
    {
        Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
        srcPtrTemp = srcPtrRow;
        dstPtrTempR = dstPtrRowR;
        dstPtrTempG = dstPtrRowG;
        dstPtrTempB = dstPtrRowB;

        int vectorLoopCount = 0;
        for (; vectorLoopCount < 672; vectorLoopCount += 48)
        {
            __m256 p[6];
            rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
            rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);
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
    __m128i mask = _mm_setr_epi8(0 , 3, 6, 9, 12, 15, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
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
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            convert_Pkd3_Pln3(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr);
            for(int c = 0; c < 3; c++)
            {
                Rpp8u *srcPtrRow;
                Rpp8u *dstPtrRow;
                Rpp32u offset;
                offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                srcPtrRow = srcPtrChannel + c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                dstPtrRow = dstPtrChannel + (c * dstDescPtr->strides.cStride);
                int alignedLen = ((alignedLength - (xOffsetsLoc[0] * 3)) / 18) * 18;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp8u *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;
                    if(offset < srcDescPtr->strides.nStride)
                    {
                        for(int j = 0; j < alignedLen; j += 18)
                        {
                            __m128i p = _mm_loadu_si128((__m128i *)srcPtrTemp);
                            __m128i px = _mm_shuffle_epi8(p, mask);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, px); 
                            srcPtrTemp += 18;
                            dstPtrTemp += 6;
                        }
                        for(int j = 0; j < (alignedLength - (alignedLen + xOffsetsLoc[c] * 3)); j+=3)
                        {
                            *dstPtrTemp = *srcPtrTemp;
                            dstPtrTemp++;
                            srcPtrTemp += 3;
                        }
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                    offset += srcDescPtr->strides.hStride;
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride);
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
                            }
                            for(int i = 0; i < (alignedLengthPerChannel - ( AligLen + xOffsetsLoc[c])); i++)
                            {
                                *dstTemp = *srcTemp;
                                dstTemp++;
                                srcTemp++;
                            }
                        }
                        srcPtrImageTemp += srcDescPtr->strides.hStride;
                        dstPtrImageTemp += srcDescPtr->strides.hStride;
                        offset += srcDescPtr->strides.hStride;
                    }
                }
            }
            convert_Pln3_Pkd3(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr);
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
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
                            }
                            for(int i = 0; i < (alignedLengthPerChannel - ( AligLen + xOffsetsLoc[c])); i++)
                            {
                                *dstTemp = *srcTemp;
                                dstTemp++;
                                srcTemp++;
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
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                Rpp8u *srcPtrRow, *srcPtrRowChn, *dstPtrRow;
                srcPtrRow = srcPtrImage + c;
                srcPtrRowChn = srcPtrImage + c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                dstPtrRow = dstPtrImage + c;
                if(offset > 0)
                {
                    for( int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        Rpp8u *srcPtrTemp, *srcPtrTempChn, *dstPtrTemp;
                        srcPtrTemp = srcPtrRow;
                        srcPtrTempChn = srcPtrRowChn;
                        dstPtrTemp = dstPtrRow;
                        int aligLen = ((alignedLength - (xOffsetsLoc[0] * 3)) / 48) * 48;
                        if(offset < srcDescPtr->strides.nStride)
                        {
                            for( int j = 0; j < aligLen; j += 48)
                            {
                                __m256 p1[6], p2[6];
                                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p1);
                                rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTempChn, p2);
                                rpp_store_f32pln3_to_u8pkd3_avx(dstPtrTemp, p1, p2);
                                dstPtrTemp += 48;
                                srcPtrTemp += 48;
                                srcPtrTempChn += 48;
                            }
                            for(int j = 0; j < (alignedLength - (aligLen + xOffsetsLoc[c] * 3)); j+=3)
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