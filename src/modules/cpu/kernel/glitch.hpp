#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>
#include <cstring>
#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

template <typename T> void convert_Pln3_Pkd3( T *srcPtrChannel, T *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
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

template <typename T> void convert_Pkd3_Pln3( T *srcPtrChannel, T *dstPtrChannel, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
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
            convert_Pkd3_Pln3<Rpp8u>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = 0; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8u *srcTemp = srcPtrImageTemp;
                        Rpp8u *dstTemp = dstPtrImageTemp;
                        Rpp32u AligLen = ((alignedLengthPerChannel - xOffsetsLoc[c]) / 32) * 32;
                        if(offset < dstDescPtr->strides.cStride)
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
            convert_Pln3_Pkd3<Rpp8u>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr);
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
                Rpp8u *dstTemp;
                dstTemp = (Rpp8u*)malloc(52 * sizeof(Rpp8u));
                srcPtrRow = dstPtrImage + c;
                srcPtrRowChn = srcPtrImage + c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
                dstPtrRow = dstPtrImage + c;
                if(offset > 2)
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
                                p1[0] = p2[0];
                                p1[1] = p2[1];
                                rpp_simd_store(rpp_store48_f32pln3_to_u8pkd3_avx, dstTemp, p1);
                                for(int i = 0; i < 3 ; i++)
                                {
                                    __m128i p;
                                    p = _mm_loadu_si128((__m128i *)dstTemp);
                                    _mm_storeu_si128((__m128i *)dstPtrTemp + (i* 16) , p); 
                                    dstTemp += 16;
                                }
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

RppStatus glitch_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
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

        Rpp8s *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp8s *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp32u alignedLength = (bufferLength / 32) * 32;
        Rpp32u alignedLengthPerChannel = (srcDescPtr->w / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            //convert_Pkd3_Pln3<Rpp8s>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8s *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = 0; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8s *srcTemp = srcPtrImageTemp;
                        Rpp8s *dstTemp = dstPtrImageTemp;
                        Rpp32u AligLen = ((alignedLengthPerChannel - xOffsetsLoc[c]) / 32) * 32;
                        if(offset < dstDescPtr->strides.cStride)
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
                Rpp8s *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = 0; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8s *srcTemp = srcPtrImageTemp;
                        Rpp8s *dstTemp = dstPtrImageTemp;
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
            //convert_Pln3_Pkd3<Rpp8s>(srcPtrChannel, dstPtrChannel, srcDescPtr, dstDescPtr);
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
            for(int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp8s *srcPtrImageTemp, *dstPtrImageTemp;
                srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->strides.cStride) + yOffsetsLoc[c] + xOffsetsLoc[c];
                dstPtrImageTemp = dstPtrImage + (c * srcDescPtr->strides.cStride);
                if(offset > 0)
                {
                    for(int j = 0; j < roi.xywhROI.roiHeight; j++)
                    {
                        Rpp8s *srcTemp = srcPtrImageTemp;
                        Rpp8s *dstTemp = dstPtrImageTemp;
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
    }
    return RPP_SUCCESS;
}