#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>
#include <cstring>
#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

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
    __m128i maskR = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45);
    __m128i maskG = _mm_setr_epi8(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46);
    __m128i maskB = _mm_setr_epi8(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47);
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
            Rpp8u *srcPtrR, *srcPtrG, *srcPtrB;
            Rpp8u *dstPtr;
            dstPtr = dstPtrImage;
            srcPtrR = srcPtrImage + (xOffsetsLoc[0] * 3) + (yOffsetsLoc[0] * 3);
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                for(int j = xOffsetsLoc[0] * 3; j < alignedLength; j += 16)
                {
                    __m128i p;
                    p = _mm_loadu_epi8((__m128i *)srcPtrR);
                    __m128i px = _mm_shuffle_epi8(p, maskR);
                    _mm_storeu_epi8((__m128i *)dstPtr, px);
                    dstPtr += 16;
                    srcPtrR += 48;
                }
            }
        }
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            for(int c = 0; c < srcDescPtr->c ; c++)
            {
                Rpp32u offset = xOffsetsLoc[c] + yOffsetsLoc[c];
                Rpp32u offsetTemp = offset;
                Rpp8u *srcRowR, *srcRowRTemp;
                srcRowRTemp = srcPtrChannel + (c * srcDescPtr->strides.cStride);
                srcRowR = srcRowRTemp + offset;
                Rpp8u *firstRowTemp = srcRowR;
                Rpp8u *firstRowOut = srcRowRTemp;

                for (int i = 0; i < roi.xywhROI.roiHeight - 1; i++)
                {
                    Rpp8u *rowTemp = srcRowR;
                    Rpp8u *outPutRow = srcRowRTemp;
                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < srcDescPtr->strides.hStride; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        if (offsetTemp < srcDescPtr->strides.cStride)
                        {
                            __m128 p[4];
                            rpp_simd_load(rpp_load16_u8_to_f32, rowTemp, p);
                            rpp_simd_store(rpp_store16_f32_to_u8, outPutRow, p);
                            rowTemp += vectorIncrementPerChannel;
                            outPutRow += vectorIncrementPerChannel;
                            offsetTemp += vectorIncrementPerChannel;
                        }
                    }
                    srcRowR += srcDescPtr->strides.hStride;
                    srcRowRTemp += srcDescPtr->strides.hStride;
                }
            }

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
                        Rpp32u AligLen = ((alignedLength - xOffsetsLoc[c]) / 32) * 32;
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < AligLen; i += 32)
                            {
                                __m256i p;
                                p = _mm256_loadu_epi8((__m256i *)srcTemp);
                                _mm256_storeu_epi8((__m256i *)dstTemp, p);
                                srcTemp += 32;
                                dstTemp += 32;
                                offset += 32;
                            }
                            for(int i = 0; i < (alignedLength - AligLen); i++)
                            {
                                *dstTemp = *srcTemp;
                                dstTemp++;
                                srcTemp++;
                                offset++;
                            }
                            srcTemp+=8;
                        }
                        if(offset < srcDescPtr->strides.cStride)
                        {
                            for(int i = 0; i < xOffsetsLoc[c]; i++){
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
    //     else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW ))
    //     {
    //         Rpp32u offsetR = yOffsetsLoc[0] + xOffsetsLoc[0];
    //         Rpp32u offsetG = yOffsetsLoc[1] + xOffsetsLoc[1];
    //         Rpp32u offsetB = yOffsetsLoc[2] + xOffsetsLoc[2];
    //         Rpp8u *srcPtrImageTempR, *srcPtrImageTempG, *srcPtrImageTempB, *dstPtrImageTempR, *dstPtrImageTempG, *dstPtrImageTempB;
    //         srcPtrImageTempR = srcPtrImage + offsetR;
    //         srcPtrImageTempG = srcPtrImage + (srcDescPtr->w * srcDescPtr->h) + offsetG;
    //         srcPtrImageTempB = srcPtrImage + (2 * srcDescPtr->w * srcDescPtr->h) + offsetB;
    //         dstPtrImageTempR = dstPtrImage;
    //         dstPtrImageTempG = dstPtrImage + (dstDescPtr->w * dstDescPtr->h);
    //         dstPtrImageTempB = dstPtrImage + (2 * dstDescPtr->w * dstDescPtr->h);
    //             for ( int i = 0; i < roi.xywhROI.roiHeight ; i++)
    //             {
    //                 for( int j = 0; j < srcDescPtr->strides.hStride ; j++)
    //                 {
    //                     if( offsetR < srcDescPtr->strides.cStride)
    //                     {
    //                         *dstPtrImageTempR = *srcPtrImageTempR;
    //                         dstPtrImageTempR++;
    //                         srcPtrImageTempR++;
    //                         offsetR++;
    //                     }
    //                     if( offsetG < srcDescPtr->strides.cStride)
    //                     {
    //                         *dstPtrImageTempG = *srcPtrImageTempG;
    //                         dstPtrImageTempG++;
    //                         srcPtrImageTempG++;
    //                         offsetG++;
    //                     }
    //                     if( offsetB < srcDescPtr->strides.cStride)
    //                     {
    //                         *dstPtrImageTempB = *srcPtrImageTempB;
    //                         dstPtrImageTempB++;
    //                         srcPtrImageTempB++;
    //                         offsetB++;
    //                     }
    //                 }
    //             }
    // }
    else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC ))
    {
        for(int c = 0 ; c < 3 ; c++)
        {
            Rpp32u offset = yOffsetsLoc[c] *3 + xOffsetsLoc[c] * 3;
            Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
            srcPtrImageTemp = srcPtrImage + c + yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
            dstPtrImageTemp = dstPtrImage + c;
            offset = yOffsetsLoc[c] * 3 + xOffsetsLoc[c] * 3;
            if (offset > 0)
            {
                for ( int i = 0; i < roi.xywhROI.roiHeight ; i++)
                {
                    for( int j = 0; j < srcDescPtr->strides.hStride ; j++)
                    {
                        if( offset < srcDescPtr->strides.nStride)
                        {
                            *dstPtrImageTemp = *srcPtrImageTemp;
                            dstPtrImageTemp += 3;
                            srcPtrImageTemp += 3;
                            offset += 3;
                        }
                    }
                }
            }
        }
    }
    }
    return RPP_SUCCESS;
}