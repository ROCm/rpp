#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <fstream>
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
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u alignedLengthPerChannel = (srcDescPtr->w / 16) * 16;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW ))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;
            for (int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);
                    rpp_simd_store(rpp_store48_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
            for (int c = 0; c < 3; c++)
            {
                Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
                Rpp32u offsetTemp = offset;
                Rpp8u *rowChannel = dstPtrChannel + dstDescPtr->strides.cStride;
                Rpp8u *rowChannelOffset = rowChannel + offset;
                Rpp8u *firstRowTemp = rowChannelOffset;
                Rpp8u *firstRowOut = rowChannel;
                rowChannelOffset += srcDescPtr->w;
                rowChannel += srcDescPtr->w;

                for (int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *rowTemp = rowChannelOffset;
                    Rpp8u *outPutRow = rowChannel;
                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < srcDescPtr->strides.hStride ; vectorLoopCount += vectorIncrementPerChannel)
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
                    rowChannelOffset += srcDescPtr->strides.hStride;
                    rowChannel += srcDescPtr->strides.hStride;
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
        memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
        for(int c = 0; c < 3; c++)
        {
            Rpp32u offset = yOffsetsLoc[c] + xOffsetsLoc[c];
            Rpp8u *srcPtrImageTemp, *dstPtrImageTemp;
            srcPtrImageTemp = srcPtrImage + (c * srcDescPtr->w * srcDescPtr->h) + yOffsetsLoc[c] + xOffsetsLoc[c];
            dstPtrImageTemp = dstPtrImage + (c * dstDescPtr->w * dstDescPtr->h);
            if(offset > 0)
            {
                for ( int i = 0; i < roi.xywhROI.roiHeight ; i++)
                {
                    for( int j = 0; j < srcDescPtr->strides.hStride ; j++)
                    {
                        if( offset < srcDescPtr->strides.cStride)
                        {
                            *dstPtrImageTemp = *srcPtrImageTemp;
                            dstPtrImageTemp++;
                            srcPtrImageTemp++;
                            offset++;
                        }
                    }
                }
            }
        }
    }
    else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC ))
    {
        memcpy(dstPtrImage, srcPtrImage, sizeof(Rpp8u) * srcDescPtr->strides.nStride);
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