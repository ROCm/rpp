/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

    //           RICAP output image profile
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-1------|------img-roi-2------|
    // |------img-roi-3------|------img-roi-4------|
    // |------img-roi-3------|------img-roi-4------|
    // |------img-roi-3------|------img-roi-4------|

RppStatus ricap_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *permutedIndices,
                                  RpptROIPtr roiPtrInputCropRegion,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi1, roi2, roi3, roi4;
        RpptROIPtr roiPtrInput1, roiPtrInput2, roiPtrInput3, roiPtrInput4;
        roiPtrInput1 = &roiPtrInputCropRegion[0];
        roiPtrInput2 = &roiPtrInputCropRegion[1];
        roiPtrInput3 = &roiPtrInputCropRegion[2];
        roiPtrInput4 = &roiPtrInputCropRegion[3];
        compute_roi_validation_host(roiPtrInput1, &roi1, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput2, &roi2, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput3, &roi3, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput4, &roi4, &roiDefault, roiType);

        Rpp8u *srcPtrImage1, *srcPtrImage2, *srcPtrImage3, *srcPtrImage4, *dstPtrImage;
        int permutedCount = batchCount * 4;
        srcPtrImage1 = srcPtr + (permutedIndices[permutedCount] * srcDescPtr->strides.nStride);
        srcPtrImage2 = srcPtr + (permutedIndices[permutedCount + 1] * srcDescPtr->strides.nStride);
        srcPtrImage3 = srcPtr + (permutedIndices[permutedCount + 2] * srcDescPtr->strides.nStride);
        srcPtrImage4 = srcPtr + (permutedIndices[permutedCount + 3] * srcDescPtr->strides.nStride);
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength1 = roi1.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength2 = roi2.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength3 = roi3.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength4 = roi4.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel1, *srcPtrChannel2, *srcPtrChannel3, *srcPtrChannel4, *dstPtrChannel;
        srcPtrChannel1 = srcPtrImage1 + (roi1.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi1.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel2 = srcPtrImage2 + (roi2.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi2.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel3 = srcPtrImage3 + (roi3.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi3.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel4 = srcPtrImage4 + (roi4.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi4.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength1 = (bufferLength1 / 48) * 48;
        Rpp32u alignedLength2 = (bufferLength2 / 48) * 48;
        Rpp32u alignedLength3 = (bufferLength3 / 48) * 48;
        Rpp32u alignedLength4 = (bufferLength4 / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Ricap with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow1 = srcPtrChannel1;
            srcPtrRow2 = srcPtrChannel2;
            srcPtrRow3 = srcPtrChannel3;
            srcPtrRow4 = srcPtrChannel4;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp1, *srcPtrTemp2, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp1 = srcPtrRow1;
                srcPtrTemp2 = srcPtrRow2;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp1, p);                             // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1 += 3)
                {
                    *dstPtrTempR = srcPtrTemp1[0];
                    *dstPtrTempG = srcPtrTemp1[1];
                    *dstPtrTempB = srcPtrTemp1[2];
                    srcPtrTemp1 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp2, p);                             // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2 += 3)
                {
                    *dstPtrTempR = srcPtrTemp2[0];
                    *dstPtrTempG = srcPtrTemp2[1];
                    *dstPtrTempB = srcPtrTemp2[2];
                    srcPtrTemp2 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow1 += srcDescPtr->strides.hStride;
                srcPtrRow2 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp3, *srcPtrTemp4, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp3 = srcPtrRow3;
                srcPtrTemp4 = srcPtrRow4;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp3, p);                             // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp3 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3 += 3)
                {
                    *dstPtrTempR = srcPtrTemp3[0];
                    *dstPtrTempG = srcPtrTemp3[1];
                    *dstPtrTempB = srcPtrTemp3[2];
                    srcPtrTemp3 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_u8pkd3_to_u8pln3, srcPtrTemp4, p);                             // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp4 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4 += 3)
                {
                    *dstPtrTempR = srcPtrTemp4[0];
                    *dstPtrTempG = srcPtrTemp4[1];
                    *dstPtrTempB = srcPtrTemp4[2];
                    srcPtrTemp4 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow3 += srcDescPtr->strides.hStride;
                srcPtrRow4 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Ricap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR1, *srcPtrRowG1, *srcPtrRowB1, *srcPtrRowR2, *srcPtrRowG2, *srcPtrRowB2;
            Rpp8u *srcPtrRowR3, *srcPtrRowG3, *srcPtrRowB3, *srcPtrRowR4, *srcPtrRowG4, *srcPtrRowB4, *dstPtrRow;
            srcPtrRowR1 = srcPtrChannel1;
            srcPtrRowG1 = srcPtrRowR1 + srcDescPtr->strides.cStride;
            srcPtrRowB1 = srcPtrRowG1 + srcDescPtr->strides.cStride;
            srcPtrRowR2 = srcPtrChannel2;
            srcPtrRowG2 = srcPtrRowR2 + srcDescPtr->strides.cStride;
            srcPtrRowB2 = srcPtrRowG2 + srcDescPtr->strides.cStride;
            srcPtrRowR3 = srcPtrChannel3;
            srcPtrRowG3 = srcPtrRowR3 + srcDescPtr->strides.cStride;
            srcPtrRowB3 = srcPtrRowG3 + srcDescPtr->strides.cStride;
            srcPtrRowR4 = srcPtrChannel4;
            srcPtrRowG4 = srcPtrRowR4 + srcDescPtr->strides.cStride;
            srcPtrRowB4 = srcPtrRowG4 + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR1, *srcPtrTempG1, *srcPtrTempB1, *srcPtrTempR2, *srcPtrTempG2, *srcPtrTempB2, *dstPtrTemp;
                srcPtrTempR1 = srcPtrRowR1;
                srcPtrTempG1 = srcPtrRowG1;
                srcPtrTempB1 = srcPtrRowB1;
                srcPtrTempR2 = srcPtrRowR2;
                srcPtrTempG2 = srcPtrRowG2;
                srcPtrTempB2 = srcPtrRowB2;

                dstPtrTemp = dstPtrRow;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTempR1, srcPtrTempG1, srcPtrTempB1, px); // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR1 += vectorIncrementPerChannel;
                    srcPtrTempG1 += vectorIncrementPerChannel;
                    srcPtrTempB1 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1++)
                {
                    dstPtrTemp[0] = *srcPtrTempR1;
                    dstPtrTemp[1] = *srcPtrTempG1;
                    dstPtrTemp[2] = *srcPtrTempB1;
                    srcPtrTempR1++;
                    srcPtrTempG1++;
                    srcPtrTempB1++;
                    dstPtrTemp += 3;
                }

                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTempR2, srcPtrTempG2, srcPtrTempB2, px); // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR2 += vectorIncrementPerChannel;
                    srcPtrTempG2 += vectorIncrementPerChannel;
                    srcPtrTempB2 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2++)
                {
                    dstPtrTemp[0] = *srcPtrTempR2;
                    dstPtrTemp[1] = *srcPtrTempG2;
                    dstPtrTemp[2] = *srcPtrTempB2;
                    srcPtrTempR2++;
                    srcPtrTempG2++;
                    srcPtrTempB2++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR1 += srcDescPtr->strides.hStride;
                srcPtrRowG1 += srcDescPtr->strides.hStride;
                srcPtrRowB1 += srcDescPtr->strides.hStride;
                srcPtrRowR2 += srcDescPtr->strides.hStride;
                srcPtrRowG2 += srcDescPtr->strides.hStride;
                srcPtrRowB2 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR3, *srcPtrTempG3, *srcPtrTempB3, *srcPtrTempR4, *srcPtrTempG4, *srcPtrTempB4, *dstPtrTemp;
                srcPtrTempR3 = srcPtrRowR3;
                srcPtrTempG3 = srcPtrRowG3;
                srcPtrTempB3 = srcPtrRowB3;
                srcPtrTempR4 = srcPtrRowR4;
                srcPtrTempG4 = srcPtrRowG4;
                srcPtrTempB4 = srcPtrRowB4;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTempR3, srcPtrTempG3, srcPtrTempB3, px); // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR3 += vectorIncrementPerChannel;
                    srcPtrTempG3 += vectorIncrementPerChannel;
                    srcPtrTempB3 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3++)
                {
                    dstPtrTemp[0] = *srcPtrTempR3;
                    dstPtrTemp[1] = *srcPtrTempG3;
                    dstPtrTemp[2] = *srcPtrTempB3;
                    srcPtrTempR3++;
                    srcPtrTempG3++;
                    srcPtrTempB3++;
                    dstPtrTemp += 3;
                }

                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTempR4, srcPtrTempG4, srcPtrTempB4, px); // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR4 += vectorIncrementPerChannel;
                    srcPtrTempG4 += vectorIncrementPerChannel;
                    srcPtrTempB4 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4++)
                {
                    dstPtrTemp[0] = *srcPtrTempR4;
                    dstPtrTemp[1] = *srcPtrTempG4;
                    dstPtrTemp[2] = *srcPtrTempB4;
                    srcPtrTempR4++;
                    srcPtrTempG4++;
                    srcPtrTempB4++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR3 += srcDescPtr->strides.hStride;
                srcPtrRowG3 += srcDescPtr->strides.hStride;
                srcPtrRowB3 += srcDescPtr->strides.hStride;
                srcPtrRowR4 += srcDescPtr->strides.hStride;
                srcPtrRowG4 += srcDescPtr->strides.hStride;
                srcPtrRowB4 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Ricap without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            for (int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRow;
                srcPtrRow1 = srcPtrChannel1;
                srcPtrRow2 = srcPtrChannel2;
                srcPtrRow3 = srcPtrChannel3;
                srcPtrRow4 = srcPtrChannel4;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp1, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp1 = srcPtrRow1;
                    srcPtrTemp2 = srcPtrRow2;
                    dstPtrTemp = dstPtrRow;

                    memcpy(dstPtrTemp, srcPtrTemp1, bufferLength1);
                    dstPtrTemp += bufferLength1;
                    memcpy(dstPtrTemp, srcPtrTemp2, bufferLength2);
                    srcPtrRow1 += srcDescPtr->strides.hStride;
                    srcPtrRow2 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp3, *srcPtrTemp4, *dstPtrTemp;
                    srcPtrTemp3 = srcPtrRow3;
                    srcPtrTemp4 = srcPtrRow4;
                    dstPtrTemp = dstPtrRow;
                    memcpy(dstPtrTemp, srcPtrTemp3, bufferLength3);
                    dstPtrTemp += bufferLength3;
                    memcpy(dstPtrTemp, srcPtrTemp4, bufferLength4);

                    srcPtrRow3 += srcDescPtr->strides.hStride;
                    srcPtrRow4 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel1 += srcDescPtr->strides.cStride;
                srcPtrChannel2 += srcDescPtr->strides.cStride;
                srcPtrChannel3 += srcDescPtr->strides.cStride;
                srcPtrChannel4 += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus ricap_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u *permutedIndices,
                                    RpptROIPtr roiPtrInputCropRegion,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi1, roi2, roi3, roi4;
        RpptROIPtr roiPtrInput1, roiPtrInput2, roiPtrInput3, roiPtrInput4;
        roiPtrInput1 = &roiPtrInputCropRegion[0];
        roiPtrInput2 = &roiPtrInputCropRegion[1];
        roiPtrInput3 = &roiPtrInputCropRegion[2];
        roiPtrInput4 = &roiPtrInputCropRegion[3];
        compute_roi_validation_host(roiPtrInput1, &roi1, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput2, &roi2, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput3, &roi3, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput4, &roi4, &roiDefault, roiType);

        Rpp32f *srcPtrImage1, *srcPtrImage2, *srcPtrImage3, *srcPtrImage4, *dstPtrImage;
        int permutedCount = batchCount * 4;
        srcPtrImage1 = srcPtr + (permutedIndices[permutedCount] * srcDescPtr->strides.nStride);
        srcPtrImage2 = srcPtr + (permutedIndices[permutedCount + 1] * srcDescPtr->strides.nStride);
        srcPtrImage3 = srcPtr + (permutedIndices[permutedCount + 2] * srcDescPtr->strides.nStride);
        srcPtrImage4 = srcPtr + (permutedIndices[permutedCount + 3] * srcDescPtr->strides.nStride);
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength1 = roi1.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength2 = roi2.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength3 = roi3.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength4 = roi4.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel1, *srcPtrChannel2, *srcPtrChannel3, *srcPtrChannel4, *dstPtrChannel;
        srcPtrChannel1 = srcPtrImage1 + (roi1.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi1.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel2 = srcPtrImage2 + (roi2.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi2.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel3 = srcPtrImage3 + (roi3.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi3.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel4 = srcPtrImage4 + (roi4.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi4.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength1 = (bufferLength1 / 12) * 12;
        Rpp32u alignedLength2 = (bufferLength2 / 12) * 12;
        Rpp32u alignedLength3 = (bufferLength3 / 12) * 12;
        Rpp32u alignedLength4 = (bufferLength4 / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        // Ricap with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow1 = srcPtrChannel1;
            srcPtrRow2 = srcPtrChannel2;
            srcPtrRow3 = srcPtrChannel3;
            srcPtrRow4 = srcPtrChannel4;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp1, *srcPtrTemp2, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp1 = srcPtrRow1;
                srcPtrTemp2 = srcPtrRow2;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrement)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp1, p);                             // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1 += 3)
                {
                    *dstPtrTempR = srcPtrTemp1[0];
                    *dstPtrTempG = srcPtrTemp1[1];
                    *dstPtrTempB = srcPtrTemp1[2];
                    srcPtrTemp1 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrement)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp2, p);                             // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2 += 3)
                {
                    *dstPtrTempR = srcPtrTemp2[0];
                    *dstPtrTempG = srcPtrTemp2[1];
                    *dstPtrTempB = srcPtrTemp2[2];
                    srcPtrTemp2 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow1 += srcDescPtr->strides.hStride;
                srcPtrRow2 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTemp3, *srcPtrTemp4, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp3 = srcPtrRow3;
                srcPtrTemp4 = srcPtrRow4;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrement)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp3, p);                             // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp3 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3 += 3)
                {
                    *dstPtrTempR = srcPtrTemp3[0];
                    *dstPtrTempG = srcPtrTemp3[1];
                    *dstPtrTempB = srcPtrTemp3[2];
                    srcPtrTemp3 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrement)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp4, p);                             // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp4 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4 += 3)
                {
                    *dstPtrTempR = srcPtrTemp4[0];
                    *dstPtrTempG = srcPtrTemp4[1];
                    *dstPtrTempB = srcPtrTemp4[2];
                    srcPtrTemp4 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow3 += srcDescPtr->strides.hStride;
                srcPtrRow4 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Ricap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcPtrRowR1, *srcPtrRowG1, *srcPtrRowB1, *srcPtrRowR2, *srcPtrRowG2, *srcPtrRowB2, *srcPtrRowR3, *srcPtrRowG3, *srcPtrRowB3, *srcPtrRowR4, *srcPtrRowG4, *srcPtrRowB4, *dstPtrRow;
            srcPtrRowR1 = srcPtrChannel1;
            srcPtrRowG1 = srcPtrRowR1 + srcDescPtr->strides.cStride;
            srcPtrRowB1 = srcPtrRowG1 + srcDescPtr->strides.cStride;
            srcPtrRowR2 = srcPtrChannel2;
            srcPtrRowG2 = srcPtrRowR2 + srcDescPtr->strides.cStride;
            srcPtrRowB2 = srcPtrRowG2 + srcDescPtr->strides.cStride;
            srcPtrRowR3 = srcPtrChannel3;
            srcPtrRowG3 = srcPtrRowR3 + srcDescPtr->strides.cStride;
            srcPtrRowB3 = srcPtrRowG3 + srcDescPtr->strides.cStride;
            srcPtrRowR4 = srcPtrChannel4;
            srcPtrRowG4 = srcPtrRowR4 + srcDescPtr->strides.cStride;
            srcPtrRowB4 = srcPtrRowG4 + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR1, *srcPtrTempG1, *srcPtrTempB1, *srcPtrTempR2, *srcPtrTempG2, *srcPtrTempB2, *dstPtrTemp;
                srcPtrTempR1 = srcPtrRowR1;
                srcPtrTempG1 = srcPtrRowG1;
                srcPtrTempB1 = srcPtrRowB1;
                srcPtrTempR2 = srcPtrRowR2;
                srcPtrTempG2 = srcPtrRowG2;
                srcPtrTempB2 = srcPtrRowB2;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrementPerChannel)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR1, srcPtrTempG1, srcPtrTempB1, p); // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                             // simd stores
                    srcPtrTempR1 += vectorIncrementPerChannel;
                    srcPtrTempG1 += vectorIncrementPerChannel;
                    srcPtrTempB1 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1++)
                {
                    dstPtrTemp[0] = *srcPtrTempR1;
                    dstPtrTemp[1] = *srcPtrTempG1;
                    dstPtrTemp[2] = *srcPtrTempB1;
                    srcPtrTempR1++;
                    srcPtrTempG1++;
                    srcPtrTempB1++;
                    dstPtrTemp += 3;
                }
                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrementPerChannel)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR2, srcPtrTempG2, srcPtrTempB2, p); // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                             // simd stores
                    srcPtrTempR2 += vectorIncrementPerChannel;
                    srcPtrTempG2 += vectorIncrementPerChannel;
                    srcPtrTempB2 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2++)
                {
                    dstPtrTemp[0] = *srcPtrTempR2;
                    dstPtrTemp[1] = *srcPtrTempG2;
                    dstPtrTemp[2] = *srcPtrTempB2;
                    srcPtrTempR2++;
                    srcPtrTempG2++;
                    srcPtrTempB2++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR1 += srcDescPtr->strides.hStride;
                srcPtrRowG1 += srcDescPtr->strides.hStride;
                srcPtrRowB1 += srcDescPtr->strides.hStride;
                srcPtrRowR2 += srcDescPtr->strides.hStride;
                srcPtrRowG2 += srcDescPtr->strides.hStride;
                srcPtrRowB2 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp32f *srcPtrTempR3, *srcPtrTempG3, *srcPtrTempB3, *srcPtrTempR4, *srcPtrTempG4, *srcPtrTempB4, *dstPtrTemp;
                srcPtrTempR3 = srcPtrRowR3;
                srcPtrTempG3 = srcPtrRowG3;
                srcPtrTempB3 = srcPtrRowB3;
                srcPtrTempR4 = srcPtrRowR4;
                srcPtrTempG4 = srcPtrRowG4;
                srcPtrTempB4 = srcPtrRowB4;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrementPerChannel)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR3, srcPtrTempG3, srcPtrTempB3, p); // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                             // simd stores
                    srcPtrTempR3 += vectorIncrementPerChannel;
                    srcPtrTempG3 += vectorIncrementPerChannel;
                    srcPtrTempB3 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3++)
                {
                    dstPtrTemp[0] = *srcPtrTempR3;
                    dstPtrTemp[1] = *srcPtrTempG3;
                    dstPtrTemp[2] = *srcPtrTempB3;
                    srcPtrTempR3++;
                    srcPtrTempG3++;
                    srcPtrTempB3++;
                    dstPtrTemp += 3;
                }
                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrementPerChannel)
                {
                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTempR4, srcPtrTempG4, srcPtrTempB4, p); // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, p);                             // simd stores
                    srcPtrTempR4 += vectorIncrementPerChannel;
                    srcPtrTempG4 += vectorIncrementPerChannel;
                    srcPtrTempB4 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4++)
                {
                    dstPtrTemp[0] = *srcPtrTempR4;
                    dstPtrTemp[1] = *srcPtrTempG4;
                    dstPtrTemp[2] = *srcPtrTempB4;
                    srcPtrTempR4++;
                    srcPtrTempG4++;
                    srcPtrTempB4++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR3 += srcDescPtr->strides.hStride;
                srcPtrRowG3 += srcDescPtr->strides.hStride;
                srcPtrRowB3 += srcDescPtr->strides.hStride;
                srcPtrRowR4 += srcDescPtr->strides.hStride;
                srcPtrRowG4 += srcDescPtr->strides.hStride;
                srcPtrRowB4 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Ricap without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u copyLengthInBytes1 = (bufferLength1) * sizeof(Rpp32f);
            Rpp32u copyLengthInBytes2 = (bufferLength2) * sizeof(Rpp32f);
            Rpp32u copyLengthInBytes3 = (bufferLength3) * sizeof(Rpp32f);
            Rpp32u copyLengthInBytes4 = (bufferLength4) * sizeof(Rpp32f);

            for (int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp32f *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRow;
                srcPtrRow1 = srcPtrChannel1;
                srcPtrRow2 = srcPtrChannel2;
                srcPtrRow3 = srcPtrChannel3;
                srcPtrRow4 = srcPtrChannel4;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp1, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp1 = srcPtrRow1;
                    srcPtrTemp2 = srcPtrRow2;
                    dstPtrTemp = dstPtrRow;
                    memcpy(dstPtrTemp, srcPtrTemp1, copyLengthInBytes1);
                    dstPtrTemp += bufferLength1;
                    memcpy(dstPtrTemp, srcPtrTemp2, copyLengthInBytes2);

                    srcPtrRow1 += srcDescPtr->strides.hStride;
                    srcPtrRow2 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
                {
                    Rpp32f *srcPtrTemp3, *srcPtrTemp4, *dstPtrTemp;
                    srcPtrTemp3 = srcPtrRow3;
                    srcPtrTemp4 = srcPtrRow4;
                    dstPtrTemp = dstPtrRow;
                    memcpy(dstPtrTemp, srcPtrTemp3, copyLengthInBytes3);
                    dstPtrTemp += bufferLength3;
                    memcpy(dstPtrTemp, srcPtrTemp4, copyLengthInBytes4);

                    srcPtrRow3 += srcDescPtr->strides.hStride;
                    srcPtrRow4 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel1 += srcDescPtr->strides.cStride;
                srcPtrChannel2 += srcDescPtr->strides.cStride;
                srcPtrChannel3 += srcDescPtr->strides.cStride;
                srcPtrChannel4 += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}


RppStatus ricap_f16_f16_host_tensor(Rpp16f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u *permutedIndices,
                                    RpptROIPtr roiPtrInputCropRegion,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi1, roi2, roi3, roi4;
        RpptROIPtr roiPtrInput1, roiPtrInput2, roiPtrInput3, roiPtrInput4;
        roiPtrInput1 = &roiPtrInputCropRegion[0];
        roiPtrInput2 = &roiPtrInputCropRegion[1];
        roiPtrInput3 = &roiPtrInputCropRegion[2];
        roiPtrInput4 = &roiPtrInputCropRegion[3];
        compute_roi_validation_host(roiPtrInput1, &roi1, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput2, &roi2, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput3, &roi3, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput4, &roi4, &roiDefault, roiType);

        Rpp16f *srcPtrImage1, *srcPtrImage2, *srcPtrImage3, *srcPtrImage4, *dstPtrImage;
        int permutedCount = batchCount * 4;
        srcPtrImage1 = srcPtr + (permutedIndices[permutedCount] * srcDescPtr->strides.nStride);
        srcPtrImage2 = srcPtr + (permutedIndices[permutedCount + 1] * srcDescPtr->strides.nStride);
        srcPtrImage3 = srcPtr + (permutedIndices[permutedCount + 2] * srcDescPtr->strides.nStride);
        srcPtrImage4 = srcPtr + (permutedIndices[permutedCount + 3] * srcDescPtr->strides.nStride);
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength1 = roi1.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength2 = roi2.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength3 = roi3.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength4 = roi4.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp16f *srcPtrChannel1, *srcPtrChannel2, *srcPtrChannel3, *srcPtrChannel4, *dstPtrChannel;
        srcPtrChannel1 = srcPtrImage1 + (roi1.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi1.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel2 = srcPtrImage2 + (roi2.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi2.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel3 = srcPtrImage3 + (roi3.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi3.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel4 = srcPtrImage4 + (roi4.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi4.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength1 = (bufferLength1 / 12) * 12;
        Rpp32u alignedLength2 = (bufferLength2 / 12) * 12;
        Rpp32u alignedLength3 = (bufferLength3 / 12) * 12;
        Rpp32u alignedLength4 = (bufferLength4 / 12) * 12;
        Rpp32u vectorIncrement = 12;
        Rpp32u vectorIncrementPerChannel = 4;

        // ricap with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow1 = srcPtrChannel1;
            srcPtrRow2 = srcPtrChannel2;
            srcPtrRow3 = srcPtrChannel3;
            srcPtrRow4 = srcPtrChannel4;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp1, *srcPtrTemp2, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp1 = srcPtrRow1;
                srcPtrTemp2 = srcPtrRow2;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                    for(int cnt = 0; cnt < 12; cnt++)
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp1 + cnt);

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1 += 3)
                {
                    *dstPtrTempR = srcPtrTemp1[0];
                    *dstPtrTempG = srcPtrTemp1[1];
                    *dstPtrTempB = srcPtrTemp1[2];
                    srcPtrTemp1 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                    for(int cnt = 0; cnt < 12; cnt++)
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp2 + cnt);

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2 += 3)
                {
                    *dstPtrTempR = srcPtrTemp2[0];
                    *dstPtrTempG = srcPtrTemp2[1];
                    *dstPtrTempB = srcPtrTemp2[2];
                    srcPtrTemp2 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow1 += srcDescPtr->strides.hStride;
                srcPtrRow2 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTemp3, *srcPtrTemp4, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp3 = srcPtrRow3;
                srcPtrTemp4 = srcPtrRow4;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                    for(int cnt = 0; cnt < 12; cnt++)
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp3 + cnt);

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp3 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3 += 3)
                {
                    *dstPtrTempR = srcPtrTemp3[0];
                    *dstPtrTempG = srcPtrTemp3[1];
                    *dstPtrTempB = srcPtrTemp3[2];
                    srcPtrTemp3 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrement)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[12];
                    for(int cnt = 0; cnt < 12; cnt++)
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTemp4 + cnt);

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pkd3_to_f32pln3, srcPtrTemp_ps, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTemp_ps, dstPtrTemp_ps + 4, dstPtrTemp_ps + 8, p);    // simd stores

                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(dstPtrTempR + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);
                        *(dstPtrTempG + cnt) = (Rpp16f) *(dstPtrTemp_ps + 4 + cnt);
                        *(dstPtrTempB + cnt) = (Rpp16f) *(dstPtrTemp_ps + 8 + cnt);
                    }

                    srcPtrTemp4 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4 += 3)
                {
                    *dstPtrTempR = srcPtrTemp4[0];
                    *dstPtrTempG = srcPtrTemp4[1];
                    *dstPtrTempB = srcPtrTemp4[2];
                    srcPtrTemp4 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow3 += srcDescPtr->strides.hStride;
                srcPtrRow4 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Ricap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *srcPtrRowR1, *srcPtrRowG1, *srcPtrRowB1, *srcPtrRowR2, *srcPtrRowG2, *srcPtrRowB2, *srcPtrRowR3, *srcPtrRowG3, *srcPtrRowB3, *srcPtrRowR4, *srcPtrRowG4, *srcPtrRowB4, *dstPtrRow;
            srcPtrRowR1 = srcPtrChannel1;
            srcPtrRowG1 = srcPtrRowR1 + srcDescPtr->strides.cStride;
            srcPtrRowB1 = srcPtrRowG1 + srcDescPtr->strides.cStride;
            srcPtrRowR2 = srcPtrChannel2;
            srcPtrRowG2 = srcPtrRowR2 + srcDescPtr->strides.cStride;
            srcPtrRowB2 = srcPtrRowG2 + srcDescPtr->strides.cStride;
            srcPtrRowR3 = srcPtrChannel3;
            srcPtrRowG3 = srcPtrRowR3 + srcDescPtr->strides.cStride;
            srcPtrRowB3 = srcPtrRowG3 + srcDescPtr->strides.cStride;
            srcPtrRowR4 = srcPtrChannel4;
            srcPtrRowG4 = srcPtrRowR4 + srcDescPtr->strides.cStride;
            srcPtrRowB4 = srcPtrRowG4 + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR1, *srcPtrTempG1, *srcPtrTempB1, *srcPtrTempR2, *srcPtrTempG2, *srcPtrTempB2, *dstPtrTemp;
                srcPtrTempR1 = srcPtrRowR1;
                srcPtrTempG1 = srcPtrRowG1;
                srcPtrTempB1 = srcPtrRowB1;
                srcPtrTempR2 = srcPtrRowR2;
                srcPtrTempG2 = srcPtrRowG2;
                srcPtrTempB2 = srcPtrRowB2;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR1 + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG1 + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB1 + cnt);
                    }

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                    srcPtrTempR1 += vectorIncrementPerChannel;
                    srcPtrTempG1 += vectorIncrementPerChannel;
                    srcPtrTempB1 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1++)
                {
                    dstPtrTemp[0] = *srcPtrTempR1;
                    dstPtrTemp[1] = *srcPtrTempG1;
                    dstPtrTemp[2] = *srcPtrTempB1;
                    srcPtrTempR1++;
                    srcPtrTempG1++;
                    srcPtrTempB1++;
                    dstPtrTemp += 3;
                }

                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR2 + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG2 + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB2 + cnt);
                    }

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                    srcPtrTempR2 += vectorIncrementPerChannel;
                    srcPtrTempG2 += vectorIncrementPerChannel;
                    srcPtrTempB2 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2++)
                {
                    dstPtrTemp[0] = *srcPtrTempR2;
                    dstPtrTemp[1] = *srcPtrTempG2;
                    dstPtrTemp[2] = *srcPtrTempB2;
                    srcPtrTempR2++;
                    srcPtrTempG2++;
                    srcPtrTempB2++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR1 += srcDescPtr->strides.hStride;
                srcPtrRowG1 += srcDescPtr->strides.hStride;
                srcPtrRowB1 += srcDescPtr->strides.hStride;
                srcPtrRowR2 += srcDescPtr->strides.hStride;
                srcPtrRowG2 += srcDescPtr->strides.hStride;
                srcPtrRowB2 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp16f *srcPtrTempR3, *srcPtrTempG3, *srcPtrTempB3, *srcPtrTempR4, *srcPtrTempG4, *srcPtrTempB4, *dstPtrTemp;
                srcPtrTempR3 = srcPtrRowR3;
                srcPtrTempG3 = srcPtrRowG3;
                srcPtrTempB3 = srcPtrRowB3;
                srcPtrTempR4 = srcPtrRowR4;
                srcPtrTempG4 = srcPtrRowG4;
                srcPtrTempB4 = srcPtrRowB4;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR3 + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG3 + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB3 + cnt);
                    }

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                    srcPtrTempR3 += vectorIncrementPerChannel;
                    srcPtrTempG3 += vectorIncrementPerChannel;
                    srcPtrTempB3 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3++)
                {
                    dstPtrTemp[0] = *srcPtrTempR3;
                    dstPtrTemp[1] = *srcPtrTempG3;
                    dstPtrTemp[2] = *srcPtrTempB3;
                    srcPtrTempR3++;
                    srcPtrTempG3++;
                    srcPtrTempB3++;
                    dstPtrTemp += 3;
                }

                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrementPerChannel)
                {
                    Rpp32f srcPtrTemp_ps[12], dstPtrTemp_ps[13];
                    for(int cnt = 0; cnt < 4; cnt++)
                    {
                        *(srcPtrTemp_ps + cnt) = (Rpp32f) *(srcPtrTempR4 + cnt);
                        *(srcPtrTemp_ps + 4 + cnt) = (Rpp32f) *(srcPtrTempG4 + cnt);
                        *(srcPtrTemp_ps + 8 + cnt) = (Rpp32f) *(srcPtrTempB4 + cnt);
                    }

                    __m128 p[4];
                    rpp_simd_load(rpp_load12_f32pln3_to_f32pln3, srcPtrTemp_ps, srcPtrTemp_ps + 4, srcPtrTemp_ps + 8, p);    // simd loads
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp_ps, p);    // simd stores

                    for(int cnt = 0; cnt < 12; cnt++)
                        *(dstPtrTemp + cnt) = (Rpp16f) *(dstPtrTemp_ps + cnt);

                    srcPtrTempR4 += vectorIncrementPerChannel;
                    srcPtrTempG4 += vectorIncrementPerChannel;
                    srcPtrTempB4 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4++)
                {
                    dstPtrTemp[0] = *srcPtrTempR4;
                    dstPtrTemp[1] = *srcPtrTempG4;
                    dstPtrTemp[2] = *srcPtrTempB4;
                    srcPtrTempR4++;
                    srcPtrTempG4++;
                    srcPtrTempB4++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR3 += srcDescPtr->strides.hStride;
                srcPtrRowG3 += srcDescPtr->strides.hStride;
                srcPtrRowB3 += srcDescPtr->strides.hStride;
                srcPtrRowR4 += srcDescPtr->strides.hStride;
                srcPtrRowG4 += srcDescPtr->strides.hStride;
                srcPtrRowB4 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Ricap without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            Rpp32u copyLengthInBytes1 = (bufferLength1) * sizeof(Rpp16f);
            Rpp32u copyLengthInBytes2 = (bufferLength2) * sizeof(Rpp16f);
            Rpp32u copyLengthInBytes3 = (bufferLength3) * sizeof(Rpp16f);
            Rpp32u copyLengthInBytes4 = (bufferLength4) * sizeof(Rpp16f);

            for (int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp16f *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRow;
                srcPtrRow1 = srcPtrChannel1;
                srcPtrRow2 = srcPtrChannel2;
                srcPtrRow3 = srcPtrChannel3;
                srcPtrRow4 = srcPtrChannel4;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp1, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp1 = srcPtrRow1;
                    srcPtrTemp2 = srcPtrRow2;
                    dstPtrTemp = dstPtrRow;

                    memcpy(dstPtrTemp, srcPtrTemp1, copyLengthInBytes1);
                    dstPtrTemp += bufferLength1;
                    memcpy(dstPtrTemp, srcPtrTemp2, copyLengthInBytes2);
                    srcPtrRow1 += srcDescPtr->strides.hStride;
                    srcPtrRow2 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
                {
                    Rpp16f *srcPtrTemp3, *srcPtrTemp4, *dstPtrTemp;
                    srcPtrTemp3 = srcPtrRow3;
                    srcPtrTemp4 = srcPtrRow4;
                    dstPtrTemp = dstPtrRow;
                    memcpy(dstPtrTemp, srcPtrTemp3, copyLengthInBytes3);
                    dstPtrTemp += bufferLength3;
                    memcpy(dstPtrTemp, srcPtrTemp4, copyLengthInBytes4);

                    srcPtrRow3 += srcDescPtr->strides.hStride;
                    srcPtrRow4 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel1 += srcDescPtr->strides.cStride;
                srcPtrChannel2 += srcDescPtr->strides.cStride;
                srcPtrChannel3 += srcDescPtr->strides.cStride;
                srcPtrChannel4 += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus ricap_i8_i8_host_tensor(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *permutedIndices,
                                  RpptROIPtr roiPtrInputCropRegion,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi1, roi2, roi3, roi4;
        RpptROIPtr roiPtrInput1, roiPtrInput2, roiPtrInput3, roiPtrInput4;
        roiPtrInput1 = &roiPtrInputCropRegion[0];
        roiPtrInput2 = &roiPtrInputCropRegion[1];
        roiPtrInput3 = &roiPtrInputCropRegion[2];
        roiPtrInput4 = &roiPtrInputCropRegion[3];
        compute_roi_validation_host(roiPtrInput1, &roi1, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput2, &roi2, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput3, &roi3, &roiDefault, roiType);
        compute_roi_validation_host(roiPtrInput4, &roi4, &roiDefault, roiType);

        Rpp8s *srcPtrImage1, *srcPtrImage2, *srcPtrImage3, *srcPtrImage4, *dstPtrImage;
        int permutedCount = batchCount * 4;
        srcPtrImage1 = srcPtr + (permutedIndices[permutedCount] * srcDescPtr->strides.nStride);
        srcPtrImage2 = srcPtr + (permutedIndices[permutedCount + 1] * srcDescPtr->strides.nStride);
        srcPtrImage3 = srcPtr + (permutedIndices[permutedCount + 2] * srcDescPtr->strides.nStride);
        srcPtrImage4 = srcPtr + (permutedIndices[permutedCount + 3] * srcDescPtr->strides.nStride);
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u bufferLength1 = roi1.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength2 = roi2.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength3 = roi3.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u bufferLength4 = roi4.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8s *srcPtrChannel1, *srcPtrChannel2, *srcPtrChannel3, *srcPtrChannel4, *dstPtrChannel;
        srcPtrChannel1 = srcPtrImage1 + (roi1.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi1.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel2 = srcPtrImage2 + (roi2.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi2.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel3 = srcPtrImage3 + (roi3.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi3.xywhROI.xy.x * layoutParams.bufferMultiplier);
        srcPtrChannel4 = srcPtrImage4 + (roi4.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi4.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength1 = (bufferLength1 / 48) * 48;
        Rpp32u alignedLength2 = (bufferLength2 / 48) * 48;
        Rpp32u alignedLength3 = (bufferLength3 / 48) * 48;
        Rpp32u alignedLength4 = (bufferLength4 / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Ricap with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow1 = srcPtrChannel1;
            srcPtrRow2 = srcPtrChannel2;
            srcPtrRow3 = srcPtrChannel3;
            srcPtrRow4 = srcPtrChannel4;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp1, *srcPtrTemp2, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp1 = srcPtrRow1;
                srcPtrTemp2 = srcPtrRow2;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp1, p);                             // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp1 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1 += 3)
                {
                    *dstPtrTempR = srcPtrTemp1[0];
                    *dstPtrTempG = srcPtrTemp1[1];
                    *dstPtrTempB = srcPtrTemp1[2];
                    srcPtrTemp1 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp2, p);                             // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp2 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2 += 3)
                {
                    *dstPtrTempR = srcPtrTemp2[0];
                    *dstPtrTempG = srcPtrTemp2[1];
                    *dstPtrTempB = srcPtrTemp2[2];
                    srcPtrTemp2 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow1 += srcDescPtr->strides.hStride;
                srcPtrRow2 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTemp3, *srcPtrTemp4, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp3 = srcPtrRow3;
                srcPtrTemp4 = srcPtrRow4;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp3, p);                             // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp3 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3 += 3)
                {
                    *dstPtrTempR = srcPtrTemp3[0];
                    *dstPtrTempG = srcPtrTemp3[1];
                    *dstPtrTempB = srcPtrTemp3[2];
                    srcPtrTemp3 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrement)
                {
                    __m128i p[3];
                    rpp_simd_load(rpp_load48_i8pkd3_to_i8pln3, srcPtrTemp4, p);                             // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, p); // simd stores
                    srcPtrTemp4 += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4 += 3)
                {
                    *dstPtrTempR = srcPtrTemp4[0];
                    *dstPtrTempG = srcPtrTemp4[1];
                    *dstPtrTempB = srcPtrTemp4[2];
                    srcPtrTemp4 += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }

                srcPtrRow3 += srcDescPtr->strides.hStride;
                srcPtrRow4 += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Ricap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcPtrRowR1, *srcPtrRowG1, *srcPtrRowB1, *srcPtrRowR2, *srcPtrRowG2, *srcPtrRowB2, *srcPtrRowR3, *srcPtrRowG3, *srcPtrRowB3, *srcPtrRowR4, *srcPtrRowG4, *srcPtrRowB4, *dstPtrRow;
            srcPtrRowR1 = srcPtrChannel1;
            srcPtrRowG1 = srcPtrRowR1 + srcDescPtr->strides.cStride;
            srcPtrRowB1 = srcPtrRowG1 + srcDescPtr->strides.cStride;
            srcPtrRowR2 = srcPtrChannel2;
            srcPtrRowG2 = srcPtrRowR2 + srcDescPtr->strides.cStride;
            srcPtrRowB2 = srcPtrRowG2 + srcDescPtr->strides.cStride;
            srcPtrRowR3 = srcPtrChannel3;
            srcPtrRowG3 = srcPtrRowR3 + srcDescPtr->strides.cStride;
            srcPtrRowB3 = srcPtrRowG3 + srcDescPtr->strides.cStride;
            srcPtrRowR4 = srcPtrChannel4;
            srcPtrRowG4 = srcPtrRowR4 + srcDescPtr->strides.cStride;
            srcPtrRowB4 = srcPtrRowG4 + srcDescPtr->strides.cStride;

            dstPtrRow = dstPtrChannel;

            for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR1, *srcPtrTempG1, *srcPtrTempB1, *srcPtrTempR2, *srcPtrTempG2, *srcPtrTempB2, *dstPtrTemp;
                srcPtrTempR1 = srcPtrRowR1;
                srcPtrTempG1 = srcPtrRowG1;
                srcPtrTempB1 = srcPtrRowB1;
                srcPtrTempR2 = srcPtrRowR2;
                srcPtrTempG2 = srcPtrRowG2;
                srcPtrTempB2 = srcPtrRowB2;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount1 = 0;
                for (; vectorLoopCount1 < alignedLength1; vectorLoopCount1 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTempR1, srcPtrTempG1, srcPtrTempB1, px); // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR1 += vectorIncrementPerChannel;
                    srcPtrTempG1 += vectorIncrementPerChannel;
                    srcPtrTempB1 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount1 < bufferLength1; vectorLoopCount1++)
                {
                    dstPtrTemp[0] = *srcPtrTempR1;
                    dstPtrTemp[1] = *srcPtrTempG1;
                    dstPtrTemp[2] = *srcPtrTempB1;
                    srcPtrTempR1++;
                    srcPtrTempG1++;
                    srcPtrTempB1++;
                    dstPtrTemp += 3;
                }

                int vectorLoopCount2 = 0;
                for (; vectorLoopCount2 < alignedLength2; vectorLoopCount2 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTempR2, srcPtrTempG2, srcPtrTempB2, px); // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR2 += vectorIncrementPerChannel;
                    srcPtrTempG2 += vectorIncrementPerChannel;
                    srcPtrTempB2 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount2 < bufferLength2; vectorLoopCount2++)
                {
                    dstPtrTemp[0] = *srcPtrTempR2;
                    dstPtrTemp[1] = *srcPtrTempG2;
                    dstPtrTemp[2] = *srcPtrTempB2;
                    srcPtrTempR2++;
                    srcPtrTempG2++;
                    srcPtrTempB2++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR1 += srcDescPtr->strides.hStride;
                srcPtrRowG1 += srcDescPtr->strides.hStride;
                srcPtrRowB1 += srcDescPtr->strides.hStride;
                srcPtrRowR2 += srcDescPtr->strides.hStride;
                srcPtrRowG2 += srcDescPtr->strides.hStride;
                srcPtrRowB2 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }

            for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
            {
                Rpp8s *srcPtrTempR3, *srcPtrTempG3, *srcPtrTempB3, *srcPtrTempR4, *srcPtrTempG4, *srcPtrTempB4, *dstPtrTemp;
                srcPtrTempR3 = srcPtrRowR3;
                srcPtrTempG3 = srcPtrRowG3;
                srcPtrTempB3 = srcPtrRowB3;
                srcPtrTempR4 = srcPtrRowR4;
                srcPtrTempG4 = srcPtrRowG4;
                srcPtrTempB4 = srcPtrRowB4;

                dstPtrTemp = dstPtrRow;

                int vectorLoopCount3 = 0;
                for (; vectorLoopCount3 < alignedLength3; vectorLoopCount3 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTempR3, srcPtrTempG3, srcPtrTempB3, px); // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR3 += vectorIncrementPerChannel;
                    srcPtrTempG3 += vectorIncrementPerChannel;
                    srcPtrTempB3 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount3 < bufferLength3; vectorLoopCount3++)
                {
                    dstPtrTemp[0] = *srcPtrTempR3;
                    dstPtrTemp[1] = *srcPtrTempG3;
                    dstPtrTemp[2] = *srcPtrTempB3;
                    srcPtrTempR3++;
                    srcPtrTempG3++;
                    srcPtrTempB3++;
                    dstPtrTemp += 3;
                }

                int vectorLoopCount4 = 0;
                for (; vectorLoopCount4 < alignedLength4; vectorLoopCount4 += vectorIncrementPerChannel)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_i8pln3_to_i8pln3, srcPtrTempR4, srcPtrTempG4, srcPtrTempB4, px); // simd loads
                    rpp_simd_store(rpp_store48_i8pln3_to_i8pkd3, dstPtrTemp, px);                             // simd stores
                    srcPtrTempR4 += vectorIncrementPerChannel;
                    srcPtrTempG4 += vectorIncrementPerChannel;
                    srcPtrTempB4 += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount4 < bufferLength4; vectorLoopCount4++)
                {
                    dstPtrTemp[0] = *srcPtrTempR4;
                    dstPtrTemp[1] = *srcPtrTempG4;
                    dstPtrTemp[2] = *srcPtrTempB4;
                    srcPtrTempR4++;
                    srcPtrTempG4++;
                    srcPtrTempB4++;
                    dstPtrTemp += 3;
                }

                srcPtrRowR3 += srcDescPtr->strides.hStride;
                srcPtrRowG3 += srcDescPtr->strides.hStride;
                srcPtrRowB3 += srcDescPtr->strides.hStride;
                srcPtrRowR4 += srcDescPtr->strides.hStride;
                srcPtrRowG4 += srcDescPtr->strides.hStride;
                srcPtrRowB4 += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Ricap without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            for (int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8s *srcPtrRow1, *srcPtrRow2, *srcPtrRow3, *srcPtrRow4, *dstPtrRow;
                srcPtrRow1 = srcPtrChannel1;
                srcPtrRow2 = srcPtrChannel2;
                srcPtrRow3 = srcPtrChannel3;
                srcPtrRow4 = srcPtrChannel4;
                dstPtrRow = dstPtrChannel;

                for (int i = 0; i < roi1.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp1, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp1 = srcPtrRow1;
                    srcPtrTemp2 = srcPtrRow2;
                    dstPtrTemp = dstPtrRow;
                    memcpy(dstPtrTemp, srcPtrTemp1, bufferLength1);
                    dstPtrTemp += bufferLength1;
                    memcpy(dstPtrTemp, srcPtrTemp2, bufferLength2);

                    srcPtrRow1 += srcDescPtr->strides.hStride;
                    srcPtrRow2 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                for (int i = 0; i < roi3.xywhROI.roiHeight; i++)
                {
                    Rpp8s *srcPtrTemp3, *srcPtrTemp4, *dstPtrTemp;
                    srcPtrTemp3 = srcPtrRow3;
                    srcPtrTemp4 = srcPtrRow4;
                    dstPtrTemp = dstPtrRow;
                    memcpy(dstPtrTemp, srcPtrTemp3, bufferLength3);
                    dstPtrTemp += bufferLength3;
                    memcpy(dstPtrTemp, srcPtrTemp4, bufferLength4);

                    srcPtrRow3 += srcDescPtr->strides.hStride;
                    srcPtrRow4 += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrChannel1 += srcDescPtr->strides.cStride;
                srcPtrChannel2 += srcDescPtr->strides.cStride;
                srcPtrChannel3 += srcDescPtr->strides.cStride;
                srcPtrChannel4 += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}
