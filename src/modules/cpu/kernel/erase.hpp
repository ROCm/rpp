/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus erase_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptRoiLtrb *anchorBoxInfoTensor,
                                  Rpp8u *colorsTensor,
                                  Rpp32u *numBoxesTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32u numBoxes = numBoxesTensor[batchCount];
        RpptRoiLtrb *anchorBoxInfo = anchorBoxInfoTensor + batchCount * numBoxes;
        Rpp8u *colors = colorsTensor + batchCount * numBoxes * srcDescPtr->c;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        Rpp8u *userPixel3 = (Rpp8u* )malloc(sizeof(Rpp8u) * 3);

        // Erase with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp, *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                bool is_erase = false;
                Rpp32u bufferLength = 0;
                Rpp8u userPixelR, userPixelG, userPixelB;
                int j;

                for (j = 0; j < roi.xywhROI.roiWidth;)
                {
                    for(int count = 0; count < numBoxes; count++)
                    {
                        Rpp32u x1 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth);
                        Rpp32u y1 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight);
                        Rpp32u x2 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth);
                        Rpp32u y2 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight);
                        userPixelR = colors[(count * 3)];
                        userPixelG = colors[(count * 3) + 1];
                        userPixelB = colors[(count * 3) + 2];
                        if(i >= y1 && i <= y2 && j >= x1 && j <= x2)
                        {
                            is_erase = true;
                            bufferLength = x2 - x1 + 1;
                            break;
                        }
                    }
                    if(is_erase && bufferLength)
                    {
                        memset(dstPtrTempR, userPixelR, bufferLength * sizeof(Rpp8u));
                        memset(dstPtrTempG, userPixelG, bufferLength * sizeof(Rpp8u));
                        memset(dstPtrTempB, userPixelB, bufferLength * sizeof(Rpp8u));
                        srcPtrTemp += 3 * bufferLength;
                        j += bufferLength;
                        dstPtrTempR += bufferLength;
                        dstPtrTempG += bufferLength;
                        dstPtrTempB += bufferLength;
                        is_erase = false;
                    }
                    else
                    {
                        *dstPtrTempR++ = srcPtrTemp[0];
                        *dstPtrTempG++ = srcPtrTemp[1];
                        *dstPtrTempB++ = srcPtrTemp[2];
                        srcPtrTemp += 3;
                        j++;
                    }
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Erase with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
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
                bool is_erase = false;
                Rpp32u bufferLengthPerChannel = 0;
                int j;

                for (j = 0; j < roi.xywhROI.roiWidth;)
                {
                    for(int count = 0; count < numBoxes; count++)
                    {
                        Rpp32u x1 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth);
                        Rpp32u y1 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight);
                        Rpp32u x2 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth);
                        Rpp32u y2 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight);
                        userPixel3[0] = colors[(count * 3)];
                        userPixel3[1] = colors[(count * 3) + 1];
                        userPixel3[2] = colors[(count * 3) + 2];

                        if(i >= y1 && i <= y2 && j >= x1 && j <= x2)
                        {
                            is_erase = true;
                            bufferLengthPerChannel = x2 - x1 + 1;
                            break;
                        }
                    }
                    if(is_erase && bufferLengthPerChannel)
                    {
                        for (int k = 0; k < bufferLengthPerChannel; k++)
                        {
                            memcpy(dstPtrTemp, userPixel3, sizeof(Rpp8u) * 3);
                            dstPtrTemp += 3;
                        }
                        j += bufferLengthPerChannel;
                        srcPtrTempR += bufferLengthPerChannel;
                        srcPtrTempG += bufferLengthPerChannel;
                        srcPtrTempB += bufferLengthPerChannel;
                        is_erase = false;
                    }
                    else
                    {
                        dstPtrTemp[0] = *srcPtrTempR++;
                        dstPtrTemp[1] = *srcPtrTempG++;
                        dstPtrTemp[2] = *srcPtrTempB++;
                        dstPtrTemp += 3;
                        j++;
                    }
                }

                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Erase without fused output-layout toggle (NHWC -> NHWC or NCHW -> NCHW)
        else
        {
            memcpy(dstPtrImage, srcPtrImage, dstDescPtr->strides.nStride * sizeof(Rpp8u));

            for(int count = 0; count < numBoxes; count++)
            {
                Rpp32u x1 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].lt.x, roi.xywhROI.xy.x, roi.xywhROI.roiWidth);
                Rpp32u y1 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].lt.y, roi.xywhROI.xy.y, roi.xywhROI.roiHeight);
                Rpp32u x2 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].rb.x, x1, roi.xywhROI.roiWidth);
                Rpp32u y2 = (Rpp32u) RPPPRANGECHECK(anchorBoxInfo[count].rb.y, y1, roi.xywhROI.roiHeight);

                Rpp32u pixelLocation = (y1 * srcDescPtr->strides.hStride) + (x1 * srcDescPtr->strides.wStride);
                Rpp32u boxHeight = y2 - y1 + 1;
                Rpp32u boxLength = x2 - x1 + 1;
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrImage + pixelLocation;// * layoutParams.channelParam;

                for (int c = 0; c < layoutParams.channelParam; c++)
                {
                    Rpp8u *dstPtrTemp2 = dstPtrTemp;
                    Rpp8u userPixel = colors[(count * layoutParams.channelParam) + c];
                    for (int i = 0; i < boxHeight; i++)
                    {
                        memset(dstPtrTemp2, userPixel, boxLength * sizeof(Rpp8u));
                        dstPtrTemp2 += dstDescPtr->strides.hStride;
                    }
                    dstPtrTemp += dstDescPtr->strides.cStride;
                }
            }
        }
        free(userPixel3);
    }

    return RPP_SUCCESS;
}