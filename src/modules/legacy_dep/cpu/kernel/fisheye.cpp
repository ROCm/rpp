/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include "host_legacy_executors.hpp"

inline void compute_image_location_host(RppiSize *batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax[m].height * batch_srcSizeMax[m].width);
    }
    *loc *= channel;
}

template <typename T>
RppStatus fisheye_base_host(T* srcPtrTemp, RppiSize srcSize, T* dstPtrTemp,
                            Rpp64u j, Rpp32u elementsPerChannel, Rpp32u elements,
                            Rpp32f newI, Rpp32f newIsquared,
                            RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f newJ, newIsrc, newJsrc, newJsquared, euclideanDistance, newEuclideanDistance, theta;
    int iSrc, jSrc, srcPosition;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(srcSize.width))) - 1.0;
        newJsquared = newJ * newJ;
        euclideanDistance = sqrt(newIsquared + newJsquared);
        if (euclideanDistance >= 0 && euclideanDistance <= 1)
        {
            newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
            newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;

            if (newEuclideanDistance <= 1.0)
            {
                theta = atan2(newI, newJ);

                newIsrc = newEuclideanDistance * sin(theta);
                newJsrc = newEuclideanDistance * cos(theta);

                iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) srcSize.height)) / 2.0);
                jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) srcSize.width)) / 2.0);

                srcPosition = (int)((iSrc * srcSize.width) + jSrc);

                if ((srcPosition >= 0) && (srcPosition < elementsPerChannel))
                {
                    *dstPtrTemp++ = *(srcPtrTemp + srcPosition);
                }
                else
                {
                    *dstPtrTemp++ = (T) 0;
                }
            }
            else
            {
                *dstPtrTemp++ = (T) 0;
            }
        }
        else
        {
            *dstPtrTemp++ = (T) 0;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        newJ = (((Rpp32f) (j * 2.0)) / ((Rpp32f)(srcSize.width))) - 1.0;
        newJsquared = newJ * newJ;
        euclideanDistance = sqrt(newIsquared + newJsquared);
        if (euclideanDistance >= 0 && euclideanDistance <= 1)
        {
            newEuclideanDistance = sqrt(1.0 - (euclideanDistance * euclideanDistance));
            newEuclideanDistance = (euclideanDistance + (1.0 - newEuclideanDistance)) / 2.0;

            if (newEuclideanDistance <= 1.0)
            {
                theta = atan2(newI, newJ);

                newIsrc = newEuclideanDistance * sin(theta);
                newJsrc = newEuclideanDistance * cos(theta);

                iSrc = (int) (((newIsrc + 1.0) * ((Rpp32f) srcSize.height)) / 2.0);
                jSrc = (int) (((newJsrc + 1.0) * ((Rpp32f) srcSize.width)) / 2.0);

                srcPosition = (int)(channel * ((iSrc * srcSize.width) + jSrc));

                if ((srcPosition >= 0) && (srcPosition < elements))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp++ = *(srcPtrTemp + srcPosition + c);
                    }
                }
                else
                {
                    memset(dstPtrTemp, 0, 3 * sizeof(T));
                    dstPtrTemp += 3;
                }
            }
            else
            {
                memset(dstPtrTemp, 0, 3 * sizeof(T));
                dstPtrTemp += 3;
            }
        }
        else
        {
            memset(dstPtrTemp, 0, 3 * sizeof(T));
            dstPtrTemp += 3;
        }
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus fisheye_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiROI *roiPoints,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    if(chnFormat == RPPI_CHN_PLANAR)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u remainingElementsAfterROI = (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32f newI, newIsquared;
            Rpp32u elementsPerChannelMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsMax = channel * elementsPerChannelMax;

            Rpp32f halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfWidth = batch_srcSize[batchCount].width / 2;

            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32s srcPositionArrayInt[4] = {0};
            Rpp32f eD[4] = {0};
            Rpp32f nED[4] = {0};

            __m128i px0, px1;
            __m128 p0, p1, p2;
            __m128 q0, pCmp1, pCmp2, pMask;
            __m128 pZero = _mm_set1_ps(0.0);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128 pMul = _mm_set1_ps(2.0 / (Rpp32f) batch_srcSize[batchCount].width);
            __m128 pMul2 = _mm_set1_ps(0.5);
            __m128 pMul3 = _mm_set1_ps(halfHeight);
            __m128 pMul4 = _mm_set1_ps(halfWidth);
            __m128 pWidthMax = _mm_set1_ps((Rpp32f) batch_srcSizeMax[batchCount].width);
            __m128i pxSrcPosition = _mm_set1_epi32(0);

            for(int c = 0; c < channel; c++)
            {
                T *srcPtrChannel, *dstPtrChannel;
                srcPtrChannel = srcPtrImage + (c * imageDimMax);
                dstPtrChannel = dstPtrImage + (c * imageDimMax);


                for(int i = 0; i < batch_srcSize[batchCount].height; i++)
                {
                    T *srcPtrTemp, *srcPtrTemp2, *dstPtrTemp;
                    srcPtrTemp = srcPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    dstPtrTemp = dstPtrChannel + (i * batch_srcSizeMax[batchCount].width);
                    srcPtrTemp2 = srcPtrChannel;

                    if (!((y1 <= i) && (i <= y2)))
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, batch_srcSize[batchCount].width * sizeof(T));

                        dstPtrTemp += batch_srcSizeMax[batchCount].width;
                        srcPtrTemp += batch_srcSizeMax[batchCount].width;
                    }
                    else
                    {
                        memcpy(dstPtrTemp, srcPtrTemp, x1 * sizeof(T));
                        srcPtrTemp += x1;
                        dstPtrTemp += x1;

                        Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                        Rpp32f newIsquared = newI * newI;

                        __m128 pNewI = _mm_set1_ps(newI);
                        __m128 pNewIsquared = _mm_set1_ps(newIsquared);

                        Rpp64u vectorLoopCount = x1;
                        for (; vectorLoopCount < alignedLength + x1; vectorLoopCount+=4)
                        {
                            pMask = _mm_set1_ps(1.0);

                            p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                            p0 = _mm_mul_ps(p0, pMul);
                            p0 = _mm_sub_ps(p0, pOne);

                            p1 = _mm_mul_ps(p0, p0);
                            p1 = _mm_add_ps(pNewIsquared, p1);
                            p1 = _mm_sqrt_ps(p1);

                            pCmp1 = _mm_cmpge_ps(p1, pZero);
                            pCmp2 = _mm_cmple_ps(p1, pOne);
                            pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                            _mm_storeu_ps(eD, pMask);

                            if(eD[0] != 0 && eD[1] != 0 && eD[2] != 0 && eD[3] != 0)
                            {
                                p2 = _mm_mul_ps(p1, p1);
                                p2 = _mm_sub_ps(pOne, p2);
                                p2 = _mm_sqrt_ps(p2);
                                p2 = _mm_sub_ps(pOne, p2);
                                p2 = _mm_add_ps(p1, p2);
                                p2 = _mm_mul_ps(p2, pMul2);

                                _mm_storeu_ps(nED, p2);

                                if (nED[0] <= 1.0 && nED[1] <= 1.0 && nED[2] <= 1.0 && nED[3] <= 1.0)
                                {
                                    q0 = atan2_ps(pNewI, p0);

                                    sincos_ps(q0, &p0, &p1);

                                    p0 = _mm_mul_ps(p2, p0);
                                    p1 = _mm_mul_ps(p2, p1);

                                    p0 = _mm_add_ps(p0, pOne);
                                    p0 = _mm_mul_ps(p0, pMul3);
                                    p1 = _mm_add_ps(p1, pOne);
                                    p1 = _mm_mul_ps(p1, pMul4);

                                    p0 = _mm_mul_ps(_mm_floor_ps(p0), pWidthMax);

                                    px0 = _mm_cvtps_epi32(p0);
                                    px1 = _mm_cvtps_epi32(_mm_floor_ps(p1));

                                    pxSrcPosition = _mm_add_epi32(px0, px1);

                                    _mm_storeu_si128((__m128i *)srcPositionArrayInt, pxSrcPosition);

                                    for (int pos = 0; pos < 4; pos++)
                                    {
                                        if ((srcPositionArrayInt[pos] >= 0) && (srcPositionArrayInt[pos] < elementsPerChannelMax))
                                        {
                                            *dstPtrTemp = *(srcPtrTemp2 + srcPositionArrayInt[pos]);
                                        }
                                        else
                                        {
                                            *dstPtrTemp = (T) 0;
                                        }
                                        dstPtrTemp++;
                                    }
                                }
                                else
                                {
                                    for (int id = 0; id < 4; id++)
                                    {
                                        if (nED[id] <= 1.0)
                                        {
                                            fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                            dstPtrTemp++;
                                        }
                                        else
                                        {
                                            *dstPtrTemp++ = (T) 0;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int id = 0; id < 4; id++)
                                {
                                    if (eD[id] != 0.0)
                                    {
                                        fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                        dstPtrTemp++;
                                    }
                                    else
                                    {
                                        *dstPtrTemp++ = (T) 0;
                                    }
                                }
                            }
                        }
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                            dstPtrTemp++;
                        }

                        srcPtrTemp += bufferLength;

                        memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                        srcPtrTemp += remainingElementsAfterROI;
                        dstPtrTemp += remainingElementsAfterROI;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
        for(int batchCount = 0; batchCount < nbatchSize; batchCount ++)
        {
            Rpp32u imageDimMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;

            Rpp32u x1 = roiPoints[batchCount].x;
            Rpp32u y1 = roiPoints[batchCount].y;
            Rpp32u x2 = x1 + roiPoints[batchCount].roiWidth - 1;
            Rpp32u y2 = y1 + roiPoints[batchCount].roiHeight - 1;
            if (x2 == -1)
            {
                x2 = batch_srcSize[batchCount].width - 1;
                roiPoints[batchCount].roiWidth = batch_srcSize[batchCount].width;
            }
            if (y2 == -1)
            {
                y2 = batch_srcSize[batchCount].height - 1;
                roiPoints[batchCount].roiHeight = batch_srcSize[batchCount].height;
            }

            Rpp32u elementsBeforeROI = channel * x1;
            Rpp32u remainingElementsAfterROI = channel * (batch_srcSize[batchCount].width - (roiPoints[batchCount].x + roiPoints[batchCount].roiWidth));

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            Rpp32f newI, newIsquared;
            Rpp32u elementsPerChannelMax = batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width;
            Rpp32u elementsMax = channel * elementsPerChannelMax;

            Rpp32f halfHeight = batch_srcSize[batchCount].height / 2;
            Rpp32f halfWidth = batch_srcSize[batchCount].width / 2;

            Rpp32u bufferLength = roiPoints[batchCount].roiWidth;
            Rpp32u alignedLength = (bufferLength / 4) * 4;

            Rpp32u elementsInBuffer = channel * bufferLength;

            Rpp32s srcPositionArrayInt[4] = {0};
            Rpp32f eD[4] = {0};
            Rpp32f nED[4] = {0};

            __m128i px0, px1;
            __m128 p0, p1, p2;
            __m128 q0, pCmp1, pCmp2, pMask;
            __m128 pZero = _mm_set1_ps(0.0);
            __m128 pOne = _mm_set1_ps(1.0);
            __m128 pThree = _mm_set1_ps(3.0);
            __m128 pMul = _mm_set1_ps(2.0 / (Rpp32f) batch_srcSize[batchCount].width);
            __m128 pMul2 = _mm_set1_ps(0.5);
            __m128 pMul3 = _mm_set1_ps(halfHeight);
            __m128 pMul4 = _mm_set1_ps(halfWidth);
            __m128 pWidthMax = _mm_set1_ps((Rpp32f) elementsInRowMax);
            __m128i pxSrcPosition = _mm_set1_epi32(0);


            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                Rpp32f newIsquared = newI * newI;

                T *srcPtrTemp, *srcPtrTemp2, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                srcPtrTemp2 = srcPtrImage;

                if (!((y1 <= i) && (i <= y2)))
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));

                    dstPtrTemp += elementsInRowMax;
                    srcPtrTemp += elementsInRowMax;
                }
                else
                {
                    memcpy(dstPtrTemp, srcPtrTemp, elementsBeforeROI * sizeof(T));
                    srcPtrTemp += elementsBeforeROI;
                    dstPtrTemp += elementsBeforeROI;

                    Rpp32f newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(batch_srcSize[batchCount].height))) - 1.0;
                    Rpp32f newIsquared = newI * newI;

                    __m128 pNewI = _mm_set1_ps(newI);
                    __m128 pNewIsquared = _mm_set1_ps(newIsquared);

                    Rpp64u vectorLoopCount = x1;
                    for (; vectorLoopCount < alignedLength + x1; vectorLoopCount+=4)
                    {
                        pMask = _mm_set1_ps(1.0);

                        p0 = _mm_setr_ps(vectorLoopCount, vectorLoopCount + 1, vectorLoopCount + 2, vectorLoopCount + 3);
                        p0 = _mm_mul_ps(p0, pMul);
                        p0 = _mm_sub_ps(p0, pOne);

                        p1 = _mm_mul_ps(p0, p0);
                        p1 = _mm_add_ps(pNewIsquared, p1);
                        p1 = _mm_sqrt_ps(p1);

                        pCmp1 = _mm_cmpge_ps(p1, pZero);
                        pCmp2 = _mm_cmple_ps(p1, pOne);
                        pMask = _mm_and_ps(pMask, _mm_and_ps(pCmp1, pCmp2));

                        _mm_storeu_ps(eD, pMask);

                        if (eD[0] != 0 && eD[1] != 0 && eD[2] != 0 && eD[3] != 0)
                        {
                            p2 = _mm_mul_ps(p1, p1);
                            p2 = _mm_sub_ps(pOne, p2);
                            p2 = _mm_sqrt_ps(p2);
                            p2 = _mm_sub_ps(pOne, p2);
                            p2 = _mm_add_ps(p1, p2);
                            p2 = _mm_mul_ps(p2, pMul2);

                            _mm_storeu_ps(nED, p2);

                            if (nED[0] <= 1.0 && nED[1] <= 1.0 && nED[2] <= 1.0 && nED[3] <= 1.0)
                            {
                                q0 = atan2_ps(pNewI, p0);

                                sincos_ps(q0, &p0, &p1);

                                p0 = _mm_mul_ps(p2, p0);
                                p1 = _mm_mul_ps(p2, p1);

                                p0 = _mm_add_ps(p0, pOne);
                                p0 = _mm_mul_ps(p0, pMul3);
                                p1 = _mm_add_ps(p1, pOne);
                                p1 = _mm_mul_ps(p1, pMul4);

                                p0 = _mm_mul_ps(_mm_floor_ps(p0), pWidthMax);

                                px0 = _mm_cvtps_epi32(p0);
                                px1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_floor_ps(p1), pThree));

                                pxSrcPosition = _mm_add_epi32(px0, px1);

                                _mm_storeu_si128((__m128i *)srcPositionArrayInt, pxSrcPosition);

                                for (int pos = 0; pos < 4; pos++)
                                {
                                    if ((srcPositionArrayInt[pos] >= 0) && (srcPositionArrayInt[pos] < elementsMax))
                                    {
                                        for (int c = 0; c < channel; c++)
                                        {
                                            *dstPtrTemp++ = *(srcPtrTemp2 + srcPositionArrayInt[pos] + c);
                                        }
                                    }
                                    else
                                    {
                                        memset(dstPtrTemp, 0, 3 * sizeof(T));
                                        dstPtrTemp += 3;
                                    }
                                }
                            }
                            else
                            {
                                for (int id = 0; id < 4; id++)
                                {
                                    if (nED[id] <= 1.0)
                                    {
                                        fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                        dstPtrTemp += 3;
                                    }
                                    else
                                    {
                                        memset(dstPtrTemp, 0, 3 * sizeof(T));
                                        dstPtrTemp += 3;
                                    }
                                }
                            }
                        }
                        else
                        {
                            for (int id = 0; id < 4; id++)
                            {
                                if (eD[id] != 0.0)
                                {
                                    fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount + id, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                                    dstPtrTemp += 3;
                                }
                                else
                                {
                                    memset(dstPtrTemp, 0, 3 * sizeof(T));
                                    dstPtrTemp += 3;
                                }
                            }
                        }
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        fisheye_base_host(srcPtrTemp2, batch_srcSize[batchCount], dstPtrTemp, vectorLoopCount, elementsPerChannelMax, elementsMax, newI, newIsquared, chnFormat, channel);
                        dstPtrTemp += 3;
                    }

                    srcPtrTemp += elementsInBuffer;

                    memcpy(dstPtrTemp, srcPtrTemp, remainingElementsAfterROI * sizeof(T));
                    srcPtrTemp += remainingElementsAfterROI;
                    dstPtrTemp += remainingElementsAfterROI;
                }
            }
        }
    }

    return RPP_SUCCESS;
}


template RppStatus fisheye_host_batch<Rpp8u>(Rpp8u*,
                                             RppiSize*,
                                             RppiSize*,
                                             Rpp8u*,
                                             RppiROI*,
                                             Rpp32u,
                                             RppiChnFormat,
                                             Rpp32u,
                                             rpp::Handle&);
