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

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
#define RPPPIXELCHECK(pixel)            (pixel < (Rpp32f) 0) ? ((Rpp32f) 0) : ((pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255))

inline void compute_image_location_host(RppiSize *batch_srcSizeMax, int batchCount, Rpp32u *loc, Rpp32u channel)
{
    for (int m = 0; m < batchCount; m++)
    {
        *loc += (batch_srcSizeMax[m].height * batch_srcSizeMax[m].width);
    }
    *loc *= channel;
}

template <typename T>
RppStatus snow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                              Rpp32f *batch_strength,
                              RppiROI *roiPoints, Rpp32u nbatchSize,
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

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32f strength = batch_strength[batchCount];

            strength = strength/100;
            int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

            Rpp32u snowDrops = (Rpp32u)(strength * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );


            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                dstPtrTemp = dstPtrImage + (i * batch_srcSizeMax[batchCount].width * channel);
                memcpy(dstPtrTemp, srcPtrTemp, batch_srcSizeMax[batchCount].width * channel * sizeof(T));
            }

            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for(int i = 0 ; i < snowDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] = RPPPIXELCHECK(dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width)] + snow_mat[0][0]) ;
                }
                for(int j = 0;j < 5;j++)
                {
                    if(row + 5 < batch_srcSize[batchCount].height && row + 5 > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < 5;m++)
                        {
                            if (column + 5 < batch_srcSizeMax[batchCount].width && column + 5 > 0)
                            {
                                dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] = RPPPIXELCHECK( dstPtrTemp[(row * batch_srcSizeMax[batchCount].width) + (column) + (k * batch_srcSizeMax[batchCount].height * batch_srcSizeMax[batchCount].width) + (batch_srcSizeMax[batchCount].width * j) + m] + snow_mat[j][m]) ;
                            }
                        }
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

            Rpp32f x1 = roiPoints[batchCount].x;
            Rpp32f y1 = roiPoints[batchCount].y;
            Rpp32f x2 = x1 + roiPoints[batchCount].roiWidth;
            Rpp32f y2 = y1 + roiPoints[batchCount].roiHeight;
            if (x2 == 0) x2 = batch_srcSize[batchCount].width;
            if (y2 == 0) y2 = batch_srcSize[batchCount].height;

            Rpp32f strength = batch_strength[batchCount];

            strength = strength/100;
            int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};

            Rpp32u snowDrops = (Rpp32u)(strength * batch_srcSize[batchCount].width * batch_srcSize[batchCount].height * channel );

            T *srcPtrImage, *dstPtrImage;
            Rpp32u loc = 0;
            compute_image_location_host(batch_srcSizeMax, batchCount, &loc, channel);
            srcPtrImage = srcPtr + loc;
            dstPtrImage = dstPtr + loc;

            Rpp32u elementsInRow = channel * batch_srcSize[batchCount].width;
            Rpp32u elementsInRowMax = channel * batch_srcSizeMax[batchCount].width;

            for(int i = 0; i < batch_srcSize[batchCount].height; i++)
            {
                T *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrImage + (i * elementsInRowMax);
                dstPtrTemp = dstPtrImage + (i * elementsInRowMax);
                memcpy(dstPtrTemp, srcPtrTemp, elementsInRow * sizeof(T));
            }
            T *srcPtrTemp, *dstPtrTemp;
            srcPtrTemp = srcPtrImage;
            dstPtrTemp = dstPtrImage;

            for(int i = 0 ; i < snowDrops ; i++)
            {
                Rpp32u row = rand() % batch_srcSize[batchCount].height;
                Rpp32u column = rand() % batch_srcSize[batchCount].width;
                Rpp32f pixel;
                for(int k = 0;k < channel;k++)
                {
                    dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] = RPPPIXELCHECK(dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k] + snow_mat[0][0]) ;
                }
                for(int j = 0;j < 5;j++)
                {
                    if(row + 5 < batch_srcSize[batchCount].height && row + 5 > 0 )
                    for(int k = 0;k < channel;k++)
                    {
                        for(int m = 0;m < 5;m++)
                        {
                            if (column + 5 < batch_srcSize[batchCount].width && column + 5 > 0)
                            {
                                dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] = RPPPIXELCHECK( dstPtrTemp[(channel * row * batch_srcSizeMax[batchCount].width) + (column * channel) + k + (channel * batch_srcSizeMax[batchCount].width * j) + (channel * m)] + snow_mat[j][m]);
                            }
                        }
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}


template RppStatus snow_host_batch<Rpp8u>(Rpp8u*,
                                          RppiSize*,
                                          RppiSize*,
                                          Rpp8u*,
                                          Rpp32f*,
                                          RppiROI*,
                                          Rpp32u,
                                          RppiChnFormat,
                                          Rpp32u,
                                          rpp::Handle&);