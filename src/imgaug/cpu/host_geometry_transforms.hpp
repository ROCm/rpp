#include <cpu/rpp_cpu_common.hpp>

/**************** Flip ***************/

template <typename T>
RppStatus flip_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                    RppiAxis flipAxis,
                    RppiChnFormat chnFormat, unsigned channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            int srcLoc = 0, dstLoc = 0;
            for (int i = (srcSize.height - 1); i >= 0; i--)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    srcLoc = (i * srcSize.width) + j;
                    for (int c = 0; c < channel; c++)
                    {
                        dstPtr[dstLoc + (c * srcSize.height * srcSize.width)] = srcPtr[srcLoc + (c * srcSize.height * srcSize.width)];
                    }
                    dstLoc += 1;
                }
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            int srcLoc = 0, dstLoc = 0;
            for (int i = (srcSize.width - 1); i >= 0; i--)
            {
                dstLoc = srcSize.width - 1 - i;
                for (int j = 0; j < srcSize.height; j++)
                {
                    srcLoc = (j * srcSize.width) + i;
                    for (int c = 0; c < channel; c++)
                    {
                        dstPtr[dstLoc + (c * srcSize.height * srcSize.width)] = srcPtr[srcLoc + (c * srcSize.height * srcSize.width)];
                    }
                    dstLoc += srcSize.width;
                }
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            Rpp8u *pInter = (Rpp8u *)malloc(channel * srcSize.width * srcSize.height * sizeof(Rpp8u));
            int srcLoc = 0, interLoc = 0;
            for (int i = (srcSize.height - 1); i >= 0; i--)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    srcLoc = (i * srcSize.width) + j;
                    for (int c = 0; c < channel; c++)
                    {
                        pInter[interLoc + (c * srcSize.height * srcSize.width)] = srcPtr[srcLoc + (c * srcSize.height * srcSize.width)];
                    }
                    interLoc += 1;
                }
            }
            int dstLoc = 0;
            interLoc = 0;
            for (int i = (srcSize.width - 1); i >= 0; i--)
            {
                dstLoc = srcSize.width - 1 - i;
                for (int j = 0; j < srcSize.height; j++)
                {
                    interLoc = (j * srcSize.width) + i;
                    for (int c = 0; c < channel; c++)
                    {
                        dstPtr[dstLoc + (c * srcSize.height * srcSize.width)] = pInter[interLoc + (c * srcSize.height * srcSize.width)];
                    }
                    dstLoc += srcSize.width;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        if (flipAxis == RPPI_HORIZONTAL_AXIS)
        {
            int srcLoc = 0, dstLoc = 0;
            for (int i = (srcSize.height - 1); i >= 0; i--)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    srcLoc = (i * channel * srcSize.width) + (channel * j);
                    for (int c = 0; c < channel; c++)
                    {
                        dstPtr[dstLoc + c] = srcPtr[srcLoc + c];
                    }
                    srcLoc += channel;
                    dstLoc += channel;
                }
            }
        }
        else if (flipAxis == RPPI_VERTICAL_AXIS)
        {
            int srcLoc = 0, dstLoc = 0;
            for (int i = (srcSize.width - 1); i >= 0; i--)
            {
                dstLoc = channel * (srcSize.width - 1 - i);
                for (int j = 0; j < srcSize.height; j++)
                {
                    srcLoc = (j * channel * srcSize.width) + (i * channel);
                    for (int c = 0; c < channel; c++)
                    {
                        dstPtr[dstLoc + c] = srcPtr[srcLoc + c];
                    }
                    dstLoc += (srcSize.width * channel);
                }
            }
        }
        else if (flipAxis == RPPI_BOTH_AXIS)
        {
            Rpp8u *pInter = (Rpp8u *)malloc(channel * srcSize.width * srcSize.height * sizeof(Rpp8u));
            int srcLoc = 0, interLoc = 0;

            for (int i = (srcSize.height - 1); i >= 0; i--)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    srcLoc = (i * channel * srcSize.width) + (channel * j);
                    for (int c = 0; c < channel; c++)
                    {
                        pInter[interLoc + c] = srcPtr[srcLoc + c];
                    }
                    srcLoc += channel;
                    interLoc += channel;
                }
            }

            int dstLoc = 0;
            interLoc = 0;


            for (int i = (srcSize.width - 1); i >= 0; i--)
            {
                dstLoc = channel * (srcSize.width - 1 - i);
                for (int j = 0; j < srcSize.height; j++)
                {
                    interLoc = (j * channel * srcSize.width) + (i * channel);
                    for (int c = 0; c < channel; c++)
                    {
                        dstPtr[dstLoc + c] = pInter[interLoc + c];
                    }
                    dstLoc += (srcSize.width * channel);
                }
            }
        }
    }

    return RPP_SUCCESS;
}




/**************** Warp Affine ***************/

template <typename T>
RppStatus warp_affine_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                       T* affine)
{
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (affine[0] * i) + (affine[1] * j) + (affine[2] * 1);
            newj = (affine[3] * i) + (affine[4] * j) + (affine[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
            if (newi > maxX)
            {
                maxX = newi;
            }
            if (newj > maxY)
            {
                maxY = newj;
            }
        }
    }
    dstSizePtr->height = ((Rpp32s)maxX - (Rpp32s)minX) + 1;
    dstSizePtr->width = ((Rpp32s)maxY - (Rpp32s)minY) + 1;

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus warp_affine_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           U* affine,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    float minX = 0, minY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (affine[0] * i) + (affine[1] * j) + (affine[2] * 1);
            newj = (affine[3] * i) + (affine[4] * j) + (affine[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
        }
    }

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    int k = (Rpp32s)((affine[0] * i) + (affine[1] * j) + (affine[2] * 1));
                    int l = (Rpp32s)((affine[3] * i) + (affine[4] * j) + (affine[5] * 1));
                    k -= (Rpp32s)minX;
                    l -= (Rpp32s)minY;
                    dstPtr[(c * dstSize.height * dstSize.width) + (k * dstSize.width) + l] = srcPtr[(c * srcSize.height * srcSize.width) + (i * srcSize.width) + j];
                }
            }
        }

        for (int c = 0; c < channel; c++)
        {
            for (int i = 1; i < dstSize.height - 1; i++)
            {
                for (int j = 1; j < dstSize.width - 1; j++)
                {
                    if (dstPtr[(c * dstSize.height * dstSize.width) + (i * dstSize.width) + j] == 0)
                    {
                        Rpp32f pixel;
                        pixel = 0.25 * (dstPtr[(c * dstSize.height * dstSize.width) + ((i - 1) * dstSize.width) + (j - 1)] +
                                        dstPtr[(c * dstSize.height * dstSize.width) + ((i - 1) * dstSize.width) + (j + 1)] +
                                        dstPtr[(c * dstSize.height * dstSize.width) + ((i + 1) * dstSize.width) + (j - 1)] +
                                        dstPtr[(c * dstSize.height * dstSize.width) + ((i + 1) * dstSize.width) + (j + 1)]);
                        dstPtr[(c * dstSize.height * dstSize.width) + (i * dstSize.width) + j] = (Rpp8u) pixel;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    int k = (Rpp32s)((affine[0] * i) + (affine[1] * j) + (affine[2] * 1));
                    int l = (Rpp32s)((affine[3] * i) + (affine[4] * j) + (affine[5] * 1));
                    k -= (Rpp32s)minX;
                    l -= (Rpp32s)minY;
                    dstPtr[c + (channel * k * dstSize.width) + (channel * l)] = srcPtr[c + (channel * i * srcSize.width) + (channel * j)];
                }
            }
        }

        for (int c = 0; c < channel; c++)
        {
            for (int i = 1; i < dstSize.height - 1; i++)
            {
                for (int j = 1; j < dstSize.width - 1; j++)
                {
                    if (dstPtr[c + (channel * i * dstSize.width) + (channel * j)] == 0)
                    {
                        Rpp32f pixel;
                        pixel = 0.25 * (dstPtr[c + (channel * (i - 1) * dstSize.width) + (channel * (j - 1))] +
                                        dstPtr[c + (channel * (i - 1) * dstSize.width) + (channel * (j + 1))] +
                                        dstPtr[c + (channel * (i + 1) * dstSize.width) + (channel * (j - 1))] +
                                        dstPtr[c + (channel * (i + 1) * dstSize.width) + (channel * (j + 1))]);
                        dstPtr[c + (channel * i * dstSize.width) + (channel * j)] = (Rpp8u) pixel;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}




/**************** Rotate ***************/

RppStatus rotate_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                  Rpp32f angleDeg)
{
    Rpp32f angleRad = RAD(angleDeg);
    Rpp32f rotate[6] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = 0;
    rotate[3] = -sin(angleRad);
    rotate[4] = cos(angleRad);
    rotate[5] = 0;

    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (rotate[0] * i) + (rotate[1] * j) + (rotate[2] * 1);
            newj = (rotate[3] * i) + (rotate[4] * j) + (rotate[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
            if (newi > maxX)
            {
                maxX = newi;
            }
            if (newj > maxY)
            {
                maxY = newj;
            }
        }
    }
    dstSizePtr->height = ((Rpp32s)maxX - (Rpp32s)minX) + 1;
    dstSizePtr->width = ((Rpp32s)maxY - (Rpp32s)minY) + 1;

    return RPP_SUCCESS;
}

template <typename T>
RppStatus rotate_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f angleDeg,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32f angleRad = RAD(angleDeg);
    Rpp32f rotate[6] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = 0;
    rotate[3] = -sin(angleRad);
    rotate[4] = cos(angleRad);
    rotate[5] = 0;

    float minX = 0, minY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (rotate[0] * i) + (rotate[1] * j) + (rotate[2] * 1);
            newj = (rotate[3] * i) + (rotate[4] * j) + (rotate[5] * 1);
            if (newi < minX)
            {
                minX = newi;
            }
            if (newj < minY)
            {
                minY = newj;
            }
        }
    }

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    int k = (Rpp32s)((rotate[0] * i) + (rotate[1] * j) + (rotate[2] * 1));
                    int l = (Rpp32s)((rotate[3] * i) + (rotate[4] * j) + (rotate[5] * 1));
                    k -= (Rpp32s)minX;
                    l -= (Rpp32s)minY;
                    dstPtr[(c * dstSize.height * dstSize.width) + (k * dstSize.width) + l] = srcPtr[(c * srcSize.height * srcSize.width) + (i * srcSize.width) + j];
                }
            }
        }

        for (int c = 0; c < channel; c++)
        {
            for (int i = 1; i < dstSize.height - 1; i++)
            {
                for (int j = 1; j < dstSize.width - 1; j++)
                {
                    if (dstPtr[(c * dstSize.height * dstSize.width) + (i * dstSize.width) + j] == 0)
                    {
                        Rpp32f pixel;
                        pixel = 0.25 * (dstPtr[(c * dstSize.height * dstSize.width) + ((i - 1) * dstSize.width) + (j - 1)] +
                                        dstPtr[(c * dstSize.height * dstSize.width) + ((i - 1) * dstSize.width) + (j + 1)] +
                                        dstPtr[(c * dstSize.height * dstSize.width) + ((i + 1) * dstSize.width) + (j - 1)] +
                                        dstPtr[(c * dstSize.height * dstSize.width) + ((i + 1) * dstSize.width) + (j + 1)]);
                        dstPtr[(c * dstSize.height * dstSize.width) + (i * dstSize.width) + j] = (Rpp8u) pixel;
                    }
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    int k = (Rpp32s)((rotate[0] * i) + (rotate[1] * j) + (rotate[2] * 1));
                    int l = (Rpp32s)((rotate[3] * i) + (rotate[4] * j) + (rotate[5] * 1));
                    k -= (Rpp32s)minX;
                    l -= (Rpp32s)minY;
                    dstPtr[c + (channel * k * dstSize.width) + (channel * l)] = srcPtr[c + (channel * i * srcSize.width) + (channel * j)];
                }
            }
        }

        for (int c = 0; c < channel; c++)
        {
            for (int i = 1; i < dstSize.height - 1; i++)
            {
                for (int j = 1; j < dstSize.width - 1; j++)
                {
                    if (dstPtr[c + (channel * i * dstSize.width) + (channel * j)] == 0)
                    {
                        Rpp32f pixel;
                        pixel = 0.25 * (dstPtr[c + (channel * (i - 1) * dstSize.width) + (channel * (j - 1))] +
                                        dstPtr[c + (channel * (i - 1) * dstSize.width) + (channel * (j + 1))] +
                                        dstPtr[c + (channel * (i + 1) * dstSize.width) + (channel * (j - 1))] +
                                        dstPtr[c + (channel * (i + 1) * dstSize.width) + (channel * (j + 1))]);
                        dstPtr[c + (channel * i * dstSize.width) + (channel * j)] = (Rpp8u) pixel;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}




/**************** Scale ***************/

RppStatus scale_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                  Rpp32f percentage)
{
    if (percentage < 0)
    {
        return RPP_ERROR;
    }
    percentage /= 100;
    dstSizePtr->height = (Rpp32s) (percentage * (Rpp32f) srcSize.height);
    dstSizePtr->width = (Rpp32s) (percentage * (Rpp32f) srcSize.width);

    return RPP_SUCCESS;
}

template <typename T>
RppStatus scale_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32f percentage,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    if (percentage < 0)
    {
        return RPP_ERROR;
    }
    percentage = round(percentage * 100.0) / 100.0;
    percentage /= 100;

    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    T *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / percentage;
                Rpp32f weightedHeight = srcLocationRow - RPPFLOOR(srcLocationRow);
                
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / percentage;
                    Rpp32f weightedWidth = srcLocationColumn - RPPFLOOR(srcLocationColumn);

                    if ((Rpp32s) RPPFLOOR(srcLocationRow) > (srcSize.height - 2))
                    {
                        srcLocationRow = srcSize.height - 2;
                    }
                    if ((Rpp32s) RPPFLOOR(srcLocationColumn) > (srcSize.width - 2))
                    {
                        srcLocationColumn = srcSize.width - 2;
                    }

                    srcPtrTopRow = srcPtr + (c * srcSize.height * srcSize.width) 
                                            + ((Rpp32s) RPPFLOOR(srcLocationRow) * srcSize.width) 
                                            + (Rpp32s) RPPFLOOR(srcLocationColumn);
                    srcPtrBottomRow = srcPtr + (c * srcSize.height * srcSize.width) 
                                                + ((Rpp32s) RPPFLOOR(srcLocationRow + 1) * srcSize.width) 
                                                + (Rpp32s) RPPFLOOR(srcLocationColumn);

                    pixel = ((*srcPtrTopRow) * (1 - weightedWidth) * (1 - weightedHeight)) 
                            + ((*(srcPtrTopRow + 1)) * (weightedWidth) * (1 - weightedHeight)) 
                            + ((*srcPtrBottomRow) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + 1)) * (weightedWidth) * (weightedHeight));
                    
                    *dstPtrTemp = (Rpp8u) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / percentage;
            Rpp32f weightedHeight = srcLocationRow - RPPFLOOR(srcLocationRow);

            for (int j = 0; j < dstSize.width; j++)
            {   
                for (int c = 0; c < channel; c++)
                {
                    srcLocationColumn = ((Rpp32f) j) / percentage;
                    Rpp32f weightedWidth = srcLocationColumn - RPPFLOOR(srcLocationColumn);

                    if ((Rpp32s) RPPFLOOR(srcLocationRow) > (srcSize.height - 2))
                    {
                        srcLocationRow = srcSize.height - 2;
                    }
                    if ((Rpp32s) RPPFLOOR(srcLocationColumn) > (srcSize.width - 2))
                    {
                        srcLocationColumn = srcSize.width - 2;
                    }
                    
                    srcPtrTopRow = srcPtr + c + channel * (((Rpp32s) RPPFLOOR(srcLocationRow) * srcSize.width) 
                                                                    + (Rpp32s) RPPFLOOR(srcLocationColumn));
                    srcPtrBottomRow = srcPtr + c + channel * (((Rpp32s) RPPFLOOR(srcLocationRow + 1) * srcSize.width) 
                                                                        + (Rpp32s) RPPFLOOR(srcLocationColumn));
                    
                    pixel = ((*srcPtrTopRow) * (1 - weightedWidth) * (1 - weightedHeight)) 
                            + ((*(srcPtrTopRow + channel)) * (weightedWidth) * (1 - weightedHeight)) 
                            + ((*srcPtrBottomRow) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + channel)) * (weightedWidth) * (weightedHeight));
                    
                    *dstPtrTemp = (Rpp8u) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}




/**************** Resize ***************/

template <typename T>
RppStatus resize_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    if (dstSize.height < 0 || dstSize.width < 0)
    {
        return RPP_ERROR;
    }
    Rpp32f hRatio = (((Rpp32f) dstSize.height) / ((Rpp32f) srcSize.height));
    Rpp32f wRatio = (((Rpp32f) dstSize.width) / ((Rpp32f) srcSize.width));
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    T *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / hRatio;
                Rpp32f weightedHeight = srcLocationRow - RPPFLOOR(srcLocationRow);
                
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    Rpp32f weightedWidth = srcLocationColumn - RPPFLOOR(srcLocationColumn);

                    if ((Rpp32s) RPPFLOOR(srcLocationRow) > (srcSize.height - 2))
                    {
                        srcLocationRow = srcSize.height - 2;
                    }
                    if ((Rpp32s) RPPFLOOR(srcLocationColumn) > (srcSize.width - 2))
                    {
                        srcLocationColumn = srcSize.width - 2;
                    }

                    srcPtrTopRow = srcPtr + (c * srcSize.height * srcSize.width) 
                                            + ((Rpp32s) RPPFLOOR(srcLocationRow) * srcSize.width) 
                                            + (Rpp32s) RPPFLOOR(srcLocationColumn);
                    srcPtrBottomRow = srcPtr + (c * srcSize.height * srcSize.width) 
                                                + ((Rpp32s) RPPFLOOR(srcLocationRow + 1) * srcSize.width) 
                                                + (Rpp32s) RPPFLOOR(srcLocationColumn);

                    pixel = ((*srcPtrTopRow) * (1 - weightedWidth) * (1 - weightedHeight)) 
                            + ((*(srcPtrTopRow + 1)) * (weightedWidth) * (1 - weightedHeight)) 
                            + ((*srcPtrBottomRow) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + 1)) * (weightedWidth) * (weightedHeight));
                    
                    *dstPtrTemp = (Rpp8u) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            Rpp32f weightedHeight = srcLocationRow - RPPFLOOR(srcLocationRow);

            for (int j = 0; j < dstSize.width; j++)
            {   
                for (int c = 0; c < channel; c++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    Rpp32f weightedWidth = srcLocationColumn - RPPFLOOR(srcLocationColumn);

                    if ((Rpp32s) RPPFLOOR(srcLocationRow) > (srcSize.height - 2))
                    {
                        srcLocationRow = srcSize.height - 2;
                    }
                    if ((Rpp32s) RPPFLOOR(srcLocationColumn) > (srcSize.width - 2))
                    {
                        srcLocationColumn = srcSize.width - 2;
                    }

                    srcPtrTopRow = srcPtr + c + channel * (((Rpp32s) RPPFLOOR(srcLocationRow) * srcSize.width) 
                                                                    + (Rpp32s) RPPFLOOR(srcLocationColumn));
                    srcPtrBottomRow = srcPtr + c + channel * (((Rpp32s) RPPFLOOR(srcLocationRow + 1) * srcSize.width) 
                                                                        + (Rpp32s) RPPFLOOR(srcLocationColumn));
                    
                    pixel = ((*srcPtrTopRow) * (1 - weightedWidth) * (1 - weightedHeight)) 
                            + ((*(srcPtrTopRow + channel)) * (weightedWidth) * (1 - weightedHeight)) 
                            + ((*srcPtrBottomRow) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + channel)) * (weightedWidth) * (weightedHeight));
                    
                    *dstPtrTemp = (Rpp8u) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}




/**************** Resize Crop ***************/

template <typename T>
RppStatus resizeCrop_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    if (dstSize.height < 0 || dstSize.width < 0)
    {
        return RPP_ERROR;
    }
    if ((RPPINRANGE(x1, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(x2, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(y1, 0, srcSize.height - 1) == 0) 
        || (RPPINRANGE(y2, 0, srcSize.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    RppiSize srcNewSize;
    int xDiff = (int) x2 - (int) x1;
    int yDiff = (int) y2 - (int) y1;
    srcNewSize.width = (Rpp32u) RPPABS(xDiff);
    srcNewSize.height = (Rpp32u) RPPABS(yDiff);
    
    Rpp8u *srcNewPtr = (Rpp8u *)calloc(channel * srcNewSize.height * srcNewSize.width, sizeof(Rpp8u));

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = RPPMIN2(y1, y2), m = 0; i < RPPMAX2(y1, y2); i++, m++)
            {
                for (int j = RPPMIN2(x1, x2), n = 0; j < RPPMAX2(x1, x2); j++, n++)
                {
                    *(srcNewPtr + (c * srcNewSize.height * srcNewSize.width) + (m * srcNewSize.width) + n) 
                        = *(srcPtr + (c * srcSize.height * srcSize.width) + (i * srcSize.width) + j);
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = RPPMIN2(y1, y2), m = 0; i < RPPMAX2(y1, y2); i++, m++)
        {
            for (int j = RPPMIN2(x1, x2), n = 0; j < RPPMAX2(x1, x2); j++, n++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *(srcNewPtr + c + (channel * m * srcNewSize.width) + (channel * n)) 
                        = *(srcPtr + c + (channel * i * srcSize.width) + (channel * j));
                }
            }
        }
    }

    Rpp32f hRatio = (((Rpp32f) dstSize.height) / ((Rpp32f) srcNewSize.height));
    Rpp32f wRatio = (((Rpp32f) dstSize.width) / ((Rpp32f) srcNewSize.width));
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    T *dstPtrTemp, *srcNewPtrTopRow, *srcNewPtrBottomRow;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / hRatio;
                Rpp32f weightedHeight = srcLocationRow - RPPFLOOR(srcLocationRow);
                
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    Rpp32f weightedWidth = srcLocationColumn - RPPFLOOR(srcLocationColumn);

                    if ((Rpp32s) RPPFLOOR(srcLocationRow) > (srcNewSize.height - 2))
                    {
                        srcLocationRow = srcNewSize.height - 2;
                    }
                    if ((Rpp32s) RPPFLOOR(srcLocationColumn) > (srcNewSize.width - 2))
                    {
                        srcLocationColumn = srcNewSize.width - 2;
                    }

                    srcNewPtrTopRow = srcNewPtr + (c * srcNewSize.height * srcNewSize.width) 
                                            + ((Rpp32s) RPPFLOOR(srcLocationRow) * srcNewSize.width) 
                                            + (Rpp32s) RPPFLOOR(srcLocationColumn);
                    srcNewPtrBottomRow = srcNewPtr + (c * srcNewSize.height * srcNewSize.width) 
                                                + ((Rpp32s) RPPFLOOR(srcLocationRow + 1) * srcNewSize.width) 
                                                + (Rpp32s) RPPFLOOR(srcLocationColumn);

                    pixel = ((*srcNewPtrTopRow) * (1 - weightedWidth) * (1 - weightedHeight)) 
                            + ((*(srcNewPtrTopRow + 1)) * (weightedWidth) * (1 - weightedHeight)) 
                            + ((*srcNewPtrBottomRow) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcNewPtrBottomRow + 1)) * (weightedWidth) * (weightedHeight));
                    
                    *dstPtrTemp = (Rpp8u) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRow = ((Rpp32f) i) / hRatio;
            Rpp32f weightedHeight = srcLocationRow - RPPFLOOR(srcLocationRow);

            for (int j = 0; j < dstSize.width; j++)
            {   
                for (int c = 0; c < channel; c++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    Rpp32f weightedWidth = srcLocationColumn - RPPFLOOR(srcLocationColumn);

                    if ((Rpp32s) RPPFLOOR(srcLocationRow) > (srcNewSize.height - 2))
                    {
                        srcLocationRow = srcNewSize.height - 2;
                    }
                    if ((Rpp32s) RPPFLOOR(srcLocationColumn) > (srcNewSize.width - 2))
                    {
                        srcLocationColumn = srcNewSize.width - 2;
                    }
                    
                    srcNewPtrTopRow = srcNewPtr + c + channel * (((Rpp32s) RPPFLOOR(srcLocationRow) * srcNewSize.width) 
                                                                    + (Rpp32s) RPPFLOOR(srcLocationColumn));
                    srcNewPtrBottomRow = srcNewPtr + c + channel * (((Rpp32s) RPPFLOOR(srcLocationRow + 1) * srcNewSize.width) 
                                                                        + (Rpp32s) RPPFLOOR(srcLocationColumn));
                    
                    pixel = ((*srcNewPtrTopRow) * (1 - weightedWidth) * (1 - weightedHeight)) 
                            + ((*(srcNewPtrTopRow + channel)) * (weightedWidth) * (1 - weightedHeight)) 
                            + ((*srcNewPtrBottomRow) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcNewPtrBottomRow + channel)) * (weightedWidth) * (weightedHeight));
                    
                    *dstPtrTemp = (Rpp8u) round(pixel);
                    dstPtrTemp ++;
                }
            }
        }
    }
    
    return RPP_SUCCESS;
}