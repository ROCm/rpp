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




/**************** Resize ***************/

template <typename T>
RppStatus resize_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    resize_kernel_host(srcPtr, srcSize, dstPtr, dstSize, chnFormat, channel);

    return RPP_SUCCESS;
}




/**************** Resize Crop ***************/

template <typename T>
RppStatus resize_crop_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    resize_crop_kernel_host(srcPtr, srcSize, dstPtr, dstSize, x1, y1, x2, y2, chnFormat, channel);

    return RPP_SUCCESS;
    
}




/**************** Rotate ***************/

RppStatus rotate_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                       Rpp32f angleDeg)
{
    Rpp32f angleRad = -RAD(angleDeg);
    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);
    
    float minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (rotate[0] * i) + (rotate[1] * j);
            newj = (rotate[2] * i) + (rotate[3] * j);
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
    Rpp32f angleRad = -RAD(angleDeg);
    Rpp32f rotate[4] = {0};
    rotate[0] = cos(angleRad);
    rotate[1] = sin(angleRad);
    rotate[2] = -sin(angleRad);
    rotate[3] = cos(angleRad);
    
    Rpp32f minX = 0, minY = 0;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    for (int i = 0; i < srcSize.height; i++)
    {
        for (int j = 0; j < srcSize.width; j++)
        {
            Rpp32f newi = 0, newj = 0;
            newi = (rotate[0] * i) + (rotate[1] * j);
            newj = (rotate[2] * i) + (rotate[3] * j);
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

    Rpp32f divisor = (rotate[1] * rotate[2]) - (rotate[0] * rotate[3]);
    Rpp32f srcLocationRow, srcLocationColumn, srcLocationRowTerm1, srcLocationColumnTerm1, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;

    Rpp32f srcLocationRowParameter = (-rotate[3] * (Rpp32s)minX) + (rotate[1] * (Rpp32s)minY);
    Rpp32f srcLocationColumnParameter = (rotate[2] * (Rpp32s)minX) + (-rotate[0] * (Rpp32s)minY);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                srcLocationRowTerm1 = -rotate[3] * i;
                srcLocationColumnTerm1 = rotate[2] * i;
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter) / divisor;
                    srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter) / divisor;
                    
                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                    
                        *dstPtrTemp = (Rpp8u) round(pixel);
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRowTerm1 = -rotate[3] * i;
            srcLocationColumnTerm1 = rotate[2] * i;
            for (int j = 0; j < dstSize.width; j++)
            {
                srcLocationRow = (srcLocationRowTerm1 + (rotate[1] * j) + srcLocationRowParameter) / divisor;
                srcLocationColumn = (srcLocationColumnTerm1 + (-rotate[0] * j) + srcLocationColumnParameter) / divisor;
                
                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                }
                else
                {
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                    
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                    
                        *dstPtrTemp = (Rpp8u) round(pixel);
                        dstPtrTemp ++;
                    }
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
    Rpp32f minX = 0, minY = 0;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
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

    Rpp32f divisor = (affine[1] * affine[3]) - (affine[0] * affine[4]);
    Rpp32f srcLocationRow, srcLocationColumn, srcLocationRowTerm1, srcLocationColumnTerm1, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;

    Rpp32f srcLocationRowParameter = (-affine[4] * (-affine[2] + (Rpp32s)minX)) + (affine[1] * (-affine[5] + (Rpp32s)minY));
    Rpp32f srcLocationColumnParameter = (affine[3] * (-affine[2] + (Rpp32s)minX)) + (-affine[0] * (-affine[5] + (Rpp32s)minY));

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {
                srcLocationRowTerm1 = -affine[4] * i;
                srcLocationColumnTerm1 = affine[3] * i;
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationRow = (srcLocationRowTerm1 + (affine[1] * j) + srcLocationRowParameter) / divisor;
                    srcLocationColumn = (srcLocationColumnTerm1 + (-affine[0] * j) + srcLocationColumnParameter) / divisor;
                    
                    if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                    else
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        
                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));
                    
                        *dstPtrTemp = (Rpp8u) round(pixel);
                        dstPtrTemp ++;
                    }
                }
            }
            srcPtrTemp += srcSize.height * srcSize.width;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < dstSize.height; i++)
        {
            srcLocationRowTerm1 = -affine[4] * i;
            srcLocationColumnTerm1 = affine[3] * i;
            for (int j = 0; j < dstSize.width; j++)
            {
                srcLocationRow = (srcLocationRowTerm1 + (affine[1] * j) + srcLocationRowParameter) / divisor;
                srcLocationColumn = (srcLocationColumnTerm1 + (-affine[0] * j) + srcLocationColumnParameter) / divisor;
                
                if (srcLocationRow < 0 || srcLocationColumn < 0 || srcLocationRow > (srcSize.height - 2) || srcLocationColumn > (srcSize.width - 2))
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = 0;
                        dstPtrTemp++;
                    }
                }
                else
                {
                    srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                    Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                    Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                    
                    srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                    srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                    Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                    for (int c = 0; c < channel; c++)
                    {
                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                    
                        *dstPtrTemp = (Rpp8u) round(pixel);
                        dstPtrTemp ++;
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}



/**************** Fish Eye Effect ***************/

template <typename T>
RppStatus fisheye_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f newI, newJ, newIsrc, newJsrc, newIsquared, newJsquared, euclideanDistance, newEuclideanDistance, theta;
    int iSrc, jSrc, srcPosition;
    Rpp32u elementsPerChannel = srcSize.height * srcSize.width;
    Rpp32u elements = channel * elementsPerChannel;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for(int i = 0; i < srcSize.height; i++)
            {
                newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(srcSize.height))) - 1.0;
                newIsquared = newI * newI;
                for(int j = 0; j < srcSize.width; j++)
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
                                *dstPtrTemp = *(srcPtrTemp + srcPosition);
                            }
                        }
                    }
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int i = 0; i < srcSize.height; i++)
        {
            newI = (((Rpp32f) (i * 2.0)) / ((Rpp32f)(srcSize.height))) - 1.0;
            newIsquared = newI * newI;
            for(int j = 0; j < srcSize.width; j++)
            {
                for(int c = 0; c < channel; c++)
                {
                    srcPtrTemp = srcPtr + c;
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
                                *dstPtrTemp = *(srcPtrTemp + srcPosition);
                            }
                        }
                    }
                    dstPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Lens Correction ***************/

template <typename T>
RppStatus lens_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                               Rpp32f strength, Rpp32f zoom, 
                               RppiChnFormat chnFormat, unsigned int channel)
{
    if (strength < 0)
    {
        return RPP_ERROR;
    }

    if (zoom < 1)
    {
        return RPP_ERROR;
    }
    
    Rpp32f halfHeight, halfWidth, newI, newJ, correctionRadius, euclideanDistance, correctedDistance, theta;
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    halfHeight = ((Rpp32f) srcSize.height) / 2.0;
    halfWidth = ((Rpp32f) srcSize.width) / 2.0;

    if (strength == 0) strength = 0.000001;

    correctionRadius = sqrt(srcSize.height * srcSize.height + srcSize.width * srcSize.width) / strength;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSize.height; i++)
            {
                newI = i - halfHeight;
                for (int j = 0; j < srcSize.width; j++)
                {
                    newJ = j - halfWidth;

                    euclideanDistance = sqrt(newI * newI + newJ * newJ);
                    
                    correctedDistance = euclideanDistance / correctionRadius;

                    if(correctedDistance == 0)
                    {
                        theta = 1;
                    }
                    else
                    {
                        theta = atan(correctedDistance) / correctedDistance;
                    }

                    srcLocationRow = halfHeight + theta * newI * zoom;
                    srcLocationColumn = halfWidth + theta * newJ * zoom;
                    
                    if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) && 
                        (srcLocationRow < srcSize.height) && (srcLocationColumn < srcSize.width))
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        if (srcLocationRowFloor > (srcSize.height - 2))
                        {
                            srcLocationRowFloor = srcSize.height - 2;
                        }

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                        srcPtrBottomRow  = srcPtrTopRow + srcSize.width;

                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }

                        pixel = ((*(srcPtrTopRow + srcLocationColumnFloor)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrTopRow + srcLocationColumnFloor + 1)) * (1 - weightedHeight) * (weightedWidth)) 
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor)) * (weightedHeight) * (1 - weightedWidth)) 
                                + ((*(srcPtrBottomRow + srcLocationColumnFloor + 1)) * (weightedHeight) * (weightedWidth));

                        *dstPtrTemp = (Rpp8u) round(pixel);
                    }
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32s elementsInRow = srcSize.width * channel;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for(int c = 0; c < channel; c++)
                {
                    newI = i - halfHeight;
                    newJ = j - halfWidth;

                    euclideanDistance = sqrt(newI * newI + newJ * newJ);
                    
                    correctedDistance = euclideanDistance / correctionRadius;

                    if(correctedDistance == 0)
                    {
                        theta = 1;
                    }
                    else
                    {
                        theta = atan(correctedDistance) / correctedDistance;
                    }

                    srcLocationRow = halfHeight + theta * newI * zoom;
                    srcLocationColumn = halfWidth + theta * newJ * zoom;

                    if ((srcLocationRow >= 0) && (srcLocationColumn >= 0) && 
                        (srcLocationRow < srcSize.height) && (srcLocationColumn < srcSize.width))
                    {
                        srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                        srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);

                        Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                        if (srcLocationRowFloor > (srcSize.height - 2))
                        {
                            srcLocationRowFloor = srcSize.height - 2;
                        }

                        srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
                        srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

                        Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;
                        if (srcLocationColumnFloor > (srcSize.width - 2))
                        {
                            srcLocationColumnFloor = srcSize.width - 2;
                        }

                        Rpp32s srcLocColFloorChanneled = channel * srcLocationColumnFloor;

                        pixel = ((*(srcPtrTopRow + c + srcLocColFloorChanneled)) * (1 - weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrTopRow + c + srcLocColFloorChanneled + channel)) * (1 - weightedHeight) * (weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled)) * (weightedHeight) * (1 - weightedWidth)) 
                            + ((*(srcPtrBottomRow + c + srcLocColFloorChanneled + channel)) * (weightedHeight) * (weightedWidth));
                        
                        *dstPtrTemp = (Rpp8u) round(pixel);
                    }
                    dstPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}
