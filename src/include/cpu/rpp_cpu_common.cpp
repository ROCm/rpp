#include "rpp_cpu_common.hpp"





// Generate Functions

RppStatus generate_gaussian_kernel_host(Rpp32f stdDev, Rpp32f* kernel, unsigned int kernelSize)
{
    Rpp32f s, sum = 0.0, multiplier;
    int bound = ((kernelSize - 1) / 2);
    unsigned int c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }

    return RPP_SUCCESS;
}

RppStatus generate_gaussian_kernel_asymmetric_host(Rpp32f stdDev, Rpp32f* kernel, Rpp32u kernelSizeX, Rpp32u kernelSizeY)
{
    Rpp32f s, sum = 0.0, multiplier;
    if (kernelSizeX % 2 == 0)
    {
        return RPP_ERROR;
    }
    if (kernelSizeY % 2 == 0)
    {
        return RPP_ERROR;
    }
    int boundX = ((kernelSizeX - 1) / 2);
    int boundY = ((kernelSizeY - 1) / 2);
    unsigned int c = 0;
    s = 1 / (2 * stdDev * stdDev);
    multiplier = (1 / M_PI) * (s);
    for (int i = -boundY; i <= boundY; i++)
    {
        for (int j = -boundX; j <= boundX; j++)
        {
            kernel[c] = multiplier * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSizeX * kernelSizeY); i++)
    {
        kernel[i] /= sum;
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus generate_bilateral_kernel_host(Rpp32f multiplierI, Rpp32f multiplierS, Rpp32f multiplier, Rpp32f* kernel, unsigned int kernelSize, int bound, 
                                         T* srcPtrWindow, RppiSize srcSizeMod, Rpp32u remainingElementsInRow, Rpp32u incrementToWindowCenter, 
                                         RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32f sum = 0.0;
    Rpp32f* kernelTemp;
    kernelTemp = kernel;
    
    T *srcPtrWindowTemp, *srcPtrWindowCenter;
    srcPtrWindowTemp = srcPtrWindow;
    srcPtrWindowCenter = srcPtrWindow + incrementToWindowCenter;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                T pixel = *srcPtrWindowCenter - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                pixel = pixel * pixel;
                *kernelTemp = multiplier * exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum = sum + *kernelTemp;
                kernelTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }

    kernelTemp = kernel;
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = *kernelTemp / sum;
        kernelTemp++;
    }
    
    return RPP_SUCCESS;
}

template <typename T>
RppStatus generate_evenly_padded_image_host(T* srcPtr, RppiSize srcSize, T* srcPtrMod, RppiSize srcSizeMod, 
                                     RppiChnFormat chnFormat, unsigned int channel)
{
    if (RPPISEVEN(srcSize.height) != RPPISEVEN(srcSizeMod.height) 
        || RPPISEVEN(srcSize.width) != RPPISEVEN(srcSizeMod.width)
        || srcSizeMod.height < srcSize.height
        || srcSizeMod.width < srcSize.width)
    {
        return RPP_ERROR;
    }
    T *srcPtrTemp, *srcPtrModTemp;
    srcPtrTemp = srcPtr;
    srcPtrModTemp = srcPtrMod;
    int bound = (srcSizeMod.height - srcSize.height) / 2;
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < srcSizeMod.width; i++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
            }
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int b = 0; b < bound; b++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
                for (int j = 0; j < srcSize.width; j++)
                {
                    *srcPtrModTemp = *srcPtrTemp;
                    srcPtrModTemp++;
                    srcPtrTemp++;
                }
                for (int b = 0; b < bound; b++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
            }
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < srcSizeMod.width; i++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
            }
        }
    }
    else if(chnFormat == RPPI_CHN_PACKED)
    {
        for (int b = 0; b < bound; b++)
        {
            for (int i = 0; i < srcSizeMod.width; i++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }

            }
        }

        for (int i = 0; i < srcSize.height; i++)
        {
            for (int b = 0; b < bound; b++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
            }
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *srcPtrModTemp = *srcPtrTemp;
                    srcPtrModTemp++;
                    srcPtrTemp++;
                }
            }
            for (int b = 0; b < bound; b++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
            }
        }

        for (int b = 0; b < bound; b++)
        {
            for (int i = 0; i < srcSizeMod.width; i++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *srcPtrModTemp = 0;
                    srcPtrModTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus generate_box_kernel_host(Rpp32f* kernel, unsigned int kernelSize)
{
    Rpp32f* kernelTemp;
    kernelTemp = kernel;
    Rpp32f kernelValue = 1.0 / (Rpp32f) (kernelSize * kernelSize);
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        *kernelTemp = kernelValue;
        kernelTemp++;
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus generate_crop_host(T* srcPtr, RppiSize srcSize, T* srcPtrSubImage, RppiSize srcSizeSubImage, T* dstPtr, 
                             RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrSubImageTemp, *dstPtrTemp;
    srcPtrSubImageTemp = srcPtrSubImage;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = (srcSize.width - srcSizeSubImage.width);
        for (int c = 0; c < channel; c++)
        {
            srcPtrSubImageTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                for (int j = 0; j < srcSizeSubImage.width; j++)
                {
                    *dstPtrTemp = *srcPtrSubImageTemp;
                    srcPtrSubImageTemp++;
                    dstPtrTemp++;
                }
                srcPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize.width - srcSizeSubImage.width);
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            for (int j = 0; j < srcSizeSubImage.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *srcPtrSubImageTemp;
                    srcPtrSubImageTemp++;
                    dstPtrTemp++;
                }
            }
            srcPtrSubImageTemp += remainingElementsInRow;
        }
    }

    return RPP_SUCCESS;
}


















// Kernels for functions

template<typename T>
RppStatus convolution_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32f* kernel, unsigned int kernelSize, Rpp32u remainingElementsInRow, 
                                       RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32f pixel = 0.0;

    T* srcPtrWindowTemp;
    srcPtrWindowTemp = srcPtrWindow;
    
    Rpp32f* kernelPtrTemp;
    kernelPtrTemp = kernel;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                pixel += ((*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRow;
        }
    }
    pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
    *dstPtrPixel = (Rpp8u) round(pixel);

    return RPP_SUCCESS;
}

template<typename T>
RppStatus histogram_kernel_host(T* srcPtr, RppiSize srcSize, Rpp32u* histogram, 
                                Rpp32u bins, 
                                unsigned int channel)
{
    Rpp32u elementsInBin = ((Rpp32u)(std::numeric_limits<T>::max()) + 1) / bins;
    int flag = 0;

    T *srcPtrTemp;
    srcPtrTemp = srcPtr;
    Rpp32u *histogramTemp;
    histogramTemp = histogram;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        flag = 0;
        for (int binCheck = 0; binCheck < bins - 1; binCheck++)
        {
            if (*srcPtrTemp >= binCheck * elementsInBin && *srcPtrTemp <= ((binCheck + 1) * elementsInBin) - 1)
            {
                *(histogramTemp + binCheck) += 1;
                flag = 1;
                break;
            }
        }
        if (flag == 0)
        {
            *(histogramTemp + bins - 1) += 1;
        }
        srcPtrTemp++;
    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    if (dstSize.height < 0 || dstSize.width < 0)
    {
        return RPP_ERROR;
    }

    Rpp32f hRatio = (((Rpp32f) (dstSize.height - 1)) / ((Rpp32f) (srcSize.height - 1)));
    Rpp32f wRatio = (((Rpp32f) (dstSize.width - 1)) / ((Rpp32f) (srcSize.width - 1)));
    Rpp32f srcLocationRow, srcLocationColumn, pixel;
    Rpp32s srcLocationRowFloor, srcLocationColumnFloor;
    T *srcPtrTemp, *dstPtrTemp, *srcPtrTopRow, *srcPtrBottomRow;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < dstSize.height; i++)
            {   
                srcLocationRow = ((Rpp32f) i) / hRatio;
                srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
                Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;
                if (srcLocationRowFloor > (srcSize.height - 2))
                {
                    srcLocationRowFloor = srcSize.height - 2;
                }

                srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * srcSize.width;
                srcPtrBottomRow  = srcPtrTopRow + srcSize.width;
                
                for (int j = 0; j < dstSize.width; j++)
                {
                    srcLocationColumn = ((Rpp32f) j) / wRatio;
                    srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
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
                    dstPtrTemp ++;
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
            srcLocationRow = ((Rpp32f) i) / hRatio;
            srcLocationRowFloor = (Rpp32s) RPPFLOOR(srcLocationRow);
            Rpp32f weightedHeight = srcLocationRow - srcLocationRowFloor;

            if (srcLocationRowFloor > (srcSize.height - 2))
            {
                srcLocationRowFloor = srcSize.height - 2;
            }

            srcPtrTopRow = srcPtrTemp + srcLocationRowFloor * elementsInRow;
            srcPtrBottomRow  = srcPtrTopRow + elementsInRow;

            for (int j = 0; j < dstSize.width; j++)
            {   
                srcLocationColumn = ((Rpp32f) j) / wRatio;
                srcLocationColumnFloor = (Rpp32s) RPPFLOOR(srcLocationColumn);
                Rpp32f weightedWidth = srcLocationColumn - srcLocationColumnFloor;

                if (srcLocationColumnFloor > (srcSize.width - 2))
                {
                    srcLocationColumnFloor = srcSize.width - 2;
                }

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

    return RPP_SUCCESS;
}

template <typename T>
RppStatus resize_crop_kernel_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize,
                           Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    RppiSize srcSizeSubImage;
    T *srcPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    T *srcPtrResize = (T*) calloc(channel * srcSizeSubImage.height * srcSizeSubImage.width, sizeof(T));

    generate_crop_host(srcPtr, srcSize, srcPtrSubImage, srcSizeSubImage, srcPtrResize, chnFormat, channel);

    resize_kernel_host(srcPtrResize, srcSizeSubImage, dstPtr, dstSize, chnFormat, channel);

    return RPP_SUCCESS;
    
}




















// Convolution Functions

template<typename T>
RppStatus convolve_image_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSize, 
                        Rpp32f* kernel, unsigned int kernelSize, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;
    
    T *srcPtrWindow, *dstPtrTemp;
    srcPtrWindow = srcPtrMod;
    dstPtrTemp = dstPtr;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRowPlanar, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += (kernelSize - 1);
            }
            srcPtrWindow += ((kernelSize - 1) * srcSizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRowPacked, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += ((kernelSize - 1) * channel);
        }
    }
    
    return RPP_SUCCESS;
}

template<typename T>
RppStatus convolve_subimage_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSizeSubImage, RppiSize srcSize, 
                        Rpp32f* kernel, unsigned int kernelSize, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u remainingElementsInRowPlanar = srcSize.width - kernelSize;
    Rpp32u remainingElementsInRowPacked = (srcSize.width - kernelSize) * channel;
    
    int widthDiffPlanar = srcSize.width - srcSizeSubImage.width;
    int widthDiffPacked = (srcSize.width - srcSizeSubImage.width) * channel;

    T *srcPtrWindow, *dstPtrTemp;
    
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrWindow = srcPtrMod + (c * srcSize.height * srcSize.width);
            dstPtrTemp = dstPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSizeSubImage.height; i++)
            {
                for (int j = 0; j < srcSizeSubImage.width; j++)
                {
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRowPlanar, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
                srcPtrWindow += widthDiffPlanar;
                dstPtrTemp += widthDiffPlanar;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrWindow = srcPtrMod;
        dstPtrTemp = dstPtr;
        for (int i = 0; i < srcSizeSubImage.height; i++)
        {
            for (int j = 0; j < srcSizeSubImage.width; j++)
            {
                for (int c = 0; c < channel; c++)
                {   
                    convolution_kernel_host(srcPtrWindow, dstPtrTemp, srcSize, 
                                                 kernel, kernelSize, remainingElementsInRowPacked, 
                                                 chnFormat, channel);
                    srcPtrWindow++;
                    dstPtrTemp++;
                }
            }
            srcPtrWindow += widthDiffPacked;
            dstPtrTemp += widthDiffPacked;
        }
    }
    
    return RPP_SUCCESS;
}
















// Compute Functions

template<typename T>
RppStatus compute_subimage_location_host(T* ptr, T** ptrSubImage, 
                                         RppiSize size, RppiSize *sizeSubImage, 
                                         unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, 
                                         RppiChnFormat chnFormat, unsigned int channel)
{
    if ((RPPINRANGE(x1, 0, size.width - 1) == 0) 
        || (RPPINRANGE(x2, 0, size.width - 1) == 0) 
        || (RPPINRANGE(y1, 0, size.height - 1) == 0) 
        || (RPPINRANGE(y2, 0, size.height - 1) == 0))
    {
        return RPP_ERROR;
    }
    
    int yDiff = (int) y2 - (int) y1;
    int xDiff = (int) x2 - (int) x1;

    sizeSubImage->height = (Rpp32u) RPPABS(yDiff) + 1;
    sizeSubImage->width = (Rpp32u) RPPABS(xDiff) + 1;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width) + RPPMIN2(x1, x2);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        *ptrSubImage = ptr + (RPPMIN2(y1, y2) * size.width * channel) + (RPPMIN2(x1, x2) * channel);
    }

    return RPP_SUCCESS;
}

template<typename T>
RppStatus compute_transpose_host(T* srcPtr, RppiSize srcSize, T* dstPtr, RppiSize dstSize, 
                                 RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtr + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < dstSize.height; i++)
            {
                for (int j = 0; j < dstSize.width; j++)
                {
                    *dstPtrTemp = *(srcPtrTemp + (j * srcSize.width) + i);
                    dstPtrTemp++;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < dstSize.height; i++)
        {
            for (int j = 0; j < dstSize.width; j++)
            {
                srcPtrTemp = srcPtr + (channel * ((j * srcSize.width) + i));
                for (int c = 0; c < channel; c++)
                {
                    *dstPtrTemp = *(srcPtrTemp + c);
                    dstPtrTemp++;
                }
            }
        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus compute_multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   unsigned int channel)
{
    U pixel;
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        pixel = ((U) srcPtr1[i]) * ((U) srcPtr2[i]);
        pixel = (pixel < (U) 255) ? pixel : ((U) 255);
        pixel = (pixel > (U) 0) ? pixel : ((U) 0);
        dstPtr[i] =(T) pixel;
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus compute_rgb_to_hsl_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, unsigned channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = 0; i < (srcSize.width * srcSize.height); i++)
        {
            float rf, gf, bf, cmax, cmin, delta, divisor;
            rf = ((float) srcPtr[i]) / 255;
            gf = ((float) srcPtr[i + (srcSize.width * srcSize.height)]) / 255;
            bf = ((float) srcPtr[i + (2 * srcSize.width * srcSize.height)]) / 255;
            cmax = ((rf > gf) && (rf > bf)) ? rf : ((gf > bf) ? gf : bf);
            cmin = ((rf < gf) && (rf < bf)) ? rf : ((gf < bf) ? gf : bf);
            divisor = cmax + cmin - 1;
            delta = cmax - cmin;

            if (delta == 0)
            {
                dstPtr[i] = 0;
            }
            else if (cmax == rf)
            {
                dstPtr[i] = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                dstPtr[i] = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                dstPtr[i] = round(60 * (((rf - gf) / delta) + 4));
            }
            
            while (dstPtr[i] > 360)
            {
                dstPtr[i] = dstPtr[i] - 360;
            }
            while (dstPtr[i] < 0)
            {
                dstPtr[i] = 360 + dstPtr[i];
            }

            if (delta == 0)
            {
                dstPtr[i + (srcSize.width * srcSize.height)] = 0;
            }
            else
            {
                dstPtr[i + (srcSize.width * srcSize.height)] = delta / (1 - RPPABS(divisor));
            }

            dstPtr[i + (2 * srcSize.width * srcSize.height)] = (cmax + cmin) / 2;

        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < (3 * srcSize.width * srcSize.height); i += 3)
        {
            float rf, gf, bf, cmax, cmin, delta, divisor;
            rf = ((float) srcPtr[i]) / 255;
            gf = ((float) srcPtr[i + 1]) / 255;
            bf = ((float) srcPtr[i + 2]) / 255;
            cmax = ((rf > gf) && (rf > bf)) ? rf : ((gf > bf) ? gf : bf);
            cmin = ((rf < gf) && (rf < bf)) ? rf : ((gf < bf) ? gf : bf);
            divisor = cmax + cmin - 1;
            delta = cmax - cmin;

            if (delta == 0)
            {
                dstPtr[i] = 0;
            }
            else if (cmax == rf)
            {
                dstPtr[i] = round(60 * fmod(((gf - bf) / delta),6));
            }
            else if (cmax == gf)
            {
                dstPtr[i] = round(60 * (((bf - rf) / delta) + 2));
            }
            else if (cmax == bf)
            {
                dstPtr[i] = round(60 * (((rf - gf) / delta) + 4));
            }
            
            while (dstPtr[i] > 360)
            {
                dstPtr[i] = dstPtr[i] - 360;
            }
            while (dstPtr[i] < 0)
            {
                dstPtr[i] = 360 + dstPtr[i];
            }

            if (delta == 0)
            {
                dstPtr[i + 1] = 0;
            }
            else
            {
                dstPtr[i + 1] = delta / (1 - RPPABS(divisor));
            }

            dstPtr[i + 2] = (cmax + cmin) / 2;

        }
    }

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus compute_hsl_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, unsigned channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = 0; i < (srcSize.width * srcSize.height); i++)
        {
            float c, x, m, rf, gf, bf;
            c = (2 * srcPtr[i + (2 * srcSize.width * srcSize.height)]) - 1;
            c = (1 - RPPABS(c)) * srcPtr[i + (srcSize.width * srcSize.height)];
            x = c * (1 - abs((fmod((srcPtr[i] / 60), 2)) - 1));
            m = srcPtr[i + (2 * srcSize.width * srcSize.height)] - c / 2;
            
            if ((0 <= srcPtr[i]) && (srcPtr[i] < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= srcPtr[i]) && (srcPtr[i] < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= srcPtr[i]) && (srcPtr[i] < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= srcPtr[i]) && (srcPtr[i] < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= srcPtr[i]) && (srcPtr[i] < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= srcPtr[i]) && (srcPtr[i] < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            dstPtr[i] = (Rpp8u) round((rf + m) * 255);
            dstPtr[i + (srcSize.width * srcSize.height)] = (Rpp8u) round((gf + m) * 255);
            dstPtr[i + (2 * srcSize.width * srcSize.height)] = (Rpp8u) round((bf + m) * 255);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        printf("\nInside\n");
        for (int i = 0; i < (3 * srcSize.width * srcSize.height); i += 3)
        {
            float c, x, m, rf, gf, bf;
            c = (2 * srcPtr[i + 2]) - 1;
            c = (1 - RPPABS(c)) * srcPtr[i + 1];
            x = c * (1 - abs((fmod((srcPtr[i] / 60), 2)) - 1));
            m = srcPtr[i + 2] - c / 2;
            
            if ((0 <= srcPtr[i]) && (srcPtr[i] < 60))
            {
                rf = c;
                gf = x;
                bf = 0;
            }
            else if ((60 <= srcPtr[i]) && (srcPtr[i] < 120))
            {
                rf = x;
                gf = c;
                bf = 0;
            }
            else if ((120 <= srcPtr[i]) && (srcPtr[i] < 180))
            {
                rf = 0;
                gf = c;
                bf = x;
            }
            else if ((180 <= srcPtr[i]) && (srcPtr[i] < 240))
            {
                rf = 0;
                gf = x;
                bf = c;
            }
            else if ((240 <= srcPtr[i]) && (srcPtr[i] < 300))
            {
                rf = x;
                gf = 0;
                bf = c;
            }
            else if ((300 <= srcPtr[i]) && (srcPtr[i] < 360))
            {
                rf = c;
                gf = 0;
                bf = x;
            }

            dstPtr[i] = (Rpp8u) round((rf + m) * 255);
            dstPtr[i + 1] = (Rpp8u) round((gf + m) * 255);
            dstPtr[i + 2] = (Rpp8u) round((bf + m) * 255);
        }
    }

    return RPP_SUCCESS;
}

#endif //RPP_CPU_COMMON_H