#ifndef RPP_CPU_COMMON_H
#define RPP_CPU_COMMON_H

#include <math.h>
#include <algorithm>

#include <rppdefs.h>

#define PI 3.14159265
#define RAD(deg)                (deg * PI / 180)
#define RPPABS(a)               ((a < 0) ? (-a) : (a))
#define RPPMIN2(a,b)            ((a < b) ? a : b)
#define RPPMIN3(a,b,c)          ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
#define RPPMAX2(a,b)            ((a > b) ? a : b)
#define RPPMAX3(a,b,c)          ((a > b) && (a > c) ?  a : ((b > c) ? b : c))
#define RPPGAUSSIAN(x, sigma)   (exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * PI * pow(sigma, 2)))
#define RPPDISTANCE(x, y, i, j) (sqrt(pow(x - i, 2) + pow(y - j, 2)))
#define RPPINRANGE(a, x, y)     ((a >= x) && (a <= y) ? 1 : 0)
#define RPPFLOOR(a)             ((int) a)
#define RPPCEIL(a)              ((int) (a + 1.0))
#define RPPISEVEN(a)            ((a % 2 == 0) ? 1 : 0)

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

template <typename T>
RppStatus generate_bilateral_kernel_host(Rpp32f sigmaI, Rpp32f sigmaS, Rpp32f* kernel, unsigned int kernelSize, 
                                         T* srcPtrWindow, RppiSize srcSizeMod, Rpp32u rowEndIncrement, 
                                         RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32f sum = 0.0, multiplierI, multiplierS;
    unsigned int count = 0;

    int bound = ((kernelSize - 1) / 2);

    multiplierI = -1 / (2 * sigmaI * sigmaI);
    multiplierS = -1 / (2 * sigmaS * sigmaS);
    
    T *srcPtrWindowTemp, *srcPtr;
    srcPtrWindowTemp = srcPtrWindow;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtr = srcPtrWindow + (bound * srcSizeMod.width) + bound;
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                Rpp8u pixel = *srcPtr - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                //pixel = pixel * pixel;
                pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
                pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                kernel[count] = exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum += kernel[count];
                count += 1;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += rowEndIncrement;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtr = srcPtrWindow + channel * ((bound * srcSizeMod.width) + bound);
        for (int i = -bound; i <= bound; i++)
        {
            for (int j = -bound; j <= bound; j++)
            {
                Rpp8u pixel = *srcPtr - *srcPtrWindowTemp;
                pixel = RPPABS(pixel);
                //pixel = pixel * pixel;
                pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
                pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                kernel[count] = exp((multiplierS * (i*i + j*j)) + (multiplierI * pixel));
                sum += kernel[count];
                count += 1;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += rowEndIncrement;
        }
    }

    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
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

template<typename T>
RppStatus convolution_kernel_host(T* srcPtrWindow, T* dstPtrPixel, RppiSize srcSize, 
                                       Rpp32f* kernel, unsigned int kernelSize, int remainingElementsInRowPlanar, int remainingElementsInRowPacked, 
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
                
                pixel += ((Rpp32f)(*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp++;
            }
            srcPtrWindowTemp += remainingElementsInRowPlanar;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int m = 0; m < kernelSize; m++)
        {
            for (int n = 0; n < kernelSize; n++)
            {
                pixel += ((Rpp32f)(*kernelPtrTemp) * (Rpp32f)(*srcPtrWindowTemp));
                kernelPtrTemp++;
                srcPtrWindowTemp += channel;
            }
            srcPtrWindowTemp += remainingElementsInRowPacked;
        }
    }
    pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
    *dstPtrPixel = (Rpp8u) round(pixel);

    return RPP_SUCCESS;
}

template<typename T>
RppStatus convolve_image_host(T* srcPtrMod, RppiSize srcSizeMod, T* dstPtr, RppiSize srcSize, 
                        Rpp32f* kernel, unsigned int kernelSize, 
                        RppiChnFormat chnFormat, unsigned int channel)
{
    int remainingElementsInRowPlanar = srcSizeMod.width - kernelSize;
    int remainingElementsInRowPacked = (srcSizeMod.width - kernelSize) * channel;
    
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
                                                 kernel, kernelSize, remainingElementsInRowPlanar, remainingElementsInRowPacked, 
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
                                                 kernel, kernelSize, remainingElementsInRowPlanar, remainingElementsInRowPacked, 
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
    int remainingElementsInRowPlanar = srcSize.width - kernelSize;
    int remainingElementsInRowPacked = (srcSize.width - kernelSize) * channel;
    
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
                                                 kernel, kernelSize, remainingElementsInRowPlanar, remainingElementsInRowPacked, 
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
                                                 kernel, kernelSize, remainingElementsInRowPlanar, remainingElementsInRowPacked, 
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

#endif //RPP_CPU_COMMON_H