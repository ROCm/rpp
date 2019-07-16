#include <cpu/rpp_cpu_common.hpp>
#include "cpu/host_geometry_transforms.hpp"
#include "cpu/host_color_model_conversions.hpp"
#include <stdlib.h>
#include <time.h>

/************ Blur************/

template <typename T>
RppStatus blur_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev, unsigned int kernelSize,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }
    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));
    int bound = ((kernelSize - 1) / 2);

    generate_gaussian_kernel_host(stdDev, kernel, kernelSize);

    RppiSize srcSizeMod;
    srcSizeMod.width = srcSize.width + (2 * bound);
    srcSizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *srcPtrMod = (Rpp8u *)calloc(srcSizeMod.width * srcSizeMod.height * channel, sizeof(Rpp8u));

    generate_evenly_padded_image_host(srcPtr, srcSize, srcPtrMod, srcSizeMod, chnFormat, channel);
    
    convolve_image_host(srcPtrMod, srcSizeMod, dstPtr, srcSize, kernel, kernelSize, chnFormat, channel);
    
    return RPP_SUCCESS;
}

/**************** Contrast ***************/

template <typename T>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for(int c = 0; c < channel; c++)
        {
            Rpp32f Min, Max;
            Min = srcPtr[c * srcSize.height * srcSize.width];
            Max = srcPtr[c * srcSize.height * srcSize.width];
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (srcPtr[i + (c * srcSize.height * srcSize.width)] < Min)
                {
                    Min = srcPtr[i + (c * srcSize.height * srcSize.width)];
                }
                if (srcPtr[i + (c * srcSize.height * srcSize.width)] > Max)
                {
                    Max = srcPtr[i + (c * srcSize.height * srcSize.width)];
                }
            }
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32f pixel = (Rpp32f) srcPtr[i + (c * srcSize.height * srcSize.width)];
                pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
                pixel = (pixel < (Rpp32f)new_max) ? pixel : ((Rpp32f)new_max);
                pixel = (pixel > (Rpp32f)new_min) ? pixel : ((Rpp32f)new_min);
                dstPtr[i + (c * srcSize.height * srcSize.width)] = (Rpp8u) pixel;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for(int c = 0; c < channel; c++)
        {
            Rpp32f Min, Max;
            Min = srcPtr[c];
            Max = srcPtr[c];
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                if (srcPtr[(channel * i) + c] < Min)
                {
                    Min = srcPtr[(channel * i) + c];
                }
                if (srcPtr[(channel * i) + c] > Max)
                {
                    Max = srcPtr[(channel * i) + c];
                }
            }
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                Rpp32f pixel = (Rpp32f) srcPtr[(channel * i) + c];
                pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
                pixel = (pixel < (Rpp32f)new_max) ? pixel : ((Rpp32f)new_max);
                pixel = (pixel > (Rpp32f)new_min) ? pixel : ((Rpp32f)new_min);
                dstPtr[(channel * i) + c] = (Rpp8u) pixel;
            }
        }
    }

    return RPP_SUCCESS;
}

/************ Brightness ************/

template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                            Rpp32f alpha, Rpp32f beta,
                                   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) * alpha + beta;
        pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
        pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}


/**************** Gamma Correction ***************/

template <typename T>
RppStatus gamma_correction_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f gamma,
                                RppiChnFormat chnFormat,   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) / 255;
        pixel = pow(pixel, gamma);
        pixel *= 255;
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Color Temperature ***************/

template <typename T>
RppStatus color_temperature_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp8s adjustmentValue,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if (channel != 3)
    {
        return RPP_ERROR;
    }
    if (adjustmentValue < -100 || adjustmentValue > 100)
    {
        return RPP_ERROR;
    }   

    Rpp32s pixel;
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
            pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
            pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
            *dstPtrTemp = (T) pixel;
            dstPtrTemp++;
            srcPtrTemp++;
        }
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *dstPtrTemp = *srcPtrTemp;
            dstPtrTemp++;
            srcPtrTemp++;
        }
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
            pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
            pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
            *dstPtrTemp = (T) pixel;
            dstPtrTemp++;
            srcPtrTemp++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
            pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
            pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
            *dstPtrTemp = (T) pixel;
            dstPtrTemp++;
            srcPtrTemp++;

            *dstPtrTemp = *srcPtrTemp;
            dstPtrTemp++;
            srcPtrTemp++;

            pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
            pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
            pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
            *dstPtrTemp = (T) pixel;
            dstPtrTemp++;
            srcPtrTemp++;
        }
    }
     
    return RPP_SUCCESS;
}

/**************** Pixelate ***************/

template <typename T>
RppStatus pixelate_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    unsigned int kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, 
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if (kernelSize % 2 == 0)
    {
        return RPP_ERROR;
    }

    unsigned int bound = ((kernelSize - 1) / 2);

    if ((RPPINRANGE(x1, bound, srcSize.width - 1 - bound) == 0) 
        || (RPPINRANGE(x2, bound, srcSize.width - 1 - bound) == 0) 
        || (RPPINRANGE(y1, bound, srcSize.height - 1 - bound) == 0) 
        || (RPPINRANGE(y2, bound, srcSize.height - 1 - bound) == 0))
    {
        return RPP_ERROR;
    }



    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *srcPtrTemp;
        srcPtrTemp++;
        dstPtrTemp++;
    }



    Rpp32f *kernel = (Rpp32f *)calloc(kernelSize * kernelSize, sizeof(Rpp32f));

    generate_box_kernel_host(kernel, kernelSize);

    RppiSize srcSizeMod, srcSizeSubImage;
    T *srcPtrMod, *srcPtrSubImage, *dstPtrSubImage;

    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);
    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    srcSizeMod.height = srcSizeSubImage.height + (2 * bound);
    srcSizeMod.width = srcSizeSubImage.width + (2* bound);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrMod = srcPtrSubImage - (bound * srcSize.width) - bound;
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrMod = srcPtrSubImage - (bound * srcSize.width * channel) - (bound * channel);
    }

    convolve_subimage_host(srcPtrMod, srcSizeMod, dstPtrSubImage, srcSizeSubImage, srcSize, kernel, kernelSize, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** Jitter Add ***************/

template <typename T>
RppStatus jitterAdd_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    unsigned int maxJitterX, unsigned int maxJitterY, 
                    RppiChnFormat chnFormat, unsigned int channel)
{
    if ((RPPINRANGE(maxJitterX, 0, srcSize.width - 1) == 0) 
        || (RPPINRANGE(maxJitterY, 0, srcSize.height - 1) == 0))
    {
        return RPP_ERROR;
    }

    Rpp8u *dstPtrForJitter = (Rpp8u *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp8u));

    T *srcPtrTemp, *dstPtrTemp;
    T *srcPtrBeginJitter, *dstPtrBeginJitter;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtrForJitter;
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *srcPtrTemp;
        srcPtrTemp++;
        dstPtrTemp++;
    }

    srand (time(NULL));
    int jitteredPixelLocDiffX, jitteredPixelLocDiffY;
    int jitterRangeX = 2 * maxJitterX;
    int jitterRangeY = 2 * maxJitterY;

    if (chnFormat == RPPI_CHN_PLANAR)
    {      
        srcPtrBeginJitter = srcPtr + (maxJitterY * srcSize.width) + maxJitterX;
        dstPtrBeginJitter = dstPtrForJitter + (maxJitterY * srcSize.width) + maxJitterX;
        for (int c = 0; c < channel; c++)
        {
            srcPtrTemp = srcPtrBeginJitter + (c * srcSize.height * srcSize.width);
            dstPtrTemp = dstPtrBeginJitter + (c * srcSize.height * srcSize.width);
            for (int i = 0; i < srcSize.height - jitterRangeY; i++)
            {
                for (int j = 0; j < srcSize.width - jitterRangeX; j++)
                {
                    jitteredPixelLocDiffX = (rand() % (jitterRangeX + 1));
                    jitteredPixelLocDiffY = (rand() % (jitterRangeY + 1));
                    jitteredPixelLocDiffX -= maxJitterX;
                    jitteredPixelLocDiffY -= maxJitterY;
                    *dstPtrTemp = *(srcPtrTemp + (jitteredPixelLocDiffY * (int) srcSize.width) + jitteredPixelLocDiffX);
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
                srcPtrTemp += jitterRangeX;
                dstPtrTemp += jitterRangeX;
            }
        }

        resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(dstPtrForJitter), srcSize, static_cast<Rpp8u*>(dstPtr), srcSize,
                            maxJitterX, maxJitterY, srcSize.width - maxJitterX - 1, srcSize.height - maxJitterY - 1,
                            RPPI_CHN_PLANAR, channel);
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        int elementsInRow = (int)(srcSize.width * channel);
        int channeledJitterRangeX = jitterRangeX * channel;
        int channeledJitterRangeY = jitterRangeY * channel;
        srcPtrBeginJitter = srcPtr + (maxJitterY * elementsInRow) + (maxJitterX * channel);
        dstPtrBeginJitter = dstPtrForJitter + (maxJitterY * elementsInRow) + (maxJitterX * channel);
        srcPtrTemp = srcPtrBeginJitter;
        dstPtrTemp = dstPtrBeginJitter;
        for (int i = 0; i < srcSize.height - jitterRangeY; i++)
        {
            for (int j = 0; j < srcSize.width - jitterRangeX; j++)
            {
                for (int c = 0; c < channel; c++)
                {
                    jitteredPixelLocDiffX = rand() % (jitterRangeX + 1);
                    jitteredPixelLocDiffY = rand() % (jitterRangeY + 1);
                    jitteredPixelLocDiffX -= maxJitterX;
                    jitteredPixelLocDiffY -= maxJitterY;
                    *dstPtrTemp = *(srcPtrTemp + (jitteredPixelLocDiffY * elementsInRow) + (jitteredPixelLocDiffX * (int) channel));
                    srcPtrTemp++;
                    dstPtrTemp++;
                }
            }
            srcPtrTemp += channeledJitterRangeX;
            dstPtrTemp += channeledJitterRangeX;
        }
        resize_crop_host<Rpp8u>(static_cast<Rpp8u*>(dstPtrForJitter), srcSize, static_cast<Rpp8u*>(dstPtr), srcSize,
                            maxJitterX, maxJitterY, srcSize.width - maxJitterX - 1, srcSize.height - maxJitterY - 1,
                            RPPI_CHN_PACKED, channel);
    }
    
    return RPP_SUCCESS;
}

/**************** Vignette ***************/

template <typename T>
RppStatus vignette_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32f *mask = (Rpp32f *)calloc(srcSize.height * srcSize.width, sizeof(Rpp32f));
    Rpp32f *maskTemp;
    maskTemp = mask;

    RppiSize kernelRowsSize, kernelColumnsSize;
    kernelRowsSize.height = srcSize.height;
    kernelRowsSize.width = 1;
    kernelColumnsSize.height = srcSize.width;
    kernelColumnsSize.width = 1;

    Rpp32f *kernelRows = (Rpp32f *)calloc(kernelRowsSize.height * kernelRowsSize.width, sizeof(Rpp32f));
    Rpp32f *kernelColumns = (Rpp32f *)calloc(kernelColumnsSize.height * kernelColumnsSize.width, sizeof(Rpp32f));

    if (kernelRowsSize.height % 2 == 0)
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelRows, kernelRowsSize.height - 1, kernelRowsSize.width);
        kernelRows[kernelRowsSize.height - 1] = kernelRows[kernelRowsSize.height - 2];
    }
    else
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelRows, kernelRowsSize.height, kernelRowsSize.width);
    }
    
    if (kernelColumnsSize.height % 2 == 0)
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelColumns, kernelColumnsSize.height - 1, kernelColumnsSize.width);
        kernelColumns[kernelColumnsSize.height - 1] = kernelColumns[kernelColumnsSize.height - 2];
    }
    else
    {
        generate_gaussian_kernel_asymmetric_host(stdDev, kernelColumns, kernelColumnsSize.height, kernelColumnsSize.width);
    }

    Rpp32f *kernelRowsTemp, *kernelColumnsTemp;
    kernelRowsTemp = kernelRows;
    kernelColumnsTemp = kernelColumns;
    
    for (int i = 0; i < srcSize.height; i++)
    {
        kernelColumnsTemp = kernelColumns;
        for (int j = 0; j < srcSize.width; j++)
        {
            *maskTemp = *kernelRowsTemp * *kernelColumnsTemp;
            maskTemp++;
            kernelColumnsTemp++;
        }
        kernelRowsTemp++;
    }

    Rpp32f max = 0;
    maskTemp = mask;
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        if (*maskTemp > max)
        {
            max = *maskTemp;
        }
        maskTemp++;
    }

    maskTemp = mask;
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        *maskTemp = *maskTemp / max;
        maskTemp++;
    }

    Rpp32f *maskFinal = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
    Rpp32f *maskFinalTemp;
    maskFinalTemp = maskFinal;
    maskTemp = mask;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            maskTemp = mask;
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    *maskFinalTemp = *maskTemp;
                    maskFinalTemp++;
                    maskTemp++;
                }
            }
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
                    *maskFinalTemp = *maskTemp;
                    maskFinalTemp++;
                }
                maskTemp++;
            }
        }
    }

    compute_multiply_host(srcPtr, maskFinal, srcSize, dstPtr, channel);
    
    return RPP_SUCCESS;
}

/**************** Fish Eye Effect ***************/

template <typename T>
RppStatus fish_eye_effect_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
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

/**************** Occlusion Add ***************/

template <typename T>
RppStatus occlusionAdd_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize1, RppiSize srcSize2, T* dstPtr, 
                            Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                            Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, 
                            RppiChnFormat chnFormat, unsigned int channel)
{
    RppiSize srcSize1SubImage, srcSize2SubImage, dstSizeSubImage;
    T *srcPtr1SubImage, *srcPtr2SubImage, *dstPtrSubImage;

    compute_subimage_location_host(srcPtr2, &srcPtr2SubImage, srcSize2, &srcSize2SubImage, src2x1, src2y1, src2x2, src2y2, chnFormat, channel);

    T *srcPtrResize = (T*) calloc(channel * srcSize2SubImage.height * srcSize2SubImage.width, sizeof(T));

    generate_crop_host(srcPtr2, srcSize2, srcPtr2SubImage, srcSize2SubImage, srcPtrResize, chnFormat, channel);

    compute_subimage_location_host(srcPtr1, &srcPtr1SubImage, srcSize1, &srcSize1SubImage, src1x1, src1y1, src1x2, src1y2, chnFormat, channel);

    T *dstPtrResize = (T*) calloc(channel * srcSize1SubImage.height * srcSize1SubImage.width, sizeof(T));
    
    resize_kernel_host(srcPtrResize, srcSize2SubImage, dstPtrResize, srcSize1SubImage, chnFormat, channel);

    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize1, &dstSizeSubImage, src1x1, src1y1, src1x2, src1y2, chnFormat, channel);

    T *srcPtr1Temp, *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize1.height * srcSize1.width); i++)
    {
        *dstPtrTemp = *srcPtr1Temp;
        srcPtr1Temp++;
        dstPtrTemp++;
    }

    T *dstPtrResizeTemp, *dstPtrSubImageTemp;
    dstPtrResizeTemp = dstPtrResize;
    dstPtrSubImageTemp = dstPtrSubImage;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u remainingElementsInRow = srcSize1.width - dstSizeSubImage.width;
        for (int c = 0;  c < channel; c++)
        {
            dstPtrSubImageTemp = dstPtrSubImage + (c * srcSize1.height * srcSize1.width);
            for (int i = 0; i < srcSize1SubImage.height; i++)
            {
                for (int j = 0; j < srcSize1SubImage.width; j++)
                {
                    *dstPtrSubImageTemp = *dstPtrResizeTemp;
                    dstPtrResizeTemp++;
                    dstPtrSubImageTemp++;
                }
                dstPtrSubImageTemp += remainingElementsInRow;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        Rpp32u remainingElementsInRow = channel * (srcSize1.width - dstSizeSubImage.width);
        for (int i = 0; i < srcSize1SubImage.height; i++)
        {
            for (int j = 0; j < srcSize1SubImage.width; j++)
            {
                for (int c = 0;  c < channel; c++)
                {
                    *dstPtrSubImageTemp = *dstPtrResizeTemp;
                    dstPtrResizeTemp++;
                    dstPtrSubImageTemp++;
                }
            }
            dstPtrSubImageTemp += remainingElementsInRow;
        }
    }
  
    return RPP_SUCCESS;
}

/**************** Snowy ***************/

template <typename T, typename U>
RppStatus snowy_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f strength,
                    RppiChnFormat chnFormat, unsigned channel, RppiFormat imageFormat)
{
    if (strength < 0 || strength > 1)
    {
        return RPP_ERROR;
    }

    if (imageFormat == RGB)
    {
        Rpp32f *srcPtrHSL = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            rgb2hsl_host(srcPtr, srcSize, srcPtrHSL, RPPI_CHN_PLANAR, 3);

            Rpp32f *srcPtrHSLTemp;
            srcPtrHSLTemp = srcPtrHSL + (2 * srcSize.height * srcSize.width);

            for (int i = 0; i < srcSize.height * srcSize.width; i++)
            {
                if (*srcPtrHSLTemp < strength)
                {
                    *srcPtrHSLTemp *= 4;
                }
                if (*srcPtrHSLTemp > 1)
                {
                    *srcPtrHSLTemp = 1;
                }
                srcPtrHSLTemp++;
            }

            hsl2rgb_host(srcPtrHSL, srcSize, dstPtr, RPPI_CHN_PLANAR, 3);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            rgb2hsl_host(srcPtr, srcSize, srcPtrHSL, RPPI_CHN_PACKED, 3);

            Rpp32f *srcPtrHSLTemp;
            srcPtrHSLTemp = srcPtrHSL + 2;

            for (int i = 0; i < srcSize.height * srcSize.width; i++)
            {
                if (*srcPtrHSLTemp < strength)
                {
                    *srcPtrHSLTemp *= 4;
                }
                if (*srcPtrHSLTemp > 1)
                {
                    *srcPtrHSLTemp = 1;
                }
                srcPtrHSLTemp = srcPtrHSLTemp + channel;
            }

            hsl2rgb_host(srcPtrHSL, srcSize, dstPtr, RPPI_CHN_PACKED, 3);
        }
    }
    else
    {
        return RPP_ERROR;
    }
    return RPP_SUCCESS;
}

/**************** Random Shadow ***************/

template <typename T>
RppStatus random_shadow_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                             Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                             Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, 
                             RppiChnFormat chnFormat, unsigned int channel)
{
    srand (time(NULL));
    RppiSize srcSizeSubImage, dstSizeSubImage, shadowSize;
    T *srcPtrSubImage, *dstPtrSubImage;
    
    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    compute_subimage_location_host(dstPtr, &dstPtrSubImage, srcSize, &dstSizeSubImage, x1, y1, x2, y2, chnFormat, channel);
    
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *srcPtrTemp;
        srcPtrTemp++;
        dstPtrTemp++;
    }

    for (int shadow = 0; shadow < numberOfShadows; shadow++)
    {
        shadowSize.height = rand() % (srcSizeSubImage.height / 2);
        shadowSize.width = rand() % (srcSizeSubImage.width / 2);
        Rpp32u shadowPosI = rand() % (srcSizeSubImage.height - shadowSize.height);
        Rpp32u shadowPosJ = rand() % (srcSizeSubImage.width - shadowSize.width);

        if (chnFormat == RPPI_CHN_PLANAR)
        {
            Rpp32u remainingElementsInRow = srcSize.width - shadowSize.width;
            for (int c = 0; c < channel; c++)
            {
                dstPtrTemp = dstPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;
                srcPtrTemp = srcPtrSubImage + (c * srcSize.height * srcSize.width) + (shadowPosI * srcSize.width) + shadowPosJ;

                for (int i = 0; i < shadowSize.height; i++)
                {
                    for (int j = 0; j < shadowSize.width; j++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                    dstPtrTemp += remainingElementsInRow;
                    srcPtrTemp += remainingElementsInRow;
                }
            }
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            dstPtrTemp = dstPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            srcPtrTemp = srcPtrSubImage + (channel * ((shadowPosI * srcSize.width) + shadowPosJ));
            Rpp32u remainingElementsInRow = channel * (srcSize.width - shadowSize.width);
            for (int i = 0; i < shadowSize.height; i++)
            {
                for (int j = 0; j < shadowSize.width; j++)
                {
                    for (int c = 0; c < channel; c++)
                    {
                        *dstPtrTemp = *srcPtrTemp / 2;
                        dstPtrTemp++;
                        srcPtrTemp++;
                    }
                }
                dstPtrTemp += remainingElementsInRow;
                srcPtrTemp += remainingElementsInRow;
            }
        }
    }
 
    return RPP_SUCCESS;
}