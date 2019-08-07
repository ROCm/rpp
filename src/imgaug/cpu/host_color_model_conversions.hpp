#include <cpu/rpp_cpu_common.hpp>

/**************** RGB2HSV ***************/

template <typename T, typename U>
RppStatus rgb_to_hsv_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_rgb_to_hsv_host(srcPtr, srcSize, dstPtr, chnFormat, channel);
    
    return RPP_SUCCESS;
}

/**************** HSV2RGB ***************/

template <typename T, typename U>
RppStatus hsv_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_hsv_to_rgb_host(srcPtr, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** Hue Modification ***************/

template <typename T, typename U>
RppStatus hueRGB_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f hueShift,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f *srcPtrHSV = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
    compute_rgb_to_hsv_host(srcPtr, srcSize, srcPtrHSV, chnFormat, channel);

    Rpp32f *srcPtrHSVTemp;
    srcPtrHSVTemp = srcPtrHSV;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *srcPtrHSVTemp = *srcPtrHSVTemp + hueShift;
            while (*srcPtrHSVTemp > 360)
            {
                *srcPtrHSVTemp = *srcPtrHSVTemp - 360;
            }
            while (*srcPtrHSVTemp < 0)
            {
                *srcPtrHSVTemp = 360 + *srcPtrHSVTemp;
            }
            srcPtrHSVTemp++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *srcPtrHSVTemp = *srcPtrHSVTemp + hueShift;
            while (*srcPtrHSVTemp > 360)
            {
                *srcPtrHSVTemp = *srcPtrHSVTemp - 360;
            }
            while (*srcPtrHSVTemp < 0)
            {
                *srcPtrHSVTemp = 360 + *srcPtrHSVTemp;
            }
            srcPtrHSVTemp = srcPtrHSVTemp + channel;
        }
    }

    compute_hsv_to_rgb_host(srcPtrHSV, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus hueHSV_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f hueShift,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *dstPtrTemp = *srcPtrTemp + hueShift;
            while (*dstPtrTemp > 360)
            {
                *dstPtrTemp = *dstPtrTemp - 360;
            }
            while (*dstPtrTemp < 0)
            {
                *dstPtrTemp = 360 + *dstPtrTemp;
            }
            srcPtrTemp++;
            dstPtrTemp++;
        }
        memcpy(dstPtrTemp, srcPtrTemp, 2 * srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *dstPtrTemp = *srcPtrTemp + hueShift;
            while (*dstPtrTemp > 360)
            {
                *dstPtrTemp = *dstPtrTemp - 360;
            }
            while (*dstPtrTemp < 0)
            {
                *dstPtrTemp = 360 + *dstPtrTemp;
            }
            srcPtrTemp++;
            dstPtrTemp++;
            for (int c = 0; c < (channel - 1); c++)
            {
                *dstPtrTemp = *srcPtrTemp;
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }

    return RPP_SUCCESS;
}

/**************** Saturation Modification ***************/

template <typename T, typename U>
RppStatus saturationRGB_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    Rpp32f *srcPtrHSV = (Rpp32f *)calloc(channel * srcSize.height * srcSize.width, sizeof(Rpp32f));
    compute_rgb_to_hsv_host(srcPtr, srcSize, srcPtrHSV, chnFormat, channel);

    Rpp32f *srcPtrHSVTemp;
    srcPtrHSVTemp = srcPtrHSV;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrHSVTemp = srcPtrHSV + (srcSize.height * srcSize.width);
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *srcPtrHSVTemp *= saturationFactor;
            *srcPtrHSVTemp = (*srcPtrHSVTemp < (Rpp32f) 1) ? *srcPtrHSVTemp : ((Rpp32f) 1);
            *srcPtrHSVTemp = (*srcPtrHSVTemp > (Rpp32f) 0) ? *srcPtrHSVTemp : ((Rpp32f) 0);
            srcPtrHSVTemp++;
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrHSVTemp = srcPtrHSV + 1;
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *srcPtrHSVTemp *= saturationFactor;
            *srcPtrHSVTemp = (*srcPtrHSVTemp < (Rpp32f) 1) ? *srcPtrHSVTemp : ((Rpp32f) 1);
            *srcPtrHSVTemp = (*srcPtrHSVTemp > (Rpp32f) 0) ? *srcPtrHSVTemp : ((Rpp32f) 0);
            srcPtrHSVTemp = srcPtrHSVTemp + channel;
        }
    }

    compute_hsv_to_rgb_host(srcPtrHSV, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

template <typename T, typename U>
RppStatus saturationHSV_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    Rpp32f saturationFactor,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        memcpy(dstPtrTemp, srcPtrTemp, srcSize.height * srcSize.width * sizeof(T));
        dstPtrTemp += srcSize.height * srcSize.width;
        srcPtrTemp += srcSize.height * srcSize.width;
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *dstPtrTemp = *srcPtrTemp * saturationFactor;
            *dstPtrTemp = (*dstPtrTemp < (Rpp32f) 1) ? *dstPtrTemp : ((Rpp32f) 1);
            *dstPtrTemp = (*dstPtrTemp > (Rpp32f) 0) ? *dstPtrTemp : ((Rpp32f) 0);
            srcPtrTemp++;
            dstPtrTemp++;
        }
        memcpy(dstPtrTemp, srcPtrTemp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp++;
            dstPtrTemp++;

            *dstPtrTemp = *srcPtrTemp * saturationFactor;
            *dstPtrTemp = (*dstPtrTemp < (Rpp32f) 1) ? *dstPtrTemp : ((Rpp32f) 1);
            *dstPtrTemp = (*dstPtrTemp > (Rpp32f) 0) ? *dstPtrTemp : ((Rpp32f) 0);
            srcPtrTemp++;
            dstPtrTemp++;

            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp++;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}

/**************** RGB2HSL ***************/

template <typename T, typename U>
RppStatus rgb_to_hsl_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_rgb_to_hsl_host(srcPtr, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}

/**************** HSL2RGB ***************/

template <typename T, typename U>
RppStatus hsl_to_rgb_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_hsl_to_rgb_host(srcPtr, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;
}


/**************** Color Temperature ***************/

template <typename T>
RppStatus color_temperature_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp8s adjustmentValue,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (channel != 1 && channel !=  3)
    {
        return RPP_ERROR;
    }
    if (adjustmentValue < -100 || adjustmentValue > 100)
    {
        return RPP_ERROR;
    }   

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    Rpp32s pixel;

    if (channel == 1)
    {
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
            pixel = RPPPIXELCHECK(pixel);
            *dstPtrTemp = (T) pixel;
            dstPtrTemp++;
            srcPtrTemp++;
        }
    }
    else if (channel == 3)
    {   
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
                pixel = RPPPIXELCHECK(pixel);
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
                pixel = RPPPIXELCHECK(pixel);
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
                pixel = RPPPIXELCHECK(pixel);
                *dstPtrTemp = (T) pixel;
                dstPtrTemp++;
                srcPtrTemp++;

                *dstPtrTemp = *srcPtrTemp;
                dstPtrTemp++;
                srcPtrTemp++;

                pixel = (Rpp32s) *srcPtrTemp + (Rpp32s) adjustmentValue;
                pixel = RPPPIXELCHECK(pixel);
                *dstPtrTemp = (T) pixel;
                dstPtrTemp++;
                srcPtrTemp++;
            }
        }
    }
    
    return RPP_SUCCESS;
}


/**************** Vignette ***************/

template <typename T>
RppStatus vignette_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32f stdDev,
                    RppiChnFormat chnFormat, Rpp32u channel)
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
    for (int i = 0; i < (srcSize.height * srcSize.width); i++)
    {
        if (*maskTemp > max)
        {
            max = *maskTemp;
        }
        maskTemp++;
    }

    maskTemp = mask;
    for (int i = 0; i < (srcSize.height * srcSize.width); i++)
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


/**************** Channel Extract ***************/

template <typename T>
RppStatus channel_extract_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    Rpp32u extractChannelNumber, 
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    if (extractChannelNumber != 0 && extractChannelNumber != 1 && extractChannelNumber != 2)
    {
        return RPP_ERROR;
    }

    T *srcPtrTemp, *dstPtrTemp;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        srcPtrTemp = srcPtr + (extractChannelNumber * srcSize.height * srcSize.width);
        memcpy(dstPtrTemp, srcPtrTemp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        srcPtrTemp = srcPtr + extractChannelNumber;
        for (int i = 0; i < srcSize.height * srcSize.width; i++)
        {
            *dstPtrTemp = *srcPtrTemp;
            srcPtrTemp = srcPtrTemp + channel;
            dstPtrTemp++;
        }
    }

    return RPP_SUCCESS;
}


/**************** Channel Combine ***************/

template <typename T>
RppStatus channel_combine_host(T* srcPtr1, T* srcPtr2, T* srcPtr3, RppiSize srcSize, T* dstPtr,
                               RppiChnFormat chnFormat, Rpp32u channel)
{

    T *srcPtr1Temp, *srcPtr2Temp, *srcPtr3Temp, *dstPtrTemp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    srcPtr3Temp = srcPtr3;
    dstPtrTemp = dstPtr;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        Rpp32u increment = srcSize.height * srcSize.width;
        memcpy(dstPtrTemp, srcPtr1Temp, srcSize.height * srcSize.width * sizeof(T));
        dstPtrTemp += increment;
        memcpy(dstPtrTemp, srcPtr2Temp, srcSize.height * srcSize.width * sizeof(T));
        dstPtrTemp += increment;
        memcpy(dstPtrTemp, srcPtr3Temp, srcSize.height * srcSize.width * sizeof(T));
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int i = 0; i < srcSize.height * srcSize.width; i++)
        {
            *dstPtrTemp = *srcPtr1Temp;
            dstPtrTemp++;
            srcPtr1Temp++;
            *dstPtrTemp = *srcPtr2Temp;
            dstPtrTemp++;
            srcPtr2Temp++;
            *dstPtrTemp = *srcPtr3Temp;
            dstPtrTemp++;
            srcPtr3Temp++;
        }
    }

    return RPP_SUCCESS;
}


/**************** Look Up Table ***************/

template <typename T, typename U>
RppStatus look_up_table_host(T* srcPtr, RppiSize srcSize, U* dstPtr,
                    U *lutPtr,
                    RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;
    U *lutPtrTemp;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            lutPtrTemp = lutPtr + (c * 256);
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                *dstPtrTemp = *(lutPtrTemp + (Rpp32u)(*srcPtrTemp));
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        lutPtrTemp = lutPtr;
        for (int i = 0; i < (srcSize.height * srcSize.width); i++)
        {
            for (int c = 0; c < channel; c++)
            {
                *dstPtrTemp = *(lutPtrTemp + c + (channel * (Rpp32u)(*srcPtrTemp)));
                srcPtrTemp++;
                dstPtrTemp++;
            }
        }
    }
    
    return RPP_SUCCESS;
}