#include <cpu/rpp_cpu_common.hpp>

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
    Rpp32f *kernel = (Rpp32f *)malloc(kernelSize * kernelSize * sizeof(Rpp32f));
    Rpp32f s, sum = 0.0;
    int bound = ((kernelSize - 1) / 2);
    unsigned int c = 0;
    s = 1 / (2 * stdDev * stdDev);
    for (int i = -bound; i <= bound; i++)
    {
        for (int j = -bound; j <= bound; j++)
        {
            kernel[c] = (1 / M_PI) * (s) * exp((-1) * (s) * (i*i + j*j));
            sum += kernel[c];
            c += 1;
        }
    }
    for (int i = 0; i < (kernelSize * kernelSize); i++)
    {
        kernel[i] /= sum;
    }
    RppiSize sizeMod;
    sizeMod.width = srcSize.width + (2 * bound);
    sizeMod.height = srcSize.height + (2 * bound);
    Rpp8u *pSrcMod = (Rpp8u *)malloc(sizeMod.width * sizeMod.height * channel * sizeof(Rpp8u));
    int srcLoc = 0, srcModLoc = 0, dstLoc = 0;

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
            }
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
                for (int j = 0; j < srcSize.width; j++)
                {
                    pSrcMod[srcModLoc] = srcPtr[srcLoc];
                    srcModLoc += 1;
                    srcLoc += 1;
                }
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
            }
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += 1;
                }
            }
        }
        dstLoc = 0;
        srcModLoc = 0;
        int count = 0;
        float pixel = 0.0;
        int *convLocs = (int *)malloc(kernelSize * kernelSize * sizeof(int));
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    count = 0;
                    pixel = 0.0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++, count++)
                        {
                            convLocs[count] = srcModLoc + (m * sizeMod.width) + n;
                        }
                    }
                    for (int k = 0; k < (kernelSize * kernelSize); k++)
                    {
                        pixel += (kernel[k] * (float)pSrcMod[convLocs[k]]);
                    }
                    pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
                    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                    dstPtr[dstLoc] = (Rpp8u) round(pixel);
                    dstLoc += 1;
                    srcModLoc += 1;
                }
                srcModLoc += (kernelSize - 1);
            }
            srcModLoc += ((kernelSize - 1) * sizeMod.width);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int c = 0; c < channel; c++)
        {
            srcModLoc = c;
            srcLoc = c;
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
            }
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
                for (int j = 0; j < srcSize.width; j++)
                {
                    pSrcMod[srcModLoc] = srcPtr[srcLoc];
                    srcModLoc += channel;
                    srcLoc += channel;
                }
                for (int b = 0; b < bound; b++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
            }
            for (int b = 0; b < bound; b++)
            {
                for (int i = 0; i < sizeMod.width; i++)
                {
                    pSrcMod[srcModLoc] = 0;
                    srcModLoc += channel;
                }
            }
            
        }
        dstLoc = 0;
        srcModLoc = 0;
        int count = 0;
        float pixel = 0.0;
        int *convLocs = (int *)malloc(kernelSize * kernelSize * sizeof(int));
        for (int c = 0; c < channel; c++)
        {
            srcModLoc = c;
            dstLoc = c;
            for (int i = 0; i < srcSize.height; i++)
            {
                for (int j = 0; j < srcSize.width; j++)
                {
                    count = 0;
                    pixel = 0.0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++, count++)
                        {
                            convLocs[count] = srcModLoc + (m * sizeMod.width * channel) + (n * channel);
                        }
                    }
                    for (int k = 0; k < (kernelSize * kernelSize); k++)
                    {
                        pixel += (kernel[k] * (float)pSrcMod[convLocs[k]]);
                    }
                    pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
                    pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
                    dstPtr[dstLoc] = (Rpp8u) round(pixel);
                    dstLoc += channel;
                    srcModLoc += channel;
                }
                srcModLoc += ((kernelSize - 1) * channel);
            }
        }
    }
    
    return RPP_SUCCESS;
}

/************ Brightness ************/

template <typename T>
RppStatus brightness_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                            Rpp32f alpha, Rpp32f beta,
                            RppiChnFormat chnFormat, unsigned int channel)
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