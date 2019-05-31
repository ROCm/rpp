#include <math.h>


template <typename T>
RppStatus host_blur(T* srcPtr, RppiSize srcSize, T* dstPtr,
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
