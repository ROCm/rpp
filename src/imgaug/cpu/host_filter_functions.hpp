#include <cpu/rpp_cpu_common.hpp>

/**************** Bilateral Filter ***************/

template <typename T>
RppStatus bilateral_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32u diameter, Rpp64f sigmaI, Rpp64f sigmaS,
                                RppiChnFormat chnFormat, unsigned int channel)
{
    if (chnFormat == RPPI_CHN_PLANAR)
    {
        for (int c = 0; c < channel; c++)
        {
            for(int i = 2; i < srcSize.height - 2; i++) {
                for(int j = 2; j < srcSize.width - 2; j++) {
                    int x = i, y = j;
                    double iFiltered = 0;
                    double wP = 0;
                    int neighbor_x = 0;
                    int neighbor_y = 0;
                    int half = diameter / 2;

                    for(int m = 0; m < diameter; m++) {
                        for(int n = 0; n < diameter; n++) {
                            neighbor_x = x - (half - m);
                            neighbor_y = y - (half - n);
                            double gi = RPPGAUSSIAN(srcPtr[(c * srcSize.height * srcSize.width) + (neighbor_x * srcSize.width) + neighbor_y] - srcPtr[(c * srcSize.height * srcSize.width) + (x * srcSize.width) + y], sigmaI);
                            double gs = RPPGAUSSIAN(RPPDISTANCE(x, y, neighbor_x, neighbor_y), sigmaS);
                            double w = gi * gs;
                            iFiltered = iFiltered + srcPtr[(c * srcSize.height * srcSize.width) + (neighbor_x * srcSize.width) + neighbor_y] * w;
                            wP = wP + w;
                        }
                    }
                    iFiltered = iFiltered / wP;
                    dstPtr[(c * srcSize.height * srcSize.width) + (x * srcSize.width) + y] = (Rpp8u) iFiltered;
                }
            }
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        for (int c = 0; c < channel; c++)
        {
            for(int i = 2; i < srcSize.height - 2; i++) {
                for(int j = 2; j < srcSize.width - 2; j++) {
                    int x = i, y = j;
                    double iFiltered = 0;
                    double wP = 0;
                    int neighbor_x = 0;
                    int neighbor_y = 0;
                    int half = diameter / 2;

                    for(int m = 0; m < diameter; m++) {
                        for(int n = 0; n < diameter; n++) {
                            neighbor_x = x - (half - m);
                            neighbor_y = y - (half - n);
                            double gi = RPPGAUSSIAN(srcPtr[c + (channel * neighbor_x * srcSize.width) + (channel * neighbor_y)] - srcPtr[c + (channel * x * srcSize.width) + (channel * y)], sigmaI);
                            double gs = RPPGAUSSIAN(RPPDISTANCE(x, y, neighbor_x, neighbor_y), sigmaS);
                            double w = gi * gs;
                            iFiltered = iFiltered + srcPtr[c + (channel * neighbor_x * srcSize.width) + (channel * neighbor_y)] * w;
                            wP = wP + w;
                        }
                    }
                    iFiltered = iFiltered / wP;
                    dstPtr[c + (channel * x * srcSize.width) + (channel * y)] = (Rpp8u) iFiltered;
                }
            }
        }
    }
    return RPP_SUCCESS;

}


// BOX FILTER
/************ BOX BLUR************/

template <typename T>
RppStatus box_filter_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    int kernelSize = 3;
    Rpp32f kernel[9]= {1,1,1,1,1,1,1,1,1};
    Rpp32f  sum = 9.0;
    int bound = ((kernelSize - 1) / 2);
    unsigned int c = 0;
   
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

