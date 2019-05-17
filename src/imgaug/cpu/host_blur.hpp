#include <algorithm>
#include <math.h>


template <typename T>
RppStatus host_blur_pln(T* srcPtr, RppiSize srcSize, T* dstPtr, unsigned int channel)
{
    float kernel_3x3[9] = {1,2,1,2,4,2,1,2,1};
    for (int i = 0; i < 9; i++)
    {
        kernel_3x3[i] *= 0.0625;
    }
    RppiSize sizeMod;
    sizeMod.width = srcSize.width + 2;
    sizeMod.height = srcSize.height + 2;

    Rpp8u *pSrcMod = (Rpp8u *)malloc(sizeMod.width * sizeMod.height * channel * sizeof(Rpp8u));

    int srcLoc = 0, srcModLoc = 0, dstLoc = 0;
    for (int c = 0; c < channel; c++)
    {
        for (int i = 0; i < sizeMod.width; i++)
        {
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 1;
        }
        for (int i = 0; i < srcSize.height; i++)
        {
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 1;
            for (int j = 0; j < srcSize.width; j++)
            {
                pSrcMod[srcModLoc] = srcPtr[srcLoc];
                srcModLoc += 1;
                srcLoc += 1;
            }
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 1;
        }
        for (int i = 0; i < sizeMod.width; i++)
        {
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 1;
        }
    }
    
    dstLoc = 0;
    srcModLoc = 0;
    int convLocs[9] = {0}, count = 0;
    float pixel = 0.0;

    for (int c = 0; c < channel; c++)
    {
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                count = 0;
                pixel = 0.0;
                for (int m = 0; m < 3; m++)
                {
                    for (int n = 0; n < 3; n++, count++)
                    {
                        convLocs[count] = srcModLoc + (m * sizeMod.width) + n;
                    }
                }
                for (int k = 0; k < 9; k++)
                {
                    pixel += (kernel_3x3[k] * (float)pSrcMod[convLocs[k]]);
                }
                pixel = std::min(pixel, (Rpp32f) 255);
                pixel = std::max(pixel, (Rpp32f) 0);
                dstPtr[dstLoc] = (Rpp8u) round(pixel);
                dstLoc += 1;
                srcModLoc += 1;
            }
            srcModLoc += 2;
        }
        srcModLoc += (2 * sizeMod.width);
    }
    return RPP_SUCCESS;
}

template <typename T>
RppStatus host_blur_pkd(T* srcPtr, RppiSize srcSize, T* dstPtr, unsigned int channel)
{
    float kernel_3x3[9] = {1,2,1,2,4,2,1,2,1};
    for (int i = 0; i < 9; i++)
    {
        kernel_3x3[i] *= 0.0625;
    }
    RppiSize sizeMod;
    sizeMod.width = srcSize.width + 2;
    sizeMod.height = srcSize.height + 2;

    Rpp8u *pSrcMod = (Rpp8u *)malloc(sizeMod.width * sizeMod.height * channel * sizeof(Rpp8u));

    int srcLoc = 0, srcModLoc = 0, dstLoc = 0;
    for (int c = 0; c < channel; c++)
    {
        srcModLoc = c;
        srcLoc = c;
        for (int i = 0; i < sizeMod.width; i++)
        {
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 3;
        }
        for (int i = 0; i < srcSize.height; i++)
        {
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 3;
            for (int j = 0; j < srcSize.width; j++)
            {
                pSrcMod[srcModLoc] = srcPtr[srcLoc];
                srcModLoc += 3;
                srcLoc += 3;
            }
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 3;
        }
        for (int i = 0; i < sizeMod.width; i++)
        {
            pSrcMod[srcModLoc] = 0;
            srcModLoc += 3;
        }
    }

    dstLoc = 0;
    srcModLoc = 0;
    int convLocs[9] = {0}, count = 0;
    float pixel = 0.0;

    for (int c = 0; c < 3; c++)
    {
        srcModLoc = c;
        dstLoc = c;
        for (int i = 0; i < srcSize.height; i++)
        {
            for (int j = 0; j < srcSize.width; j++)
            {
                count = 0;
                pixel = 0.0;
                for (int m = 0; m < 3; m++)
                {
                    for (int n = 0; n < 3; n++, count++)
                    {
                        convLocs[count] = srcModLoc + (m * sizeMod.width * 3) + (n * 3);
                    }
                }
                for (int k = 0; k < 9; k++)
                {
                    pixel += (kernel_3x3[k] * (float)pSrcMod[convLocs[k]]);
                }
                pixel = std::min(pixel, (Rpp32f) 255);
                pixel = std::max(pixel, (Rpp32f) 0);
                dstPtr[dstLoc] = (Rpp8u) round(pixel);
                dstLoc += 3;
                srcModLoc += 3;
            }
            srcModLoc += 6;
        }
    }
    return RPP_SUCCESS;
}