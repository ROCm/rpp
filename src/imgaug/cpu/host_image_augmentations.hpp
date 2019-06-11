#include <algorithm>
#include <math.h>

using namespace std;

/************ Blur************/
// Blur planar host implementation

template <typename T>
RppStatus blur_pln_host(T* srcPtr, RppiSize srcSize, T* dstPtr, unsigned int channel)
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

// Blur packed host implementation
template <typename T>
RppStatus blur_pkd_host(T* srcPtr, RppiSize srcSize, T* dstPtr, unsigned int channel)
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

/************ Brightness ************/
// Brightness host implementation
template <typename T>
RppStatus brightness_contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                                Rpp32f alpha, Rpp32f beta, unsigned int channel, RppiChnFormat chnFormat)
{
    //logic is planar/packed independent
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((Rpp32f) srcPtr[i]) * alpha + beta;
        pixel = std::min(pixel, (Rpp32f) 255);
        pixel = std::max(pixel, (Rpp32f) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

// Contrast host implementation

template <typename T>
RppStatus contrast_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                        Rpp32u new_min, Rpp32u new_max, unsigned int channel)
{
    for(int c = 0; c < channel; c++)
    {
        Rpp32f Min = (Rpp32f) *std::min_element(srcPtr + (c * srcSize.width * srcSize.height), srcPtr + ((c + 1) * srcSize.width * srcSize.height));
        Rpp32f Max = (Rpp32f) *std::max_element(srcPtr + (c * srcSize.width * srcSize.height), srcPtr + ((c + 1) * srcSize.width * srcSize.height));

        for (int i = 0; i < (srcSize.width * srcSize.height); i++)
        {
            Rpp32f pixel = (Rpp32f) srcPtr[i + (c * srcSize.width * srcSize.height)];
            pixel = ((pixel - Min) * ((new_max - new_min) / (Max - Min))) + new_min;
            pixel = std::min(pixel, (Rpp32f)new_max);
            pixel = std::max(pixel, (Rpp32f)new_min);
            dstPtr[i + (c * srcSize.width * srcSize.height)] = (Rpp8u) pixel;
        }
    }
    
    return RPP_SUCCESS;

}