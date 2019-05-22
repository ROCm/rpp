#include <algorithm>
#include <math.h>

template <typename T>
RppStatus 
host_rgb2hsv_pln(T *srcPtr, RppiSize srcSize, T *dstPtr)
{
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        float rf, gf, bf, cmax, cmin, delta;
        rf = ((float) srcPtr[i]) / 255;
        gf = ((float) srcPtr[i + (srcSize.width * srcSize.height)]) / 255;
        bf = ((float) srcPtr[i + (2 * srcSize.width * srcSize.height)]) / 255;
        cmax = std::max(std::max(rf, gf), bf);
        cmin = std::min(std::min(rf, gf), bf);
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

        if (cmax == 0)
        {
            dstPtr[i + (srcSize.width * srcSize.height)] = 0;
        }
        else
        {
            dstPtr[i + (srcSize.width * srcSize.height)] = delta / cmax;
        }

        dstPtr[i + (2 * srcSize.width * srcSize.height)] = cmax;

    }

    return RPP_SUCCESS;
}

template <typename T>
RppStatus 
host_rgb2hsv_pkd(T *srcPtr, RppiSize srcSize, T *dstPtr)
{
    for (int i = 0; i < (3 * srcSize.width * srcSize.height); i += 3)
    {
        float rf, gf, bf, cmax, cmin, delta;
        rf = ((float) srcPtr[i]) / 255;
        gf = ((float) srcPtr[i + 1]) / 255;
        bf = ((float) srcPtr[i + 2]) / 255;
        cmax = std::max(std::max(rf, gf), bf);
        cmin = std::min(std::min(rf, gf), bf);
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

        if (cmax == 0)
        {
            dstPtr[i + 1] = 0;
        }
        else
        {
            dstPtr[i + 1] = delta / cmax;
        }

        dstPtr[i + 2] = cmax;

    }
    
    return RPP_SUCCESS;
}