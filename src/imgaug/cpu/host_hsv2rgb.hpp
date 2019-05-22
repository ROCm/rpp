#include <algorithm>
#include <math.h>

template <typename T>
RppStatus host_hsv2rgb_pln(T *srcPtr, RppiSize srcSize, T *dstPtr)
{
    for (int i = 0; i < (srcSize.width * srcSize.height); i++)
    {
        float c, x, m, rf, gf, bf;
        c = srcPtr[i + (2 * srcSize.width * srcSize.height)] * srcPtr[i + (srcSize.width * srcSize.height)];
        x = c * (1 - abs((fmod((srcPtr[i] / 60), 2)) - 1));
        m = srcPtr[i + (2 * srcSize.width * srcSize.height)] - c;
        
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

    return RPP_SUCCESS;
}

template <typename T>
RppStatus host_hsv2rgb_pkd(T *srcPtr, RppiSize srcSize, T *dstPtr)
{
    for (int i = 0; i < (3 * srcSize.width * srcSize.height); i += 3)
    {
        float c, x, m, rf, gf, bf;
        c = dstPtr[i + 2] * dstPtr[i + 1];
        x = c * (1 - abs((fmod((dstPtr[i] / 60), 2)) - 1));
        m = dstPtr[i + 2] - c;
        
        if ((0 <= dstPtr[i]) && (dstPtr[i] < 60))
        {
            rf = c;
            gf = x;
            bf = 0;
        }
        else if ((60 <= dstPtr[i]) && (dstPtr[i] < 120))
        {
            rf = x;
            gf = c;
            bf = 0;
        }
        else if ((120 <= dstPtr[i]) && (dstPtr[i] < 180))
        {
            rf = 0;
            gf = c;
            bf = x;
        }
        else if ((180 <= dstPtr[i]) && (dstPtr[i] < 240))
        {
            rf = 0;
            gf = x;
            bf = c;
        }
        else if ((240 <= dstPtr[i]) && (dstPtr[i] < 300))
        {
            rf = x;
            gf = 0;
            bf = c;
        }
        else if ((300 <= dstPtr[i]) && (dstPtr[i] < 360))
        {
            rf = c;
            gf = 0;
            bf = x;
        }

        dstPtr[i] = (Rpp8u) round((rf + m) * 255);
        dstPtr[i + 1] = (Rpp8u) round((gf + m) * 255);
        dstPtr[i + 2] = (Rpp8u) round((bf + m) * 255);
    }

    return RPP_SUCCESS;
}