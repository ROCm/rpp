#include <cpu/rpp_cpu_common.hpp>

/**************** Min ***************/

template <typename T>
RppStatus min_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = RPPMIN2(srcPtr1[i], srcPtr2[i]);
    }

    return RPP_SUCCESS;

}

/**************** Max ***************/

template <typename T>
RppStatus max_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = RPPMAX2(srcPtr1[i], srcPtr2[i]);
    }

    return RPP_SUCCESS;

}

/**************** MinMax ***************/

template <typename T>
RppStatus minMax_host(T* srcPtr, RppiSize srcSize, T* maskPtr,
                      Rpp8u* min, Rpp8u* max, 
                      RppiChnFormat chnFormat, unsigned int channel)
{
    *min = 255;
    *max = 0;
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        if (maskPtr[i] != 0 && maskPtr[i] != 1)
        {
            return RPP_ERROR;
        }
        else
        {
            if (maskPtr[i] == 1)
            {
                if (srcPtr[i] > *max)
                {
                    *max = srcPtr[i];
                }
                if (srcPtr[i] < *min)
                {
                    *min = srcPtr[i];
                }
            }
        }
    }

    return RPP_SUCCESS;

}

/**************** MinMaxLoc ***************/

template <typename T>
RppStatus minMaxLoc_host(T* srcPtr, RppiSize srcSize, T* maskPtr,
                      Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc,
                      RppiChnFormat chnFormat, unsigned int channel)
{
    *min = 255;
    *max = 0;
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        if (maskPtr[i] != 0 && maskPtr[i] != 1)
        {
            return RPP_ERROR;
        }
        else
        {
            if (maskPtr[i] == 1)
            {
                if (srcPtr[i] > *max)
                {
                    *max = srcPtr[i];
                    *maxLoc = &srcPtr[i];
                }
                if (srcPtr[i] < *min)
                {
                    *min = srcPtr[i];
                    *minLoc = &srcPtr[i];
                }
            }
        }
    }

    return RPP_SUCCESS;

}

/**************** MeanStdDev ***************/

template <typename T>
RppStatus meanStd_host(T* srcPtr, RppiSize srcSize,
                      Rpp32f* mean, Rpp32f* stdDev, 
                      RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        *mean += (Rpp32f) srcPtr[i];
    }
    *mean /= ((Rpp32f)(channel * srcSize.width * srcSize.height));
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        *stdDev += (((Rpp32f) srcPtr[i] - *mean) * ((Rpp32f) srcPtr[i] - *mean));
    }
    *stdDev /= ((Rpp32f)((channel * srcSize.width * srcSize.height) - 1));
    *stdDev = sqrt(*stdDev);


    return RPP_SUCCESS;

}