#include <cpu/rpp_cpu_common.hpp>

/**************** Absolute Difference ***************/

template <typename T>
RppStatus absolute_difference_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32s pixel = ((Rpp32s) srcPtr1[i]) - ((Rpp32s) srcPtr2[i]);
        pixel = RPPABS(pixel);
        pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
        pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Accumulate Weighted ***************/

template <typename T>
RppStatus accumulate_weighted_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize,
                                   Rpp32f alpha,
                                    RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32f pixel = ((1 - alpha) * ((Rpp32f) srcPtr1[i])) + (alpha * ((Rpp32s) srcPtr2[i]));
        pixel = (pixel < (Rpp32f) 255) ? pixel : ((Rpp32f) 255);
        pixel = (pixel > (Rpp32f) 0) ? pixel : ((Rpp32f) 0);
        srcPtr1[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Accumulate ***************/

template <typename T>
RppStatus accumulate_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize,
                                RppiChnFormat chnFormat,unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32s pixel = ((Rpp32s) srcPtr1[i]) + ((Rpp32s) srcPtr2[i]);
        pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
        pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
        srcPtr1[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Add ***************/

template <typename T>
RppStatus add_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                    RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32s pixel = ((Rpp32s) srcPtr1[i]) + ((Rpp32s) srcPtr2[i]);
        pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
        pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Subtract ***************/

template <typename T>
RppStatus subtract_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                                    RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        Rpp32s pixel = ((Rpp32s) srcPtr1[i]) - ((Rpp32s) srcPtr2[i]);
        pixel = (pixel < (Rpp32s) 255) ? pixel : ((Rpp32s) 255);
        pixel = (pixel > (Rpp32s) 0) ? pixel : ((Rpp32s) 0);
        dstPtr[i] =(Rpp8u) pixel;
    }

    return RPP_SUCCESS;

}

/**************** Mean & Standard Deviation ***************/

template <typename T>
RppStatus mean_stddev_host(T* srcPtr, RppiSize srcSize,
                            Rpp32f *mean, Rpp32f *stddev, 
                            RppiChnFormat chnFormat, unsigned int channel)
{
    int i;
    *mean = 0;
    *stddev = 0;
    T* srcPtrTemp=srcPtr;
    for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        *mean += *srcPtr;
        srcPtr++;
    }
    *mean = (*mean)/(srcSize.height * srcSize.width * channel);

    for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        *stddev += (((*mean)-(*srcPtrTemp)) * ((*mean)-(*srcPtrTemp)));
        srcPtrTemp++;
    }
    *stddev = sqrt((*stddev)/(srcSize.height * srcSize.width * channel));
    return RPP_SUCCESS;
}