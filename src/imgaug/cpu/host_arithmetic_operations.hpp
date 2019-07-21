#include <cpu/rpp_cpu_common.hpp>

/**************** Absolute Difference ***************/

template <typename T, typename U>
RppStatus absolute_difference_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) - ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPABS(pixel);
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Accumulate Weighted ***************/

template <typename T, typename U>
RppStatus accumulate_weighted_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                                   Rpp32f alpha,
                                   Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        Rpp32f pixel = ((1 - alpha) * ((Rpp32f) (*srcPtr1Temp))) + (alpha * ((Rpp32f) (*srcPtr2Temp)));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }

    return RPP_SUCCESS;

}

/**************** Accumulate ***************/

template <typename T, typename U>
RppStatus accumulate_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize,
                                   Rpp32u channel)
{
    T *srcPtr1Temp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;

    Rpp32f pixel;
    
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) + ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *srcPtr1Temp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
    }

    return RPP_SUCCESS;

}

/**************** Add ***************/

template <typename T, typename U>
RppStatus add_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) + ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Subtract ***************/

template <typename T, typename U>
RppStatus subtract_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) - ((Rpp32f) (*srcPtr2Temp));
        pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
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