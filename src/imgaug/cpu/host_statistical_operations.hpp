#include <cpu/rpp_cpu_common.hpp>
#include <limits>


/**************** Histogram ***************/

template <typename T>
RppStatus histogram_host(T* srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                         Rpp32u channel)
{
    histogram_kernel_host(srcPtr, srcSize, outputHistogram, bins - 1, channel);

    return RPP_SUCCESS;

}

/**************** Thresholding ***************/

template <typename T, typename U>
RppStatus thresholding_host(T* srcPtr, RppiSize srcSize, U* dstPtr, 
                                 U min, U max, 
                                 RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_threshold_host(srcPtr, srcSize, dstPtr, min, max, 1, chnFormat, channel);

    return RPP_SUCCESS;

}

/**************** Min ***************/

template <typename T, typename U>
RppStatus min_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMIN2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Max ***************/

template <typename T, typename U>
RppStatus max_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = RPPMAX2(*srcPtr1Temp, ((T)*srcPtr2Temp));
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Min Max Loc ***************/

template <typename T>
RppStatus minMaxLoc_host(T* srcPtr, RppiSize srcSize, 
                         Rpp8u* min, Rpp8u* max, Rpp32s* minLoc, Rpp32s* maxLoc, 
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    *min = 255;
    *max = 0;

    T *srcPtrTemp, *minLocPtrTemp, *maxLocPtrTemp;
    srcPtrTemp = srcPtr;
    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        if (*srcPtrTemp > *max)
        {
            *max = *srcPtrTemp;
            maxLocPtrTemp = srcPtrTemp;
        }
        if (*srcPtrTemp < *min)
        {
            *min = *srcPtrTemp;
            minLocPtrTemp = srcPtrTemp;
        }
        srcPtrTemp++;
    }
    *minLoc = minLocPtrTemp - srcPtr;
    *maxLoc = maxLocPtrTemp - srcPtr;

    return RPP_SUCCESS;

}