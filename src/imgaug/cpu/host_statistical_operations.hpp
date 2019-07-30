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

/**************** Equalize Histogram ***************/

template <typename T>
RppStatus equalize_histogram_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                                  Rpp32u channel)
{
    Rpp32u *histogram = (Rpp32u *) calloc(256, sizeof(Rpp32u));
    T *lookUpTable = (T *) calloc (256, sizeof(T));
    Rpp32u *histogramTemp;
    T *lookUpTableTemp;
    Rpp32f multiplier = 255.0 / ((Rpp32f)(channel * srcSize.height * srcSize.width));

    histogram_kernel_host(srcPtr, srcSize, histogram, 255, channel);

    Rpp32u sum = 0;
    histogramTemp = histogram;
    lookUpTableTemp = lookUpTable;
    
    for (int i = 0; i < 256; i++)
    {
        sum += *histogramTemp;
        *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
        histogramTemp++;
        lookUpTableTemp++;
    }

    T *srcPtrTemp, *dstPtrTemp;
    srcPtrTemp = srcPtr;
    dstPtrTemp = dstPtr;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
        srcPtrTemp++;
        dstPtrTemp++;
    }
    
    return RPP_SUCCESS;

}