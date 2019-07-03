#include <cpu/rpp_cpu_common.hpp>
#include <limits>

/**************** Histogram ***************/

template <typename T>
RppStatus histogram_host(T* srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                         RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrTemp;
    srcPtrTemp = srcPtr;
    Rpp32u *histogram = (Rpp32u *) calloc(bins, sizeof(Rpp32u));
    Rpp32u *outputHistogramTemp;
    outputHistogramTemp = outputHistogram;
    for (int c = 0; c < channel; c++)
    {
        memset (histogram,0,bins * sizeof(Rpp32u));

        Rpp32u *histogramTemp;
        histogramTemp = histogram;

        histogram_kernel_host(srcPtrTemp, srcSize, histogram, bins, 0, chnFormat, channel);

        for (int i = 0; i < bins; i++)
        {
            *outputHistogramTemp = *histogramTemp;
            outputHistogramTemp++;
            histogramTemp++;
        }
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            srcPtrTemp += (srcSize.height * srcSize.width);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            srcPtrTemp += channel;
        }
    }

    return RPP_SUCCESS;

}

/**************** Equalize Histogram ***************/

template <typename T>
RppStatus equalize_histogram_host(T* srcPtr, RppiSize srcSize, T* dstPtr, 
                         RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrChannelBegin, *dstPtrChannelBegin;
    srcPtrChannelBegin = srcPtr;
    dstPtrChannelBegin = dstPtr;
    Rpp32u bins = (Rpp32u)(std::numeric_limits<unsigned char>::max()) + 1;
    Rpp32u *histogram = (Rpp32u *) calloc(bins, sizeof(Rpp32u));
    T *lookUpTable = (T *) calloc (bins, sizeof(T));
    Rpp32f multiplier = 255.0 / ((Rpp32f)(srcSize.height * srcSize.width));

    for (int c = 0; c < channel; c++)
    {
        memset (histogram,0,bins * sizeof(Rpp32u));
        memset (lookUpTable,0,bins * sizeof(T));

        histogram_kernel_host(srcPtrChannelBegin, srcSize, histogram, bins, 0, chnFormat, channel);

        Rpp32u sum = 0;
        Rpp32u *histogramTemp;
        T *lookUpTableTemp;
        histogramTemp = histogram;
        lookUpTableTemp = lookUpTable;
        
        for (int i = 0; i < bins; i++)
        {
            sum += *histogramTemp;
            *lookUpTableTemp = (T)round(((Rpp32f)sum) * multiplier);
            histogramTemp++;
            lookUpTableTemp++;
        }

        T *srcPtrTemp, *dstPtrTemp;
        srcPtrTemp = srcPtrChannelBegin;
        dstPtrTemp = dstPtrChannelBegin;
        
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
                srcPtrTemp++;
                dstPtrTemp++;
            }
            srcPtrChannelBegin += (srcSize.height * srcSize.width);
            dstPtrChannelBegin += (srcSize.height * srcSize.width);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            for (int i = 0; i < (srcSize.height * srcSize.width); i++)
            {
                *dstPtrTemp = *(lookUpTable + *srcPtrTemp);
                srcPtrTemp += 3;
                dstPtrTemp += 3;
            }
            srcPtrChannelBegin++;
            dstPtrChannelBegin++;
        }
    }

    return RPP_SUCCESS;

}

/**************** Histogram Subimage ***************/

template <typename T>
RppStatus histogram_subimage_host(T* srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                  unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2, 
                                  RppiChnFormat chnFormat, unsigned int channel)
{
    T *srcPtrSubImage;
    RppiSize srcSizeSubImage;
    compute_subimage_location_host(srcPtr, &srcPtrSubImage, srcSize, &srcSizeSubImage, x1, y1, x2, y2, chnFormat, channel);

    Rpp32u increment = srcSize.width - srcSizeSubImage.width;

    T *srcPtrTemp;
    srcPtrTemp = srcPtrSubImage;
    Rpp32u *histogram = (Rpp32u *) calloc(bins, sizeof(Rpp32u));
    Rpp32u *outputHistogramTemp;
    outputHistogramTemp = outputHistogram;
    for (int c = 0; c < channel; c++)
    {
        memset (histogram,0,bins * sizeof(Rpp32u));

        Rpp32u *histogramTemp;
        histogramTemp = histogram;

        histogram_kernel_host(srcPtrTemp, srcSizeSubImage, histogram, bins, increment, chnFormat, channel);

        for (int i = 0; i < bins; i++)
        {
            *outputHistogramTemp = *histogramTemp;
            outputHistogramTemp++;
            histogramTemp++;
        }
        if (chnFormat == RPPI_CHN_PLANAR)
        {
            srcPtrTemp += (srcSize.height * srcSize.width);
        }
        else if (chnFormat == RPPI_CHN_PACKED)
        {
            srcPtrTemp += channel;
        }
    }

    return RPP_SUCCESS;

}