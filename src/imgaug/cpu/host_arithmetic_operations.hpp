#include <cpu/rpp_cpu_common.hpp>

/**************** Absolute Difference ***************/

template <typename T, typename U>
RppStatus absolute_difference_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                                   RppiChnFormat chnFormat, Rpp32u channel)
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
                                   RppiChnFormat chnFormat, Rpp32u channel)
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
                          RppiChnFormat chnFormat, Rpp32u channel)
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
                   RppiChnFormat chnFormat, Rpp32u channel)
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
                        RppiChnFormat chnFormat, Rpp32u channel)
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

/**************** Magnitude ***************/

template <typename T>
RppStatus magnitude_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                         RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_magnitude_host(srcPtr1, srcPtr2, srcSize, dstPtr, chnFormat, channel);

    return RPP_SUCCESS;

}

/**************** Multiply ***************/

template <typename T, typename U>
RppStatus multiply_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                        RppiChnFormat chnFormat, Rpp32u channel)
{
    compute_multiply_host(srcPtr1, srcPtr2, srcSize, dstPtr, channel);

    return RPP_SUCCESS;

}

/**************** Phase ***************/

template <typename T, typename U>
RppStatus phase_host(T* srcPtr1, U* srcPtr2, RppiSize srcSize, T* dstPtr,
                   RppiChnFormat chnFormat, Rpp32u channel)
{
    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;

    Rpp32f pixel;

    for (int i = 0; i < (channel * srcSize.height * srcSize.width); i++)
    {
        pixel = atan(((Rpp32f) (*srcPtr1Temp)) / ((Rpp32f) (*srcPtr2Temp)));
        //pixel = RPPPIXELCHECK(pixel);
        *dstPtrTemp =(T) pixel;
        srcPtr1Temp++;
        srcPtr2Temp++;
        dstPtrTemp++;
    }

    return RPP_SUCCESS;

}

/**************** Tensor Add ***************/

template <typename T, typename U>
RppStatus tensor_add_host(T* srcPtr1, U* srcPtr2, T* dstPtr, 
                          Rpp32u tensorDimension, Rpp32u *tensorDimensionValues)
{
    Rpp32u *tensorDimensionValuesTemp;
    tensorDimensionValuesTemp = tensorDimensionValues;

    Rpp32u tensorSize = 1;
    for(int i = 0; i < tensorDimension; i++)
    {
        tensorSize *= *tensorDimensionValuesTemp;
        tensorDimensionValuesTemp++;
    }

    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;
    Rpp32f pixel;

    for (int i = 0; i < tensorSize; i++)
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

/**************** Tensor Subtract ***************/

template <typename T, typename U>
RppStatus tensor_subtract_host(T* srcPtr1, U* srcPtr2, T* dstPtr, 
                          Rpp32u tensorDimension, Rpp32u *tensorDimensionValues)
{
    Rpp32u *tensorDimensionValuesTemp;
    tensorDimensionValuesTemp = tensorDimensionValues;

    Rpp32u tensorSize = 1;
    for(int i = 0; i < tensorDimension; i++)
    {
        tensorSize *= *tensorDimensionValuesTemp;
        tensorDimensionValuesTemp++;
    }

    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;
    Rpp32f pixel;

    for (int i = 0; i < tensorSize; i++)
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

/**************** Tensor Multiply ***************/

template <typename T, typename U>
RppStatus tensor_multiply_host(T* srcPtr1, U* srcPtr2, T* dstPtr, 
                          Rpp32u tensorDimension, Rpp32u *tensorDimensionValues)
{
    Rpp32u *tensorDimensionValuesTemp;
    tensorDimensionValuesTemp = tensorDimensionValues;

    Rpp32u tensorSize = 1;
    for(int i = 0; i < tensorDimension; i++)
    {
        tensorSize *= *tensorDimensionValuesTemp;
        tensorDimensionValuesTemp++;
    }

    T *srcPtr1Temp, *dstPtrTemp;
    U *srcPtr2Temp;
    srcPtr1Temp = srcPtr1;
    srcPtr2Temp = srcPtr2;
    dstPtrTemp = dstPtr;
    Rpp32f pixel;

    for (int i = 0; i < tensorSize; i++)
    {
        pixel = ((Rpp32f) (*srcPtr1Temp)) * ((Rpp32f) (*srcPtr2Temp));
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
    Rpp32f *meanTemp, *stdDevTemp;
    meanTemp = mean;
    stdDevTemp = stddev;
    T* srcPtrTemp = srcPtr;
    *meanTemp = 0;
    *stdDevTemp = 0;
    for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        *meanTemp += *srcPtr;
        srcPtr++;
    }
    *meanTemp = (*meanTemp) / (srcSize.height * srcSize.width * channel);

    for(i = 0; i < (srcSize.height * srcSize.width * channel); i++)
    {
        *stdDevTemp += (((*meanTemp) - (*srcPtrTemp)) * ((*meanTemp) - (*srcPtrTemp)));
        srcPtrTemp++;
    }
    *stdDevTemp = sqrt((*stdDevTemp) / (srcSize.height * srcSize.width * channel));
    return RPP_SUCCESS;
}