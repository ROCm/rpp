#include <cpu/rpp_cpu_common.hpp>

/**************** Bitwise And ***************/

template <typename T>
RppStatus bitwise_AND_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = srcPtr1[i] & srcPtr2[i];
    }

    return RPP_SUCCESS;

}

/**************** Bitwise Not ***************/

template <typename T>
RppStatus bitwise_NOT_host(T* srcPtr, RppiSize srcSize, T* dstPtr,
                           RppiChnFormat chnFormat, unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = ~srcPtr[i];
    }

    return RPP_SUCCESS;

}

/**************** Bitwise Exclusive Or ***************/

template <typename T>
RppStatus exclusive_OR_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                            RppiChnFormat chnFormat,  unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] = srcPtr1[i] ^ srcPtr2[i];
    }

    return RPP_SUCCESS;

}

/**************** Bitwise Inclusive Or ***************/

template <typename T>
RppStatus inclusive_OR_host(T* srcPtr1, T* srcPtr2, RppiSize srcSize, T* dstPtr,
                            RppiChnFormat chnFormat,  unsigned int channel)
{
    for (int i = 0; i < (channel * srcSize.width * srcSize.height); i++)
    {
        dstPtr[i] =srcPtr1[i] | srcPtr2[i] ;
    }

    return RPP_SUCCESS;

}