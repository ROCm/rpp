#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_hue.hpp"

#include <iostream>

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueRGB_pln<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueRGB_pkd<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueHSV_pln<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hueHSV_pkd<Rpp32f>(static_cast<Rpp32f*>(srcPtr), srcSize,
                            static_cast<Rpp32f*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}