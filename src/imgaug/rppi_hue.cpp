#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_hue.hpp"

#include <iostream>

RppStatus
rppi_hue_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f hueShift)
{

    host_hue<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), hueShift);
    return RPP_SUCCESS;

}
