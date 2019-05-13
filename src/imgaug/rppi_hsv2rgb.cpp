#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_hsv2rgb.hpp"

#include <iostream>

RppStatus
rppi_hsv2rgb_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{

    host_hsv2rgb<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr));
    return RPP_SUCCESS;

}
