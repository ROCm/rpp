#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_rotate.hpp"

#include <iostream>

RppStatus
rppi_rotate_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          RppiSize sizeDst, Rpp32f angleRad)
{
    int channel = 1;
    host_rotate<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), sizeDst, angleRad, channel);
    return RPP_SUCCESS;

}
