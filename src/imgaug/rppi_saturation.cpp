#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_saturation.hpp"

#include <iostream>

RppStatus
rppi_saturation_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f saturationFactor)
{

    host_saturation<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                            static_cast<Rpp8u*>(dstPtr), saturationFactor);
    return RPP_SUCCESS;

}
