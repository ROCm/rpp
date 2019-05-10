#include <rppdefs.h>
#include <rppi_image_augumentation_functions.h>

#include "cpu/host_contrast.hpp"

#include <iostream>

RppStatus
rppi_contrast_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u new_min, Rpp32u new_max)
{

    host_contrast<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize,
                                    static_cast<Rpp8u*>(dstPtr), new_min ,new_max );
    return RPP_SUCCESS;

}
