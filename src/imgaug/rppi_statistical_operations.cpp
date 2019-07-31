#include <rppi_statistical_operations.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
#include <hip/rpp_hip_common.hpp>

#elif defined(OCL_COMPILE)
#include <cl/rpp_cl_common.hpp>
#include "cl/cl_declarations.hpp"
#endif //backend
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono; 

#include "cpu/host_statistical_operations.hpp" 
 
// ----------------------------------------
// Host histogram functions calls 
// ----------------------------------------


RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins)
{
    histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, 
                          1);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins)
{
    histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, 
                          3);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins)
{
    histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, 
                          3);

    return RPP_SUCCESS;

}
 
// ----------------------------------------
// Host thresholding functions calls 
// ----------------------------------------


RppStatus
rppi_thresholding_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp8u min, Rpp8u max)
{
    thresholding_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                             min, max, 
                             RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_thresholding_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp8u min, Rpp8u max)
{
    thresholding_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                             min, max, 
                             RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_thresholding_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp8u min, Rpp8u max)
{
    thresholding_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                             min, max, 
                             RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}
 
// ----------------------------------------
// Host min functions calls 
// ----------------------------------------


RppStatus
rppi_min_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    min_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_min_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    min_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_min_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    min_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}
 
// ----------------------------------------
// Host max functions calls 
// ----------------------------------------


RppStatus
rppi_max_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    max_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    1);

    return RPP_SUCCESS;

}

RppStatus
rppi_max_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    max_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}

RppStatus
rppi_max_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr)
{
    max_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr1), static_cast<Rpp8u*>(srcPtr2), srcSize, static_cast<Rpp8u*>(dstPtr),
                                    3);

    return RPP_SUCCESS;

}
 
// ----------------------------------------
// Host minMaxLoc functions calls 
// ----------------------------------------


RppStatus
rppi_minMaxLoc_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32s* minLoc, Rpp32s* maxLoc)
{
    minMaxLoc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                       min, max, minLoc, maxLoc, 
                       RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_minMaxLoc_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32s* minLoc, Rpp32s* maxLoc)
{
    minMaxLoc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                       min, max, minLoc, maxLoc, 
                       RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_minMaxLoc_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32s* minLoc, Rpp32s* maxLoc)
{
    minMaxLoc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                       min, max, minLoc, maxLoc, 
                       RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}
