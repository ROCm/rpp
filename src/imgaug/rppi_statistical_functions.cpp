#include <rppdefs.h>
#include <rppi_statistical_functions.h>

#include "cpu/host_statistical_functions.hpp"


/******* Min ********/

// GPU calls for Min function

// Host calls for Min function

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




/******* Max ********/

// GPU calls for Max function

// Host calls for Max function

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




/******* MinMax ********/

// GPU calls for MinMax function

// Host calls for MinMax function

RppStatus
rppi_minMax_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max)
{
    minMax_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(maskPtr), 
                       min, max, 
                       RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_minMax_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max)
{
    minMax_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(maskPtr), 
                       min, max, 
                       RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_minMax_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max)
{
    minMax_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(maskPtr), 
                       min, max, 
                       RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* MinMaxLoc ********/

// GPU calls for MinMaxLoc function

// Host calls for MinMaxLoc function

RppStatus
rppi_minMaxLoc_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc)
{
    minMaxLoc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(maskPtr), 
                       min, max, minLoc, maxLoc, 
                       RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_minMaxLoc_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc)
{
    minMaxLoc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(maskPtr), 
                       min, max, minLoc, maxLoc, 
                       RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_minMaxLoc_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc)
{
    minMaxLoc_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(maskPtr), 
                       min, max, minLoc, maxLoc, 
                       RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* MeanStdDev ********/

// GPU calls for MeanStdDev function

// Host calls for MeanStdDev function

RppStatus
rppi_meanStd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f* mean, Rpp32f* stdDev)
{
    meanStd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                       mean, stdDev, 
                       RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_meanStd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f* mean, Rpp32f* stdDev)
{
    meanStd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                       mean, stdDev, 
                       RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_meanStd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f* mean, Rpp32f* stdDev)
{
    meanStd_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                       mean, stdDev, 
                       RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




