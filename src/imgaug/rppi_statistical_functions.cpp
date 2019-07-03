#include <rppdefs.h>
#include <rppi_statistical_functions.h>

#include "cpu/host_statistical_functions.hpp"


/******* Histogram ********/

// GPU calls for Histogram function

// Host calls for Histogram function

RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins)
{
    histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, 
                          RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins)
{
    histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, 
                          RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins)
{
    histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, 
                          RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* Equalize Histogram ********/

// GPU calls for Equalize Histogram function

// Host calls for Equalize Histogram function

RppStatus
rppi_equalize_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    equalize_histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                       RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_equalize_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    equalize_histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                       RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_equalize_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr)
{
    equalize_histogram_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, static_cast<Rpp8u*>(dstPtr), 
                       RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}




/******* Histogram Subimage ********/

// GPU calls for Histogram Subimage function

// Host calls for Histogram Subimage function

RppStatus
rppi_histogram_subimage_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                     unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    histogram_subimage_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, x1, y1, x2, y2, 
                          RPPI_CHN_PLANAR, 1);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_subimage_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                     unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    histogram_subimage_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, x1, y1, x2, y2, 
                          RPPI_CHN_PLANAR, 3);

    return RPP_SUCCESS;

}

RppStatus
rppi_histogram_subimage_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                     unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2)
{
    histogram_subimage_host<Rpp8u>(static_cast<Rpp8u*>(srcPtr), srcSize, 
                          outputHistogram, bins, x1, y1, x2, y2, 
                          RPPI_CHN_PACKED, 3);

    return RPP_SUCCESS;

}