#ifndef RPPI_STATISTICAL_FUNCTIONS_H
#define RPPI_STATISTICAL_FUNCTIONS_H
#include "rppdefs.h"

#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

//------------------------- Histogram Related -------------------------


// --------------------
// Histogram
// --------------------

// Host function declarations

RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins);

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins);

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins);

// Gpu function declarations




// --------------------
// Equalize Histogram
// --------------------

// Host function declarations

RppStatus
rppi_equalize_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_equalize_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_equalize_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// Gpu function declarations




// --------------------
// Histogram Subimage
// --------------------

// Host function declarations

RppStatus
rppi_histogram_subimage_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                     unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

RppStatus
rppi_histogram_subimage_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                     unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

RppStatus
rppi_histogram_subimage_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, 
                                     unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

// Gpu function declarations





#ifdef __cplusplus
}
#endif

#endif /* RPPI_STATISTICAL_FUNCTIONS_H */