#ifndef RPPI_STATISTICAL_FUNCTIONS_H
#define RPPI_STATISTICAL_FUNCTIONS_H
#include "rppdefs.h"

#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// --------------------
// Min
// --------------------

// Gpu function declarations

// Host function declarations

RppStatus
rppi_min_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_min_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_min_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);


// --------------------
// Max
// --------------------

// Gpu function declarations

// Host function declarations

RppStatus
rppi_max_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_max_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_max_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);


// --------------------
// MinMax
// --------------------

// Gpu function declarations

// Host function declarations

RppStatus
rppi_minMax_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max);

RppStatus
rppi_minMax_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max);

RppStatus
rppi_minMax_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max);


// --------------------
// MinMaxLoc
// --------------------

// Gpu function declarations

// Host function declarations

RppStatus
rppi_minMaxLoc_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc);

RppStatus
rppi_minMaxLoc_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc);

RppStatus
rppi_minMaxLoc_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc);


// --------------------
// MeanStdDev
// --------------------

// Gpu function declarations

// Host function declarations

RppStatus
rppi_meanStd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f* mean, Rpp32f* stdDev);

RppStatus
rppi_meanStd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f* mean, Rpp32f* stdDev);

RppStatus
rppi_meanStd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32f* mean, Rpp32f* stdDev);


#endif /* RPPI_STATISTICAL_FUNCTIONS_H */