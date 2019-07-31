#ifndef RPPI_STATISTICAL_OPERATIONS
#define RPPI_STATISTICAL_OPERATIONS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host histogram functions declaration 
// ----------------------------------------
/* Computes histogrma of image
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] outputHistogram output array of histogram
*param[in] bins number of bins
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins);

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins);

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins);


// ----------------------------------------
// Host thresholding functions declaration 
// ----------------------------------------

RppStatus
rppi_thresholding_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp8u min, Rpp8u max);

RppStatus
rppi_thresholding_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp8u min, Rpp8u max);

RppStatus
rppi_thresholding_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp8u min, Rpp8u max);

// ----------------------------------------
// Host min functions declaration 
// ----------------------------------------

RppStatus
rppi_min_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_min_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_min_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

// ----------------------------------------
// Host max functions declaration 
// ----------------------------------------

RppStatus
rppi_max_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_max_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_max_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr);

// ----------------------------------------
// Host minMaxLoc functions declaration 
// ----------------------------------------

RppStatus
rppi_minMaxLoc_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc);

RppStatus
rppi_minMaxLoc_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc);

RppStatus
rppi_minMaxLoc_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t maskPtr, Rpp8u* min, Rpp8u* max, Rpp8u** minLoc, Rpp8u** maxLoc);


 
#ifdef __cplusplus
}
#endif
#endif
