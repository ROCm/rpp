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
// Host integral functions declaration 
// ----------------------------------------

RppStatus
rppi_integral_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_integral_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_integral_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);


 
#ifdef __cplusplus
}
#endif
#endif
