#ifndef RPPI_FILTER_OPERATIONS
#define RPPI_FILTER_OPERATIONS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host bilateral_filter functions declaration 
// ----------------------------------------
/* Apllies bilateral filtering to the input image.
param[in] srcPtr1 input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] filterSize size of filter which uses the neighbouring pixels value  for filtering.
*param[in] sigmaI filter sigma value in color space and value should be between 0 and 20
*param[in] sigmaS filter sigma value in coordinate space and value should be between 0 and 20
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_bilateral_filter_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS);

RppStatus
rppi_bilateral_filter_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS);

RppStatus
rppi_bilateral_filter_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS);

// ----------------------------------------
// Host box_filter functions declaration 
// ----------------------------------------
/* Apllies box filtering to the input image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] kernelSize size of filter which uses the neighbouring pixels value  for filtering.
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_box_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_box_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_box_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

// ----------------------------------------
// Host sobel_filter functions declaration 
// ----------------------------------------

RppStatus
rppi_sobel_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u sobelType);

RppStatus
rppi_sobel_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u sobelType);

RppStatus
rppi_sobel_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u sobelType);

// ----------------------------------------
// Host median_filter functions declaration 
// ----------------------------------------

RppStatus
rppi_median_filter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_median_filter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_median_filter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

// ----------------------------------------
// GPU bilateral_filter functions declaration 
// ----------------------------------------
/* Apllies bilateral filtering to the input image.
param[in] srcPtr1 input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] filterSize size of filter which uses the neighbouring pixels value  for filtering.
*param[in] sigmaI filter sigma value in color space and value should be between 0 and 20
*param[in] sigmaS filter sigma value in coordinate space and value should be between 0 and 20
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_bilateral_filter_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS, RppHandle_t rppHandle) ;

RppStatus
rppi_bilateral_filter_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS, RppHandle_t rppHandle) ;

RppStatus
rppi_bilateral_filter_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u filterSize,Rpp64f sigmaI,Rpp64f sigmaS, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
