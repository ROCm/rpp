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
// Host custom_convolution functions declaration 
// ----------------------------------------

RppStatus
rppi_custom_convolution_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize kernelSize);

RppStatus
rppi_custom_convolution_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize kernelSize);

RppStatus
rppi_custom_convolution_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize kernelSize);

// ----------------------------------------
// Host non_max_suppression functions declaration 
// ----------------------------------------

RppStatus
rppi_non_max_suppression_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_non_max_suppression_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

RppStatus
rppi_non_max_suppression_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize);

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

// ----------------------------------------
// gpu median_filter functions declaration 
// ----------------------------------------

//KERNAL SIZE = 1,3,5,7.

RppStatus
rppi_median_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_median_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_median_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

// ----------------------------------------
// gpu non_max_suppression functions declaration 
// ----------------------------------------

RppStatus
rppi_non_max_suppression_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_non_max_suppression_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_non_max_suppression_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

// ----------------------------------------
// gpu custom_convolution functions declaration 
// ----------------------------------------

RppStatus
rppi_custom_convolution_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_custom_convolution_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_custom_convolution_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppPtr_t kernel, RppiSize kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_histogram_equalize_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_histogram_equalize_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_histogram_equalize_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_gaussian_filter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_gaussian_filter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

RppStatus
rppi_gaussian_filter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f stdDev, Rpp32u kernelSize, RppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif
