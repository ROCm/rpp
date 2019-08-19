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
/* Computes a Box filter over a window of the input image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[[in] kernelSize kernelSize size of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_box_filter_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_box_filter_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_box_filter_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

// ----------------------------------------
// Host sobel_filter functions declaration 
// ----------------------------------------
/* Implements the Sobel Image Filter Kernel.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] sobelType sobelType 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_sobel_filter_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u sobelType);

RppStatus
rppi_sobel_filter_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u sobelType);

RppStatus
rppi_sobel_filter_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u sobelType);

// ----------------------------------------
// Host non_max_suppression functions declaration 
// ----------------------------------------
/*This function uses a N x N box around the output pixel used to determine value. If the centre pixel is the maximum it will be retained else it will be replaced with zero.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] kernelSize kernelSize size of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_non_max_suppression_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_non_max_suppression_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_non_max_suppression_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

// ----------------------------------------
// Host median_filter functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x,y) srcSize = median(xi,yi)
x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
bound = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_median_filter_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_median_filter_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_median_filter_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

// ----------------------------------------
// Host custom_convolution functions declaration 
// ----------------------------------------
/* Applies a N x M convolution on every input pixel and stores it in the destination.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] kernelSize kernelSize dimension of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_custom_convolution_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize);

RppStatus
rppi_custom_convolution_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize);

RppStatus
rppi_custom_convolution_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize);

// ----------------------------------------
// Host gaussian_filter functions declaration 
// ----------------------------------------
/* Applies gaussian filter over every pixel in the input image and stores it in the destination image.
*param [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev kernel 
*param[in] kernelSize stdDev standard deviation value to populate gaussian kernel
*param[in] kernelSize size of the kernel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_gaussian_filter_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

RppStatus
rppi_gaussian_filter_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

RppStatus
rppi_gaussian_filter_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

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
// GPU box_filter functions declaration 
// ----------------------------------------
/* Computes a Box filter over a window of the input image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[[in] kernelSize kernelSize size of the kernel
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_box_filter_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_box_filter_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_box_filter_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU sobel_filter functions declaration 
// ----------------------------------------
/* Implements the Sobel Image Filter Kernel.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] sobelType sobelType 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_sobel_filter_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u sobelType, RppHandle_t rppHandle) ;

RppStatus
rppi_sobel_filter_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u sobelType, RppHandle_t rppHandle) ;

RppStatus
rppi_sobel_filter_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u sobelType, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU non_max_suppression functions declaration 
// ----------------------------------------
/*This function uses a N x N box around the output pixel used to determine value. If the centre pixel is the maximum it will be retained else it will be replaced with zero.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] kernelSize kernelSize size of the kernel
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_non_max_suppression_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_non_max_suppression_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_non_max_suppression_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU median_filter functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x,y) srcSize = median(xi,yi)
x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
bound = (kernelsize + 1) / 2
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] kernelSize dimension of the kernel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_median_filter_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_median_filter_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_median_filter_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU custom_convolution functions declaration 
// ----------------------------------------
/* Applies a N x M convolution on every input pixel and stores it in the destination.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] kernelSize kernelSize dimension of the kernel
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_custom_convolution_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_custom_convolution_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_custom_convolution_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppPtr_t kernel,RppiSize kernelSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU gaussian_filter functions declaration 
// ----------------------------------------
/* Applies gaussian filter over every pixel in the input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev kernel 
*param[in] kernelSize stdDev standard deviation value to populate gaussian kernel
*param[in] rppHandle kernelSize size of the kernel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_gaussian_filter_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_gaussian_filter_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_gaussian_filter_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

#ifdef __cplusplus
}
#endif
#endif
