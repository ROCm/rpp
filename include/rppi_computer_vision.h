#ifndef RPPI_COMPUTER_VISION
#define RPPI_COMPUTER_VISION
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------
// Host local_binary_pattern functions declaration 
// ----------------------------------------
/*Extracts LBP image from an input image and stores it in the output image.
 srcPtr Local binary pattern is defined as: Each pixel (y
x) srcSize generate an 8 bit value describing the local binary pattern around the pixel
 dstPtr by comparing the pixel value with its 8 neighbours (selected neighbours of the 3x3 or 5x5 window).
*param [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_local_binary_pattern_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_local_binary_pattern_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_local_binary_pattern_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host data_object_copy functions declaration 
// ----------------------------------------
/* Performs a deep copy from the input image to the output image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_data_object_copy_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_data_object_copy_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_data_object_copy_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host gaussian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a Gaussian Image Pyramid from an input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
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
rppi_gaussian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

// ----------------------------------------
// Host laplacian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a laplacian Image Pyramid from an input image and stores it in the destination image.
*param srcPtr [in] srcPtr input image
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
rppi_laplacian_image_pyramid_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize);

// ----------------------------------------
// GPU local_binary_pattern functions declaration 
// ----------------------------------------
/*Extracts LBP image from an input image and stores it in the output image.
 srcPtr Local binary pattern is defined as: Each pixel (y
x) srcSize generate an 8 bit value describing the local binary pattern around the pixel
 dstPtr by comparing the pixel value with its 8 neighbours (selected neighbours of the 3x3 or 5x5 window).
*param rppHandle [in] srcPtr input image
*param[in]  srcSize dimensions of the images
*param[out] dstPtr output image
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_local_binary_pattern_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_local_binary_pattern_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_local_binary_pattern_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU data_object_copy functions declaration 
// ----------------------------------------
/* Performs a deep copy from the input image to the output image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_data_object_copy_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_data_object_copy_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_data_object_copy_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU gaussian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a Gaussian Image Pyramid from an input image and stores it in the destination image.
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
rppi_gaussian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_gaussian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_gaussian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU laplacian_image_pyramid functions declaration 
// ----------------------------------------
/* Computes a laplacian Image Pyramid from an input image and stores it in the destination image.
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
rppi_laplacian_image_pyramid_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_laplacian_image_pyramid_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_laplacian_image_pyramid_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev,Rpp32u kernelSize, RppHandle_t rppHandle) ; 

#ifdef __cplusplus
}
#endif
#endif
