#ifndef RPPI_MORPHOLOGICAL_TRANSFORMS
#define RPPI_MORPHOLOGICAL_TRANSFORMS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif



// ----------------------------------------
// Host dilate functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x srcPtr
y) srcSize = max(xi
yi) dstPtr
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
rppi_dilate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_dilate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_dilate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

// ----------------------------------------
// Host erode functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x srcPtr
y) srcSize = min(xi
yi) dstPtr
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
rppi_erode_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_erode_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

RppStatus
rppi_erode_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize);

// ----------------------------------------
// GPU dilate functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x srcPtr
y) srcSize = max(xi
yi) dstPtr
x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
bound rppHandle = (kernelsize + 1) / 2
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
rppi_dilate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_dilate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_dilate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU erode functions declaration 
// ----------------------------------------
/* This function uses a N x N box around the output pixel used to determine value.
dest(x srcPtr
y) srcSize = min(xi
yi) dstPtr
x-bound kernelSize < xi < x+bound and x-bound < xi < x+bound
bound rppHandle = (kernelsize + 1) / 2
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
rppi_erode_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_erode_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;

RppStatus
rppi_erode_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
