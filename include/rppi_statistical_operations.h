#ifndef RPPI_STATISTICAL_OPERATIONS
#define RPPI_STATISTICAL_OPERATIONS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host min functions declaration 
// ----------------------------------------
/* Computes pixel wise minimum on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_min_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_min_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_min_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host max functions declaration 
// ----------------------------------------
/* Computes pixel wise maximum on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_max_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_max_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_max_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host histogram functions declaration 
// ----------------------------------------
/* Computes the histogram of image and stores it in the histogram array of size bins.
*param srcPtr [in] srcPtr input image
*param[in] srcSize srcSize dimensions of the images
*param[out] outputHistogram outputHistogram pointer to store the histogram of the input image
*param[in] bins bins size of output histogram 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_histogram_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins);

RppStatus
rppi_histogram_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins);

RppStatus
rppi_histogram_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins);

// ----------------------------------------
// Host min_max_loc functions declaration 
// ----------------------------------------
/* This function finds the minimum and maximum values and its corresponding location and stores it in the output variables.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] min min minimum pixel value in the input image
 max *param[out] max maximum pixel value in the input image
 minLoc *param[out] minLoc minimum pixel's index in the input image
 maxLoc *param[out] max maximum pixel's index in the input image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_min_max_loc_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc);

RppStatus
rppi_min_max_loc_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc);

RppStatus
rppi_min_max_loc_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc);

// ----------------------------------------
// GPU min functions declaration 
// ----------------------------------------
/* Computes pixel wise minimum on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_min_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_min_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_min_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU max functions declaration 
// ----------------------------------------
/* Computes pixel wise maximum on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_max_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_max_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_max_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU histogram functions declaration 
// ----------------------------------------
/* Computes the histogram of image and stores it in the histogram array of size bins.
*param srcPtr [in] srcPtr input image
*param[in] srcSize srcSize dimensions of the images
*param[out] outputHistogram outputHistogram pointer to store the histogram of the input image
*param[in] bins bins size of output histogram 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_histogram_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins, RppHandle_t rppHandle) ;

RppStatus
rppi_histogram_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins, RppHandle_t rppHandle) ;

RppStatus
rppi_histogram_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32u* outputHistogram,Rpp32u bins, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU min_max_loc functions declaration 
// ----------------------------------------
/* This function finds the minimum and maximum values and its corresponding location and stores it in the output variables.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] min min minimum pixel value in the input image
 max *param[out] max maximum pixel value in the input image
 minLoc *param[out] minLoc minimum pixel's index in the input image
 maxLoc *param[out] max maximum pixel's index in the input image
*param[in] rppHandle  rppHandle OpenCL handle 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_min_max_loc_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc, RppHandle_t rppHandle) ;

RppStatus
rppi_min_max_loc_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc, RppHandle_t rppHandle) ;

RppStatus
rppi_min_max_loc_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp8u* min,Rpp8u* max,Rpp32u* minLoc,Rpp32u* maxLoc, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
