#ifndef RPPI_COLOR_MODEL_CONVERSIONS
#define RPPI_COLOR_MODEL_CONVERSIONS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host color_temperature functions declaration 
// ----------------------------------------
/* Changes color temperature of an image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] adjustmentValue adjustmentValue adjustment value used in color temperature correction which should range between -100 - 100
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_color_temperature_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue);

RppStatus
rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue);

RppStatus
rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue);

// ----------------------------------------
// Host vignette functions declaration 
// ----------------------------------------
/* Introduces vignette effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev stdDev standard deviation for the gaussian function used in the vignette (decides amount of vignette)
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_vignette_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev);

RppStatus
rppi_vignette_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev);

RppStatus
rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev);


// ----------------------------------------
// GPU color_temperature functions declaration 
// ----------------------------------------
/* Changes color temperature of an image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] adjustmentValue adjustmentValue adjustment value used in color temperature correction which should range between -100 - 100
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_color_temperature_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue, RppHandle_t rppHandle) ;

RppStatus
rppi_color_temperature_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue, RppHandle_t rppHandle) ;

RppStatus
rppi_color_temperature_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32s adjustmentValue, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU vignette functions declaration 
// ----------------------------------------
/* Introduces vignette effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] stdDev stdDev standard deviation for the gaussian function used in the vignette (decides amount of vignette)
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_vignette_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) ;

RppStatus
rppi_vignette_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) ;

RppStatus
rppi_vignette_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) ;

 
#ifdef __cplusplus
}
#endif
#endif
