#ifndef RPPI_IMAGE_AUGMENTATIONS.H
#define RPPI_IMAGE_AUGMENTATIONS.H
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host blur functions declaration 
// ----------------------------------------
/* Uses Gaussian for blurring the image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] stdDev standard deviation value to populate gaussian kernels
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_blur_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev);

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev);

RppStatus
rppi_blur_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev);

// ----------------------------------------
// Host contrast functions declaration 
// ----------------------------------------
/* Computes contrast of the image using contrast stretch technique.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] newMin minimum pixel value for contrast stretch
*param[in] newMax maxium pixel value for contrast stretch
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax);

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax);

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax);

// ----------------------------------------
// Host brightness functions declaration 
// ----------------------------------------
/* Computes brightness of an image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] alpha alpha for brightness calculation and value should be between 0 and 20
*param[in] beta beta value for brightness calculation and value should be between 0 and 255
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_brightness_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta);

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta);

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta);

// ----------------------------------------
// Host gamma_correction functions declaration 
// ----------------------------------------
/* Computes gamma correction for an image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
param[in] gamma gamma value used in gamma correction
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma);

RppStatus
rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma);

RppStatus
rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma);

// ----------------------------------------
// Host exposure functions declaration 
// ----------------------------------------

RppStatus
rppi_exposureRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor);

RppStatus
rppi_exposureRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor);

RppStatus
rppi_exposureHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor);

RppStatus
rppi_exposureHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f exposureFactor);

// ----------------------------------------
// Host color_temperature functions declaration 
// ----------------------------------------

RppStatus
rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue);

RppStatus
rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue);

// ----------------------------------------
// Host pixelate functions declaration 
// ----------------------------------------

RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

// ----------------------------------------
// Host jitterAdd functions declaration 
// ----------------------------------------

RppStatus
rppi_jitterAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

RppStatus
rppi_jitterAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

RppStatus
rppi_jitterAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

// ----------------------------------------
// Host vignette functions declaration 
// ----------------------------------------

RppStatus
rppi_vignette_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev);

RppStatus
rppi_vignette_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev);

RppStatus
rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev);

// ----------------------------------------
// Host fish_eye_effect functions declaration 
// ----------------------------------------

RppStatus
rppi_fish_eye_effect_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_fish_eye_effect_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_fish_eye_effect_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// ----------------------------------------
// Host lens_correction functions declaration 
// ----------------------------------------

RppStatus
rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

RppStatus
rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

RppStatus
rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

// ----------------------------------------
// Host occlusionAdd functions declaration 
// ----------------------------------------

RppStatus
rppi_occlusionAdd_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2);

RppStatus
rppi_occlusionAdd_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2);

RppStatus
rppi_occlusionAdd_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2);

// ----------------------------------------
// Host snowy functions declaration 
// ----------------------------------------

RppStatus
rppi_snowyRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength);

RppStatus
rppi_snowyRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength);

// ----------------------------------------
// Host random_shadow functions declaration 
// ----------------------------------------

RppStatus
rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY);

RppStatus
rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY);

RppStatus
rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
                                Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY);

// ----------------------------------------
// GPU blur functions declaration 
// ----------------------------------------
/* Uses Gaussian for blurring the image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] stdDev standard deviation value to populate gaussian kernels
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_blur_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) ;

RppStatus
rppi_blur_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) ;

RppStatus
rppi_blur_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f stdDev, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU contrast functions declaration 
// ----------------------------------------
/* Computes contrast of the image using contrast stretch technique.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] newMin minimum pixel value for contrast stretch
*param[in] newMax maxium pixel value for contrast stretch
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_contrast_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle) ;

RppStatus
rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle) ;

RppStatus
rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u newMin,Rpp32u newMax, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU brightness functions declaration 
// ----------------------------------------
/* Computes brightness of an image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] alpha alpha for brightness calculation and value should be between 0 and 20
*param[in] beta beta value for brightness calculation and value should be between 0 and 255
*param[in] rppHandle OpenCL handle 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_brightness_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle) ;

RppStatus
rppi_brightness_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle) ;

RppStatus
rppi_brightness_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha,Rpp32f beta, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU gamma_correction functions declaration 
// ----------------------------------------
/* Computes gamma correction for an image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
param[in] gamma gamma value used in gamma correction
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_gamma_correction_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle) ;

RppStatus
rppi_gamma_correction_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle) ;

RppStatus
rppi_gamma_correction_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f gamma, RppHandle_t rppHandle) ;
 
#ifdef __cplusplus
}
#endif
#endif
