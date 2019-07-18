#ifndef RPPI_IMAGE_AUGMENTATIONS
#define RPPI_IMAGE_AUGMENTATIONS
 
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
// Host fog functions declaration 
// ----------------------------------------
/* Introduces foggy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] fogValue fogValue float value to decide the amount of foggy effect to be added which should range between 0 - 1
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_fog_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue);

RppStatus
rppi_fog_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue);

RppStatus
rppi_fog_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue);

// ----------------------------------------
// Host rain functions declaration 
// ----------------------------------------
/* Introduces rainy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rainValue rainValue float value to decide the amount of rainy effect to be added which should range between 0 - 1
*param[in] rainWidth rainWidth width of the rain line
*param[in] rainHeight rainHeight height of the rain line
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rain_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainValue,Rpp32u rainWidth,Rpp32u rainHeight);

RppStatus
rppi_rain_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainValue,Rpp32u rainWidth,Rpp32u rainHeight);

RppStatus
rppi_rain_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainValue,Rpp32u rainWidth,Rpp32u rainHeight);

// ----------------------------------------
// Host snow functions declaration 
// ----------------------------------------
/* Introduces snowy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] snowValue snowValue float value to decide the amount of snowy effect to be added which should range between 0 - 1
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_snow_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue);

RppStatus
rppi_snow_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue);

RppStatus
rppi_snow_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue);

// ----------------------------------------
// Host random_shadow functions declaration 
// ----------------------------------------
/* Adds multiple random shadows [rectangle shaped shadows] in the image to the roi area.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] Rpp32u x1 value of roi
*param[in] validate_int_range y1 value of roi
 srcSize.width - 1 *param[in] x2 value of roi
 y1 *param[in]y2 value of roi
*param[in] x2 numberOfShadows number of shadows to be added in the roi region
*param[in] y2 maxSizeX shadow's maximum width
*param[in] numberOfShadows maxSizeY shadow's maximum height
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY);

RppStatus
rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY);

RppStatus
rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY);

// ----------------------------------------
// Host blend functions declaration 
// ----------------------------------------
/* Blends two source image and stores it in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] alpha alpha transperancy factor of the images where alpha is for image1 and 1-alpha is for image2
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_blend_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha);

RppStatus
rppi_blend_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha);

RppStatus
rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha);

// ----------------------------------------
// Host pixelate functions declaration 
// ----------------------------------------
/* pixelates the roi region of the image  
*param srcPtr [in] srcPtr1 input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] Rpp32u x1 x1 value of roi
*param[in] validate_int_range y1 y1 value of roi
 srcSize.width - 1 *param[in] x2 x2 value of roi
 y1 *param[in] y2 y2 value of roi
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2 );

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

// ----------------------------------------
// Host random_crop_letterbox functions declaration 
// ----------------------------------------
/* Crops the roi region
 srcPtr adds border and stores it in destination
*param srcSize [in] srcPtr input image
*param[in] dstPtr  srcSize dimensions of the images
*param[out] dstSize dstPtr output image
*param[in] x1 x1 value of roi
*param[in] y1 y1 value of roi
 x2 *param[in] x2 value of roi
 y2 *param[in]y2 value of roi
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_random_crop_letterbox_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_random_crop_letterbox_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_random_crop_letterbox_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

// ----------------------------------------
// Host occlusion functions declaration 
// ----------------------------------------
/* Adds occlusion in a region of the first image by taking a crop from a region in the second image 
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcSize  srcSize1 dimensions of the first input image
*param srcPtr2 [in] srcPtr2 input image2
*param[in] srcSize2  srcSize2 dimensions of the secong input image
*param[out] dstPtr dstPtr output image
*param[in] src1x1 src1x1 value of roi in image1
*param[in] src1y1 src1y1 value of roi in image1
 src1x2 *param[in] src1x2 value of roi in image1
 src1y2 *param[in] src1y2 value of roi in image1
*param[in] src2x1 src2x1 value of roi in image2
*param[in] src2y1 src2y1 value of roi in image2
 src2x2 *param[in] src2x2 value of roi in image2
 src2y2 *param[in] src2y2 value of roi in image2
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_occlusion_u8_pln1_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2);

RppStatus
rppi_occlusion_u8_pln3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2);

RppStatus
rppi_occlusion_u8_pkd3_host(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2);

// ----------------------------------------
// Host exposure functions declaration 
// ----------------------------------------
/*Changes exposure of an image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] exposureValue exposureFactor factor used in exposure correction which should range between -4 - 4
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_exposure_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue);

RppStatus
rppi_exposure_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue);

RppStatus
rppi_exposure_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue);


// ----------------------------------------
// Host jitterAdd functions declaration 
// ----------------------------------------
/* Introduces jitter in the entire image
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image
*param[in] maxJitterX maximum jitter range in the x direction (number of pixels)
*param[in] maxJitterY maximum jitter range in the y direction (number of pixels)
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_jitter_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

RppStatus
rppi_jitter_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

RppStatus
rppi_jitter_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

// ----------------------------------------
// Host add noise functions  declaration
// ----------------------------------------

//Host function declaration
RppStatus
rppi_noise_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter);

RppStatus
rppi_noise_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter);

RppStatus
rppi_noise_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter);

// // ----------------------------------------
// // Host blend functions  declaration
// // ----------------------------------------

// RppStatus
// rppi_blend_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha) ;

// RppStatus
// rppi_blend_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha) ;

// RppStatus
// rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha) ;

// // ----------------------------------------
// // Host add noise functions  declaration
// // ----------------------------------------

// //Host function declaration
// RppStatus
// rppi_noiseAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter);

// RppStatus
// rppi_noiseAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter);

// RppStatus
// rppi_noiseAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter);

// // ----------------------------------------
// // Host fog functions declaration 
// // ----------------------------------------

// RppStatus
// rppi_fog_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue);

// RppStatus
// rppi_fog_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue);

// RppStatus
// rppi_fog_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f fogValue);

// // ----------------------------------------
// // Host fog functions declaration 
// // ----------------------------------------

// RppStatus
// rppi_rain_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, 
//                                             Rpp32u rainWidth, Rpp32u rainHeight);

// RppStatus
// rppi_rain_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue,
//                                              Rpp32u rainWidth, Rpp32u rainHeight);

// RppStatus
// rppi_rain_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, Rpp32u rainWidth, Rpp32u rainHeight);
// // Host color_temperature functions declaration 
// // ----------------------------------------
// /* Changes color temperature of an image.
// param[in] srcPtr input image
// *param[in] srcSize dimensions of the image
// *param[out] dstPtr output image
// param[in] adjustmentValue adjustment value used in color temperature correction
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                           Rpp8s adjustmentValue);

// RppStatus
// rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                           Rpp8s adjustmentValue);

// // ----------------------------------------
// // Host pixelate functions declaration 
// // ----------------------------------------
// /* Pixelates a region of an image
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *param[in] dstSize dimensions of the output images
// *param[in] x1 x1 value of roi
// *param[in] y1 y1 value of roi
// *param[in] x2 x2 value of roi
// *param[in] y2 y2 value of roi
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                              Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

// RppStatus
// rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                              Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

// RppStatus
// rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                              Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

// // ----------------------------------------
// // Host jitterAdd functions declaration 
// // ----------------------------------------
// /* Introduces jitter in the entire image
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *param[in] dstSize dimensions of the output images
// *param[in] maxJitterX maximum jitter range in the x direction (number of pixels)
// *param[in] maxJitterY maximum jitter range in the y direction (number of pixels)
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_jitterAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                              unsigned int maxJitterX, unsigned int maxJitterY);

// RppStatus
// rppi_jitterAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                              unsigned int maxJitterX, unsigned int maxJitterY);

// RppStatus
// rppi_jitterAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                              unsigned int maxJitterX, unsigned int maxJitterY);

// // ----------------------------------------
// // Host vignette functions declaration 
// // ----------------------------------------
// /* Introduces vignette effect in the entire image
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *param[in] stdDev standard deviation for the gaussian function used in the vignette (decides amount of vignette)
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_vignette_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                           Rpp32f stdDev);

// RppStatus
// rppi_vignette_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                           Rpp32f stdDev);

// RppStatus
// rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                           Rpp32f stdDev);

// // ----------------------------------------
// // Host fish_eye_effect functions declaration 
// // ----------------------------------------
// /* Introduces fish eye effect in the entire image
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_fish_eye_effect_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// RppStatus
// rppi_fish_eye_effect_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// RppStatus
// rppi_fish_eye_effect_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// // ----------------------------------------
// // Host lens_correction functions declaration 
// // ----------------------------------------
// /* Introduces lens correction in the lens distorted images
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *param[in] strength strength of lens correction needed
// *param[in] zoom extent to which zoom-out is needed
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

// RppStatus
// rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

// RppStatus
// rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

// // ----------------------------------------
// // Host occlusionAdd functions declaration 
// // ----------------------------------------
// /* Introduces occlusion in a region of the first image by taking a crop from a region in the second image
// *param[in] srcPtr1 input image 1
// *param[in] srcPtr2 input image 2
// *param[in] srcSize1 dimensions of the input image 1
// *param[in] srcSize2 dimensions of the input image 2
// *param[out] dstPtr output image
// *param[in] src1x1 x1 value of roi in image 1
// *param[in] src1y1 y1 value of roi in image 1
// *param[in] src1x2 x2 value of roi in image 1
// *param[in] src1y2 y2 value of roi in image 1
// *param[in] src2x1 x1 value of roi in image 2
// *param[in] src2y1 y1 value of roi in image 2
// *param[in] src2x2 x2 value of roi in image 2
// *param[in] src2y2 y2 value of roi in image 2
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_occlusionAdd_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
//                                Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
//                                Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2);

// RppStatus
// rppi_occlusionAdd_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
//                                Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
//                                Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2);

// RppStatus
// rppi_occlusionAdd_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
//                                Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
//                                Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2);

// // ----------------------------------------
// // Host snowy functions declaration 
// // ----------------------------------------
// /* Introduces snowy effect in the entire image
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *param[in] strength strength of snowy effect desired
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_snowyRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                          Rpp32f strength);

// RppStatus
// rppi_snowyRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
//                          Rpp32f strength);

// // ----------------------------------------
// // Host random_shadow functions declaration 
// // ----------------------------------------
// /* Introduces random shadow effect in the image
// *param[in] srcPtr input image
// *param[in] srcSize dimensions of the input images
// *param[out] dstPtr output image
// *param[in] x1 x1 value of roi
// *param[in] y1 y1 value of roi
// *param[in] x2 x2 value of roi
// *param[in] y2 y2 value of roi
// *param[in] numberOfShadows total number of shadows desired
// *param[in] maxSizeX maximum x dimension of shadow
// *param[in] maxSizeY maximum y dimension of shadow
// *returns a  RppStatus enumeration. 
// *retval RPP_SUCCESS : No error succesful completion
// *retval RPP_ERROR : Error 
// */

// RppStatus
// rppi_random_shadow_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                                 Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
//                                 Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY);

// RppStatus
// rppi_random_shadow_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                                 Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
//                                 Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY);

// RppStatus
// rppi_random_shadow_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
//                                 Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, 
//                                 Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY);
// RppStatus
// rppi_rain_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, 
//                                                 Rpp32u rainWidth, Rpp32u rainHeight);


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

// ----------------------------------------
// GPU fog functions declaration 
// ----------------------------------------
/* Introduces foggy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] fogValue fogValue float value to decide the amount of foggy effect to be added which should range between 0 - 1
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_fog_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue, RppHandle_t rppHandle) ;

RppStatus
rppi_fog_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue, RppHandle_t rppHandle) ;

RppStatus
rppi_fog_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f fogValue, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU rain functions declaration 
// ----------------------------------------
/* Introduces rainy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rainValue rainValue float value to decide the amount of rainy effect to be added which should range between 0 - 1
*param[in] rainWidth rainWidth width of the rain line
*param[in] rainHeight rainHeight height of the rain line
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rain_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainValue,Rpp32u rainWidth,Rpp32u rainHeight, RppHandle_t rppHandle) ;

RppStatus
rppi_rain_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainValue,Rpp32u rainWidth,Rpp32u rainHeight, RppHandle_t rppHandle) ;

RppStatus
rppi_rain_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f rainValue,Rpp32u rainWidth,Rpp32u rainHeight, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU snow functions declaration 
// ----------------------------------------
/* Introduces snowy effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] snowValue snowValue float value to decide the amount of snowy effect to be added which should range between 0 - 1
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_snow_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle) ;

RppStatus
rppi_snow_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle) ;

RppStatus
rppi_snow_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f snowValue, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU random_shadow functions declaration 
// ----------------------------------------
/* Adds multiple random shadows [rectangle shaped shadows] in the image to the roi area.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] Rpp32u x1 value of roi
*param[in] validate_int_range y1 value of roi
 srcSize.width - 1 *param[in] x2 value of roi
 y1 *param[in]y2 value of roi
*param[in] x2 numberOfShadows number of shadows to be added in the roi region
*param[in] y2 maxSizeX shadow's maximum width
*param[in] numberOfShadows maxSizeY shadow's maximum height
*param[in] maxSizeX OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle) ;

RppStatus
rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle) ;

RppStatus
rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,Rpp32u numberOfShadows,Rpp32u maxSizeX,Rpp32u maxSizeY, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU blend functions declaration 
// ----------------------------------------
/* Blends two source image and stores it in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] alpha alpha transperancy factor of the images where alpha is for image1 and 1-alpha is for image2
param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_blend_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle) ;

RppStatus
rppi_blend_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle) ;

RppStatus
rppi_blend_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f alpha, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU pixelate functions declaration 
// ----------------------------------------
/* pixelates the roi region of the image  
*param srcPtr [in] srcPtr1 input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] Rpp32u x1 x1 value of roi
*param[in] validate_int_range y1 y1 value of roi
 srcSize.width - 1 *param[in] x2 x2 value of roi
 y1 *param[in] y2 y2 value of roi
*param[in] x2 OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2,RppHandle_t rppHandle) ;

RppStatus
rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u x1 ,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU random_crop_letterbox functions declaration 
// ----------------------------------------
/* Crops the roi region
 srcPtr adds border and stores it in destination
*param srcSize [in] srcPtr input image
*param[in] dstPtr  srcSize dimensions of the images
*param[out] dstSize dstPtr output image
*param[in] x1 x1 value of roi
*param[in] y1 y1 value of roi
 x2 *param[in] x2 value of roi
 y2 *param[in]y2 value of roi
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_random_crop_letterbox_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_random_crop_letterbox_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_random_crop_letterbox_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU occlusion functions declaration 
// ----------------------------------------
/* Adds occlusion in a region of the first image by taking a crop from a region in the second image 
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcSize  srcSize1 dimensions of the first input image
*param srcPtr2 [in] srcPtr2 input image2
*param[in] srcSize2  srcSize2 dimensions of the secong input image
*param[out] dstPtr dstPtr output image
*param[in] src1x1 src1x1 value of roi in image1
*param[in] src1y1 src1y1 value of roi in image1
 src1x2 *param[in] src1x2 value of roi in image1
 src1y2 *param[in] src1y2 value of roi in image1
*param[in] src2x1 src2x1 value of roi in image2
*param[in] src2y1 src2y1 value of roi in image2
 src2x2 *param[in] src2x2 value of roi in image2
 src2y2 *param[in] src2y2 value of roi in image2
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_occlusion_u8_pln1_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle) ;

RppStatus
rppi_occlusion_u8_pln3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle) ;

RppStatus
rppi_occlusion_u8_pkd3_gpu(RppPtr_t srcPtr1,RppiSize srcSize,RppPtr_t srcPtr2,RppiSize srcSize2,RppPtr_t dstPtr,Rpp32u src1x1,Rpp32u src1y1,Rpp32u src1x2,Rpp32u src1y2,Rpp32u src2x1,Rpp32u src2y1,Rpp32u src2x2,Rpp32u src2y2, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU exposure functions declaration 
// ----------------------------------------
/*Changes exposure of an image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] exposureValue exposureFactor factor used in exposure correction which should range between -4 - 4
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle) ;

RppStatus
rppi_exposure_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle) ;

RppStatus
rppi_exposure_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f exposureValue, RppHandle_t rppHandle) ;


// ----------------------------------------
// GPU jitter functions declaration 
// ----------------------------------------
/* Introduces jitter in the entire image
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image
*param[in] minJitter minimum jitter value that needs to be added to the pixels and it should range in 0 - 255
*param[in] maxJitter maximum jitter value that needs to be added to the pixels and it should range in 0 - 255
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
rppi_jitter_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle);
RppStatus
rppi_jitter_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle);
RppStatus
rppi_jitter_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u minJitter, Rpp32u maxJitter, RppHandle_t rppHandle);


// ----------------------------------------
// GPU add noise functions  declaration
// ----------------------------------------
/* Introduces noise in the entire image using gaussian or salt and pepper.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*param[in] noiseType takes RppiNoise as input
*typedef enum{
    GAUSSIAN,
    SNP
} RppiNoise;
*param[in] noiseParameter parameters for Gaussian or salt and pepper
SNP (salt and pepper) 
pointer to a float variable and it should range between 0 - 1
GAUSSIAN
pointer to a structure 
typedef struct {
    Rpp32f mean;
    Rpp32f sigma;
} RppiGaussParameter;
mean and sigma greater than 1
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/
RppStatus
rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, 
                                    Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle);

RppStatus
rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, 
                                Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle);

RppStatus
rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, 
                        Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle);
//---------------------------------------
RppStatus
rppi_noiseAdd_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle);

RppStatus
rppi_noiseAdd_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle);

RppStatus
rppi_noiseAdd_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle);

// // ----------------------------------------
// // GPU blend functions declaration 
// // ----------------------------------------

// RppStatus
// rppi_blend_u8_pln1_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, RppHandle_t rppHandle);

// RppStatus
// rppi_blend_u8_pln3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, RppHandle_t rppHandle) ;

// RppStatus
// rppi_blend_u8_pkd3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f alpha, RppHandle_t rppHandle) ;

// // ----------------------------------------
// // GPU pixelate functions declaration 
// // ----------------------------------------
// RppStatus
// rppi_pixelate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;
// RppStatus
// rppi_pixelate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;
// RppStatus
// rppi_pixelate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u kernelSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;
// // ----------------------------------------
// // GPU snow functions declaration 
// // ----------------------------------------
// RppStatus
// rppi_snow_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f snowCoefficient, RppHandle_t rppHandle);
// RppStatus
// rppi_snow_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f snowCoefficient, RppHandle_t rppHandle);
// RppStatus
// rppi_snow_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f snowCoefficient, RppHandle_t rppHandle);
// // GPU add noise functions  declaration
// // ----------------------------------------

// RppStatus
// rppi_noiseAdd_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle);

// RppStatus
// rppi_noiseAdd_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle);

// RppStatus
// rppi_noiseAdd_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiNoise noiseType, void * noiseParameter, RppHandle_t rppHandle);

// // ----------------------------------------
// // Exposure modification functions  declaration
// // ----------------------------------------

// RppStatus
// rppi_exposure_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f exposureValue, RppHandle_t rppHandle);

// RppStatus
// rppi_exposure_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f exposureValue, RppHandle_t rppHandle);

// RppStatus
// rppi_exposure_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f exposureValue, RppHandle_t rppHandle);

// // ----------------------------------------
// // Rainy functions  declaration
// // ----------------------------------------
// RppStatus
// rppi_rain_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, 
//                                         Rpp32u rainWidth, Rpp32u rainHeight, RppHandle_t rppHandle);

// RppStatus
// rppi_rain_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue,
//                                         Rpp32u rainWidth, Rpp32u rainHeight,RppHandle_t rppHandle);

// RppStatus
// rppi_rain_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, Rpp32f rainValue, 
//                                         Rpp32u rainWidth, Rpp32u rainHeight,RppHandle_t rppHandle);

// // ----------------------------------------
// // Random Shadow functions  declaration
// // ----------------------------------------

// RppStatus
// rppi_random_shadow_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle);

// RppStatus
// rppi_random_shadow_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle);

// RppStatus
// rppi_random_shadow_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2, Rpp32u numberOfShadows, Rpp32u maxSizeX, Rpp32u maxSizeY, RppHandle_t rppHandle);

//Occlusion
//---------------------------------------
RppStatus
rppi_occlusionAdd_u8_pln1_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle);

RppStatus
rppi_occlusionAdd_u8_pln3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle);

RppStatus
rppi_occlusionAdd_u8_pkd3_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize1, RppiSize srcSize2, RppPtr_t dstPtr, 
                               Rpp32u src1x1, Rpp32u src1y1, Rpp32u src1x2, Rpp32u src1y2, 
                               Rpp32u src2x1, Rpp32u src2y1, Rpp32u src2x2, Rpp32u src2y2, RppHandle_t rppHandle);

//------------------------------------
//Histogram Balance
//--------------------------------------
RppStatus
rppi_histogram_balance_u8_pln1_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                    RppiSize srcSize, RppHandle_t rppHandle);

RppStatus
rppi_histogram_balance_u8_pln1_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                    RppiSize srcSize, RppHandle_t rppHandle);

RppStatus
rppi_histogram_balance_u8_pln1_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, 
                                    RppiSize srcSize, RppHandle_t rppHandle);
                                    
#ifdef __cplusplus
}
#endif
#endif
