#ifndef RPPI_COLOR_MODEL_CONVERSIONS
#define RPPI_COLOR_MODEL_CONVERSIONS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host rgb_to_hsv functions declaration 
// ----------------------------------------
/* Converts RGB image to HSV image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rgb_to_hsv_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_rgb_to_hsv_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_rgb_to_hsv_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host hsv_to_rgb functions declaration 
// ----------------------------------------
/* Converts HSV image to RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hsv_to_rgb_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_hsv_to_rgb_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_hsv_to_rgb_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host hueRGB functions declaration 
// ----------------------------------------
/* Computes hue value and updates it in RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] hueShift hue shift for hue calculation
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hueRGB_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift);

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift);

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift);

// ----------------------------------------
// Host hueHSV functions declaration 
// ----------------------------------------
/* Computes hue value and updates it in HSV image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] hueShift hue shift for hue calculation
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hueHSV_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift);

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift);

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift);

// ----------------------------------------
// Host saturationRGB functions declaration 
// ----------------------------------------
/* Computes saturation value and updates it in RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] saturationFactor saturationFactor for saturation calculation
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_saturationRGB_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor);

RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor);

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor);

// ----------------------------------------
// Host saturationHSV functions declaration 
// ----------------------------------------
/* Computes saturation value and updates it in HSV image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] saturationFactor saturationFactor for saturation calculation
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_saturationHSV_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor);

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor);

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor);

// ----------------------------------------
// Host rgb_to_hsl functions declaration 
// ----------------------------------------
/* Converts RGB image to HSL image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rgb_to_hsl_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_rgb_to_hsl_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_rgb_to_hsl_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host hsl_to_rgb functions declaration 
// ----------------------------------------
/* Converts HSL image to RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hsl_to_rgb_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_hsl_to_rgb_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_hsl_to_rgb_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

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
// Host channel_extract functions declaration 
// ----------------------------------------
/* Extract a single channel from given RGB image and stores it in destination grey scale image. 
/*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*param[in] extractChannelNumber extractChannelNumber The channel to be extracted and it could be
*0---> R channel
 *1---> G channel
*2---> B channel
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_channel_extract_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u extractChannelNumber);

RppStatus
rppi_channel_extract_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u extractChannelNumber);

RppStatus
rppi_channel_extract_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u extractChannelNumber);

// ----------------------------------------
// Host channel_combine functions declaration 
// ----------------------------------------
/* Combines 3 greyscale images to produce a single RGB image. 
/*param srcPtr1 [in] srcPtr1 Input image1
*param srcPtr2 [in] srcPtr2 Input image2
*param srcPtr3 [in] srcPtr3 Input image3
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_channel_combine_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppPtr_t srcPtr3,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_channel_combine_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppPtr_t srcPtr3,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_channel_combine_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppPtr_t srcPtr3,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host look_up_table functions declaration 
// ----------------------------------------
/* This function uses each pixel in an image to index into a LUT and put the indexed LUT value into the output image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] lutPtr lutPtr contains the input look up table values
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_look_up_table_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp8u* lutPtr);

RppStatus
rppi_look_up_table_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp8u* lutPtr);

RppStatus
rppi_look_up_table_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp8u* lutPtr);

// ----------------------------------------
// GPU rgb_to_hsv functions declaration 
// ----------------------------------------
/* Converts RGB image to HSV image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rgb_to_hsv_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_rgb_to_hsv_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_rgb_to_hsv_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU hsv_to_rgb functions declaration 
// ----------------------------------------
/* Converts HSV image to RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hsv_to_rgb_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_hsv_to_rgb_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_hsv_to_rgb_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU hueRGB functions declaration 
// ----------------------------------------
/* Computes hue value and updates it in RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] hueShift hue shift for hue calculation
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hueRGB_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) ;

RppStatus
rppi_hueRGB_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) ;

RppStatus
rppi_hueRGB_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU hueHSV functions declaration 
// ----------------------------------------
/* Computes hue value and updates it in HSV image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] hueShift hue shift for hue calculation
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_hueHSV_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) ;

RppStatus
rppi_hueHSV_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) ;

RppStatus
rppi_hueHSV_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f hueShift, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU saturationRGB functions declaration 
// ----------------------------------------
/* Computes saturation value and updates it in RGB image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] saturationFactor saturationFactor for saturation calculation
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_saturationRGB_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) ;

RppStatus
rppi_saturationRGB_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) ;

RppStatus
rppi_saturationRGB_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU saturationHSV functions declaration 
// ----------------------------------------
/* Computes saturation value and updates it in HSV image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] saturationFactor saturationFactor for saturation calculation
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_saturationHSV_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) ;

RppStatus
rppi_saturationHSV_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) ;

RppStatus
rppi_saturationHSV_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f saturationFactor, RppHandle_t rppHandle) ;


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

// ----------------------------------------
// GPU channel_extract functions declaration 
// ----------------------------------------
/* Extract a single channel from given RGB image and stores it in destination grey scale image. 
/*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*param[in] extractChannelNumber extractChannelNumber The channel to be extracted and it could be
*0---> rppHandle R channel
 *1---> G channel
*2---> B channel
*param[in]  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_channel_extract_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u extractChannelNumber, RppHandle_t rppHandle) ;

RppStatus
rppi_channel_extract_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u extractChannelNumber, RppHandle_t rppHandle) ;

RppStatus
rppi_channel_extract_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32u extractChannelNumber, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU channel_combine functions declaration 
// ----------------------------------------
/* Combines 3 greyscale images to produce a single RGB image. 
/*param srcPtr1 [in] srcPtr1 Input image1
*param srcPtr2 [in] srcPtr2 Input image2
*param srcPtr3 [in] srcPtr3 Input image3
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr Output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_channel_combine_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppPtr_t srcPtr3,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_channel_combine_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppPtr_t srcPtr3,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_channel_combine_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppPtr_t srcPtr3,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU look_up_table functions declaration 
// ----------------------------------------
/* This function uses each pixel in an image to index into a LUT and put the indexed LUT value into the output image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] lutPtr lutPtr contains the input look up table values
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_look_up_table_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp8u* lutPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_look_up_table_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp8u* lutPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_look_up_table_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp8u* lutPtr, RppHandle_t rppHandle) ;

 
#ifdef __cplusplus
}
#endif
#endif
