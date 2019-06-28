#ifndef RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#define RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif



//------------------------- Smoothening -------------------------


// --------------------
// Gaussian Blur
// --------------------

// Host function declarations

RppStatus
rppi_blur3x3_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize,
                          RppPtr_t dstPtr, Rpp32f stdDev);

RppStatus
rppi_blur3x3_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f stdDev);

RppStatus
rppi_blur3x3_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f stdDev);

// Gpu function declarations

RppStatus
rppi_blur3x3_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f stdDev, RppHandle_t rppHandle);

RppStatus
rppi_blur3x3_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f stdDev, RppHandle_t rppHandle);

RppStatus
rppi_blur3x3_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f stdDev, RppHandle_t rppHandle);



//------------------------- Image adjustments -------------------------


// --------------------
// Contrast
// --------------------

// Host function declarations

RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min , Rpp32u new_max);

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min , Rpp32u new_max);

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min , Rpp32u new_max);

// Gpu function declarations

RppStatus
rppi_contrast_u8_pln1_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32u min, Rpp32u max,
                        RppHandle_t rppHandle );

RppStatus
rppi_contrast_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax, RppHandle_t rppHandle);

RppStatus
rppi_contrast_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax, RppHandle_t rppHandle);


// --------------------
// Brightness
// --------------------

// Host function declarations

RppStatus
rppi_brightness_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

RppStatus
rppi_brightness_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

RppStatus
rppi_brightness_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

// Gpu function declarations

RppStatus
rppi_brightness_u8_pln1_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle );

RppStatus
rppi_brightness_u8_pln3_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle );

RppStatus
rppi_brightness_u8_pkd3_gpu( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle );


// --------------------
// Hue
// --------------------

// Host function declarations

RppStatus
rppi_hueRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift);

RppStatus
rppi_hueRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift);

RppStatus
rppi_hueHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift);

RppStatus
rppi_hueHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift);

// Device function declarations

RppStatus
rppi_hueRGB_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle);
RppStatus
rppi_hueHSV_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle);
RppStatus
rppi_hueRGB_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle);

RppStatus
rppi_hueHSV_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f hueShift,  RppHandle_t rppHandle);
// --------------------
// Saturation
// --------------------

// Host function declarations

RppStatus
rppi_saturationRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor );

RppStatus
rppi_saturationRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor );

RppStatus
rppi_saturationHSV_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor );

RppStatus
rppi_saturationHSV_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor );

// Device function declarations

RppStatus
rppi_saturationRGB_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle);
RppStatus
rppi_saturationHSV_u8_pln3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle);
RppStatus
rppi_saturationRGB_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle);
RppStatus
rppi_saturationHSV_u8_pkd3_gpu (RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        Rpp32f saturationFactor,  RppHandle_t rppHandle);


//------------------------- Color Space Conversions -------------------------


////////////////////////// rgbtohsv conversion//////////////////////
/*Parameters
srcPtr is of type Rppu8 *
dstPtr is of type Rpp32f *
srcSize is the size of both source and destination images (Rpp32u height, Rpp32u width)
*/
// Host function declarations

RppStatus
rppi_rgb2hsv_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_rgb2hsv_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// Gpu function declarations

RppStatus
rppi_rgb2hsv_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            RppHandle_t rppHandle);

RppStatus
rppi_rgb2hsv_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            RppHandle_t rppHandle);


// --------------------
// HSV to RGB
// --------------------

// Host function declarations

RppStatus
rppi_hsv2rgb_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_hsv2rgb_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// Gpu function declarations

RppStatus
rppi_hsv2rgb_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        RppHandle_t rppHandle);

RppStatus
rppi_hsv2rgb_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                        RppHandle_t rppHandle);

// --------------------
// Gamma correction
// --------------------

// Gpu function declarations
RppStatus
rppi_gamma_correction_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma, RppHandle_t rppHandle);
RppStatus
rppi_gamma_correction_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma, RppHandle_t rppHandle);
RppStatus
rppi_gamma_correction_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma, RppHandle_t rppHandle);


// Host function declarations
RppStatus
rppi_gamma_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma);
RppStatus
rppi_gamma_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma);
RppStatus
rppi_gamma_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                             Rpp32f gamma);

// ----------------------------------------
// Host blend functions  declaration
// ----------------------------------------


RppStatus
rppi_blend_u8_pln1_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, 
                        RppPtr_t dstPtr, Rpp32f alpha) ;

RppStatus
rppi_blend_u8_pln3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, 
                        RppPtr_t dstPtr, Rpp32f alpha) ;

RppStatus
rppi_blend_u8_pkd3_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize srcSize, 
                        RppPtr_t dstPtr, Rpp32f alpha) ;

// ----------------------------------------
// Host add noise functions  declaration
// ----------------------------------------

//Host function declaration
RppStatus
rppi_noiseAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                            RppiNoise noiseType, void * noiseParameter);
RppStatus
rppi_noiseAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                            RppiNoise noiseType, void * noiseParameter);
RppStatus
rppi_noiseAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                            RppiNoise noiseType, void * noiseParameter);

#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */
