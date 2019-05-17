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
                            RppPtr_t dstPtr);

RppStatus
rppi_blur3x3_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr);

RppStatus
rppi_blur3x3_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr);

// Gpu function declarations

RppStatus
rppi_blur3x3_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_blur3x3_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);

RppStatus
rppi_blur3x3_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);



//------------------------- Image adjustments -------------------------


// --------------------
// Contrast
// --------------------

// Host function declarations

RppStatus
rppi_contrast_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min = 0, Rpp32u new_max =  225);

RppStatus
rppi_contrast_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min = 0, Rpp32u new_max =  225);

RppStatus
rppi_contrast_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min = 0, Rpp32u new_max =  225);

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
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta,
                            RppHandle_t handle );

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
rppi_hue_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift = 0);


// --------------------
// Saturation
// --------------------

// Host function declarations

RppStatus
rppi_saturation_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor = 1);


//------------------------- Geometric Transformations -------------------------


// --------------------
// Flip
// --------------------

// Host function declarations

RppStatus rppi_flip_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis);

RppStatus rppi_flip_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis);

RppStatus rppi_flip_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis);

// Gpu function declarations

RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

RppStatus
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

RppStatus
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);


// --------------------
// Rotate
// --------------------

// Host function declarations

RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          RppiSize sizeDst, Rpp32f angleRad = 0);


//------------------------- Color Space Conversions -------------------------


// --------------------
// RGB to HSV
// --------------------

/*NOTE: Parameters:
srcPtr is of type Rppu8 *
dstPtr is of type Rpp32f *
srcSize is the size of both source and destination images (Rpp32u height, Rpp32u width)
*/

// Host function declarations

RppStatus
rppi_rgb2hsv_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// Gpu function declarations

RppStatus
rppi_rgb2hsv_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);


// --------------------
// HSV to RGB
// --------------------

// Gpu function declarations

RppStatus
rppi_hsv2rgb_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);








#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */
