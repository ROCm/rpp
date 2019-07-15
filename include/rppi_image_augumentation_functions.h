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
rppi_blur_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize);

RppStatus
rppi_blur_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize);

RppStatus
rppi_blur_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev, Rpp32u kernelSize);

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

// --------------------
// Exposure
// --------------------

// Host function declarations

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

// --------------------
// Color Temperature
// --------------------

// Host function declarations

RppStatus
rppi_color_temperature_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue);

RppStatus
rppi_color_temperature_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp8s adjustmentValue);











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
rppi_rotate_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);


// --------------------
// Warp Affine
// --------------------

// Host function declarations

RppStatus
rppi_warp_affine_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                  RppPtr_t affine);

RppStatus
rppi_warp_affine_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine);

RppStatus
rppi_warp_affine_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine);

RppStatus
rppi_warp_affine_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine);


// --------------------
// Scale
// --------------------

// Host function declarations

RppStatus
rppi_scale_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f percentage);

RppStatus
rppi_scale_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage);

RppStatus
rppi_scale_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage);

RppStatus
rppi_scale_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage);


// --------------------
// Resize
// --------------------

// Host function declarations

RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize);

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize);

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize);



// --------------------
// Resize Crop
// --------------------

// Host function declarations

RppStatus
rppi_resizeCrop_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);

RppStatus
rppi_resizeCrop_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);

RppStatus
rppi_resizeCrop_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);



// --------------------
// Random Crop Letterbox
// --------------------

// Host function declarations

RppStatus
rppi_random_crop_letterbox_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, 
                                        Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);

RppStatus
rppi_random_crop_letterbox_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, 
                                        Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);

RppStatus
rppi_random_crop_letterbox_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize, 
                                        Rpp32u x1, Rpp32u y1, Rpp32u x2, Rpp32u y2);











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
// Gamma Correction
// --------------------

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



// --------------------
// RGB to HSL
// --------------------

// Host function declarations

RppStatus
rppi_rgb2hsl_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_rgb2hsl_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);



// --------------------
// HSL to RGB
// --------------------

// Host function declarations

RppStatus
rppi_hsl2rgb_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_hsl2rgb_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);












//------------------------- Other Augmentations -------------------------



// --------------------
// Pixelate
// --------------------

// Host function declarations

RppStatus
rppi_pixelate_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

RppStatus
rppi_pixelate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

RppStatus
rppi_pixelate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             Rpp32u kernelSize, unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);



// --------------------
// Jitter Add
// --------------------

// Host function declarations

RppStatus
rppi_jitterAdd_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

RppStatus
rppi_jitterAdd_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);

RppStatus
rppi_jitterAdd_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, 
                             unsigned int maxJitterX, unsigned int maxJitterY);



// --------------------
// Vignette Effect
// --------------------

// Host function declarations

RppStatus
rppi_vignette_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev);

RppStatus
rppi_vignette_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev);

RppStatus
rppi_vignette_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          Rpp32f stdDev);



// --------------------
// Fish Eye Effect
// --------------------

// Host function declarations

RppStatus
rppi_fish_eye_effect_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_fish_eye_effect_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

RppStatus
rppi_fish_eye_effect_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);



// --------------------
// Lens Correction
// --------------------

// Host function declarations

RppStatus
rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

RppStatus
rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);

RppStatus
rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, Rpp32f strength, Rpp32f zoom);



// --------------------
// Occlusion Add
// --------------------

// Host function declarations

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



// --------------------
// Snowy
// --------------------

// Host function declarations

RppStatus
rppi_snowyRGB_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength);

RppStatus
rppi_snowyRGB_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                         Rpp32f strength);



// --------------------
// Random Shadow
// --------------------

// Host function declarations

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



#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */
