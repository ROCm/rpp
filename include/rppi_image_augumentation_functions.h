#ifndef RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#define RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

//----------------------- Basic enhancements -----------------------------------

RppStatus
rppi_brighten_1C8U_pln( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta,
                        RppHandle_t handle  );

// brightness host function declaration  for single channel
RppStatus
rppi_brighten_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta,
                            RppHandle_t handle );

// brightness host function declaration for three channel
RppStatus
rppi_brighten_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

//-------------------------- Smoothening ---------------------------------------
RppStatus
rppi_blur3x3_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr);
RppStatus
rppi_blur3x3_1C8U_pln(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);


RppStatus
rppi_blur3x3_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);

//------------------------- Colors space HSV ----------------------------------

// RGB2HSV host function declaration
RppStatus
rppi_rgb2hsv_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// HSV2RGB host function declaration
RppStatus
rppi_hsv2rgb_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

//contrast host function declaration for single channel
RppStatus
rppi_contrast_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min = 0, Rpp32u new_max =  225);

//Hue host function declaration
RppStatus
rppi_hue_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift = 0);

//Saturation host function declaration
RppStatus
rppi_saturation_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor = 1);


//----------------------Affine Transforms --------------------------------------

//Rotate host function declaration for single channel
RppStatus
rppi_rotate_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          RppiSize sizeDst, Rpp32f angleRad = 0);




#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */
