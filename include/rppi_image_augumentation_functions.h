#ifndef RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#define RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

////////////////////////// rgbtohsv conversion//////////////////////
/*Parameters
srcPtr is of type Rpp8u *
dstPtr is of type Rpp32f *
srcSize is the size of both source and destination images (Rpp32u height, Rpp32u width)
*/
RppStatus
rppi_rgb2hsv_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

// RGB2HSV host function declaration
RppStatus
rppi_rgb2hsv_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);

//Hue host function declaration
RppStatus
rppi_hue_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       Rpp32f hueShift = 0);

//Saturation host function declaration
RppStatus
rppi_saturation_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                              Rpp32f saturationFactor = 1);



//-------------------------- Smoothening ---------------------------------------
RppStatus
rppi_blur3x3_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr);

RppStatus
rppi_blur3x3_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr);

RppStatus
rppi_blur3x3_1C8U_pln(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);
RppStatus
rppi_blur3x3_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);
RppStatus
rppi_blur3x3_3C8U_pkd(RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, RppHandle_t rppHandle);


//----------------------Image adjustments--------------------------------

//contrast host function declaration for single channel
RppStatus
rppi_contrast_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min = 0, Rpp32u new_max =  225);

RppStatus
rppi_contrast_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,RppPtr_t dstPtr,
                            Rpp32u new_min = 0, Rpp32u new_max =  225);

RppStatus
rppi_contrast_1C8U_pln( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32u min, Rpp32u max,
                        RppHandle_t rppHandle );

RppStatus
rppi_contrast_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                            Rpp32u newMin, Rpp32u newMax, RppHandle_t rppHandle);


// brightness host function declaration  for single channel
RppStatus
rppi_brighten_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta,
                            RppHandle_t handle );

// brightness host function declaration for three channel
RppStatus
rppi_brighten_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

RppStatus
rppi_brighten_1C8U_pln( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle );

RppStatus
rppi_brighten_3C8U_pln( RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr,
                        Rpp32f alpha, Rpp32s beta,
                        RppHandle_t rppHandle );



//----------------------Affine Transforms --------------------------------------

//Rotate host function declaration for single channel
RppStatus
rppi_rotate_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                          RppiSize sizeDst, Rpp32f angleRad = 0);

//Flip host function declaration for single channel input
RppStatus rppi_flip_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis);

//Flip host function declaration for single channel input
RppStatus rppi_flip_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                                   RppiAxis flipAxis);

RppStatus
rppi_flip_1C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

RppStatus
rppi_flip_3C8U_pln(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);

RppStatus
rppi_flip_u8_pkd3(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                    RppiAxis flipAxis, RppHandle_t rppHandle);


#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */
