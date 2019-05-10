#ifndef RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#define RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

// RGB2HSV host function declaration
RppStatus
rppi_rgb2hsv_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr);


RppStatus
rppi_brighten_1C8U_pln( RppHandle_t handle, RppPtr_t srcPtr, RppiSize srcSize,
                        RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta );

// brightness host function declaration  for single channel
RppStatus
rppi_brighten_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

// brightness host function declaration for three channel
RppStatus
rppi_brighten_3C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr, Rpp32f alpha, Rpp32s beta);

//blur host function declaration for single channel
RppStatus
rppi_blur3x3_1C8U_pln_host(RppPtr_t *srcPtr, RppiSize srcSize,
                            RppPtr_t *dstPtr);

//contrast function declaration for single channel
RppStatus
rppi_contrast_1C8U_pln_host(RppPtr_t srcPtr, RppiSize srcSize,
                            RppPtr_t dstPtr,Rpp32u new_min, Rpp32u new_max);

#ifdef __cplusplus
}
#endif

#endif /* RPPI_IMAGE_AUGUMENTATION_FUNCTIONS_H */
