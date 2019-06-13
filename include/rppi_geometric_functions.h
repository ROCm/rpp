#ifndef RPPI_GEOMETRIC_FUNCTIONS_H
#define RPPI_GEOMETRIC_FUNCTIONS_H


/**
 * \file rppi_geometry_functions.h
 * Image Geometry Transform Primitives.
 */

#include "rppdefs.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * 1 channel 8-bit unsigned image flip function.
 *
 * \param srcPtr \ref source_image_pointer.
 * \param rSrcStep \ref source_image_line_step.
 * \param dstPtr \ref destination_image_pointer.
 * \param rDstStep \ref destination_image_line_step.
 * \param oROI \ref roi_specification.
 * \param flip Specifies the axis about which the image is to be mirrored.
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
 */



// --------------------
// Flip
// --------------------

// Host function declarations

RppStatus 
rppi_flip_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
                       RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr,
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
// Warp Affine
// --------------------

// Host function declarations

RppStatus
rppi_warp_affine_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                  RppPtr_t affine);
RppStatus
rppi_warp_affine_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine);
RppStatus
rppi_warp_affine_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              RppPtr_t affine);


// --------------------
// Rotate
// --------------------

// Host function declarations

RppStatus
rppi_rotate_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f angleDeg);
RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);
RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg);

// --------------------
// Resize
// --------------------

// Host function declarations

RppStatus
rppi_resize_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                             Rpp32f percentage);
RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage);
RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage);
RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f percentage);

#ifdef __cplusplus
}
#endif
#endif /* RPP_FILTERING_FUNCTIONS_H */
