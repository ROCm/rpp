#ifndef RPPI_GEOMETRIC_FUNCTIONS_H
#define RPPI_GEOMETRIC_FUNCTIONS_H


/**
 * \file rppi_geometry_functions.h
 * Image Geometry Transform Primitives.
 */
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
RppStatus
Rppi_Flip_8u_pkd1_host(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pln1(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pkd3(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pln3(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pkd1_host(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pln1_host(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pkd3_host(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);

RppStatus
Rppi_Flip_8u_pln3_host(const RppPtr_t  srcPtr, int rSrcStep,
                        RppPtr_t  dstPtr, int rDstStep,
                  RppiSize oROI, RppiAxis flip);


RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle);

RppStatus
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle);

RppStatus
rppi_rotate_u8_pkd1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f angleDeg, RppHandle_t rppHandle);

RppStatus
rppi_warp_affine_u8_pln1_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f *affine, RppHandle_t rppHandle);

RppStatus
rppi_warp_affine_u8_pln3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f *affine, RppHandle_t rppHandle);

RppStatus
rppi_warp_affine_u8_pkd3_gpu(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                         Rpp32f *affine, RppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif /* RPP_FILTERING_FUNCTIONS_H */