#ifndef RPPI_GEOMETRY_TRANSFORMS
#define RPPI_GEOMETRY_TRANSFORMS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host flip functions declaration 
// ----------------------------------------
/* Flips the image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] flipAxis flip axis and value should range beetween 0 and 2
 0 ---> horizontal flip
 1 ---> vertical flip
 2 ---> horizontal + vertical flip
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_flip_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis);

RppStatus
rppi_flip_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis);

// ----------------------------------------
// Host resize functions declaration 
// ----------------------------------------
/* Resizes the input image to the destination dimension.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_resize_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize);

RppStatus
rppi_resize_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize);

RppStatus
rppi_resize_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize);

// ----------------------------------------
// Host resize_crop functions declaration 
// ----------------------------------------
/* Crops the image to the roi area and resizes to the destination size
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] dstSize dimensions of the output images
*param[in] x1 x1 value of roi
*param[in] y1 y1 value of roi
*param[in] x2 x2 value of roi
*param[in] y2 y2 value of roi
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_resize_crop_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_resize_crop_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

RppStatus
rppi_resize_crop_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2);

// ----------------------------------------
// Host rotate functions declaration 
// ----------------------------------------
/* Rotates the input image according to the angle specified
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image where rotated image is stored
*param[in] dstSize dimensions of the output images
*param[in] angleDeg angle for rotation
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rotate_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg);

RppStatus
rppi_rotate_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg);

// ----------------------------------------
// Host warp_affine functions declaration 
// ----------------------------------------
/* Rotates translates and sheers the input image according to the affine values.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[in] dstPtr output image
*param[in] affine affine transformation matrix
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_warp_affine_output_size_host(RppiSize srcSize, RppiSize *dstSizePtr,
                                  Rpp32f* affine);

RppStatus
rppi_warp_affine_u8_pln1_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              Rpp32f* affine);

RppStatus
rppi_warp_affine_u8_pln3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              Rpp32f* affine);

RppStatus
rppi_warp_affine_u8_pkd3_host(RppPtr_t srcPtr, RppiSize srcSize, RppPtr_t dstPtr, RppiSize dstSize,
                              Rpp32f* affine);




// ----------------------------------------
// Host fisheye functions declaration 
// ----------------------------------------
/* Add fish eye effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_fisheye_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_fisheye_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_fisheye_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host lens_correction functions declaration 
// ----------------------------------------
/* Does lens correction in the lens distorted images.
*param srcPtr [in/out] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] strength strength strength of lens correction needed which should be greater than 0
*param[in] zoom zoom extent to which zoom-out is needed which should be greater than 1
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_lens_correction_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f strength,Rpp32f zoom);

RppStatus
rppi_lens_correction_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f strength,Rpp32f zoom);

RppStatus
rppi_lens_correction_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f strength,Rpp32f zoom);


// ----------------------------------------
// GPU flip functions declaration 
// ----------------------------------------
/* Flips the image.
param[in] srcPtr input image
*param[in] srcSize dimensions of the image
*param[out] dstPtr output image
*param[in] flipAxis flip axis and value should range beetween 0 and 2
 0 ---> horizontal flip
 1 ---> vertical flip
 2 ---> horizontal + vertical flip
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_flip_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) ;

RppStatus
rppi_flip_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) ;

RppStatus
rppi_flip_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiAxis flipAxis, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU resize functions declaration 
// ----------------------------------------
/* Resizes the input image to the destination dimension.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image where resized image is stored
*param[in] dstSize dimensions of the output images
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_resize_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU resize_crop functions declaration 
// ----------------------------------------
/* Crops the image to the roi area and resizes to the destination size
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image where resized image is stored
*param[in] dstSize dimensions of the output images
*param[in] x1 x1 value of roi
*param[in] y1 y1 value of roi
*param[in] x2 x2 value of roi
*param[in] y2 y2 value of roi
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_resize_crop_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_crop_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

RppStatus
rppi_resize_crop_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32u x1,Rpp32u y1,Rpp32u x2,Rpp32u y2, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU rotate functions declaration 
// ----------------------------------------
/* Rotates the input image according to the angle specified
*param[in] srcPtr input image
*param[in] srcSize dimensions of the input images
*param[out] dstPtr output image where rotated image is stored
*param[in] dstSize dimensions of the output images
*param[in] angleDeg angle for rotation
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_rotate_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) ;

RppStatus
rppi_rotate_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) ;

RppStatus
rppi_rotate_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,RppiSize dstSize,Rpp32f angleDeg, RppHandle_t rppHandle) ;


// ----------------------------------------
// GPU fisheye functions declaration 
// ----------------------------------------
/* Add fish eye effect in the entire image.
*param srcPtr [in] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_fisheye_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_fisheye_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_fisheye_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU lens_correction functions declaration 
// ----------------------------------------
/* Does lens correction in the lens distorted images.
*param srcPtr [in/out] srcPtr input image
*param[in] srcSize  srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] strength strength strength of lens correction needed which should be greater than 0
*param[in] zoom zoom extent to which zoom-out is needed which should be greater than 1
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_lens_correction_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f strength,Rpp32f zoom, RppHandle_t rppHandle) ;

RppStatus
rppi_lens_correction_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f strength,Rpp32f zoom, RppHandle_t rppHandle) ;

RppStatus
rppi_lens_correction_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,RppPtr_t dstPtr,Rpp32f strength,Rpp32f zoom, RppHandle_t rppHandle) ;

 
// ----------------------------------------
// GPU warp_affine functions declaration 
// ----------------------------------------
/* Rotates translates and sheers the input image according to the affine values.
*param[in] srcPtr input image
*param[in] srcSize dimensions of the images
*param[in] dstPtr output image
*param[in] affine affine transformation matrix
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

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
#endif
