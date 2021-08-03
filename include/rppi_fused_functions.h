#ifndef RPPI_FUSED_FUNCTIONS_H
#define RPPI_FUSED_FUNCTIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C"
{
#endif // cpusplus

/******************** color_twist ********************/

// Performs a fused color twist augmentation for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image batch
// *param[in] alpha Array containing an Rpp32f alpha for each image in the batch (0 <= alpha <= 20 for brightness modification)
// *param[in] beta Array containing an Rpp32f beta for each image in the batch (0 <= beta <= 255 for contrast modification)
// *param[in] hueShift Array containing an Rpp32f hue shift in degrees for each image in the batch (0 <= hueShift <= 359 for hue modification)
// *param[in] saturationFactor Array containing an Rpp32f saturation factor for each image in the batch (saturationFactor >= 0 for saturation modification)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_color_twist_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_color_twist_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32f *alpha, Rpp32f *beta, Rpp32f *hueShift, Rpp32f *saturationFactor, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** crop ********************/

// Performs crop operations for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the input batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the output batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] crop_pos_x Array containing an Rpp32u crop start x-position for each image in the batch (0 <= crop_pos_x[n] < srcSize[n].width)
// *param[in] crop_pos_y Array containing an Rpp32u crop start y-position for each image in the batch (0 <= crop_pos_y[n] < srcSize[n].height)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_crop_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** crop_mirror_normalize ********************/

// Performs fused crop mirror normalize operations for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the input batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the output batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] crop_pos_x Array containing an Rpp32u crop start x-position for each image in the batch (0 <= crop_pos_x[n] < srcSize[n].width)
// *param[in] crop_pos_y Array containing an Rpp32u crop start y-position for each image in the batch (0 <= crop_pos_y[n] < srcSize[n].height)
// *param[in] mean Array containing an Rpp32f mean for each image in the batch (mean[n] >= 0)
// *param[in] stdDev Array containing an Rpp32f standard deviation for each image in the batch (stdDev[n] >= 0)
// *param[in] mirrorFlag Array containing an Rpp32u mirror flag to set mirroring on/off for each image in the batch (mirrorFlag[n] = 0/1)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_crop_mirror_normalize_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *std_dev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_crop_mirror_normalize_u8_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *crop_pos_x, Rpp32u *crop_pos_y, Rpp32f *mean, Rpp32f *stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** resize_crop_mirror ********************/

// Performs fused resize crop mirror operations for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the input batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the output batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] xRoiBegin Array containing an Rpp32u crop start x-position for each image in the batch (0 <= xRoiBegin[n] < srcSize[n].width)
// *param[in] xRoiEnd Array containing an Rpp32u crop end x-position for each image in the batch (0 <= xRoiEnd[n] < srcSize[n].width)
// *param[in] yRoiBegin Array containing an Rpp32u crop start y-position for each image in the batch (0 <= yRoiBegin[n] < srcSize[n].height)
// *param[in] yRoiEnd Array containing an Rpp32u crop end y-position for each image in the batch (0 <= yRoiEnd[n] < srcSize[n].height)
// *param[in] mirrorFlag Array containing an Rpp32u mirror flag to set mirroring on/off for each image in the batch (mirrorFlag[n] = 0/1)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_resize_crop_mirror_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f16_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f16_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f16_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f32_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f32_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f32_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_i8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_i8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_i8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle,Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f16_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f16_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f16_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f32_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f32_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_f32_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_i8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_i8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_crop_mirror_i8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32u *xRoiBegin, Rpp32u *xRoiEnd, Rpp32u *yRoiBegin, Rpp32u *yRoiEnd, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** resize_mirror_normalize ********************/

// Performs fused resize mirror normalize operations for a batch of images

// *param[in] srcPtr Input image batch
// *param[in] srcSize Array containing an RppiSize for each image in the input batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the input batch
// *param[out] dstPtr Output image batch
// *param[in] dstSize Array containing an RppiSize for each image in the output batch
// *param[in] maxDstSize A single RppiSize which is the maxWidth and maxHeight for all images in the output batch
// *param[in] batch_mean Array containing an Rpp32f mean for each image in the batch (batch_mean[n] >= 0)
// *param[in] batch_stdDev Array containing an Rpp32f standard deviation for each image in the batch (batch_stdDev[n] >= 0)
// *param[in] mirrorFlag Array containing an Rpp32u mirror flag to set mirroring on/off for each image in the batch (mirrorFlag[n] = 0/1)
// *param[in] outputFormatToggle An Rpp32u flag to set layout toggling on/off for each image in the batch (outputFormatToggle = 0/1)
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_resize_mirror_normalize_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_mirror_normalize_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *mirrorFlag,Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_resize_mirror_normalize_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, RppiSize *dstSize, RppiSize maxDstSize, Rpp32f *batch_mean, Rpp32f *batch_stdDev, Rpp32u *mirrorFlag, Rpp32u outputFormatToggle, Rpp32u nbatchSize, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif

#endif // RPPI_FUSED_FUNCTIONS_H
