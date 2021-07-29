#ifndef RPPI_ARITHMATIC_OPERATIONS_H
#define RPPI_ARITHMATIC_OPERATIONS_H

#include "rppdefs.h"
#include "rpp.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** add ********************/

// Adds two images

// *param[in] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_add_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_add_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** subtract ********************/

// Subtracts two images

// *param[in] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_subtract_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_subtract_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** multiply ********************/

// Multiplies two images

// *param[in] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_multiply_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_multiply_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** absolute_difference ********************/

// Computes absolute differnece of two images

// *param[in] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_absolute_difference_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_absolute_difference_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** phase ********************/

// Computes phase of two images

// *param[in] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_phase_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_phase_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** magnitude ********************/

// Computes magnitude of two images

// *param[in] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[out] dstPtr Output image
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_magnitude_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_magnitude_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, RppPtr_t dstPtr, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** accumulate ********************/

// Accumulates two images into the first image

// *param[in/out] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_accumulate_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** accumulate_weighted ********************/

// Alpha-blends two images into the first image

// *param[in/out] srcPtr1 Input image1
// *param[in] srcPtr2 Input image2
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] alpha Array of floats containing one alpha value for each image in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_accumulate_weighted_u8_pln1_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pln3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pln1_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pln3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_weighted_u8_pkd3_batchPD_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32f *alpha, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** accumulate_squared ********************/

// Squares pixel values of an image in-place

// *param[in/out] srcPtr Input image
// *param[in] srcSize Array containing an RppiSize for each image in the batch
// *param[in] maxSrcSize A single RppiSize which is the maxWidth and maxHeight for all images in the batch
// *param[in] nbatchSize Batch size or the number of images in the batch
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_accumulate_squared_u8_pln1_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pln3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pkd3_batchPD_gpu(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pln1_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pln3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);
RppStatus rppi_accumulate_squared_u8_pkd3_batchPD_host(RppPtr_t srcPtr, RppiSize *srcSize, RppiSize maxSrcSize, Rpp32u nbatchSize, rppHandle_t rppHandle);

/******************** tensor_add ********************/

// Adds corresponding elements of two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_add_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
RppStatus rppi_tensor_add_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);

/******************** tensor_subtract ********************/

// Subtracts corresponding elements of two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_subtract_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
RppStatus rppi_tensor_subtract_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);

/******************** tensor_multiply ********************/

// Multiplies corresponding elements of two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimension Number of dimensions in the tensor
// *param[in] tensorDimensionValues Array of length - "tensorDimension", containing size of each dimension in the tensor
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);
RppStatus rppi_tensor_multiply_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);

/******************** tensor_matrix_multiply ********************/

// Performs matrix multiplication on two tensors

// *param[in] srcPtr1 Input tensor1
// *param[in] srcPtr2 Input tensor2
// *param[out] dstPtr Output tensor
// *param[in] tensorDimensionValues1 Array containing dimensions of tensor1
// *param[in] tensorDimensionValues2 Array containing dimensions of tensor2
// *param[in] rppHandle OpenCL-handle/HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : No error, Succesful completion
// *retval RPP_ERROR : Error

RppStatus rppi_tensor_matrix_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, RppPtr_t tensorDimensionValues1, RppPtr_t tensorDimensionValues2, rppHandle_t rppHandle);
RppStatus rppi_tensor_matrix_multiply_u8_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, RppPtr_t tensorDimensionValues1, RppPtr_t tensorDimensionValues2, rppHandle_t rppHandle);






// TO BE MOVED

// RppStatus
// rppi_tensor_table_look_up_u8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, Rpp8u *look_up_table, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);

// RppStatus
// rppi_tensor_convert_bit_depth_u8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppConvertBitDepthMode convert_mode, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, rppHandle_t rppHandle);

RppStatus
rppi_tensor_transpose_u8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle);

RppStatus
rppi_tensor_transpose_f16_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle);

RppStatus
rppi_tensor_transpose_f32_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle);

RppStatus
rppi_tensor_transpose_i8_gpu(RppPtr_t srcPtr, RppPtr_t dstPtr, RppPtr_t in_tensor_dims, RppPtr_t perm, rppHandle_t rppHandle);



#ifdef __cplusplus
}
#endif
#endif