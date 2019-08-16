#ifndef RPPI_ARITHMETIC_OPERATIONS
#define RPPI_ARITHMETIC_OPERATIONS
 
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif


// ----------------------------------------
// Host absolute_difference functions declaration 
// ----------------------------------------
/* Computes the absolute difference between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_absolute_difference_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_absolute_difference_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_absolute_difference_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host accumulate_weighted functions declaration 
// ----------------------------------------
/* Accumulates a weighted value from  input images and stores it in the first input image.
*param[in/out] srcPtr1 input image where the accumulated value will be stored
*param[in] srcPtr2 input image
*param[in] srcSize dimensions of the images
*param[in] alpha weight float value which should range between 0 - 1
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_accumulate_weighted_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha);

RppStatus
rppi_accumulate_weighted_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha);

RppStatus
rppi_accumulate_weighted_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha);

// ----------------------------------------
// Host accumulate functions declaration 
// ----------------------------------------
/* Computes the accumulation between two input images and stores it in the first input image.
*param[in/out] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_accumulate_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize);

RppStatus
rppi_accumulate_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize);

RppStatus
rppi_accumulate_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize);

// ----------------------------------------
// Host add functions declaration 
// ----------------------------------------
/* Computes the addition between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_add_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_add_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_add_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host subtract functions declaration 
// ----------------------------------------
/* Computes the subtraction between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_subtract_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_subtract_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_subtract_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);


// ----------------------------------------
// Host magnitude functions declaration 
// ----------------------------------------
/* Implements the Gradient Magnitude Computation on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_magnitude_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_magnitude_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_magnitude_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host multiply functions declaration 
// ----------------------------------------
/* Computes element wise multiplication between two images and stores it in the destination.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_multiply_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_multiply_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_multiply_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

// ----------------------------------------
// Host accumulate_squared functions declaration 
// ----------------------------------------
/* Computes the squared accumulation of the image and stores it in the same.
*param srcPtr [in/out] input image
*param[in] srcSize dimensions of the images
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_accumulate_squared_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize);

RppStatus
rppi_accumulate_squared_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize);

RppStatus
rppi_accumulate_squared_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize);

// ----------------------------------------
// Host mean_stddev functions declaration 
// ----------------------------------------
/* Computes the mean pixel value and the standard deviation of the pixels in the input image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the images
*param[out] mean mean mean of the pixel values in the input image
*param[out] stdDev stddev standard deviation of the pixels values in the input image 
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_mean_stddev_u8_pln1_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32f* mean,Rpp32f* stdDev);

RppStatus
rppi_mean_stddev_u8_pln3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32f* mean,Rpp32f* stdDev);

RppStatus
rppi_mean_stddev_u8_pkd3_host(RppPtr_t srcPtr,RppiSize srcSize,Rpp32f* mean,Rpp32f* stdDev);

// ----------------------------------------
// Host phase functions declaration 
// ----------------------------------------
/* Implements the Gradient Phase Computation phase on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_phase_u8_pln1_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_phase_u8_pln3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);

RppStatus
rppi_phase_u8_pkd3_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr);


// ----------------------------------------
// Host tensor functions declaration 
// ----------------------------------------

RppStatus
rppi_tensor_add_u8_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues);

RppStatus
rppi_tensor_subtract_u8_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues);

RppStatus
rppi_tensor_multiply_u8_host(RppPtr_t srcPtr1,RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues);


// ----------------------------------------
// GPU absolute_difference functions declaration 
// ----------------------------------------
/* Computes the absolute difference between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_absolute_difference_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_absolute_difference_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_absolute_difference_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;


// ----------------------------------------
// GPU accumulate_weighted functions declaration 
// ----------------------------------------
/* Accumulates a weighted value from  input images and stores it in the first input image.
*param[in/out] srcPtr1 input image where the accumulated value will be stored
*param[in] srcPtr2 input image
*param[in] srcSize dimensions of the images
*param[in] alpha weight float value which should range between 0 - 1
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_accumulate_weighted_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_weighted_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_weighted_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,Rpp64f alpha, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU accumulate functions declaration 
// ----------------------------------------
/* Computes the accumulation between two input images and stores it in the first input image.
*param[in/out] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_accumulate_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU add functions declaration 
// ----------------------------------------
/* Computes the addition between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_add_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_add_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_add_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU subtract functions declaration 
// ----------------------------------------
/* Computes the subtraction between two images.
*param[in] srcPtr1 input image1
*param[in] srcPtr2 input image2 
*param[in] srcSize dimensions of the images
*param[out] dstPtr output image
*param[in] rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_subtract_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_subtract_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_subtract_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU magnitude functions declaration 
// ----------------------------------------
/* Implements the Gradient Magnitude Computation on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_magnitude_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_magnitude_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_magnitude_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU multiply functions declaration 
// ----------------------------------------
/* Computes element wise multiplication between two images and stores it in the destination.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_multiply_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_multiply_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_multiply_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU accumulate_squared functions declaration 
// ----------------------------------------
/* Computes the squared accumulation of the image and stores it in the same.
*param srcPtr [in/out] input image
*param[in] srcSize dimensions of the images
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_accumulate_squared_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_squared_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize, RppHandle_t rppHandle) ;

RppStatus
rppi_accumulate_squared_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU mean_stddev functions declaration 
// ----------------------------------------
/* Computes the mean pixel value and the standard deviation of the pixels in the input image.
*param srcPtr [in] input image
*param[in] srcSize dimensions of the images
*param[out] mean mean mean of the pixel values in the input image
*param[out] stdDev stddev standard deviation of the pixels values in the input image 
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_mean_stddev_u8_pln1_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32f* mean,Rpp32f* stdDev, RppHandle_t rppHandle) ;

RppStatus
rppi_mean_stddev_u8_pln3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32f* mean,Rpp32f* stdDev, RppHandle_t rppHandle) ;

RppStatus
rppi_mean_stddev_u8_pkd3_gpu(RppPtr_t srcPtr,RppiSize srcSize,Rpp32f* mean,Rpp32f* stdDev, RppHandle_t rppHandle) ;

// ----------------------------------------
// GPU phase functions declaration 
// ----------------------------------------
/* Implements the Gradient Phase Computation phase on input images and stores the result in destination image.
*param srcPtr1 [in] srcPtr1 input image1
*param[in] srcPtr2 srcPtr2 input image2 
*param[in] srcSize srcSize dimensions of the images
*param[out] dstPtr dstPtr output image
*param[in] rppHandle  rppHandle OpenCL handle
*returns a  RppStatus enumeration. 
*retval RPP_SUCCESS : No error succesful completion
*retval RPP_ERROR : Error 
*/

RppStatus
rppi_phase_u8_pln1_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_phase_u8_pln3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;

RppStatus
rppi_phase_u8_pkd3_gpu(RppPtr_t srcPtr1,RppPtr_t srcPtr2,RppiSize srcSize,RppPtr_t dstPtr, RppHandle_t rppHandle) ;


// ----------------------------------------
// GPU Tensor functions declaration 
// ----------------------------------------

RppStatus
rppi_tensor_add_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, RppHandle_t rppHandle) ;

RppStatus
rppi_tensor_subtract_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, RppHandle_t rppHandle) ;

RppStatus
rppi_tensor_multiply_u8_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RppPtr_t dstPtr, Rpp32u tensorDimension, RppPtr_t tensorDimensionValues, RppHandle_t rppHandle) ;


#ifdef __cplusplus
}
#endif
#endif
