#ifndef RPP_HIP_HOST_DECLS_H
#define RPP_HIP_HOST_DECLS_H

#include "rpp.h"
#include "hip/rpp/handle.hpp"
#include "hip/rpp_hip_common.hpp"

// advanced_augmentations


// arithmetic_operations

RppStatus hip_exec_add_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_absolute_difference_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_subtract_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_accumulate_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_accumulate_weighted_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_accumulate_squared_batch(Rpp8u *srcPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_magnitude_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_multiply_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_phase_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_tensor_add(Rpp32u tensorDimension, Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u gdim1, Rpp32u gdim2, Rpp32u gdim3);
RppStatus hip_exec_tensor_subtract(Rpp32u tensorDimension, Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u gdim1, Rpp32u gdim2, Rpp32u gdim3);
RppStatus hip_exec_tensor_multiply(Rpp32u tensorDimension, Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u gdim1, Rpp32u gdim2, Rpp32u gdim3);
RppStatus hip_exec_tensor_matrix_multiply(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u a, Rpp32u b, Rpp32u c, Rpp32u d, Rpp32u gdim1, Rpp32u gdim2, Rpp32u gdim3);

// color_model_conversions


// computer_vision

RppStatus hip_exec_local_binary_pattern_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gaussian_image_pyramid_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gaussian_image_pyramid_pkd_batch(Rpp8u *srcPtr, Rpp8u *srcPtr1, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_gaussian_image_pyramid_pln_batch(Rpp8u *srcPtr, Rpp8u *srcPtr1, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_laplacian_image_pyramid_pkd_batch(Rpp8u *srcPtr1, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_laplacian_image_pyramid_pln_batch(Rpp8u *srcPtr1, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);


// filter_operations

RppStatus hip_exec_sobel_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_box_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_median_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_non_max_suppression_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_bilateral_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gaussian_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_custom_convolution_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *d_kernel, RppiSize kernelSize, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);


// fused_functions


// geometry_transforms


// image_augmentations

RppStatus hip_exec_brightness_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gamma_correction_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_contrast_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width, Rpp32u min, Rpp32u max);
RppStatus hip_exec_blend_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_pixelate_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_jitter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_noise_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_snow_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_exposure_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_rain_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_fog_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_random_shadow_packed(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u channel, Rpp32u column1, Rpp32u row1, Rpp32u column2, Rpp32u row2, Rpp32s i);
RppStatus hip_exec_random_shadow_planar(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u channel, Rpp32u column1, Rpp32u row1, Rpp32u column2, Rpp32u row2, Rpp32s i);

// logical_operations

RppStatus hip_exec_bitwise_AND_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_bitwise_NOT_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_exclusive_OR_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_inclusive_OR_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);

// morphological_transforms

RppStatus hip_exec_dilate_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_erode_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);

// statistical_operations

RppStatus hip_exec_thresholding_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_min_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_max_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);


#endif //RPP_HIP_HOST_DECLS_H