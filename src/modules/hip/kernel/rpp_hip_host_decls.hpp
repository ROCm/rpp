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

RppStatus hip_exec_color_temperature_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_vignette_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_channel_extract_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_channel_combine_batch(Rpp8u *srcPtr1, Rpp8u *srcPtr2, Rpp8u *srcPtr3, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_hueRGB_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_saturationRGB_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_look_up_table_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp8u *hipLutPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_tensor_look_up_table_batch(Rpp32u tensorDimension, Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp8u *hipLutPtr, rpp::Handle& handle, Rpp32u gdim1, Rpp32u gdim2, Rpp32u gdim3);

// computer_vision

RppStatus hip_exec_local_binary_pattern_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gaussian_image_pyramid_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gaussian_image_pyramid_pkd_batch(Rpp8u *srcPtr, Rpp8u *srcPtr1, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_gaussian_image_pyramid_pln_batch(Rpp8u *srcPtr, Rpp8u *srcPtr1, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_laplacian_image_pyramid_pkd_batch(Rpp8u *srcPtr1, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_laplacian_image_pyramid_pln_batch(Rpp8u *srcPtr1, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32f *kernelArray, Rpp32u max_height, Rpp32u max_width, Rpp32u batchIndex, Rpp32s i);
RppStatus hip_exec_ced_non_max_suppression(Rpp8u *srcPtr, Rpp8u *sobelX, Rpp8u *sobelY, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i);
RppStatus hip_exec_canny_edge(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i);
RppStatus hip_exec_canny_ced_pln3_to_pln1(Rpp8u *srcPtr, Rpp8u *gsin, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel);
RppStatus hip_exec_canny_ced_pkd3_to_pln1(Rpp8u *srcPtr, Rpp8u *gsin, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel);
RppStatus hip_exec_canny_ced_pln1_to_pkd3(Rpp8u *gsout, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel);
RppStatus hip_exec_canny_ced_pln1_to_pln3(Rpp8u *gsout, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel);
RppStatus hip_exec_harris_corner_detector_strength(Rpp8u *sobelX, Rpp8u *sobelY, Rpp32f *dstFloat, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i);
RppStatus hip_exec_harris_corner_detector_nonmax_supression(Rpp32f *input, Rpp32f *output, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32s i);
RppStatus hip_exec_harris_corner_detector_pkd(Rpp8u *input, Rpp32f *inputFloat, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel);
RppStatus hip_exec_harris_corner_detector_pln(Rpp8u *input, Rpp32f *inputFloat, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel);


// filter_operations

RppStatus hip_exec_sobel_pln(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32u sobelType);
RppStatus hip_exec_sobel_pkd(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, rpp::Handle& handle, Rpp32u channel, Rpp32u sobelType);
RppStatus hip_exec_sobel_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_box_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_median_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_non_max_suppression_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_bilateral_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_gaussian_pln(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, Rpp32f *kernelArray, rpp::Handle& handle, Rpp32u channel, Rpp32s i);
RppStatus hip_exec_gaussian_pkd(Rpp8u *srcPtr, Rpp8u *dstPtr, Rpp32u height, Rpp32u width, Rpp32f *kernelArray, rpp::Handle& handle, Rpp32u channel, Rpp32s i);
RppStatus hip_exec_gaussian_filter_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_custom_convolution_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *d_kernel, RppiSize kernelSize, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);


// fused_functions

RppStatus hip_exec_color_twist_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_color_twist_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_color_twist_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_color_twist_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_mirror_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_mirror_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_mirror_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_mirror_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_crop_mirror_normalize_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);


// geometry_transforms

RppStatus hip_exec_flip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_fisheye_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_lens_correction_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_scale_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_random_crop_letterbox_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32u padding, Rpp32u type, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_warp_perspective_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *perspective, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch_u8_fp16(Rpp8u *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch_u8_fp32(Rpp8u *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch_u8_int8(Rpp8u *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_resize_crop_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32u padding, Rpp32u type, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_rotate_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_rotate_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_rotate_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_rotate_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_warp_affine_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_warp_affine_batch_fp16(Rpp16f *srcPtr, Rpp16f *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_warp_affine_batch_fp32(Rpp32f *srcPtr, Rpp32f *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);
RppStatus hip_exec_warp_affine_batch_int8(Rpp8s *srcPtr, Rpp8s *dstPtr, rpp::Handle& handle, Rpp32f *affine, RPPTensorFunctionMetaData &tensor_info, Rpp32s in_plnpkdind, Rpp32s out_plnpkdind, Rpp32u max_height, Rpp32u max_width);


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