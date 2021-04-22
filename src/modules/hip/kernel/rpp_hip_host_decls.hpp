#ifndef RPP_HIP_HOST_DECLS_H
#define RPP_HIP_HOST_DECLS_H

#include "rpp.h"
#include "hip/rpp/handle.hpp"
#include "hip/rpp_hip_common.hpp"

// advanced_augmentations


// arithmetic_operations


// color_model_conversions


// computer_vision


// filter_operations


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


// morphological_transforms


// statistical_operations


#endif //RPP_HIP_HOST_DECLS_H