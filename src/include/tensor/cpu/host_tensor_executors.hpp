/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef HOST_TENSOR_EXECUTORS_HPP
#define HOST_TENSOR_EXECUTORS_HPP

#include "rppdefs.h"
#include "rpp_cpu_common.hpp"

/**************************************** ARITHMETIC OPERATIONS ****************************************/

// -------------------- add_scalar --------------------

RppStatus add_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptGenericDescPtr srcGenericDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32f *addTensor,
                                         RpptROI3DPtr roiGenericPtrSrc,
                                         RpptRoi3DType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

// -------------------- fused_multiply_add_scalar --------------------

RppStatus fused_multiply_add_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                        RpptGenericDescPtr srcGenericDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptGenericDescPtr dstGenericDescPtr,
                                                        Rpp32f *mulTensor,
                                                        Rpp32f *addTensor,
                                                        RpptROI3DPtr roiGenericPtrSrc,
                                                        RpptRoi3DType roiType,
                                                        RppLayoutParams layoutParams,
                                                        rpp::Handle& handle);

// -------------------- log_generic --------------------

RppStatus log_generic_host_tensor(Rpp8u *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

RppStatus log_generic_host_tensor(Rpp8s *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

RppStatus log_generic_host_tensor(Rpp32f *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

RppStatus log_generic_host_tensor(Rpp16f *srcPtr,
                                  RpptGenericDescPtr srcGenericDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptGenericDescPtr dstGenericDescPtr,
                                  Rpp32u *roiTensor,
                                  rpp::Handle& handle);

// -------------------- magnitude --------------------

RppStatus magnitude_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                      Rpp8u *srcPtr2,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus magnitude_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                        Rpp32f *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

RppStatus magnitude_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                        Rpp16f *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

RppStatus magnitude_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                      Rpp8s *srcPtr2,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

// -------------------- multiply_scalar --------------------

RppStatus multiply_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptGenericDescPtr srcGenericDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptGenericDescPtr dstGenericDescPtr,
                                              Rpp32f *mulTensor,
                                              RpptROI3DPtr roiGenericPtrSrc,
                                              RpptRoi3DType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle);

// -------------------- subtract_scalar --------------------

RppStatus subtract_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptGenericDescPtr srcGenericDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptGenericDescPtr dstGenericDescPtr,
                                              Rpp32f *subtractTensor,
                                              RpptROI3DPtr roiGenericPtrSrc,
                                              RpptRoi3DType roiType,
                                              RppLayoutParams layoutParams,
                                              rpp::Handle& handle);

/**************************************** AUDIO AUGMENTATIONS ****************************************/

#ifdef AUDIO_SUPPORT

// -------------------- down_mixing --------------------

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcDimsTensor,
                                  bool normalizeWeights,
                                  rpp::Handle& handle);

// -------------------- mel_filter_bank --------------------

RppStatus mel_filter_bank_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      Rpp32f maxFreqVal,    // check unused
                                      Rpp32f minFreqVal,
                                      RpptMelScaleFormula melFormula,
                                      Rpp32s numFilter,
                                      Rpp32f sampleRate,
                                      bool normalize,
                                      rpp::Handle& handle);

// -------------------- non_silent_region_detection --------------------

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcLengthTensor,
                                                  Rpp32s *detectedIndexTensor,
                                                  Rpp32s *detectionLengthTensor,
                                                  Rpp32f cutOffDB,
                                                  Rpp32s windowLength,
                                                  Rpp32f referencePower,
                                                  Rpp32s resetInterval,
                                                  rpp::Handle& handle);

// -------------------- pre_emphasis_filter --------------------

RppStatus pre_emphasis_filter_host_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s *srcLengthTensor,
                                          Rpp32f *coeffTensor,
                                          Rpp32u borderType,
                                          rpp::Handle& handle);

// -------------------- resample --------------------

RppStatus resample_host_tensor(Rpp32f *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *inRateTensor,
                               Rpp32f *outRateTensor,
                               Rpp32s *srcDimsTensor,
                               RpptResamplingWindow &window,
                               rpp::Handle& handle);

// -------------------- spectrogram --------------------

RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  bool centerWindows,
                                  bool reflectPadding,
                                  Rpp32f *windowFunction,
                                  Rpp32s nfft,
                                  Rpp32s power,
                                  Rpp32s windowLength,
                                  Rpp32s windowStep,
                                  rpp::Handle& handle);

// -------------------- to_decibels --------------------

RppStatus to_decibels_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptImagePatchPtr srcDims,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude,
                                  rpp::Handle& handle);

#endif // AUDIO_SUPPORT

/**************************************** BITWISE OPERATIONS ****************************************/

// -------------------- bitwise_and --------------------

RppStatus bitwise_and_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                        Rpp8u *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& handle);

// -------------------- bitwise_not --------------------

RppStatus bitwise_not_u8_u8_host_tensor(Rpp8u *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& Handle);

// -------------------- bitwise_or --------------------

RppStatus bitwise_or_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                       Rpp8u *srcPtr2,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& Handle);

// -------------------- bitwise_xor --------------------

RppStatus bitwise_xor_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                        Rpp8u *srcPtr2,
                                        RpptDescPtr srcDescPtr,
                                        Rpp8u *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams layoutParams,
                                        rpp::Handle& Handle);

/**************************************** EFFECTS AUGMENTATIONS ****************************************/

// -------------------- gridmask --------------------

RppStatus gridmask_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u tileWidth,
                                     Rpp32f gridRatio,
                                     Rpp32f gridAngle,
                                     RpptUintVector2D translateVector,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus gridmask_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u tileWidth,
                                       Rpp32f gridRatio,
                                       Rpp32f gridAngle,
                                       RpptUintVector2D translateVector,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus gridmask_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u tileWidth,
                                       Rpp32f gridRatio,
                                       Rpp32f gridAngle,
                                       RpptUintVector2D translateVector,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus gridmask_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u tileWidth,
                                     Rpp32f gridRatio,
                                     Rpp32f gridAngle,
                                     RpptUintVector2D translateVector,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

// -------------------- spatter --------------------

RppStatus spatter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptRGB spatterColor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus spatter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptRGB spatterColor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus spatter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp16f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptRGB spatterColor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus spatter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8s *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptRGB spatterColor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

// -------------------- noise_salt_and_pepper --------------------

RppStatus salt_and_pepper_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *noiseProbabilityTensor,
                                                  Rpp32f *saltProbabilityTensor,
                                                  Rpp32f *saltValueTensor,
                                                  Rpp32f *pepperValueTensor,
                                                  RpptXorwowState *xorwowInitialStatePtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

RppStatus salt_and_pepper_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *noiseProbabilityTensor,
                                                    Rpp32f *saltProbabilityTensor,
                                                    Rpp32f *saltValueTensor,
                                                    Rpp32f *pepperValueTensor,
                                                    RpptXorwowState *xorwowInitialStatePtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus salt_and_pepper_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *noiseProbabilityTensor,
                                                    Rpp32f *saltProbabilityTensor,
                                                    Rpp32f *saltValueTensor,
                                                    Rpp32f *pepperValueTensor,
                                                    RpptXorwowState *xorwowInitialStatePtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus salt_and_pepper_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *noiseProbabilityTensor,
                                                  Rpp32f *saltProbabilityTensor,
                                                  Rpp32f *saltValueTensor,
                                                  Rpp32f *pepperValueTensor,
                                                  RpptXorwowState *xorwowInitialStatePtr,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

// -------------------- noise_shot --------------------

RppStatus shot_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *shotNoiseFactorTensor,
                                       RpptXorwowState *xorwowInitialStatePtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus shot_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *shotNoiseFactorTensor,
                                         RpptXorwowState *xorwowInitialStatePtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus shot_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32f *shotNoiseFactorTensor,
                                         RpptXorwowState *xorwowInitialStatePtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus shot_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *shotNoiseFactorTensor,
                                       RpptXorwowState *xorwowInitialStatePtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

// -------------------- noise_gaussian --------------------

RppStatus gaussian_noise_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus gaussian_noise_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus gaussian_noise_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *meanTensor,
                                             Rpp32f *stdDevTensor,
                                             RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus gaussian_noise_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *meanTensor,
                                           Rpp32f *stdDevTensor,
                                           RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus gaussian_noise_voxel_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                 RpptGenericDescPtr srcGenericDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptGenericDescPtr dstGenericDescPtr,
                                                 Rpp32f *meanTensor,
                                                 Rpp32f *stdDevTensor,
                                                 RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                 RpptROI3DPtr roiGenericPtrSrc,
                                                 RpptRoi3DType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus gaussian_noise_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptGenericDescPtr srcGenericDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptGenericDescPtr dstGenericDescPtr,
                                                   Rpp32f *meanTensor,
                                                   Rpp32f *stdDevTensor,
                                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                   RpptROI3DPtr roiGenericPtrSrc,
                                                   RpptRoi3DType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle);

// -------------------- non_linear_blend --------------------

RppStatus non_linear_blend_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                             Rpp8u *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8u *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *stdDevTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle &handle);

RppStatus non_linear_blend_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                               Rpp32f *srcPtr2,
                                               RpptDescPtr srcDescPtr,
                                               Rpp32f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *stdDevTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle &handle);

RppStatus non_linear_blend_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                             Rpp8s *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp8s *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *stdDevTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle &handle);

RppStatus non_linear_blend_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                               Rpp16f *srcPtr2,
                                               RpptDescPtr srcDescPtr,
                                               Rpp16f *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               Rpp32f *stdDevTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle &handle);

// -------------------- jitter --------------------

RppStatus jitter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *kernelSizeTensor,
                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus jitter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *kernelSizeTensor,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus jitter_f16_f16_host_tensor(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp16f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u *kernelSizeTensor,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus jitter_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *kernelSizeTensor,
                                   RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

// -------------------- glitch --------------------

RppStatus glitch_u8_u8_host_tensor(Rpp8u *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8u *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus glitch_f32_f32_host_tensor(Rpp32f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptChannelOffsets *rgbOffsets,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus glitch_f16_f16_host_tensor(Rpp16f *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp16f *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptChannelOffsets *rgbOffsets,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus glitch_i8_i8_host_tensor(Rpp8s *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp8s *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptChannelOffsets *rgbOffsets,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

// -------------------- water --------------------

RppStatus water_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus water_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *amplitudeXTensor,
                                    Rpp32f *amplitudeYTensor,
                                    Rpp32f *frequencyXTensor,
                                    Rpp32f *frequencyYTensor,
                                    Rpp32f *phaseXTensor,
                                    Rpp32f *phaseYTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus water_f16_f16_host_tensor(Rpp16f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *amplitudeXTensor,
                                    Rpp32f *amplitudeYTensor,
                                    Rpp32f *frequencyXTensor,
                                    Rpp32f *frequencyYTensor,
                                    Rpp32f *phaseXTensor,
                                    Rpp32f *phaseYTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus water_i8_i8_host_tensor(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- ricap --------------------

RppStatus ricap_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *permutedIndices,
                                  RpptROIPtr roiPtrInputCropRegion,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus ricap_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u *permutedIndices,
                                    RpptROIPtr roiPtrInputCropRegion,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus ricap_f16_f16_host_tensor(Rpp16f *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32u *permutedIndices,
                                    RpptROIPtr roiPtrInputCropRegion,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus ricap_i8_i8_host_tensor(Rpp8s *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *permutedIndices,
                                  RpptROIPtr roiPtrInputCropRegion,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- vignette --------------------

RppStatus vignette_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *vignetteIntensityTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus vignette_f32_f32_host_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *vignetteIntensityTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

RppStatus vignette_i8_i8_host_tensor(Rpp8s *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8s *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *vignetteIntensityTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus vignette_f16_f16_host_tensor(Rpp16f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp16f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *vignetteIntensityTensor,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

// -------------------- erase --------------------

template <typename T>
RppStatus erase_host_tensor(T *srcPtr,
                            RpptDescPtr srcDescPtr,
                            T *dstPtr,
                            RpptDescPtr dstDescPtr,
                            RpptRoiLtrb *anchorBoxInfoTensor,
                            T *colorsTensor,
                            Rpp32u *numBoxesTensor,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            RppLayoutParams layoutParams,
                            rpp::Handle& handle);

// -------------------- fog --------------------

RppStatus fog_u8_u8_host_tensor(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *fogAlphaMask,
                                Rpp32f *fogIntensityMask,
                                Rpp32f *intensityFactor,
                                Rpp32f *grayFactor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle);

RppStatus fog_f16_f16_host_tensor(Rpp16f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp16f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *intensityFactor,
                                  Rpp32f *grayFactor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus fog_f32_f32_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f* fogAlphaMask,
                                  Rpp32f* fogIntensityMask,
                                  Rpp32f *intensityFactor,
                                  Rpp32f *grayFactor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus fog_i8_i8_host_tensor(Rpp8s *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8s *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f* fogAlphaMask,
                                Rpp32f* fogIntensityMask,
                                Rpp32f *intensityFactor,
                                Rpp32f *grayFactor,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle);

// -------------------- rain --------------------

RppStatus rain_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32f slantAngle,
                                 Rpp32f *alphaValues,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus rain_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f rainPercentage,
                                   Rpp32u rainWidth,
                                   Rpp32u rainHeight,
                                   Rpp32f slantAngle,
                                   Rpp32f *alphaValues,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus rain_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f rainPercentage,
                                   Rpp32u rainWidth,
                                   Rpp32u rainHeight,
                                   Rpp32f slantAngle,
                                   Rpp32f *alphaValues,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus rain_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32f rainPercentage,
                                 Rpp32u rainWidth,
                                 Rpp32u rainHeight,
                                 Rpp32f slantAngle,
                                 Rpp32f *alphaValues,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

/**************************************** GEOMETRIC AUGMENTATIONS ****************************************/

// -------------------- crop --------------------

RppStatus crop_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus crop_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus crop_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus crop_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

// -------------------- crop_and_patch --------------------

RppStatus crop_and_patch_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                           Rpp8u *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrDst,
                                           RpptROIPtr cropRoiTensor,
                                           RpptROIPtr patchRoiTensor,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus crop_and_patch_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                             Rpp32f *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrDst,
                                             RpptROIPtr cropRoiTensor,
                                             RpptROIPtr patchRoiTensor,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus crop_and_patch_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                             Rpp16f *srcPtr2,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptROIPtr roiTensorPtrDst,
                                             RpptROIPtr cropRoiTensor,
                                             RpptROIPtr patchRoiTensor,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus crop_and_patch_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                           Rpp8s *srcPtr2,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrDst,
                                           RpptROIPtr cropRoiTensor,
                                           RpptROIPtr patchRoiTensor,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

// -------------------- crop_mirror_normalize --------------------

RppStatus crop_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8u *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *offsetTensor,
                                                  Rpp32f *multiplierTensor,
                                                  Rpp32u *mirrorTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

RppStatus crop_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *offsetTensor,
                                                    Rpp32f *multiplierTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus crop_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp16f *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    Rpp32f *offsetTensor,
                                                    Rpp32f *multiplierTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus crop_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp8s *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *offsetTensor,
                                                  Rpp32f *multiplierTensor,
                                                  Rpp32u *mirrorTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams layoutParams,
                                                  rpp::Handle& handle);

RppStatus crop_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *offsetTensor,
                                                   Rpp32f *multiplierTensor,
                                                   Rpp32u *mirrorTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle);

RppStatus crop_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *offsetTensor,
                                                   Rpp32f *multiplierTensor,
                                                   Rpp32u *mirrorTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams layoutParams,
                                                   rpp::Handle& handle);

// -------------------- flip --------------------

RppStatus flip_u8_u8_host_tensor(Rpp8u *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8u *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus flip_f32_f32_host_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *horizontalTensor,
                                   Rpp32u *verticalTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus flip_f16_f16_host_tensor(Rpp16f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp16f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32u *horizontalTensor,
                                   Rpp32u *verticalTensor,
                                   RpptROIPtr roiTensorPtrSrc,
                                   RpptRoiType roiType,
                                   RppLayoutParams layoutParams,
                                   rpp::Handle& handle);

RppStatus flip_i8_i8_host_tensor(Rpp8s *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp8s *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 Rpp32u *horizontalTensor,
                                 Rpp32u *verticalTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 RppLayoutParams layoutParams,
                                 rpp::Handle& handle);

RppStatus flip_voxel_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptGenericDescPtr srcGenericDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptGenericDescPtr dstGenericDescPtr,
                                         Rpp32u *horizontalTensor,
                                         Rpp32u *verticalTensor,
                                         Rpp32u *depthTensor,
                                         RpptROI3DPtr roiGenericPtrSrc,
                                         RpptRoi3DType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);

RppStatus flip_voxel_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptGenericDescPtr srcGenericDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptGenericDescPtr dstGenericDescPtr,
                                       Rpp32u *horizontalTensor,
                                       Rpp32u *verticalTensor,
                                       Rpp32u *depthTensor,
                                       RpptROI3DPtr roiGenericPtrSrc,
                                       RpptRoi3DType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

// -------------------- remap --------------------

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus remap_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8u *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *rowRemapTable,
                                     Rpp32f *colRemapTable,
                                     RpptDescPtr remapTableDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus remap_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *rowRemapTable,
                                       Rpp32f *colRemapTable,
                                       RpptDescPtr remapTableDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

RppStatus remap_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     Rpp8s *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32f *rowRemapTable,
                                     Rpp32f *colRemapTable,
                                     RpptDescPtr remapTableDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     RppLayoutParams layoutParams,
                                     rpp::Handle& handle);

RppStatus remap_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *rowRemapTable,
                                       Rpp32f *colRemapTable,
                                       RpptDescPtr remapTableDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus remap_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *rowRemapTable,
                                           Rpp32f *colRemapTable,
                                           RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle);

RppStatus remap_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *rowRemapTable,
                                             Rpp32f *colRemapTable,
                                             RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

RppStatus remap_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *rowRemapTable,
                                           Rpp32f *colRemapTable,
                                           RpptDescPtr remapTableDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams ,
                                           rpp::Handle& handle);

RppStatus remap_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *rowRemapTable,
                                             Rpp32f *colRemapTable,
                                             RpptDescPtr remapTableDescPtr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams layoutParams,
                                             rpp::Handle& handle);

// -------------------- resize --------------------

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle);

RppStatus resize_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle);

RppStatus resize_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle);

RppStatus resize_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus resize_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8u *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams,
                                            rpp::Handle& handle);

RppStatus resize_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams,
                                              rpp::Handle& handle);

RppStatus resize_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp16f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptImagePatchPtr dstImgSize,
                                              RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType,
                                              RppLayoutParams srcLayoutParams,
                                              rpp::Handle& handle);

RppStatus resize_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            Rpp8s *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            RpptImagePatchPtr dstImgSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams srcLayoutParams,
                                            rpp::Handle& handle);

template <typename T>
RppStatus resize_separable_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f * tempPtr,
                                       RpptDescPtr tempDescPtr,
                                       RpptImagePatchPtr dstImgSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams,
                                       RpptInterpolationType interpolationType,
                                       rpp::Handle& handle);

// -------------------- resize_crop_mirror --------------------

RppStatus resize_crop_mirror_u8_u8_host_tensor(Rpp8u *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8u *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               RpptImagePatchPtr dstImgSize,
                                               Rpp32u *mirrorTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

RppStatus resize_crop_mirror_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp32f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize,
                                                 Rpp32u *mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus resize_crop_mirror_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp16f *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 RpptImagePatchPtr dstImgSize,
                                                 Rpp32u *mirrorTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams layoutParams,
                                                 rpp::Handle& handle);

RppStatus resize_crop_mirror_i8_i8_host_tensor(Rpp8s *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               Rpp8s *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               RpptImagePatchPtr dstImgSize,
                                               Rpp32u *mirrorTensor,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               RppLayoutParams layoutParams,
                                               rpp::Handle& handle);

// -------------------- resize_mirror_normalize --------------------

RppStatus resize_mirror_normalize_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8u *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSize,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus resize_mirror_normalize_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp32f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSize,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams layoutParams,
                                                      rpp::Handle& handle);

RppStatus resize_mirror_normalize_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp16f *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      RpptImagePatchPtr dstImgSize,
                                                      Rpp32f *meanTensor,
                                                      Rpp32f *stdDevTensor,
                                                      Rpp32u *mirrorTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams layoutParams,
                                                      rpp::Handle& handle);

RppStatus resize_mirror_normalize_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                    RpptDescPtr srcDescPtr,
                                                    Rpp8s *dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptImagePatchPtr dstImgSize,
                                                    Rpp32f *meanTensor,
                                                    Rpp32f *stdDevTensor,
                                                    Rpp32u *mirrorTensor,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType,
                                                    RppLayoutParams layoutParams,
                                                    rpp::Handle& handle);

RppStatus resize_mirror_normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp32f *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     RpptImagePatchPtr dstImgSize,
                                                     Rpp32f *meanTensor,
                                                     Rpp32f *stdDevTensor,
                                                     Rpp32u *mirrorTensor,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     RpptRoiType roiType,
                                                     RppLayoutParams layoutParams,
                                                     rpp::Handle& handle);

RppStatus resize_mirror_normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp16f *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     RpptImagePatchPtr dstImgSize,
                                                     Rpp32f *meanTensor,
                                                     Rpp32f *stdDevTensor,
                                                     Rpp32u *mirrorTensor,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     RpptRoiType roiType,
                                                     RppLayoutParams layoutParams,
                                                     rpp::Handle& handle);

// -------------------- lens_correction --------------------

void compute_lens_correction_remap_tables_host_tensor(RpptDescPtr srcDescPtr,
                                                      Rpp32f *rowRemapTable,
                                                      Rpp32f *colRemapTable,
                                                      RpptDescPtr tableDescPtr,
                                                      Rpp32f *cameraMatrixTensor,
                                                      Rpp32f *distortionCoeffsTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      rpp::Handle& handle);

// -------------------- phase --------------------

RppStatus phase_u8_u8_host_tensor(Rpp8u *srcPtr1,
                                  Rpp8u *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

RppStatus phase_f32_f32_host_tensor(Rpp32f *srcPtr1,
                                    Rpp32f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus phase_f16_f16_host_tensor(Rpp16f *srcPtr1,
                                    Rpp16f *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    Rpp16f *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle);

RppStatus phase_i8_i8_host_tensor(Rpp8s *srcPtr1,
                                  Rpp8s *srcPtr2,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8s *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams,
                                  rpp::Handle& handle);

// -------------------- slice --------------------

template<typename T>
RppStatus slice_host_tensor(T *srcPtr,
                            RpptGenericDescPtr srcGenericDescPtr,
                            T *dstPtr,
                            RpptGenericDescPtr dstGenericDescPtr,
                            Rpp32s *anchorTensor,
                            Rpp32s *shapeTensor,
                            T* fillValue,
                            bool enablePadding,
                            Rpp32u *roiTensor,
                            RppLayoutParams layoutParams,
                            rpp::Handle& handle);

// -------------------- transpose --------------------

RppStatus transpose_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle);

template<typename T>
RppStatus transpose_generic_host_tensor(T *srcPtr,
                                        RpptGenericDescPtr srcGenericDescPtr,
                                        T *dstPtr,
                                        RpptGenericDescPtr dstGenericDescPtr,
                                        Rpp32u *permTensor,
                                        Rpp32u *roiTensor,
                                        rpp::Handle& handle);

// -------------------- warp_affine --------------------

/************* NEAREST NEIGHBOUR INTERPOLATION *************/

RppStatus warp_affine_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8u *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle);

RppStatus warp_affine_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp32f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams,
                                             rpp::Handle& handle);

RppStatus warp_affine_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           Rpp8s *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *affineTensor,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams srcLayoutParams,
                                           rpp::Handle& handle);

RppStatus warp_affine_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp16f *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             Rpp32f *affineTensor,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             RppLayoutParams srcLayoutParams,
                                             rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus warp_affine_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8u *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *affineTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams srcLayoutParams,
                                                 rpp::Handle& handle);

RppStatus warp_affine_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp32f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *affineTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams srcLayoutParams,
                                                   rpp::Handle& handle);

RppStatus warp_affine_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                 RpptDescPtr srcDescPtr,
                                                 Rpp8s *dstPtr,
                                                 RpptDescPtr dstDescPtr,
                                                 Rpp32f *affineTensor,
                                                 RpptROIPtr roiTensorPtrSrc,
                                                 RpptRoiType roiType,
                                                 RppLayoutParams srcLayoutParams,
                                                 rpp::Handle& handle);

RppStatus warp_affine_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                   RpptDescPtr srcDescPtr,
                                                   Rpp16f *dstPtr,
                                                   RpptDescPtr dstDescPtr,
                                                   Rpp32f *affineTensor,
                                                   RpptROIPtr roiTensorPtrSrc,
                                                   RpptRoiType roiType,
                                                   RppLayoutParams srcLayoutParams,
                                                   rpp::Handle& handle);

// -------------------- warp_perspective --------------------

/************* NEAREST NEIGHBOR INTERPOLATION *************/

RppStatus warp_perspective_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp8u *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32f *perspectiveTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams srcLayoutParams,
                                                rpp::Handle& handle);

RppStatus warp_perspective_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32f *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *perspectiveTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams srcLayoutParams,
                                                  rpp::Handle& handle);

RppStatus warp_perspective_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp8s *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                Rpp32f *perspectiveTensor,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                RppLayoutParams srcLayoutParams,
                                                rpp::Handle& handle);

RppStatus warp_perspective_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp16f *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  Rpp32f *perspectiveTensor,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  RppLayoutParams srcLayoutParams,
                                                  rpp::Handle& handle);

/************* BILINEAR INTERPOLATION *************/

RppStatus warp_perspective_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp8u *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      Rpp32f *perspectiveTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams srcLayoutParams,
                                                      rpp::Handle& handle);

RppStatus warp_perspective_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp32f *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32f *perspectiveTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams srcLayoutParams,
                                                        rpp::Handle& handle);

RppStatus warp_perspective_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                      RpptDescPtr srcDescPtr,
                                                      Rpp8s *dstPtr,
                                                      RpptDescPtr dstDescPtr,
                                                      Rpp32f *perspectiveTensor,
                                                      RpptROIPtr roiTensorPtrSrc,
                                                      RpptRoiType roiType,
                                                      RppLayoutParams srcLayoutParams,
                                                      rpp::Handle& handle);

RppStatus warp_perspective_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                        RpptDescPtr srcDescPtr,
                                                        Rpp16f *dstPtr,
                                                        RpptDescPtr dstDescPtr,
                                                        Rpp32f *perspectiveTensor,
                                                        RpptROIPtr roiTensorPtrSrc,
                                                        RpptRoiType roiType,
                                                        RppLayoutParams srcLayoutParams,
                                                        rpp::Handle& handle);

#endif // HOST_TENSOR_EXECUTORS_HPP
