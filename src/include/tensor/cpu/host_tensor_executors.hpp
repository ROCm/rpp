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

#endif // HOST_TENSOR_EXECUTORS_HPP
