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

#ifndef HIP_TENSOR_EXECUTORS_HPP
#define HIP_TENSOR_EXECUTORS_HPP

#include <hip/hip_runtime.h>
#include "rpp_hip_load_store.hpp"

/**************************************** ARITHMETIC OPERATIONS ****************************************/

// -------------------- add_scalar --------------------

RppStatus hip_exec_add_scalar_tensor(Rpp32f *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32f *addTensor,
                                     rpp::Handle& handle);

// -------------------- fused_multiply_add_scalar --------------------

RppStatus hip_exec_fused_multiply_add_scalar_tensor(Rpp32f *srcPtr,
                                                    RpptGenericDescPtr srcGenericDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptGenericDescPtr dstGenericDescPtr,
                                                    RpptROI3DPtr roiGenericPtrSrc,
                                                    Rpp32f *mulTensor,
                                                    Rpp32f *addTensor,
                                                    rpp::Handle& handle);

// -------------------- log_generic --------------------

template <typename T, typename U>
RppStatus hip_exec_log_generic_tensor(T *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      U *dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      uint *roiTensor,
                                      rpp::Handle& handle);

// -------------------- magnitude --------------------

template <typename T>
RppStatus hip_exec_magnitude_tensor(T *srcPtr1,
                                    T *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    T *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rpp::Handle& handle);

// -------------------- multiply_scalar --------------------

RppStatus hip_exec_multiply_scalar_tensor(Rpp32f *srcPtr,
                                          RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          Rpp32f *mulTensor,
                                          rpp::Handle& handle);

// -------------------- subtract_scalar --------------------

RppStatus hip_exec_subtract_scalar_tensor(Rpp32f *srcPtr,
                                          RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          Rpp32f *subtractTensor,
                                          rpp::Handle& handle);

/**************************************** AUDIO AUGMENTATIONS ****************************************/

#ifdef AUDIO_SUPPORT

// -------------------- down_mixing --------------------

RppStatus hip_exec_down_mixing_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      bool normalizeWeights,
                                      rpp::Handle& handle);

// -------------------- mel_filter_bank --------------------

RppStatus hip_exec_mel_filter_bank_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s* srcDimsTensor,
                                          Rpp32f maxFreqVal,
                                          Rpp32f minFreqVal,
                                          RpptMelScaleFormula melFormula,
                                          Rpp32s numFilter,
                                          Rpp32f sampleRate,
                                          bool normalize,
                                          rpp::Handle& handle);

// -------------------- non_silent_region_detection --------------------

RppStatus hip_exec_non_silent_region_detection_tensor(Rpp32f *srcPtr,
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

RppStatus hip_exec_pre_emphasis_filter_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32f *coeffTensor,
                                              Rpp32s *srcLengthTensor,
                                              RpptAudioBorderType borderType,
                                              rpp::Handle& handle);

// -------------------- resample --------------------

RppStatus hip_exec_resample_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *inRateTensor,
                                   Rpp32f *outRateTensor,
                                   Rpp32s *srcDimsTensor,
                                   RpptResamplingWindow &window,
                                   rpp::Handle& handle);

// -------------------- spectrogram --------------------

RppStatus hip_exec_spectrogram_tensor(Rpp32f* srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f* dstPtr,
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

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr srcDims,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle);

#endif // AUDIO_SUPPORT

#endif // HIP_TENSOR_EXECUTORS_HPP
