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

#ifndef HIP_TENSOR_STATISTICAL_OPERATIONS_HPP

#include <hip/hip_runtime.h>
#include "rpp_hip_common_load_store.hpp"

template <typename T>
RppStatus hip_exec_normalize_tensor(T *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    T *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u axisMask,
                                    Rpp32f *meanTensor,
                                    Rpp32f *stdDevTensor,
                                    Rpp8u computeMeanStddev,
                                    Rpp32f scale,
                                    Rpp32f shift,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle);

template <typename T, typename U>
RppStatus hip_exec_tensor_max(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *maxArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);

template <typename T, typename U>
RppStatus hip_exec_tensor_mean(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *tensorMeanArr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle);

template <typename T, typename U>
RppStatus hip_exec_tensor_min(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *minArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle &handle);

template <typename T>
RppStatus hip_exec_tensor_stddev(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 Rpp32f *imageStddevArr,
                                 Rpp32f *meanTensor,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle);

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln1_hip(T *srcPtr,
                                    uint2 srcStridesNH,
                                    U *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln1_hip<Rpp8u, Rpp32u>(Rpp8u *srcPtr,
                                                   uint2 srcStridesNH,
                                                   Rpp32u *tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln1_hip<Rpp8s, Rpp32s>(Rpp8s *srcPtr,
                                                   uint2 srcStridesNH,
                                                   Rpp32s *tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pln3_hip(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    U *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln3_hip<Rpp8u, Rpp32u>(Rpp8u *srcPtr,
                                                   uint3 srcStridesNH,
                                                   Rpp32u *tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pln3_hip<Rpp8s, Rpp32s>(Rpp8s *srcPtr,
                                                   uint3 srcStridesNH,
                                                   Rpp32s *tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32
template <typename T, typename U>
__global__ void tensor_sum_pkd3_hip(T *srcPtr,
                                    uint2 srcStridesNH,
                                    U *tensorSumArr,
                                    RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pkd3_hip<Rpp8u, Rpp32u>(Rpp8u *srcPtr,
                                                   uint2 srcStridesNH,
                                                   Rpp32u *tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

template <>
__global__ void tensor_sum_pkd3_hip<Rpp8s, Rpp32s>(Rpp8s *srcPtr,
                                                   uint2 srcStridesNH,
                                                   Rpp32s *tensorSumArr,
                                                   RpptROIPtr roiTensorPtrSrc);

// Handle f16/f32 datatype
template <typename T, typename U>
RppStatus hip_exec_tensor_sum(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              U *tensorSumArr,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle);

template<>
RppStatus hip_exec_tensor_sum<Rpp8u, Rpp64u>(Rpp8u *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp64u *tensorSumArr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle);

template<>
RppStatus hip_exec_tensor_sum<Rpp8s, Rpp64s>(Rpp8s *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             Rpp64s *tensorSumArr,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_threshold_tensor(T *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    T *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32f *minTensor,
                                    Rpp32f *maxTensor,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rpp::Handle& handle);

#endif // HIP_TENSOR_STATISTICAL_OPERATIONS_HPP
