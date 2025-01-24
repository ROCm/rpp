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

#ifndef HIP_TENSOR_ARITHMETIC_OPERATIONS_HPP
#define HIP_TENSOR_ARITHMETIC_OPERATIONS_HPP

#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

RppStatus hip_exec_add_scalar_tensor(Rpp32f *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     Rpp32f *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32f *addTensor,
                                     rpp::Handle& handle);

RppStatus hip_exec_fused_multiply_add_scalar_tensor(Rpp32f *srcPtr,
                                                    RpptGenericDescPtr srcGenericDescPtr,
                                                    Rpp32f *dstPtr,
                                                    RpptGenericDescPtr dstGenericDescPtr,
                                                    RpptROI3DPtr roiGenericPtrSrc,
                                                    Rpp32f *mulTensor,
                                                    Rpp32f *addTensor,
                                                    rpp::Handle& handle);

template <typename T, typename U>
RppStatus hip_exec_log_generic_tensor(T *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      U *dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      uint *roiTensor,
                                      rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_magnitude_tensor(T *srcPtr1,
                                    T *srcPtr2,
                                    RpptDescPtr srcDescPtr,
                                    T *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    rpp::Handle& handle);

RppStatus hip_exec_multiply_scalar_tensor(Rpp32f *srcPtr,
                                          RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          Rpp32f *mulTensor,
                                          rpp::Handle& handle);

RppStatus hip_exec_subtract_scalar_tensor(Rpp32f *srcPtr,
                                          RpptGenericDescPtr srcGenericDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptGenericDescPtr dstGenericDescPtr,
                                          RpptROI3DPtr roiGenericPtrSrc,
                                          Rpp32f *subtractTensor,
                                          rpp::Handle& handle);

#endif // HIP_TENSOR_ARITHMETIC_OPERATIONS_HPP
