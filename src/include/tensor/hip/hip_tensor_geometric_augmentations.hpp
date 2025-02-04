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

#ifndef HIP_TENSOR_GEOMETRIC_AUGMENTATIONS_HPP
#define HIP_TENSOR_GEOMETRIC_AUGMENTATIONS_HPP

#include <hip/hip_runtime.h>
#include "rpp_hip_common_load_store.hpp"

template <typename T>
RppStatus hip_exec_crop_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_crop_and_patch_tensor(T *srcPtr1,
                                         T *srcPtr2,
                                         RpptDescPtr srcDescPtr,
                                         T *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptROIPtr cropTensorPtr,
                                         RpptROIPtr patchTensorPtr,
                                         RpptRoiType roiType,
                                         rpp::Handle& handle);

template <typename T, typename U>
RppStatus hip_exec_crop_mirror_normalize_tensor(T *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                U *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_flip_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_flip_voxel_tensor(T *srcPtr,
                                     RpptGenericDescPtr srcGenericDescPtr,
                                     T *dstPtr,
                                     RpptGenericDescPtr dstGenericDescPtr,
                                     RpptROI3DPtr roiGenericPtrSrc,
                                     Rpp32u *horizontalTensor,
                                     Rpp32u *verticalTensor,
                                     Rpp32u *depthTensor,
                                     RpptRoi3DType roiType,
                                     rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_remap_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32f *rowRemapTable,
                                Rpp32f *colRemapTable,
                                RpptDescPtr remapTableDescPtr,
                                RpptInterpolationType interpolationType,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_resize_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptImagePatchPtr dstImgSize,
                                 RpptInterpolationType interpolationType,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_resize_crop_mirror_tensor(T *srcPtr,
                                             RpptDescPtr srcDescPtr,
                                             T *dstPtr,
                                             RpptDescPtr dstDescPtr,
                                             RpptImagePatchPtr dstImgSizes,
                                             RpptInterpolationType interpolationType,
                                             RpptROIPtr roiTensorPtrSrc,
                                             RpptRoiType roiType,
                                             rpp::Handle& handle);

template <typename T, typename U>
RppStatus hip_exec_resize_mirror_normalize_tensor(T *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  U *dstPtr,
                                                  RpptDescPtr dstDescPtr,
                                                  RpptImagePatchPtr dstImgSizes,
                                                  RpptInterpolationType interpolationType,
                                                  RpptROIPtr roiTensorPtrSrc,
                                                  RpptRoiType roiType,
                                                  rpp::Handle& handle);

RppStatus hip_exec_lens_correction_tensor(RpptDescPtr dstDescPtr,
                                          Rpp32f *rowRemapTable,
                                          Rpp32f *colRemapTable,
                                          RpptDescPtr remapTableDescPtr,
                                          Rpp32f *cameraMatrix,
                                          Rpp32f *distanceCoeffs,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_phase_tensor(T *srcPtr1,
                                T *srcPtr2,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_slice_tensor(T *srcPtr,
                                RpptGenericDescPtr srcGenericDescPtr,
                                T *dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr,
                                Rpp32s *anchorTensor,
                                Rpp32s *shapeTensor,
                                T *fillValue,
                                bool enablePadding,
                                Rpp32u *roiTensor,
                                rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_transpose_tensor(T *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    T *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u *permTensor,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_warp_affine_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32f *affineTensor,
                                      RpptInterpolationType interpolationType,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rpp::Handle& handle);

template <typename T>
RppStatus hip_exec_warp_perspective_tensor(T *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *perspectiveTensor,
                                           RpptInterpolationType interpolationType,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rpp::Handle& handle);

#endif // HIP_TENSOR_GEOMETRIC_AUGMENTATIONS_HPP
