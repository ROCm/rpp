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

#include "resize_mirror_normalize.cpp"

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