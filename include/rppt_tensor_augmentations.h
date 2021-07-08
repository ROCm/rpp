/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RPPT_TENSOR_AUGMENTATIONS_H
#define RPPT_TENSOR_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------
// CPU brightness functions declaration
// ----------------------------------------
/* Computes brightness of a tensor.
*param[in] srcPtr input tensor memory
*param[in] srcDesc source tensor descriptor
*param[in] dstPtr output tensor memory
*param[in] dstDesc output tensor descriptor
*param[in] roiTensorSrc source (of size n * 4 where 4 values represent (x,y,w,h))
*param[in] alphaTensor alpha values for brightness calculation and value should be between 0 and 20 (of size n/batch_size)
*param[in] betaTensor beta  values for brightness calculation and value should be between 0 and 255 (of size n)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppt_brightness_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alphaTensor, Rpp32f *betaTensor, RpptROIPtr roiTensorPtrSrc, rppHandle_t rppHandle);

// ----------------------------------------
// GPU brightness functions declaration
// ----------------------------------------
/* Computes brightness of an image.
*param[in] srcPtr input tensor memory
*param[in] srcDesc source tensor descriptor
*param[in] dstPtr output tensor memory
*param[in] dstDesc output tensor descriptor
*param[in] roiTensorSrc source  (of size n*4 where 4 values represent (x,y,w,h))
*param[in] alphaTensor alpha values for brightness calculation and value should be between 0 and 20 (of size n/batch_size)
*param[in] betaTensor beta  values for brightness calculation and value should be between 0 and 255 (of size n)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
 rppt_brightness_gpu(RppPtr_t srcPtr, RpptDesc srcDesc, RppPtr_t dstPtr, RpptDesc dstDesc, Rpp32u *roiTensorSrc, Rpp32f* alphaTensor, Rpp32f* betaTensor, rppHandle_t rppHandle);


#ifdef __cplusplus
}
#endif
#endif