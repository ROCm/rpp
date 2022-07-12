/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RPPT_DATA_EXCHANGE_OPERATIONS_H
#define RPPT_DATA_EXCHANGE_OPERATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** copy ********************/

// Copy operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_copy_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_copy_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** swap_channels ********************/

// Swap R and B channels to toggle RGB<->BGR for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_swap_channels_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_swap_channels_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** color_to_greyscale ********************/

// Color to greyscale operation for a RGB/BGR NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcSubpixelLayout A RpptSubpixelLayout type enum to specify source subpixel layout (RGBtype or BGRtype)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_color_to_greyscale_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptSubpixelLayout srcSubpixelLayout, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_color_to_greyscale_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptSubpixelLayout srcSubpixelLayout, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_DATA_EXCHANGE_OPERATIONS_H
