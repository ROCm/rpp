/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RPPT_TENSOR_STATISTICAL_OPERATIONS_H
#define RPPT_TENSOR_STATISTICAL_OPERATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** image_min ********************/

// Image min finder operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] imageMinArr destination array of minimum length (srcPtr->n)
// *param[in] imageMinArrLength length of provided destination array (minimum length = srcPtr->n)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_image_min_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageMinArr, Rpp32u imageMinArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** image_max ********************/

// Image max finder operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] imageMaxArr destination array of minimum length (srcPtr->n)
// *param[in] imageMaxArrLength length of provided destination array (minimum length = srcPtr->n)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

#ifdef GPU_SUPPORT
RppStatus rppt_image_max_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageMaxArr, Rpp32u imageMaxArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_STATISTICAL_OPERATIONS_H