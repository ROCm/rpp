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

/*!
 * \file
 * \brief RPPT Tensor Statistical Augmentation Functions.
 *
 * \defgroup group_tensor_stat Operations: AMD RPP Tensor Statistical Operations
 * \brief RPP Tensor Statistical Augmentations.
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief rppt_tensor_sum_host
* \details Tensor sum finder operation for a NCHW/NHWC layout tensor
* \ingroup group_tensor_stat
* \param [in] srcPtr source tensor memory
* \param [in] srcDescPtr source tensor descriptor (srcDescPtr->w can be a maximum of 3840, srcDescPtr->h can be a maximum of 2160)
* \param [out] tensorSumArr destination array of minimum length (srcPtr->n * srcPtr->c)
* \param [in] tensorSumArrLength length of provided destination array (minimum length = srcPtr->n * srcPtr->c)
* \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
* \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
* \return <tt> Rppt_Status enum</tt>.
*/
RppStatus rppt_tensor_sum_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorSumArr, Rpp32u tensorSumArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_tensor_sum_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorSumArr, Rpp32u tensorSumArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_STATISTICAL_OPERATIONS_H
