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
/* Brightness augmentation for a NCHW/NHWC layout tensor
*param[in] srcPtr source tensor memory
*param[in] srcDesc source tensor descriptor
*param[out] dstPtr destination tensor memory
*param[in] dstDesc destination tensor descriptor
*param[in] alphaTensor alpha values for brightness calculation (1D tensor of size batchSize with 0 <= alpha <= 20 for each image in batch)
*param[in] betaTensor beta values for brightness calculation (1D tensor of size batchSize with 0 <= beta <= 255 for each image in batch)
*param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
*param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
rppt_brightness_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alphaTensor, Rpp32f *betaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

// ----------------------------------------
// GPU brightness functions declaration
// ----------------------------------------
/* Brightness augmentation for a NCHW/NHWC layout tensor
*param[in] srcPtr source tensor memory
*param[in] srcDesc source tensor descriptor
*param[out] dstPtr destination tensor memory
*param[in] dstDesc destination tensor descriptor
*param[in] alphaTensor alpha values for brightness calculation (1D tensor of size batchSize with 0 <= alpha <= 20 for each image in batch)
*param[in] betaTensor beta values for brightness calculation (1D tensor of size batchSize with 0 <= beta <= 255 for each image in batch)
*param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
*param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
rppt_brightness_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alphaTensor, Rpp32f *betaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

// ----------------------------------------
// CPU color_jitter functions declaration
// ----------------------------------------
/* Color Jitter augmentation for a NCHW/NHWC layout tensor
*param[in] srcPtr source tensor memory
*param[in] srcDesc source tensor descriptor
*param[out] dstPtr destination tensor memory
*param[in] dstDesc destination tensor descriptor
*param[in] brightnessTensor brightness modification parameter for each image in batch (1D tensor of size batchSize with 0 <= brightnessTensor[i] <= 1 for each image in batch)
*param[in] contrastTensor contrast modification parameter for each image in batch (1D tensor of size batchSize with 0 <= contrastTensor[i] <= 1 for each image in batch)
*param[in] hueTensor hue modification parameter for each image in batch (1D tensor of size batchSize with 0 <= hueTensor[i] <= 1 for each image in batch)
*param[in] saturationTensor saturation modification parameter for each image in batch (1D tensor of size batchSize with 0 <= saturationTensor[i] <= 1 for each image in batch)
*param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
*param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : succesful completion
*retval RPP_ERROR : Error
*/
RppStatus
rppt_color_jitter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *brightnessTensor, Rpp32f *contrastTensor, Rpp32f *hueTensor, Rpp32f *saturationTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

// ----------------------------------------
// GPU brightness functions declaration
// ----------------------------------------
/* Brightness augmentation for a NCHW/NHWC layout tensor
*param[in] srcPtr source tensor memory
*param[in] srcDesc source tensor descriptor
*param[out] dstPtr destination tensor memory
*param[in] dstDesc destination tensor descriptor
*param[in] alphaTensor alpha values for brightness calculation (1D tensor of size batchSize with 0 <= alpha <= 20 for each image in batch)
*param[in] betaTensor beta values for brightness calculation (1D tensor of size batchSize with 0 <= beta <= 255 for each image in batch)
*param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
*param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
*returns a  RppStatus enumeration.
*retval RPP_SUCCESS : succesful completion
*retval RPP_ERROR : Error
*/
// RppStatus
// rppt_brightness_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alphaTensor, Rpp32f *betaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif