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

/******************** image_mean ********************/

// Image mean finder operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor (srcDescPtr->w can be a maximum of 3840, srcDescPtr->h can be a maximum of 2160)
// *param[out] imageMeanArr destination array of minimum length (srcPtr->n * srcPtr->c)
// *param[in] imageMeanArrLength length of provided destination array (minimum length = srcPtr->n * srcPtr->c)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_image_mean_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageMeanArr, Rpp32u imageMeanArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_image_mean_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageMeanArr, Rpp32u imageMeanArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** image_stddev ********************/

// Image stddev finder operation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor (srcDescPtr->w can be a maximum of 3840, srcDescPtr->h can be a maximum of 2160)
// *param[out] imageStddevArr destination array of minimum length (srcPtr->n * srcPtr->c)
// *param[in] imageStddevArrLength length of provided destination array (minimum length = srcPtr->n * srcPtr->c)
// *param[in] meanTensor mean values for stddev calculation (1D tensor of size batchSize * 4 in format (MeanR, MeanG, MeanB, MeanImage) for each image in batch)
// *param[in] flag to select one among 0- channel stddev / 1- image stddev / 2- both.
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_image_stddev_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageStddevArr, Rpp32u imageStddevArrLength, Rpp32f *meanTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_image_stddev_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageStddevArr, Rpp32u imageStddevArrLength, Rpp32f *meanTensor, int flag, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT


#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_STATISTICAL_OPERATIONS_H