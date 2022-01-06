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

#ifndef RPPT_TENSOR_AUGMENTATIONS_H
#define RPPT_TENSOR_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** brightness ********************/

// Brightness augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] alphaTensor alpha values for brightness calculation (1D tensor of size batchSize with 0 <= alpha <= 20 for each image in batch)
// *param[in] betaTensor beta values for brightness calculation (1D tensor of size batchSize with 0 <= beta <= 255 for each image in batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_brightness_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alphaTensor, Rpp32f *betaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
RppStatus rppt_brightness_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alphaTensor, Rpp32f *betaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** gamma_correction ********************/

// Gamma correction augmentation for a NCHW/NHWC layout tensor
// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] gammaTensor gamma values for gamma correction calculation (1D tensor of size batchSize with gamma >= 0 for each image in batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_gamma_correction_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *gammaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
RppStatus rppt_gamma_correction_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *gammaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** blend ********************/

// Alpha blending augmentation for a NCHW/NHWC layout tensor
// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] alphaTensor alpha values for alpha-blending (1D tensor of size batchSize with the transparency factor transparency factor 0 <= alpha <= 1 for each image in batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_blend_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alpha, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
RppStatus rppt_blend_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *alpha, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** color_twist ********************/

// Color Twist augmentation for a NCHW/NHWC layout tensor
// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] brightnessTensor brightness modification parameter for color_jitter calculation (1D tensor of size batchSize with 0 < brightnessTensor[i] <= 20 for each image in batch)
// *param[in] contrastTensor contrast modification parameter for color_jitter calculation (1D tensor of size batchSize with 0 < contrastTensor[i] <= 255 for each image in batch)
// *param[in] hueTensor hue modification parameter for color_jitter calculation (1D tensor of size batchSize with 0 <= hueTensor[i] <= 359 for each image in batch)
// *param[in] saturationTensor saturation modification parameter for color_jitter calculation (1D tensor of size batchSize with saturationTensor[i] >= 0 for each image in batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus
rppt_color_twist_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *brightnessTensor, Rpp32f *contrastTensor, Rpp32f *hueTensor, Rpp32f *saturationTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** color_jitter ********************/

// Color Jitter augmentation for a NCHW/NHWC layout tensor
// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] brightnessTensor brightness modification parameter for color_jitter calculation (1D tensor of size batchSize with brightnessTensor[i] >= 0 for each image in batch)
// *param[in] contrastTensor contrast modification parameter for color_jitter calculation (1D tensor of size batchSize with contrastTensor[i] >= 0 for each image in batch)
// *param[in] hueTensor hue modification parameter for color_jitter calculation (1D tensor of size batchSize with -0.05 <= hueTensor[i] <= 0.05 for each image in batch)
// *param[in] saturationTensor saturation modification parameter for color_jitter calculation (1D tensor of size batchSize with saturationTensor[i] >= 0 for each image in batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus
rppt_color_jitter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *brightnessTensor, Rpp32f *contrastTensor, Rpp32f *hueTensor, Rpp32f *saturationTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** color_cast ********************/

// Color cast augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] rgbTensor R/G/B values for color casting calculation (2D tensor of size sizeof(RpptRGB) * batchSize with 0 <= rgbTensor[n].<R/G/B> <= 255 for each image in batch)
// *param[in] alphaTensor alpha values for color casting calculation (1D tensor of size sizeof(Rpp32f) * batchSize with alphaTensor[i] >= 0 for each image in batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_color_cast_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB *rgbTensor, Rpp32f *alphaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
RppStatus rppt_color_cast_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB *rgbTensor, Rpp32f *alphaTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** box_filter ********************/

// Box Filter augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor (srcDescPtr->offsetInBytes must be at least 12 * (kernelSize / 2))
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] kernelSize kernel size for box_filter (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_box_filter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** erode ********************/

// Erode augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor (srcDescPtr->offsetInBytes must be at least 12 * (kernelSize / 2), and any offset/row-padding/col-padding other than the image region should be set to 0xFF)
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] kernelSize kernel size for erode (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_erode_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** dilate ********************/

// Dilate augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor (srcDescPtr->offsetInBytes must be at least 12 * (kernelSize / 2))
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] kernelSize kernel size for dilate (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_dilate_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** crop ********************/

// Crop augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_crop_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
RppStatus rppt_crop_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/******************** gridmask ********************/

// Gridmask augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] tileWidth tileWidth value for gridmask calculation = width of black square + width of spacing until next black square on grid (a single Rpp32u number with tileWidth <= min(srcDescPtr->w, srcDescPtr->h) that applies to all images in the batch)
// *param[in] gridRatio gridRatio value for gridmask calculation = black square width / tileWidth (a single Rpp32f number with 0 <= gridRatio <= 1 that applies to all images in the batch)
// *param[in] gridAngle gridAngle value for gridmask calculation = grid rotation angle in radians (a single Rpp32f number that applies to all images in the batch)
// *param[in] translateVector translateVector for gridmask calculation = grid X and Y translation lengths in pixels (a single RpptUintVector2D x,y value pair that applies to all images in the batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_gridmask_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
RppStatus rppt_gridmask_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif