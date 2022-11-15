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

#ifndef RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H
#define RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** gridmask ********************/

// Gridmask augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] tileWidth tileWidth value for gridmask calculation = width of black square + width of spacing until next black square on grid (a single Rpp32u number with tileWidth <= min(srcDescPtr->w, srcDescPtr->h) that applies to all images in the batch)
// *param[in] gridRatio gridRatio value for gridmask calculation = black square width / tileWidth (a single Rpp32f number with 0 <= gridRatio <= 1 that applies to all images in the batch)
// *param[in] gridAngle gridAngle value for gridmask calculation = grid rotation angle in radians (a single Rpp32f number that applies to all images in the batch)
// *param[in] translateVector translateVector for gridmask calculation = grid X and Y translation lengths in pixels (a single RpptUintVector2D x,y value pair that applies to all images in the batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_gridmask_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_gridmask_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u tileWidth, Rpp32f gridRatio, Rpp32f gridAngle, RpptUintVector2D translateVector, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** spatter ********************/

// Spatter augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor (srcDescPtr->w must be a maximum of 1920, srcDescPtr->h must be a maximum of 1080)
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] spatterColor RGB values to use for the spatter augmentation (A single set of 3 Rpp8u values as RpptRGB that applies to all images in the batch)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_spatter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB spatterColor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_spatter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptRGB spatterColor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** salt_and_pepper_noise ********************/

// Salt and Pepper Noise augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] noiseProbailityTensor noiseProbaility values to decide if a destination pixel is a noise-pixel, or equal to source (1D tensor of size batchSize with 0 <= noiseProbailityTensor[i] <= 1 for each image in batch)
// *param[in] saltProbailityTensor saltProbaility values to decide if a given destination noise-pixel is salt or pepper (1D tensor of size batchSize with 0 <= saltProbailityTensor[i] <= 1 for each image in batch)
// *param[in] saltValueTensor A user-defined salt noise value (1D tensor of size batchSize with 0 <= saltValueTensor[i] <= 1 for each image in batch)
// *param[in] pepperValueTensor A user-defined pepper noise value (1D tensor of size batchSize with 0 <= pepperValueTensor[i] <= 1 for each image in batch)
// *param[in] seed A user-defined seed value (single Rpp32u value)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_salt_and_pepper_noise_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *noiseProbabilityTensor, Rpp32f *saltProbabilityTensor, Rpp32f *saltValueTensor, Rpp32f *pepperValueTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_salt_and_pepper_noise_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *noiseProbabilityTensor, Rpp32f *saltProbabilityTensor, Rpp32f *saltValueTensor, Rpp32f *pepperValueTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** shot_noise ********************/

// Shot Noise augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] shotNoiseFactorTensor shotNoiseFactor values for each image, which are used to compute the lambda values in a poisson distribution (1D tensor of size batchSize with shotNoiseFactorTensor[i] >= 0 for each image in batch)
// *param[in] seed A user-defined seed value (single Rpp32u value)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_shot_noise_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *shotNoiseFactorTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_shot_noise_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *shotNoiseFactorTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** gaussian_noise ********************/

// Gaussian Noise augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDesc destination tensor descriptor
// *param[in] meanTensor mean values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor of size batchSize with meanTensor[i] >= 0 for each image in batch)
// *param[in] stdDevTensor stdDev values for each image, which are used to compute the generalized Box-Mueller transforms in a gaussian distribution (1D tensor of size batchSize with stdDevTensor[i] >= 0 for each image in batch)
// *param[in] seed A user-defined seed value (single Rpp32u value)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_gaussian_noise_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_gaussian_noise_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u seed, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_EFFECTS_AUGMENTATIONS_H
