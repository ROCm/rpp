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

#ifndef RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
#define RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H

/*!
 * \file
 * \brief RPPT Tensor Geometric Augmentation Functions.
 *
 * \defgroup group_tensor_geometric Operations: AMD RPP Tensor Geometric Operations
 * \brief Tensor Geometric Augmentations.
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Crop augmentation HOST
 * \details Crop augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_crop_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Crop augmentation GPU
 * \details Crop augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_crop_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT


/*! \brief Crop Mirror Normalize augmentation HOST
 * \details Crop Mirror Normalize augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] offsetTensor offset value for each image in the batch (offsetTensor[n] <= 0)
 * \param [in] multiplierTensor multiplier value for each image in the batch (multiplierTensor[n] > 0)
 * \param [in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_crop_mirror_normalize_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *offsetTensor, Rpp32f *multiplierTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Crop Mirror Normalize augmentation GPU
 * \details Crop Mirror Normalize augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] offsetTensor offset value for each image in the batch (offsetTensor[n] <= 0)
 * \param [in] multiplierTensor multiplier value for each image in the batch (multiplierTensor[n] > 0)
 * \param [in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_crop_mirror_normalize_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *offsetTensor, Rpp32f *multiplierTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT


/*! \brief Warp Affine augmentation HOST
 * \details Warp Affine augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] affineTensor affine matrix values for transformation calculation (2D tensor of size batchSize * 6 for each image in batch)
 * \param [in] interpolationType Interpolation type used (RpptInterpolationType::XYWH or RpptRoiType::LTRB)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_warp_affine_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *affineTensor, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Warp Affine augmentation GPU
 * \details Warp Affine augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] affineTensor affine matrix values for transformation calculation (2D tensor of size batchSize * 6 for each image in batch)
 * \param [in] interpolationType Interpolation type used (RpptInterpolationType::XYWH or RpptRoiType::LTRB)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_warp_affine_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *affineTensor, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Flip augmentation HOST
 * \details Flip augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] horizontalTensor horizontal flag value to set horizontal flip on/off for each image in the batch (horizontalTensor[n] = 0/1)
 * \param [in] verticalTensor vertical flag value to set vertical flip on/off for each image in the batch (verticalTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_flip_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Flip augmentation GPU
 * \details Flip augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] horizontalTensor horizontal flag value to set horizontal flip on/off for each image in the batch (horizontalTensor[n] = 0/1)
 * \param [in] verticalTensor vertical flag value to set vertical flip on/off for each image in the batch (verticalTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_flip_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Resize augmentation HOST
 * \details Resize augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] dstImgSizes destination image size
 * \param [in] interpolationType resize interpolation type
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_resize_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Resize augmentation GPU
 * \details Resize augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] dstImgSizes destination image size
 * \param [in] interpolationType resize interpolation type
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_resize_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Resize Mirror Normalize augmentation HOST
 * \details Resize Mirror Normalize augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] dstImgSizes destination image size
 * \param [in] interpolationType resize interpolation type
 * \param [in] meanTensor mean value for each image in the batch (meanTensor[n] >= 0, 1D tensor of size = batchSize for greyscale images, size = batchSize * 3 for RGB images))
 * \param [in] stdDevTensor standard deviation value for each image in the batch (stdDevTensor[n] >= 0, 1D tensor of size = batchSize for greyscale images, size = batchSize * 3 for RGB images)
 * \param [in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_resize_mirror_normalize_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Resize Mirror Normalize augmentation GPU
 * \details Resize Mirror Normalize augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] dstImgSizes destination image size
 * \param [in] interpolationType resize interpolation type
 * \param [in] meanTensor mean value for each image in the batch (meanTensor[n] >= 0, 1D tensor of size = batchSize for greyscale images, size = batchSize * 3 for RGB images))
 * \param [in] stdDevTensor standard deviation value for each image in the batch (stdDevTensor[n] >= 0, 1D tensor of size = batchSize for greyscale images, size = batchSize * 3 for RGB images)
 * \param [in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_resize_mirror_normalize_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Resize Crop Mirror augmentation HOST
 * \details Resize Crop Mirror augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] dstImgSizes destination image size
 * \param [in] interpolationType resize interpolation type
 * \param [in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_resize_crop_mirror_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Resize Crop Mirror augmentation GPU
 * \details Resize Crop Mirror augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] dstImgSizes destination image size
 * \param [in] interpolationType resize interpolation type
 * \param [in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_resize_crop_mirror_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Rotate augmentation HOST
 * \details Rotate augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] angle values for image rotation in degrees (positive deg-anticlockwise/negative deg-clockwise)
 * \param [in] interpolationType resize interpolation type
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_rotate_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *angle, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
/*! \brief Rotate augmentation GPU
 * \details Rotate augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] angle values for image rotation in degrees (positive deg-anticlockwise/negative deg-clockwise)
 * \param [in] interpolationType resize interpolation type
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_rotate_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *angle, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** slice ********************/

// slice for a generic 4D tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstGenericDescPtr destination tensor descriptor
// *param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
// *param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_slice_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_slice_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** flip_voxel ********************/

// Flip augmentation for a generic 4D Tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstGenericDescPtr destination tensor descriptor
// *param[in] horizontalTensor horizontal flag value to set horizontal flip on/off for each image in the batch (horizontalTensor[n] = 0/1)
// *param[in] verticalTensor vertical flag value to set vertical flip on/off for each image in the batch (verticalTensor[n] = 0/1)
// *param[in] depthTensor depth flag value to set vertical flip on/off for each video in the batch (verticalTensor[n] = 0/1)
// *param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
// *param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_flip_voxel_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, Rpp32u *depthTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_flip_voxel_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, Rpp32u *depthTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** normalize_generic ********************/

// Normalizes the input generic ND buffer by removing the mean and dividing by the standard deviation

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstGenericDescPtr destination tensor descriptor
// *param[in] axisMask axis along which normalization needs to be done
// *param[in] meanTensor values to be subtracted from input
// *param[in] stdDevTensor standard deviation values to scale the input
// *param[in] scale value to be multiplied with data after subtracting from mean
// *param[in] shift value to be added finally
// *param[in] roiTensor values to represent dimensions of input tensor
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_normalize_generic_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask,
                                      Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32f scale, Rpp32f shift, Rpp32u *roiTensor, rppHandle_t rppHandle);


#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
