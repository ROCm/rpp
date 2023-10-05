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

#ifndef RPPT_TENSOR_MORPHOLOGICAL_OPERATIONS_H
#define RPPT_TENSOR_MORPHOLOGICAL_OPERATIONS_H

/*!
 * \file
 * \brief RPPT Tensor Morphological Augmentation Functions.
 *
 * \defgroup group_tensor_morph Operations: AMD RPP Tensor Morphological Operations
 * \brief Tensor Morphological Augmentations.
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Erode augmentation HOST
 * \details Erode augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] kernelSize kernel size for erode (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_morph
 */
#ifdef GPU_SUPPORT
/*! \brief Erode augmentation GPU
 * \details Erode augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] kernelSize kernel size for erode (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_morph
 */
RppStatus rppt_erode_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Dilate augmentation HOST
 * \details Dilate augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] kernelSize kernel size for dilate (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_morph
 */
#ifdef GPU_SUPPORT
/*! \brief Dilate augmentation GPU
 * \details Dilate augmentation for a NCHW/NHWC layout tensor
 * \param [in] srcPtr source tensor memory
 * \param [in] srcDescPtr source tensor descriptor
 * \param [out] dstPtr destination tensor memory
 * \param [in] dstDescPtr destination tensor descriptor
 * \param [in] kernelSize kernel size for dilate (a single Rpp32u odd number with kernelSize = 3/5/7/9 that applies to all images in the batch)
 * \param [in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_morph
 */
RppStatus rppt_dilate_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u kernelSize, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_MORPHOLOGICAL_OPERATIONS_H
