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

#ifndef RPPT_TENSOR_ARITHMETIC_OPERATIONS_H
#define RPPT_TENSOR_ARITHMETIC_OPERATIONS_H

/*!
 * \file
 * \brief RPPT Tensor Arithmetic operation Functions.
 *
 * \defgroup group_tensor_arithmetic Operations: AMD RPP Tensor Arithmetic Operations
 * \brief Tensor Color Augmentations.
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief  Fmadd augmentation HOST
 * \details This function performs the fmadd operation on a batch of 4D tensors.
 *          It multiplies each element of the source tensor by a corresponding element in the 'mulTensor',
 *          adds a corresponding element from the 'addTensor', and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] mulTensor mul values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] addTensor add values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Host-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_arithmetic
 */
RppStatus rppt_fused_multiply_add_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *mulTensor, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);


/*! \brief  Fmadd augmentation GPU
 * \details This function performs the fmadd operation on a batch of 4D tensors.
 *          It multiplies each element of the source tensor by a corresponding element in the 'mulTensor',
 *          adds a corresponding element from the 'addTensor', and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] mulTensor mul values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] addTensor add values for fmadd calculation (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Hip-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_arithmetic
 */

#ifdef GPU_SUPPORT
RppStatus rppt_fused_multiply_add_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *mulTensor, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief  add augmentation HOST
 * \details This function performs the add operation on a batch of 4D tensors.
 *          It adds a corresponding element from the 'addTensor' to sourceTensor, and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] addTensor add values for used for addition (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Host-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_arithmetic
 */
RppStatus rppt_add_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

/*! \brief  add augmentation GPU
 * \details This function performs the add operation on a batch of 4D tensors.
 *          It adds a corresponding element from the 'addTensor' to sourceTensor, and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] addTensor add values for used for addition (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Hip-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_arithmetic
 */

#ifdef GPU_SUPPORT
RppStatus rppt_add_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *addTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** subtract_scalar ********************/

/*! \brief  subtract augmentation HOST
 * \details This function performs the subtraction operation on a batch of 4D tensors.
 *          It subtracts a corresponding element from the 'subtractTensor' from sourceTensor, and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] subtractTensor subtract values for used for subtraction (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Host-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_arithmetic
 */

RppStatus rppt_subtract_scalar_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *subtractTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);

/*! \brief  subtract augmentation GPU
 * \details This function performs the subtraction operation on a batch of 4D tensors.
 *          It subtracts a corresponding element from the 'subtractTensor' from sourceTensor, and stores the result in the destination tensor.
 *          Support added for f32 -> f32 dataype.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] subtractTensor subtract values for used for subtraction (1D tensor of batchSize Rpp32f values)
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Hip-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_arithmetic
 */

#ifdef GPU_SUPPORT
RppStatus rppt_subtract_scalar_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32f *subtractTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_ARITHMETIC_OPERATIONS_H
