/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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

/*! \brief Bitwise AND computation on HOST backend for a NCHW/NHWC layout tensor
 * \details This function computes bitwise AND of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 *          srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 *          dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.jpg Sample Input1
 * \image html img150x150_2.jpg Sample Input2
 * \image html bitwise_and_operation_img150x150.jpg Sample Output
 * \param [in] srcPtr1 source1 tensor in HOST memory
 * \param [in] srcPtr2 source2 tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HOST memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_bitwise_and_host(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

/*! \brief Bitwise AND computation on HIP backend for a NCHW/NHWC layout tensor
 * \details The phase augmentation computes phase of corresponding pixels for a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \image html img150x150.jpg Sample Input1
 * \image html img150x150_2.jpg Sample Input2
 * \image html phase_augmentation_img150x150.jpg Sample Output
 * \param [in] srcPtr1 source1 tensor in HIP memory
 * \param [in] srcPtr2 source2 tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] dstPtr destination tensor in HIP memory
 * \param [in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = same as that of srcDescPtr)
 * \param [in] roiTensorSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_bitwise_and_gpu(RppPtr_t srcPtr1, RppPtr_t srcPtr2, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_ARITHMETIC_OPERATIONS_H
