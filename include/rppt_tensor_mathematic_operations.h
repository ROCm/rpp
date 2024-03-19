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

#ifndef RPPT_TENSOR_MATHEMATIC_OPERATIONS_H
#define RPPT_TENSOR_MATHEMATIC_OPERATIONS_H

/*!
 * \file
 * \brief RPPT Tensor Mathematic operation Functions.
 *
 * \defgroup group_tensor_mathematic Operations: AMD RPP Tensor Mathematic Operations
 * \brief Tensor Mathematic Operations.
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Mathematic Operations.
 * \defgroup group_tensor_mathematic_operations RPPT Tensor Operations - Mathematic Operations.
 * \brief RPPT Tensor Operations - Mathematic Operations.
 */

/*! \addtogroup group_rppt_tensor_mathematic_operations
 * @{
 */

/*! \brief Logarithmic Generic operation on HOST backend
 * \details Computes Log of the input for a given ND Tensor.
 *          Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes.
 * \param [in] srcPtr source tensor in HOST memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_log_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *roiTensor, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Logarithmic Generic operation on HIP backend
 * \details Computes Log of the input for a given ND Tensor.
 *          Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes.
 * \param [in] srcPtr source tensor in HIP memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor in HIP memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_log_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *roiTensor, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_MATHEMATIC_OPERATIONS_H
