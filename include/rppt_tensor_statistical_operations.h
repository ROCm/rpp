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

#ifndef RPPT_TENSOR_STATISTICAL_OPERATIONS_H
#define RPPT_TENSOR_STATISTICAL_OPERATIONS_H

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \file
 * \brief RPPT Tensor Operations - Statistical Operations.
 * \defgroup group_rppt_tensor_statistical_operations RPPT Tensor Operations - Statistical Operations.
 * \brief RPPT Tensor Operations - Statistical Operations.
 */

/*! \addtogroup group_rppt_tensor_statistical_operations
 * @{
 */

/*! \brief Tensor sum operation on HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor sum is a reduction operation that finds the channel-wise (R sum / G sum / B sum) and total sum for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] tensorSumArr destination array in HOST memory
 * \param [in] tensorSumArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then tensorSumArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorSrc[i].xywhROI.roiWidth <= 3840 and roiTensorSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_sum_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorSumArr, Rpp32u tensorSumArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Tensor sum operation on HIP backend for a NCHW/NHWC layout tensor
 * \details The tensor sum is a reduction operation that finds the channel-wise (R sum / G sum / B sum) and total sum for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] tensorSumArr destination array in HIP memory
 * \param [in] tensorSumArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then tensorSumArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorSrc[i].xywhROI.roiWidth <= 3840 and roiTensorSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_sum_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t tensorSumArr, Rpp32u tensorSumArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Tensor min operation on HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor min is a reduction operation that finds the channel-wise (R min / G min / B min) and overall min for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] minArr destination array in HOST memory
 * \param [in] minArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then tensorSumArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorSrc[i].xywhROI.roiWidth <= 3840 and roiTensorSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_min_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t minArr, Rpp32u minArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Tensor min operation on HIP backend for a NCHW/NHWC layout tensor
 * \details The tensor min is a reduction operation that finds the channel-wise (R min / G min / B min) and overall min for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] minArr destination array in HIP memory
 * \param [in] minArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then tensorSumArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorSrc[i].xywhROI.roiWidth <= 3840 and roiTensorSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_min_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageMinArr, Rpp32u imageMinArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! \brief Tensor max operation on HOST backend for a NCHW/NHWC layout tensor
 * \details The tensor max is a reduction operation that finds the channel-wise (R max / G max / B max) and overall max for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HOST memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] maxArr destination array in HOST memory
 * \param [in] maxArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then tensorSumArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorSrc ROI data in HOST memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorSrc[i].xywhROI.roiWidth <= 3840 and roiTensorSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_max_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t maxArr, Rpp32u maxArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Tensor max operation on HIP backend for a NCHW/NHWC layout tensor
 * \details The tensor max is a reduction operation that finds the channel-wise (R max / G max / B max) and overall max for each image in a batch of RGB(3 channel) / greyscale(1 channel) images with an NHWC/NCHW tensor layout.<br>
 * - srcPtr depth ranges - Rpp8u (0 to 255), Rpp16f (0 to 1), Rpp32f (0 to 1), Rpp8s (-128 to 127).
 * - dstPtr depth ranges - Will be same depth as srcPtr.
 * \param [in] srcPtr source tensor in HIP memory
 * \param [in] srcDescPtr source tensor descriptor (Restrictions - numDims = 4, offsetInBytes >= 0, dataType = U8/F16/F32/I8, layout = NCHW/NHWC, c = 1/3)
 * \param [out] maxArr destination array in HIP memory
 * \param [in] maxArrLength length of provided destination array (Restrictions - if srcDescPtr->c == 1 then tensorSumArrLength >= srcDescPtr->n, and if srcDescPtr->c == 3 then tensorSumArrLength >= srcDescPtr->n * 4)
 * \param [in] roiTensorSrc ROI data in HIP memory, for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y)) | (Restrictions - roiTensorSrc[i].xywhROI.roiWidth <= 3840 and roiTensorSrc[i].xywhROI.roiHeight <= 2160)
 * \param [in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_tensor_max_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t imageMaxArr, Rpp32u imageMaxArrLength, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/*! @}
 */

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_STATISTICAL_OPERATIONS_H
