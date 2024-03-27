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

#ifndef RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#define RPPT_TENSOR_AUDIO_AUGMENTATIONS_H

/*!
 * \file
 * \brief RPPT Tensor Operations - Audio Augmentations.
 * \defgroup group_rppt_tensor_audio_augmentations RPPT Tensor Operations - Audio Augmentations.
 * \brief RPPT Tensor Operations - Audio Augmentations.
 */

/*! \addtogroup group_rppt_tensor_audio_augmentations
 * @{
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Non Silent Region Detection augmentation on HOST backend
 * \details Non Silent Region Detection augmentation for 1D audio buffer
            \n Finds the starting index and length of non silent region in the audio buffer by comparing the
            calculated short-term power with cutoff value passed
 * \param[in] srcPtr source tensor in HOST memory
 * \param[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
 * \param[in] srcLengthTensor source audio buffer length (1D tensor in HOST memory, of size batchSize)
 * \param[out] detectedIndexTensor beginning index of non silent region (1D tensor in HOST memory, of size batchSize)
 * \param[out] detectionLengthTensor length of non silent region  (1D tensor in HOST memory, of size batchSize)
 * \param[in] cutOffDB cutOff in dB below which the signal is considered silent
 * \param[in] windowLength window length used for computing short-term power of the signal
 * \param[in] referencePower reference power that is used to convert the signal to dB
 * \param[in] resetInterval number of samples after which the moving mean average is recalculated to avoid precision loss
 * \param[in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcLengthTensor, Rpp32f *detectedIndexTensor, Rpp32f *detectionLengthTensor, Rpp32f cutOffDB, Rpp32s windowLength, Rpp32f referencePower, Rpp32s resetInterval, rppHandle_t rppHandle);

/*! \brief To Decibels augmentation on HOST backend
 * \details To Decibels augmentation for 1D audio buffer converts magnitude values to decibel values
 * \param[in] srcPtr source tensor in HOST memory
 * \param[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
 * \param[in] srcDims source tensor sizes for each element in batch (2D tensor in HOST memory, of size batchSize * 2)
 * \param[in] cutOffDB  minimum or cut-off ratio in dB
 * \param[in] multiplier factor by which the logarithm is multiplied
 * \param[in] referenceMagnitude Reference magnitude if not provided maximum value of input used as reference
 * \param[in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_to_decibels_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude, rppHandle_t rppHandle);

/*! \brief Pre Emphasis Filter augmentation on HOST backend
 * \details Pre Emphasis Filter augmentation for audio data
 * \param[in] srcPtr source tensor in HOST memory
 * \param[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
 * \param[out] dstPtr destination tensor in HOST memory
 * \param[in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
 * \param[in] srcLengthTensor source audio buffer length (1D tensor in HOST memory, of size batchSize)
 * \param[in] coeffTensor preemphasis coefficient (1D tensor in HOST memory, of size batchSize)
 * \param[in] borderType border value policy
 * \param[in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32f *coeffTensor, RpptAudioBorderType borderType, rppHandle_t rppHandle);

/*! \brief Down Mixing augmentation on HOST backend
* \details Down Mixing augmentation for audio data
* \param[in] srcPtr source tensor in HOST memory
* \param[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
* \param[out] dstPtr destination tensor in HOST memory
* \param[in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
* \param[in] srcDimsTensor source audio buffer length and number of channels (1D tensor in HOST memory, of size batchSize * 2)
* \param[in] normalizeWeights bool flag to specify if normalization of weights is needed
* \param[in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
* \return A <tt> \ref RppStatus</tt> enumeration.
* \retval RPP_SUCCESS Successful completion.
* \retval RPP_ERROR* Unsuccessful completion.
*/
RppStatus rppt_down_mixing_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcDimsTensor, bool normalizeWeights, rppHandle_t rppHandle);

/*! \brief Resample augmentation on HOST backend
* \details Resample augmentation for audio data
* \param[in] srcPtr source tensor in HOST memory
* \param[in] srcDescPtr source tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
* \param[out] dstPtr destination tensor in HOST memory
* \param[in] dstDescPtr destination tensor descriptor (Restrictions - numDims = 3, offsetInBytes >= 0, dataType = F32)
* \param[in] inRate Input sampling rate (1D tensor in HOST memory, of size batchSize)
* \param[in] outRate Output sampling rate (1D tensor in HOST memory, of size batchSize)
* \param[in] srcDimsTensor source audio buffer length and number of channels (1D tensor in HOST memory, of size batchSize * 2)
* \param[in] window Resampling window (struct of type RpptRpptResamplingWindow)
* \param[in] rppHandle RPP HOST handle created with <tt>\ref rppCreateWithBatchSize()</tt>
* \return A <tt> \ref RppStatus</tt> enumeration.
* \retval RPP_SUCCESS Successful completion.
* \retval RPP_ERROR* Unsuccessful completion.
*/
RppStatus rppt_resample_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *inRateTensor, Rpp32f *outRateTensor, Rpp32s *srcDimsTensor, RpptResamplingWindow &window, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
