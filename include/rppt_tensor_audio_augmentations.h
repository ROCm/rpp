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

#ifndef RPPT_TENSOR_AUDIO_AUGMENTATIONS_H
#define RPPT_TENSOR_AUDIO_AUGMENTATIONS_H

/*!
 * \file
 * \brief RPPT Tensor Audio Augmentation Functions.
 *
 * \defgroup group_tensor_audio Operations: AMD RPP Tensor Audio Operations
 * \brief Tensor Audio Augmentations.
 */

#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Non Silent Region Detection augmentation HOST
 * \details Non Silent Region Detection augmentation for 1D audio buffer
            \n Finds the starting index and length of non silent region in the audio buffer by comparing the
            calculated short-term power with cutoff value passed
 * \param[in] srcPtr source tensor memory
 * \param[in] srcDescPtr source tensor descriptor
 * \param[in] srcLengthTensor source audio buffer length (tensor of batchSize values)
 * \param[out] detectedIndexTensor beginning index of non silent region (tensor of batchSize values)
 * \param[out] detectionLengthTensor length of non silent region  (tensor of batchSize values)
 * \param[in] cutOffDB cutOff(dB) below which the signal is considered silent
 * \param[in] windowLength window length used for computing short-term power of the signal
 * \param[in] referencePower reference power that is used to convert the signal to dB
 * \param[in] resetInterval number of samples after which the moving mean average is recalculated to avoid loss of precision
 * \param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_audio
 */
RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcLengthTensor, Rpp32f *detectedIndexTensor, Rpp32f *detectionLengthTensor, Rpp32f cutOffDB, Rpp32s windowLength, Rpp32f referencePower, Rpp32s resetInterval, rppHandle_t rppHandle);

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_AUGMENTATIONS_H