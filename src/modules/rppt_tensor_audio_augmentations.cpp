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

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_audio_augmentations.h"
#include "cpu/host_tensor_audio_augmentations.hpp"

/******************** non_silent_region_detection ********************/

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                Rpp32s *srcLengthTensor,
                                                Rpp32s *detectedIndexTensor,
                                                Rpp32s *detectionLengthTensor,
                                                Rpp32f cutOffDB,
                                                Rpp32s windowLength,
                                                Rpp32f referencePower,
                                                Rpp32s resetInterval,
                                                rppHandle_t rppHandle)
{
    if (srcDescPtr->dataType == RpptDataType::F32)
    {
        non_silent_region_detection_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                                srcDescPtr,
                                                srcLengthTensor,
                                                detectedIndexTensor,
                                                detectionLengthTensor,
                                                cutOffDB,
                                                windowLength,
                                                referencePower,
                                                resetInterval,
                                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

    return RPP_SUCCESS;
}

/******************** to_decibels ********************/

RppStatus rppt_to_decibels_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptImagePatchPtr srcDims,
                                Rpp32f cutOffDB,
                                Rpp32f multiplier,
                                Rpp32f referenceMagnitude,
                                rppHandle_t rppHandle)
{
    if (multiplier == 0)
        return RPP_ERROR_ZERO_DIVISION;
    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        to_decibels_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                srcDescPtr,
                                static_cast<Rpp32f*>(dstPtr),
                                dstDescPtr,
                                srcDims,
                                cutOffDB,
                                multiplier,
                                referenceMagnitude,
                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE;
    }
}

/******************** pre_emphasis_filter ********************/

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        RppPtr_t dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32s *srcLengthTensor,
                                        Rpp32f *coeffTensor,
                                        RpptAudioBorderType borderType,
                                        rppHandle_t rppHandle)
{
    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        pre_emphasis_filter_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                        srcDescPtr,
                                        static_cast<Rpp32f*>(dstPtr),
                                        dstDescPtr,
                                        srcLengthTensor,
                                        coeffTensor,
                                        borderType,
                                        rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** down_mixing ********************/

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32s *srcDimsTensor,
                                bool  normalizeWeights,
                                rppHandle_t rppHandle)
{
    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        down_mixing_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                srcDescPtr,
                                static_cast<Rpp32f*>(dstPtr),
                                dstDescPtr,
                                srcDimsTensor,
                                normalizeWeights,
                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** spectrogram ********************/

RppStatus rppt_spectrogram_host(RppPtr_t srcPtr,
                                RpptDescPtr srcDescPtr,
                                RppPtr_t dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32s *srcLengthTensor,
                                bool centerWindows,
                                bool reflectPadding,
                                Rpp32f *windowFunction,
                                Rpp32s nfft,
                                Rpp32s power,
                                Rpp32s windowLength,
                                Rpp32s windowStep,
                                rppHandle_t rppHandle)
{
    if ((dstDescPtr->layout != RpptLayout::NFT) && (dstDescPtr->layout != RpptLayout::NTF)) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        spectrogram_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                srcDescPtr,
                                static_cast<Rpp32f*>(dstPtr),
                                dstDescPtr,
                                srcLengthTensor,
                                centerWindows,
                                reflectPadding,
                                windowFunction,
                                nfft,
                                power,
                                windowLength,
                                windowStep,
                                rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** mel_filter_bank ********************/

RppStatus rppt_mel_filter_bank_host(RppPtr_t srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    RppPtr_t dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    Rpp32s* srcDimsTensor,
                                    Rpp32f maxFreq,
                                    Rpp32f minFreq,
                                    RpptMelScaleFormula melFormula,
                                    Rpp32s numFilter,
                                    Rpp32f sampleRate,
                                    bool normalize,
                                    rppHandle_t rppHandle)
{
    if (srcDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if (dstDescPtr->layout != RpptLayout::NFT) return RPP_ERROR_INVALID_DST_LAYOUT;

    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        mel_filter_bank_host_tensor(static_cast<Rpp32f*>(srcPtr),
                                    srcDescPtr,
                                    static_cast<Rpp32f*>(dstPtr),
                                    dstDescPtr,
                                    srcDimsTensor,
                                    maxFreq,
                                    minFreq,
                                    melFormula,
                                    numFilter,
                                    sampleRate,
                                    normalize,
                                    rpp::deref(rppHandle));
        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}

/******************** resample ********************/

RppStatus rppt_resample_host(RppPtr_t srcPtr,
                             RpptDescPtr srcDescPtr,
                             RppPtr_t dstPtr,
                             RpptDescPtr dstDescPtr,
                             Rpp32f *inRateTensor,
                             Rpp32f *outRateTensor,
                             Rpp32s *srcDimsTensor,
                             RpptResamplingWindow &window,
                             rppHandle_t rppHandle)
{
    if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        resample_host_tensor(static_cast<Rpp32f*>(srcPtr),
                             srcDescPtr,
                             static_cast<Rpp32f*>(dstPtr),
                             dstDescPtr,
                             inRateTensor,
                             outRateTensor,
                             srcDimsTensor,
                             window,
                             rpp::deref(rppHandle));

        return RPP_SUCCESS;
    }
    else
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
}
