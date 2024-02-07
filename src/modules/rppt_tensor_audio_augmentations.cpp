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
                                                Rpp32f *detectedIndexTensor,
                                                Rpp32f *detectionLengthTensor,
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
