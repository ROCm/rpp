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

#include "rppdefs.h"
#include "rppi_validate.hpp"
#include "rppt_tensor_arithmetic_operations.h"
#include "cpu/host_tensor_arithmetic_operations.hpp"

/******************** fmadd_scalar ********************/

RppStatus rppt_fmadd_scalar_host(RppPtr_t srcPtr,
                                 RpptGenericDescPtr srcGenericDescPtr,
                                 RppPtr_t dstPtr,
                                 RpptGenericDescPtr dstGenericDescPtr,
                                 Rpp32f *mulTensor,
                                 Rpp32f *addTensor,
                                 RpptROI3DPtr roiGenericPtrSrc,
                                 RpptRoi3DType roiType,
                                 rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcGenericDescPtr->layout, srcGenericDescPtr->dims[1]);

    if (srcGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_SRC_DATATYPE;
    if (dstGenericDescPtr->dataType != RpptDataType::F32) return RPP_ERROR_INVALID_DST_DATATYPE;
    if ((srcGenericDescPtr->layout != RpptLayout::NCDHW) && (srcGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_SRC_LAYOUT;
    if ((dstGenericDescPtr->layout != RpptLayout::NCDHW) && (dstGenericDescPtr->layout != RpptLayout::NDHWC)) return RPP_ERROR_INVALID_DST_LAYOUT;
    if (srcGenericDescPtr->layout != dstGenericDescPtr->layout) return RPP_ERROR_INVALID_ARGUMENTS;

    if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        fmadd_scalar_f32_f32_host_tensor((Rpp32f*) (static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                         srcGenericDescPtr,
                                         (Rpp32f*) (static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                         dstGenericDescPtr,
                                         mulTensor,
                                         addTensor,
                                         roiGenericPtrSrc,
                                         roiType,
                                         layoutParams,
                                         rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}
