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
#include "rppt_tensor_advanced_augmentations.h"
#include "cpu/host_tensor_advanced_augmentations.hpp"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    //#include "hip/hip_tensor_advanced_augmentations.hpp"
#endif // HIP_COMPILE

/******************** erase ********************/

RppStatus rppt_erase_host(RppPtr_t srcPtr,
                          RpptDescPtr srcDescPtr,
                          RppPtr_t dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptRoiLtrb *anchorBoxInfoTensor,
                          RppPtr_t colorsTensor,
                          Rpp32u *numBoxesTensor,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);
    if ((srcDescPtr->dataType == RpptDataType::U8) && (dstDescPtr->dataType == RpptDataType::U8))
    {
        erase_host_tensor(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes,
                          srcDescPtr,
                          static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes,
                          dstDescPtr,
                          anchorBoxInfoTensor,
                          static_cast<Rpp8u*>(colorsTensor),
                          numBoxesTensor,
                          roiTensorPtrSrc,
                          roiType,
                          layoutParams,
                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F16) && (dstDescPtr->dataType == RpptDataType::F16))
    {
        erase_host_tensor(reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                          srcDescPtr,
                          reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(dstPtr) + srcDescPtr->offsetInBytes),
                          dstDescPtr,
                          anchorBoxInfoTensor,
                          static_cast<Rpp16f*>(colorsTensor),
                          numBoxesTensor,
                          roiTensorPtrSrc,
                          roiType,
                          layoutParams,
                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::F32) && (dstDescPtr->dataType == RpptDataType::F32))
    {
        erase_host_tensor(reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offsetInBytes),
                          srcDescPtr,
                          reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offsetInBytes),
                          dstDescPtr,
                          anchorBoxInfoTensor,
                          static_cast<Rpp32f*>(colorsTensor),
                          numBoxesTensor,
                          roiTensorPtrSrc,
                          roiType,
                          layoutParams,
                          rpp::deref(rppHandle));
    }
    else if ((srcDescPtr->dataType == RpptDataType::I8) && (dstDescPtr->dataType == RpptDataType::I8))
    {
        erase_host_tensor(static_cast<Rpp8s*>(srcPtr) + srcDescPtr->offsetInBytes,
                          srcDescPtr,
                          static_cast<Rpp8s*>(dstPtr) + dstDescPtr->offsetInBytes,
                          dstDescPtr,
                          anchorBoxInfoTensor,
                          static_cast<Rpp8s*>(colorsTensor),
                          numBoxesTensor,
                          roiTensorPtrSrc,
                          roiType,
                          layoutParams,
                          rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
}
