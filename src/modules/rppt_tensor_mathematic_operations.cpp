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
#include "rppt_tensor_mathematic_operations.h"
//#include "cpu/host_tensor_mathematic_operations.hpp"

#ifdef HIP_COMPILE
    #include <hip/hip_fp16.h>
    #include "hip/hip_tensor_mathematic_operations.hpp"
#endif // HIP_COMPILE

/********************************************************************************************************************/
/*********************************************** GPU_SUPPORT = ON ***********************************************/
/********************************************************************************************************************/

#ifdef GPU_SUPPORT

/******************** log ********************/

RppStatus rppt_log_gpu(RppPtr_t srcPtr,
                       RpptGenericDescPtr srcGenericDescPtr,
                       RppPtr_t dstPtr,
                       RpptGenericDescPtr dstGenericDescPtr,
                       Rpp32u *roiTensor,
                       rppHandle_t rppHandle)
{
#ifdef HIP_COMPILE
    if ((srcGenericDescPtr->dataType == RpptDataType::U8) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log_generic_tensor(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F16) && (dstGenericDescPtr->dataType == RpptDataType::F16))
    {
        hip_exec_log_generic_tensor(reinterpret_cast<half*>(static_cast<Rpp8u*>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<half*>(static_cast<Rpp8u*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::F32) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log_generic_tensor(reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(srcPtr) + srcGenericDescPtr->offsetInBytes),
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }
    else if ((srcGenericDescPtr->dataType == RpptDataType::I8) && (dstGenericDescPtr->dataType == RpptDataType::F32))
    {
        hip_exec_log_generic_tensor(static_cast<Rpp8s*>(srcPtr) + srcGenericDescPtr->offsetInBytes,
                                    srcGenericDescPtr,
                                    reinterpret_cast<Rpp32f *>(static_cast<Rpp8s*>(dstPtr) + dstGenericDescPtr->offsetInBytes),
                                    dstGenericDescPtr,
                                    roiTensor,
                                    rpp::deref(rppHandle));
    }

    return RPP_SUCCESS;
#elif defined(OCL_COMPILE)
    return RPP_ERROR_NOT_IMPLEMENTED;
#endif // backend
}

#endif // GPU_SUPPORT
