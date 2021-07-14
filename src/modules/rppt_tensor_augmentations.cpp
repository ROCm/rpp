/*
Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <rppt_tensor_augmentations.h>
#include <rppdefs.h>
#include "rppi_validate.hpp"

#ifdef HIP_COMPILE
    #include "hip/hip_declarations.hpp"
#elif defined(OCL_COMPILE)
    #include <cl/rpp_cl_common.hpp>
    #include "cl/cl_declarations.hpp"
#endif //backend

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;

#include "cpu/host_tensor_augmentations.hpp"

RppStatus
rppt_brightness_gpu(RppPtr_t srcPtr,
                    RpptDescPtr srcDescPtr,
                    RppPtr_t dstPtr,
                    RpptDescPtr dstDescPtr,
                    Rpp32f *alphaTensor,
                    Rpp32f *betaTensor,
                    RpptROIPtr roiTensorPtrSrc,
                    rppHandle_t rppHandle)
{
    // global_id(0) for width, global_id(1) for height and global_id(2) for n
    // if there are 3 channels process all in the same kernel

#ifdef OCL_COMPILE
    if (srcDescPtr->layout == RpptLayout::NCHW)
    {
        // copy roi from roiTensorSrc to gpu mem
        // copy alpha and beta tensor to corresponding gpu mem
        // brightness_tensor_nchw_cl(static_cast<cl_mem>(srcPtr),
        //                           c,
        //                           uint4 in_stride,
        //                           uint4 out_stride,
        //                           static_cast<cl_mem>(dstPtr),
        //                           static_cast<cl_mem>(src_roi_mem),
        //                           static_cast<cl_mem>(alpha_mem),
        //                           static_cast<cl_mem>(beta_mem),
        //                           rpp::deref(rppHandle));
    }
    else
    {
        // brightness_tensor_nhwc_cl(static_cast<cl_mem>(srcPtr),
        //                           c,
        //                           uint4 in_stride,
        //                           uint4 out_stride,
        //                           static_cast<cl_mem>(dstPtr),
        //                           static_cast<cl_mem>(alpha_mem),
        //                           static_cast<cl_mem>(alpha_mem),
        //                           rpp::deref(rppHandle));
    }
#elif defined (HIP_COMPILE)

    // Rpp32u paramIndex = 0;
    // copy_param_float(alpha, rpp::deref(rppHandle), paramIndex++);
    // copy_param_float(beta, rpp::deref(rppHandle), paramIndex++);

    // get_srcBatchIndex (rpp::deref(rppHandle), 3, RPPI_CHN_PACKED);

    // RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    // brightness_hip_tensor<Rpp8u>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offset,
    //                               srcDescPtr,
    //                               static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offset,
    //                               dstDescPtr,
    //                             //   alphaTensor,
    //                             //   betaTensor,
    //                             //   roiTensorPtrSrc,
    //                               rppHandle,
    //                               layoutParams);


    if (srcDescPtr->layout == RpptLayout::NCHW)
    {
        // copy roi from roiTensorSrc to gpu mem
        // copy alpha and beta tensor to corresponding gpu mem
        // brightness_tensor_nchw_hip(static_cast<cl_mem>(srcPtr),
        //                            c,
        //                            uint4 in_stride,
        //                            uint4 out_stride,
        //                            static_cast<cl_mem>(dstPtr),
        //                            static_cast<cl_mem>(src_roi_mem),
        //                            static_cast<cl_mem>(alpha_mem),
        //                            static_cast<cl_mem>(beta_mem),
        //                            rpp::deref(rppHandle));
    }
    else
    {
        // brightness_tensor_nhwc_hip(static_cast<cl_mem>(srcPtr),
        //                            c,
        //                            uint4 in_stride,
        //                            uint4 out_stride,
        //                            static_cast<cl_mem>(dstPtr),
        //                            static_cast<cl_mem>(alpha_mem),
        //                            static_cast<cl_mem>(alpha_mem),
        //                            rpp::deref(rppHandle));
    }
#endif //BACKEND

    return RPP_SUCCESS;
}

// Assumption: source and destination are on the same layout
RppStatus
rppt_brightness_host(RppPtr_t srcPtr,
                     RpptDescPtr srcDescPtr,
                     RppPtr_t dstPtr,
                     RpptDescPtr dstDescPtr,
                     Rpp32f *alphaTensor,
                     Rpp32f *betaTensor,
                     RpptROIPtr roiTensorPtrSrc,
                     RpptRoiType roiType,
                     rppHandle_t rppHandle)
{
    RppLayoutParams layoutParams = get_layout_params(srcDescPtr->layout, srcDescPtr->c);

    brightness_host_tensor<Rpp8u>(static_cast<Rpp8u*>(srcPtr) + srcDescPtr->offset,
                                  srcDescPtr,
                                  static_cast<Rpp8u*>(dstPtr) + dstDescPtr->offset,
                                  dstDescPtr,
                                  alphaTensor,
                                  betaTensor,
                                  roiTensorPtrSrc,
                                  roiType,
                                  layoutParams);

    return RPP_SUCCESS;
}
