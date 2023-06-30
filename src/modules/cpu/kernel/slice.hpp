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
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus slice_f32_f32_host_tensor(Rpp32f *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    RpptROI3DPtr roiGenericPtrSrc,
                                    RpptRoi3DType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI3D roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[4], (Rpp32s)srcGenericDescPtr->dims[2]};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        Rpp32u bufferLength = roi.xyzwhdROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcGenericDescPtr->strides[2]) + (roi.xyzwhdROI.xyz.y * srcGenericDescPtr->strides[3]) + (roi.xyzwhdROI.xyz.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        // Slice without fused output-layout toggle (NDHWC -> NDHWC or NCDHW -> NCDHW)
        for(int c = 0; c < layoutParams.channelParam; c++)
        {
            Rpp32f *srcPtrDepth, *dstPtrDepth;
            srcPtrDepth = srcPtrChannel;
            dstPtrDepth = dstPtrChannel;

            for(int i = 0; i < roi.xyzwhdROI.roiDepth; i++)
            {
                Rpp32f *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrDepth;
                dstPtrRow = dstPtrDepth;

                for(int j = 0; j < roi.xyzwhdROI.roiHeight; j++)
                {
                    memcpy(dstPtrRow, srcPtrRow, bufferLength);
                    srcPtrRow += srcGenericDescPtr->strides[3];
                    dstPtrRow += dstGenericDescPtr->strides[3];
                }
                srcPtrDepth += srcGenericDescPtr->strides[2];
                dstPtrDepth += dstGenericDescPtr->strides[2];
            }

            srcPtrChannel += srcGenericDescPtr->strides[1];
            dstPtrChannel += srcGenericDescPtr->strides[1];
        }
    }

    return RPP_SUCCESS;
}
