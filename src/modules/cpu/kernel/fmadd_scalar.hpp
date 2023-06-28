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

RppStatus fmadd_scalar_f32_f32_host_tensor(Rpp32f *srcPtr,
                                           RpptGenericDescPtr srcDescPtr,
                                           Rpp32f *dstPtr,
                                           RpptGenericDescPtr dstDescPtr,
                                           Rpp32f *mulTensor,
                                           Rpp32f *addTensor,
                                           RpptRoiXyzwhd roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           RppLayoutParams layoutParams,
                                           rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, 0, (Rpp32s)srcDescPtr->dims[3], (Rpp32s)srcDescPtr->dims[4], (Rpp32s)srcDescPtr->dims[2]};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->dims[0]; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType); // To change

        Rpp32f mulParam = mulTensor[batchCount];
        Rpp32f addParam = addTensor[batchCount];

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides[0];

        Rpp32u bufferLength = roi.RpptRoiXyzwhd.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xyzwhdROI.xyz.z * srcDescPtr->strides.dStride) + (roi.xyzwhdROI.xyz.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

#if __AVX2__
        Rpp32u alignedLength; // = (bufferLength / 24) * 24;
        //Rpp32u vectorIncrement = 24;
        Rpp32u vectorIncrementPerChannel = 8;

        __m256 pFmaddParams[2];
        pFmaddParams[0] = _mm256_set1_ps(mulParam);
        pFmaddParams[1] = _mm256_set1_ps(addParam);
#endif
        // Fmadd without fused output-layout toggle single channel(NCDHW -> NCDHW)
        if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCDHW) && (dstDescPtr->layout == RpptLayout::NCDHW))
        {
#if __AVX2__
            alignedLength = bufferLength & ~7;
#endif
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
                    Rpp32f *srcPtrTemp, *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
#if __AVX2__
                        __m256 p[1];

                        rpp_simd_load(rpp_load8_f32_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_fmadd_8_host(p, pFmaddParams);                     // fmadd adjustment
                        rpp_simd_store(rpp_store8_f32_to_f32_avx, dstPtrTemp, p);  // simd stores                                    // simd stores                                         // simd stores
#endif
                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = RPPPIXELCHECKF32((*srcPtrTemp * mulParam) + addParam);

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }

                    srcPtrRow += srcDescPtr->strides[3];
                    dstPtrRow += dstDescPtr->strides[3];
                }

                srcPtrDepth += srcDescPtr->strides[2];
                dstPtrDepth += dstDescPtr->strides[2];
            }
        }
    }

    return RPP_SUCCESS;
}
