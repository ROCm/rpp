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

RppStatus slice_audio_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32s *srcDimsTensor,
                            Rpp32f *anchorTensor,
                            Rpp32f *shapeTensor,
                            Rpp32f *fillValues,
                            rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s sampleBatchCount = batchCount * 2;

        // Slice for 1D input
        if (srcDescPtr->strides.wStride == 1 && dstDescPtr->strides.wStride == 1) {
            Rpp32s srcBufferLength = srcDimsTensor[sampleBatchCount];
            Rpp32f anchorRaw = anchorTensor[batchCount];
            Rpp32f shapeRaw = shapeTensor[batchCount];
            Rpp32f fillValue = fillValues[0];

            Rpp32s anchor = std::llround(anchorRaw);
            Rpp32s shape = std::llround(shapeRaw);

            if (anchor == 0 && shape == srcBufferLength) {
            // Do a memcpy if output dimension matches input dimension
                memcpy(dstPtrTemp, srcPtrTemp, shape * sizeof(Rpp32f));
            } else {
                Rpp32s vectorIncrement = 8;
                Rpp32s alignedLength = (shape / 8) * 8;
                __m256 pFillValue = _mm256_set1_ps(fillValue);

                bool needPad = (anchor < 0) || ((anchor + shape) > srcBufferLength);
                Rpp32s dstIdx = 0;
                if (needPad) {
                    // out of bounds (left side)
                    Rpp32s numIndices = std::abs(std::min(anchor, 0));
                    Rpp32s leftPadLength = std::min(numIndices, shape);
                    Rpp32s alignedLeftPadLength = (leftPadLength / 8) * 8;

                    for (; dstIdx < alignedLeftPadLength; dstIdx += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrTemp, pFillValue);
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; dstIdx < leftPadLength; dstIdx++)
                    {
                        *dstPtrTemp = fillValue;
                        dstPtrTemp++;
                    }

                    anchor += leftPadLength;
                }

                // within input bounds
                Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
                Rpp32s dstLengthInBounds = std::max(shape - dstIdx, 0);
                Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
                memcpy(dstPtrTemp, &srcPtrTemp[anchor], (size_t)(lengthInBounds * sizeof(Rpp32f)));
                dstIdx += lengthInBounds;
                dstPtrTemp += lengthInBounds;

                if (needPad) {
                    // out of bounds (right side)
                    for (; dstIdx < alignedLength; dstIdx += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrTemp, pFillValue);
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; dstIdx < shape; dstIdx++)
                    {
                        *dstPtrTemp = fillValue;
                        dstPtrTemp++;
                    }
                }
            }
        } else {
            Rpp32f anchorRaw[2], shapeRaw[2];
            Rpp32s anchor[2], shape[2];
            anchorRaw[0] = anchorTensor[sampleBatchCount];
            anchorRaw[1] = anchorTensor[sampleBatchCount + 1];
            shapeRaw[0] = shapeTensor[sampleBatchCount];
            shapeRaw[1] = shapeTensor[sampleBatchCount + 1];
            Rpp32f fillValue = fillValues[0];

            anchor[0] = std::llround(anchorRaw[0]);
            shape[0] = std::llround(shapeRaw[0]);
            anchor[1] = std::llround(anchorRaw[1]);
            shape[1] = std::llround(shapeRaw[1]);

            Rpp32s rowBound = std::min(srcDimsTensor[sampleBatchCount], shape[0]);
            Rpp32s colBound = std::min(srcDimsTensor[sampleBatchCount + 1], shape[1]);
            bool needRowPad = (anchor[0] < 0) || ((anchor[0] + shape[0]) > srcDimsTensor[sampleBatchCount]);
            bool needColPad = (anchor[1] < 0) || ((anchor[1] + shape[1]) > srcDimsTensor[sampleBatchCount + 1]);

            Rpp32s vectorIncrement = 8;
            __m256 pFillValue = _mm256_set1_ps(fillValue);
            Rpp32s alignedCol = (shape[1] / vectorIncrement) * vectorIncrement;
            Rpp32s alignedColMax = (dstDescPtr->strides.wStride / vectorIncrement) * vectorIncrement;

            srcPtrTemp = srcPtrTemp + anchor[0] * srcDescPtr->strides.wStride;
            int row = 0;
            for (; row < rowBound; row++) {
                int col = 0;
                Rpp32f *srcPtrRow = srcPtrTemp + row * srcDescPtr->strides.wStride + anchor[1];
                Rpp32f *dstPtrRow = dstPtrTemp;
                memcpy(dstPtrRow, &srcPtrRow[col], colBound * sizeof(Rpp32f));
                col += colBound;
                dstPtrRow += colBound;

                // Fill the columns which are beyond the input width with fill value specified
                if (col < shape[1] && needColPad) {
                    for (; col < alignedCol; col += vectorIncrement) {
                        _mm256_storeu_ps(dstPtrRow, pFillValue);
                        dstPtrRow += vectorIncrement;
                    }
                    for (; col < shape[1]; col++)
                        *dstPtrRow++ = fillValue;
                }

                dstPtrTemp += dstDescPtr->strides.wStride;
            }

            // Fill the rows which are beyond the input height with fill value specified
            if (row < shape[0]  && needRowPad) {
                for (; row < shape[0]; row++) {
                    Rpp32f *dstPtrRow = dstPtrTemp;
                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedColMax; vectorLoopCount += vectorIncrement) {
                        _mm256_storeu_ps(dstPtrRow, pFillValue);
                        dstPtrRow += vectorIncrement;
                    }
                    for (; vectorLoopCount < dstDescPtr->strides.wStride; vectorLoopCount++)
                        *dstPtrRow++ = fillValue;

                    dstPtrTemp += dstDescPtr->strides.wStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}