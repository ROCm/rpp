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
#include <omp.h>

Rpp32f getSquare(Rpp32f &value)
{
    return (value * value);
}

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
                                                  RpptDescPtr srcDescPtr,
                                                  Rpp32s *srcLengthTensor,
                                                  Rpp32f *detectedIndexTensor,
                                                  Rpp32f *detectionLengthTensor,
                                                  Rpp32f cutOffDB,
                                                  Rpp32s windowLength,
                                                  Rpp32f referencePower,
                                                  Rpp32s resetInterval,
                                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    const Rpp32f cutOff = std::pow(10.0f, cutOffDB * 0.1f);

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32s srcLength = srcLengthTensor[batchCount];

        // mmsBuffer length is equal to input audio length and can vary dynamically for each input in a batch
        // preallocating a static buffer for entire batchsize will be too big, so allocate mmsBuffer for each sample dynamically
        Rpp32f *mmsBuffer = new Rpp32f[srcLength];
        bool referenceMax = (referencePower == 0.0f);

        // set reset interval based on the user input
        Rpp32s resetLength = (resetInterval == -1) ? srcLength : resetInterval;

        // calculate moving mean square of input
        Rpp32f meanFactor = 1.0f / windowLength;
        Rpp32s windowBegin = -windowLength + 1;
        for (Rpp32s outPos = 0; outPos < srcLength;)
        {
            Rpp32f sumOfSquares = 0.0f;
            for (Rpp32s i = std::max<Rpp32s>(windowBegin, 0); i < outPos; i++)
                sumOfSquares += getSquare(srcPtrTemp[i]);

            Rpp32s intervalEndIdx = std::min<Rpp32s>(srcLength, outPos + resetLength);
            for (; outPos < intervalEndIdx; outPos++, windowBegin++)
            {
                sumOfSquares += getSquare(srcPtrTemp[outPos]);
                mmsBuffer[outPos] = sumOfSquares * meanFactor;
                if (windowBegin >= 0)
                    sumOfSquares -= getSquare(srcPtrTemp[windowBegin]);
            }
        }

        // convert cutoff from DB to magnitude
        Rpp32f base = (referenceMax) ? *std::max_element(mmsBuffer, mmsBuffer + srcLength) : referencePower;
        Rpp32f cutOffMag = base * cutOff;

        // calculate begining index, length of non silent region from the mms buffer
        Rpp32s endIdx = srcLength;
        Rpp32s beginIdx = endIdx;
        Rpp32s detectBegin, detectEnd;
        for(int i = 0; i < endIdx; i++)
        {
            if(mmsBuffer[i] >= cutOffMag)
            {
                beginIdx = i;
                break;
            }
        }
        if(beginIdx == endIdx)
        {
            detectBegin = 0;
            detectEnd = 0;
        }
        else
        {
            for(int i = endIdx - 1; i >= beginIdx; i--)
            {
                if(mmsBuffer[i] >= cutOffMag)
                {
                    endIdx = i;
                    break;
                }
            }
            detectBegin = beginIdx;
            detectEnd = endIdx - beginIdx + 1;
        }

        // extend non silent region
        if(detectBegin != 0 && detectEnd != 0)
        {
            Rpp32s newBegin = std::max<Rpp32s>(detectBegin - (windowLength - 1), 0);
            detectEnd += detectBegin - newBegin;
            detectBegin = newBegin;
        }

        detectedIndexTensor[batchCount] = detectBegin;
        detectionLengthTensor[batchCount] = detectEnd;
        delete[] mmsBuffer;
    }
    return RPP_SUCCESS;
}