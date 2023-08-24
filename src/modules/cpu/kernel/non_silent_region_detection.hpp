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

Rpp32f getMax(Rpp32f *values, Rpp32s srcLength)
{
    Rpp32f max = values[0];
    for(int i = 1; i < srcLength; i++)
        max = std::max(max, values[i]);
    return max;
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

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32s srcLength = srcLengthTensor[batchCount];
        bool referenceMax = (referencePower == 0.0f);

        // set reset interval based on the user input
        Rpp32s resetLength = (resetInterval == -1) ? srcLength : resetInterval;

        // Calculate buffer size for mms array and allocate mms buffer
        Rpp32s mmsBufferSize = srcLength;
        Rpp32f *mmsBuffer = static_cast<Rpp32f *>(calloc(mmsBufferSize, sizeof(Rpp32f)));

        // Calculate moving mean square of input array and store srcPtrTemp mms buffer
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

        // Convert cutOff from DB to magnitude
        Rpp32f base = (referenceMax) ? getMax(mmsBuffer, mmsBufferSize) : referencePower;
        Rpp32f cutOffMag = base * std::pow(10.0f, cutOffDB * 0.1f);

        // Calculate begining index, length of non silent region from the mms buffer
        Rpp32s endIdx = mmsBufferSize;
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

        // Extend non silent region
        if(detectBegin != 0 && detectEnd != 0)
        {
            Rpp32s newBegin = std::max<Rpp32s>(detectBegin - (windowLength - 1), 0);
            detectEnd += detectBegin - newBegin;
            detectBegin = newBegin;
        }

        detectedIndexTensor[batchCount] = detectBegin;
        detectionLengthTensor[batchCount] = detectEnd;
        free(mmsBuffer);
    }

    return RPP_SUCCESS;
}