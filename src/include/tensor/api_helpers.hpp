/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#ifndef API_HELPERS_HPP
#define API_HELPERS_HPP

#include "rpp.h"
#include "rppdefs.h"
#include <random>

// sets descriptor dimensions and strides for descriptor used for fog augmentation
inline void set_fog_mask_descriptor(RpptDescPtr descPtr, Rpp32s batchSize, Rpp32s maxHeight, Rpp32s maxWidth, Rpp32s numChannels)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = 0;
    descPtr->dataType = RpptDataType::F32;
    descPtr->layout = RpptLayout::NCHW;
    descPtr->n = batchSize;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;
    descPtr->strides = {descPtr->c * descPtr->w * descPtr->h,  1, descPtr->w, 1};
}

inline void init_grid_dropout(int batchCount, RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, Rpp32u gridH, Rpp32u gridW, Rpp32u &maxHoleW, Rpp32u &maxHoleH, Rpp32f holeRatio, bool randomOffset)
{
    int seed = randomOffset ? std::random_device{}() : 42;
    std::mt19937 rng(seed);

    for(int i=0; i< batchCount; i++)
    {
        Rpp32u roiW = roiTensorPtrSrc[i].xywhROI.roiWidth;
        Rpp32u roiH = roiTensorPtrSrc[i].xywhROI.roiHeight;
        Rpp32s x_base = roiTensorPtrSrc[i].xywhROI.xy.x;
        Rpp32s y_base = roiTensorPtrSrc[i].xywhROI.xy.y;

        Rpp32u cellW = roiW / gridW;
        Rpp32u cellH = roiH / gridH;
        Rpp32u holeW = static_cast<Rpp32u>(cellW * holeRatio);
        Rpp32u holeH = static_cast<Rpp32u>(cellH * holeRatio);
        if (holeW > maxHoleW)
            maxHoleW = holeW;
        if (holeH > maxHoleH)
            maxHoleH = holeH;

        std::uniform_int_distribution<int> distX(0, (cellW > holeW) ? cellW - holeW : 0);
        std::uniform_int_distribution<int> distY(0, (cellH > holeH) ? cellH - holeH : 0);

        int boxOffset = i * gridH * gridW;
        for (Rpp32u row = 0; row < gridH; ++row)
        {
            for (Rpp32u col = 0; col < gridW; ++col)
            {
                Rpp32s cellX = x_base + col * cellW;
                Rpp32s cellY = y_base + row * cellH;

                Rpp32s offsetX = 0, offsetY = 0;
                if (randomOffset && (cellW > holeW) && (cellH > holeH))
                {
                    offsetX = distX(rng);
                    offsetY = distY(rng);
                }

                Rpp32s x1 = std::min(cellX + offsetX, x_base + (Rpp32s)roiW - 1);
                Rpp32s y1 = std::min(cellY + offsetY, y_base + (Rpp32s)roiH - 1);
                Rpp32s x2 = std::min(x1 + (Rpp32s)holeW - 1, x_base + (Rpp32s)roiW - 1);
                Rpp32s y2 = std::min(y1 + (Rpp32s)holeH - 1, y_base + (Rpp32s)roiH - 1);

                int boxIdx = boxOffset + (row * gridW + col);
                anchorBoxInfoTensor[boxIdx].lt.x = x1;
                anchorBoxInfoTensor[boxIdx].lt.y = y1;
                anchorBoxInfoTensor[boxIdx].rb.x = x2;
                anchorBoxInfoTensor[boxIdx].rb.y = y2;
            }
        }
    }
}

#endif /* API_HELPERS_HPP */
