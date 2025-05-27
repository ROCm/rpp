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

#include "hip_tensor_executors.hpp"

// -------------------- median_filter device helper --------------------
template <int kernelSize>
__device__ void median_filter_row_hip_compute(uchar *srcPtr, d_float8 *median_f8)
{
    const int paddedKernelWidth = kernelSize + 7; // padded row size for aligned memory access
    const int uint32LoadCountPerRow = (kernelSize + 10) / 4; // Number of 32-bit loads required to read each row
    const int windowSize = kernelSize * kernelSize; // Total number of pixels in the filter window
    const int medianIndex = (windowSize - 1) / 2; // Index of the median value in the sorted list
    const int sortSteps = medianIndex + 1; // Number of steps required for partial sort to find the median

    float src[kernelSize * paddedKernelWidth], medianVals[8];

    // Load and unpack image data into float using bitwise operations
    for (int i = 0; i < kernelSize; ++i)
    {
        uchar *rowPtr = srcPtr + i * SMEM_LENGTH_X; // Pointer to the start of the current row
        uint32_t *uintPtr = (uint32_t *)rowPtr;

        // Read the row in 32-bit chunks and unpack 4 uchar values from each
        for (int j = 0; j < uint32LoadCountPerRow; ++j)
        {
            uint32_t val = uintPtr[j];
            for (int k = 0; k < 4; ++k)
            {
                int posInRow = j * 4 + k;
                if (posInRow >= paddedKernelWidth)
                    break;
                src[i * paddedKernelWidth + posInRow] = (float)((val >> (k * 8)) & 0xFF); // Extract byte and convert to float
            }
        }
    }

    // Perform median filtering for 8 separate output pixels
    for (int filter = 0; filter < 8; ++filter)
    {
        float window[windowSize];
        const int offsetX = filter; // Column offset for the current filter output

        // Extract window values from the padded source buffer
        for (int i = 0; i < windowSize; ++i)
        {
            int row = i / kernelSize;
            int col = i % kernelSize;
            window[i] = src[row * paddedKernelWidth + offsetX + col];
        }

        // Partial selection sort to find the median value in the window
        for (int i = 0; i < sortSteps; ++i)
        {
            int minIdx = i;
            for (int j = i + 1; j < windowSize; ++j)
                minIdx = (window[j] < window[minIdx]) ? j : minIdx;

            float temp = window[i];
            window[i] = window[minIdx];
            window[minIdx] = temp;
        }

        // Store the median result for the current filter
        medianVals[filter] = window[medianIndex];
    }

    // Store the 8 median values as two float4 vectors
    median_f8->f4[0] = make_float4(medianVals[0], medianVals[1], medianVals[2], medianVals[3]);
    median_f8->f4[1] = make_float4(medianVals[4], medianVals[5], medianVals[6], medianVals[7]);
}

template <typename T>
__global__ void median_filter_3x3_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Compute input pixel coordinates with edge replication
    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[8]; // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    // Process the remaining 2 channels for RGB images
    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[8]; // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + srcStridesNCH.y + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[8]; // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[8]; // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (srcStridesNCH.y) + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }
        else
        {
            // Nearest-neighbor padding
            T tempBuffer[8]; // Temporary storage for 8 pixels
            for (int i = 0; i < 8; i++)
            {
                int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (2 * srcStridesNCH.y) + (clampedY * srcStridesNCH.z) + clampedX;
                tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pkd3_pln3_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pkd3_pln3_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for(int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pkd3_pln3_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pkd3_pln3_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;

    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0, rgbOffset = 0; i < 8; i++, rgbOffset += 3)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clampedY * srcStridesNH.y) + (clampedX * 3);

            tempBuffer[rgbOffset] = srcPtr[clampedIdx];         // R
            tempBuffer[rgbOffset + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[rgbOffset + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pln3_pkd3_hip_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        rpp_hip_load8_to_uchar8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<3>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pln3_pkd3_hip_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        rpp_hip_load8_to_uchar8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pln3_pkd3_hip_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        rpp_hip_load8_to_uchar8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pln3_pkd3_hip_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer0[8], tempBuffer1[8], tempBuffer2[8];

        for (int i = 0; i < 8; i++)
        {
            int clampedX = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clampedY = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx0 = (id_z * srcStridesNCH.x) + (clampedY * srcStridesNCH.z) + clampedX;
            int clampedIdx1 = clampedIdx0 + srcStridesNCH.y;
            int clampedIdx2 = clampedIdx1 + srcStridesNCH.y;

            tempBuffer0[i] = srcPtr[clampedIdx0];
            tempBuffer1[i] = srcPtr[clampedIdx1];
            tempBuffer2[i] = srcPtr[clampedIdx2];
        }

        rpp_hip_load8_to_uchar8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_median_filter_tensor(T *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        T *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32u kernelSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + kernelSize + 7) >> 3;
    int globalThreads_y = dstDescPtr->h + kernelSize;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = ((dstDescPtr->strides.hStride + kernelSize) / 3 + 7 ) >> 3;

        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(median_filter_3x3_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(median_filter_5x5_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(median_filter_7x7_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(median_filter_9x9_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(median_filter_3x3_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(median_filter_5x5_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(median_filter_7x7_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(median_filter_9x9_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(median_filter_3x3_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(median_filter_5x5_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(median_filter_7x7_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(median_filter_9x9_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(median_filter_3x3_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(median_filter_5x5_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(median_filter_7x7_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(median_filter_9x9_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}

template RppStatus hip_exec_median_filter_tensor<Rpp8u>(Rpp8u*,
                                                        RpptDescPtr,
                                                        Rpp8u*,
                                                        RpptDescPtr,
                                                        Rpp32u,
                                                        RpptROIPtr,
                                                        RpptRoiType,
                                                        rpp::Handle&);

template RppStatus hip_exec_median_filter_tensor<half>(half*,
                                                       RpptDescPtr,
                                                       half*,
                                                       RpptDescPtr,
                                                       Rpp32u,
                                                       RpptROIPtr,
                                                       RpptRoiType,
                                                       rpp::Handle&);

template RppStatus hip_exec_median_filter_tensor<Rpp32f>(Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32f*,
                                                         RpptDescPtr,
                                                         Rpp32u,
                                                         RpptROIPtr,
                                                         RpptRoiType,
                                                         rpp::Handle&);

template RppStatus hip_exec_median_filter_tensor<Rpp8s>(Rpp8s*,
                                                        RpptDescPtr,
                                                        Rpp8s*,
                                                        RpptDescPtr,
                                                        Rpp32u,
                                                        RpptROIPtr,
                                                        RpptRoiType,
                                                        rpp::Handle&);
