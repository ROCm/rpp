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
#include "rpp_hip_math.hpp"

// -------------------- median_filter device helpers --------------------

template <typename T>
__device__ void median_filter_3x3_row_hip_compute(T* src_smem, d_float8* median_f8)
{
    using VectorType = typename FilterDispatch<T>::VectorType;
    // Load 3 rows of shared memory into vectorized uint3/ int3 format
    VectorType row0 = *((VectorType*)&src_smem[0 * SMEM_LENGTH_X]);
    VectorType row1 = *((VectorType*)&src_smem[1 * SMEM_LENGTH_X]);
    VectorType row2 = *((VectorType*)&src_smem[2 * SMEM_LENGTH_X]);

    float3 val0_f3, val1_f3, val2_f3, valz_f3;

    // ======================
    // Pixel : 3x3 window
    // [ row0.x[0], row0.x[1], row0.x[2] ]
    // [ row1.x[0], row1.x[1], row1.x[2] ]
    // [ row2.x[0], row2.x[1], row2.x[2] ]
    // ======================

    // pixel 0
    valz_f3.x = rpp_hip_unpack0(row0.x);
    valz_f3.y = rpp_hip_unpack1(row0.x);
    valz_f3.z = rpp_hip_unpack2(row0.x);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row1.x);
    valz_f3.y = rpp_hip_unpack1(row1.x);
    valz_f3.z = rpp_hip_unpack2(row1.x);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row2.x);
    valz_f3.y = rpp_hip_unpack1(row2.x);
    valz_f3.z = rpp_hip_unpack2(row2.x);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    // Compute 3x3 median:
    // - Take max of row mins, median of row medians, min of row maxes
    // - Then, compute median of those three
    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[0] = rpp_hip_median3(valz_f3);

    // pixel 1
    valz_f3.x = rpp_hip_unpack1(row0.x);
    valz_f3.y = rpp_hip_unpack2(row0.x);
    valz_f3.z = rpp_hip_unpack3(row0.x);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row1.x);
    valz_f3.y = rpp_hip_unpack2(row1.x);
    valz_f3.z = rpp_hip_unpack3(row1.x);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row2.x);
    valz_f3.y = rpp_hip_unpack2(row2.x);
    valz_f3.z = rpp_hip_unpack3(row2.x);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[1] = rpp_hip_median3(valz_f3);

    // pixel 2
    valz_f3.x = rpp_hip_unpack2(row0.x);
    valz_f3.y = rpp_hip_unpack3(row0.x);
    valz_f3.z = rpp_hip_unpack0(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row1.x);
    valz_f3.y = rpp_hip_unpack3(row1.x);
    valz_f3.z = rpp_hip_unpack0(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row2.x);
    valz_f3.y = rpp_hip_unpack3(row2.x);
    valz_f3.z = rpp_hip_unpack0(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[2] = rpp_hip_median3(valz_f3);

    // pixel 3
    valz_f3.x = rpp_hip_unpack3(row0.x);
    valz_f3.y = rpp_hip_unpack0(row0.y);
    valz_f3.z = rpp_hip_unpack1(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row1.x);
    valz_f3.y = rpp_hip_unpack0(row1.y);
    valz_f3.z = rpp_hip_unpack1(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);;

    valz_f3.x = rpp_hip_unpack3(row2.x);
    valz_f3.y = rpp_hip_unpack0(row2.y);
    valz_f3.z = rpp_hip_unpack1(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[3] = rpp_hip_median3(valz_f3);

    // pixel 4
    valz_f3.x = rpp_hip_unpack0(row0.y);
    valz_f3.y = rpp_hip_unpack1(row0.y);
    valz_f3.z = rpp_hip_unpack2(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row1.y);
    valz_f3.y = rpp_hip_unpack1(row1.y);
    valz_f3.z = rpp_hip_unpack2(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack0(row2.y);
    valz_f3.y = rpp_hip_unpack1(row2.y);
    valz_f3.z = rpp_hip_unpack2(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[4] = rpp_hip_median3(valz_f3);

    // pixel 5
    valz_f3.x = rpp_hip_unpack1(row0.y);
    valz_f3.y = rpp_hip_unpack2(row0.y);
    valz_f3.z = rpp_hip_unpack3(row0.y);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row1.y);
    valz_f3.y = rpp_hip_unpack2(row1.y);
    valz_f3.z = rpp_hip_unpack3(row1.y);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack1(row2.y);
    valz_f3.y = rpp_hip_unpack2(row2.y);
    valz_f3.z = rpp_hip_unpack3(row2.y);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[5] = rpp_hip_median3(valz_f3);

    // pixel 6
    valz_f3.x = rpp_hip_unpack2(row0.y);
    valz_f3.y = rpp_hip_unpack3(row0.y);
    valz_f3.z = rpp_hip_unpack0(row0.z);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row1.y);
    valz_f3.y = rpp_hip_unpack3(row1.y);
    valz_f3.z = rpp_hip_unpack0(row1.z);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack2(row2.y);
    valz_f3.y = rpp_hip_unpack3(row2.y);
    valz_f3.z = rpp_hip_unpack0(row2.z);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[6] = rpp_hip_median3(valz_f3);

    // pixel 7
    valz_f3.x = rpp_hip_unpack3(row0.y);
    valz_f3.y = rpp_hip_unpack0(row0.z);
    valz_f3.z = rpp_hip_unpack1(row0.z);
    val0_f3.x = rpp_hip_min3(valz_f3);
    val0_f3.y = rpp_hip_median3(valz_f3);
    val0_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row1.y);
    valz_f3.y = rpp_hip_unpack0(row1.z);
    valz_f3.z = rpp_hip_unpack1(row1.z);
    val1_f3.x = rpp_hip_min3(valz_f3);
    val1_f3.y = rpp_hip_median3(valz_f3);
    val1_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_unpack3(row2.y);
    valz_f3.y = rpp_hip_unpack0(row2.z);
    valz_f3.z = rpp_hip_unpack1(row2.z);
    val2_f3.x = rpp_hip_min3(valz_f3);
    val2_f3.y = rpp_hip_median3(valz_f3);
    val2_f3.z = rpp_hip_max3(valz_f3);

    valz_f3.x = rpp_hip_max3(make_float3(val0_f3.x, val1_f3.x, val2_f3.x));
    valz_f3.y = rpp_hip_median3(make_float3(val0_f3.y, val1_f3.y, val2_f3.y));
    valz_f3.z = rpp_hip_min3(make_float3(val0_f3.z, val1_f3.z, val2_f3.z));
    median_f8->f1[7] = rpp_hip_median3(valz_f3);
}

template <>
__device__ void median_filter_3x3_row_hip_compute<float>(float* src_smem, d_float8* median_f8)
{
    float* row0Ptr = src_smem;
    float* row1Ptr = row0Ptr + SMEM_LENGTH_X;
    float* row2Ptr = row1Ptr + SMEM_LENGTH_X;
    float3 minVal_f3, maxVal_f3, medianVal_f3;

    #pragma unroll
    for (int px = 0; px < 8; ++px)
    {
        float3 row0_f3 = make_float3(row0Ptr[px], row0Ptr[px + 1], row0Ptr[px + 2]);
        float3 row1_f3 = make_float3(row1Ptr[px], row1Ptr[px + 1], row1Ptr[px + 2]);
        float3 row2_f3 = make_float3(row2Ptr[px], row2Ptr[px + 1], row2Ptr[px + 2]);

        minVal_f3.x = rpp_hip_min3(row0_f3);
        medianVal_f3.x = rpp_hip_median3(row0_f3);
        maxVal_f3.x = rpp_hip_max3(row0_f3);

        minVal_f3.y = rpp_hip_min3(row1_f3);
        medianVal_f3.y = rpp_hip_median3(row1_f3);
        maxVal_f3.y = rpp_hip_max3(row1_f3);

        minVal_f3.z = rpp_hip_min3(row2_f3);
        medianVal_f3.z = rpp_hip_median3(row2_f3);
        maxVal_f3.z = rpp_hip_max3(row2_f3);

        float maxOfMin = rpp_hip_max3(make_float3(minVal_f3.x, minVal_f3.y, minVal_f3.z));
        float median   = rpp_hip_median3(make_float3(medianVal_f3.x, medianVal_f3.y, medianVal_f3.z));
        float minOfMax = rpp_hip_min3(make_float3(maxVal_f3.x, maxVal_f3.y, maxVal_f3.z));

        median_f8->f1[px] = rpp_hip_median3(make_float3(minOfMax, median, maxOfMin));
    }
}

template<int kernelSize, typename T>
__device__ __forceinline__ float compute_median(T *window)
{
    constexpr int windowSize = kernelSize * kernelSize;
    constexpr int medianIndex = (windowSize - 1) / 2;    // median position

    int leftIdx  = 0;
    int rightIdx = windowSize - 1;

    // Hoare Quick-Select -----------------------------------------------------
    while (leftIdx < rightIdx)
    {
        // 1. choose midVal – median-of-3 (first, mid, last) is good enough
        int midIdx   = (leftIdx + rightIdx) >> 1;
        float3 val_f3 = make_float3(static_cast<float>(window[leftIdx]), static_cast<float>(window[midIdx]), static_cast<float>(window[rightIdx]));
        float midVal = rpp_hip_median3(val_f3);

        // 2. partition
        int i = leftIdx;
        int j = rightIdx;
        while (i <= j)
        {
            while (window[i] < midVal)  ++i;
            while (window[j] > midVal)  --j;

            if (i <= j)
            {
                T tmp = window[i];
                window[i]      = window[j];
                window[j]      = tmp;
                ++i;
                --j;
            }
        }

        // 3. shrink the search interval toward the median slot
        if (medianIndex <= j)
            rightIdx = j;
        else if (i <= medianIndex)
            leftIdx = i;
        else
            break;               // midVal is the median
    }

    return static_cast<float>(window[medianIndex]);
}

template <int kernelSize, typename T>
__device__ void median_filter_row_hip_compute(T *srcPtr, d_float8 *median_f8)
{
    using bufType = typename std::conditional<std::is_same<T, uchar>::value, uint32_t, int32_t>::type;
    const int paddedKernelWidth = kernelSize + 7; // padded row size for aligned memory access
    const int loadCountPerRow = (kernelSize + 10) / 4; // Number of 32-bit loads required to read each row
    const int windowSize = kernelSize * kernelSize;

    T src[kernelSize * paddedKernelWidth];

    // Load and unpack image data from shared memory into float array
    for (int i = 0; i < kernelSize; ++i)
    {
        //pointer to the start of the current row in shared memory (SMEM_LENGTH_X assumed defined)
        bufType *srcPtrRowUint = (bufType *)(srcPtr + i * SMEM_LENGTH_X);

        for (int j = 0; j < loadCountPerRow; ++j)
        {
            bufType val = srcPtrRowUint[j];
            // Unpack 4 bytes from each 32-bit int
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                int posInRow = (j << 2) + k; // same as j*4 + k, but faster with shift
                if (posInRow >= paddedKernelWidth)
                    break;
                src[i * paddedKernelWidth + posInRow] = static_cast<T>((val >> (k << 3)) & 0xFF);
            }
        }
    }

    // Compute median for 8 different filter positions on this row
    for (int filter = 0; filter < 8; ++filter)
    {
        T window[windowSize];
        const int offsetX = filter; // offset in columns for this pixel's filter window

        // Extract the window from src buffer with padding offset
        for (int i = 0; i < windowSize; ++i)
            window[i] = src[(i / kernelSize) * paddedKernelWidth + offsetX + (i % kernelSize)];

        // Calculate median using templated function (uses sort network for 3x3, partial selection for others)
        float medianVal = compute_median<kernelSize>(window);
        // Store median results into output struct (d_float8 assumed to have f1[8] float array)
        median_f8->f1[filter] = medianVal;
    }
}

template <int kernelSize>
__device__ void median_filter_row_hip_compute(float* src_smem, d_float8* median_f8)
{
    constexpr int paddedKernelWidth = kernelSize + 7;
    constexpr int loadCountPerRow = kernelSize + 10;
    constexpr int windowSize = kernelSize * kernelSize;

    float src[kernelSize * paddedKernelWidth];

    // Load shared memory into local array
    for (int i = 0; i < kernelSize; ++i)
    {
        float* srcRow = src_smem + i * SMEM_LENGTH_X;
        #pragma unroll
        for (int j = 0; j < loadCountPerRow; ++j)
        {
            src[i * paddedKernelWidth + j] = srcRow[j];
        }
    }

    // Compute 8 median values
    for (int filter = 0; filter < 8; ++filter)
    {
        float window[windowSize];
        int base = filter;

        for (int k = 0; k < windowSize; ++k)
        {
            int row = k / kernelSize;
            int col = k % kernelSize;
            window[k] = src[row * paddedKernelWidth + base + col];
        }

        median_f8->f1[filter] = compute_median<kernelSize>(window);
    }
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
        FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
        FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<5>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
        FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<7>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
        FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        {
            FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
            FilterDispatch<T>::rpp_hip_load8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_row_hip_compute<9>(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    SharedType *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(srcPtr + srcIdx, src_smem_channel);
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
        FilterDispatch<T>::rpp_hip_load24_pkd3_to_pln3(tempBuffer, src_smem_channel);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

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
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

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
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

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
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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
    using SharedType = typename FilterDispatch<T>::SharedType;
    __shared__ SharedType src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

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
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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

        FilterDispatch<T>::rpp_hip_load8(tempBuffer0, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer1, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        FilterDispatch<T>::rpp_hip_load8(tempBuffer2, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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
