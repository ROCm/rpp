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

#include "host_tensor_executors.hpp"

#include <type_traits>
#if __AVX2__
#include <immintrin.h>
#endif

/* median filter algorithm explanation for U8 PLN1 3x3 kernel size variant
Let’s consider a 3x32 image input:

x  x  x  x  x  x  x  x  x  x  ..  x  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  1  2  3  4  5  6  7  8  9  .. 32  x
x  x  x  x  x  x  x  x  x  x  ..  x  x

padLength = 1 (kernelSize / 2)

Below steps are followed for computing each output pixel in the ROI:
1. For each pixel location (i, j), collect a 3x3 neighborhood of pixels centered at (i, j)
   - Apply nearest-neighbor padding at borders
   - Extract values into a temporary array of 9 elements

2. Sort the 9 values:
   e.g., for 3x3 window: [2, 4, 3, 1, 5, 6, 3, 7, 2] → sorted → [1, 2, 2, 3, 3, 4, 5, 6, 7]

3. Pick the median (middle) value:
   - median = element at index 4 (zero-based), i.e., value 3 in the above example

4. Assign this median to the output pixel at (i, j)

This process is repeated for each pixel in the ROI.
- For single-channel (PLN1), apply per pixel.
- For multi-channel, median is computed independently per channel.

Note: Unlike box filter, there is no arithmetic averaging or SIMD optimization here due to sorting-based computation.
*/

/* =====================================================================================
 * OpenCV-matching median sort-net for 3x3 and 5x5.
 * - Same compare-swap sequences as OpenCV `medianBlur_SortNet` (median_blur.simd.hpp)
 * - Same border handling as existing RPP code: clamp (replicate)
 * ===================================================================================== */

template <typename T>
inline void rpp_minmax_op_scalar(T &a, T &b)
{
    T t = a;
    a = std::min(a, b);
    b = std::max(b, t);
}

// 3x3 sorting network (OpenCV m==3). After ops, p[4] is median.
template <typename T>
inline T rpp_median_3x3_sortnet(T *p)
{
#define op(i, j) rpp_minmax_op_scalar(p[i], p[j])
    op(1, 2); op(4, 5); op(7, 8); op(0, 1); op(3, 4); op(6, 7);
    op(1, 2); op(4, 5); op(7, 8); op(0, 3); op(5, 8); op(4, 7);
    op(3, 6); op(1, 4); op(2, 5); op(4, 7);
    op(4, 2); op(6, 4); op(4, 2);
#undef op
    return p[4];
}

// 5x5 sorting network (OpenCV m==5). After ops, p[12] is median.
template <typename T>
inline T rpp_median_5x5_sortnet(T *p)
{
#define op(i, j) rpp_minmax_op_scalar(p[i], p[j])
    op(1, 2); op(0, 1); op(1, 2); op(4, 5); op(3, 4);
    op(4, 5); op(0, 3); op(2, 5); op(2, 3); op(1, 4);
    op(1, 2); op(3, 4); op(7, 8); op(6, 7); op(7, 8);
    op(10, 11); op(9, 10); op(10, 11); op(6, 9); op(8, 11);
    op(8, 9); op(7, 10); op(7, 8); op(9, 10); op(0, 6);
    op(4, 10); op(4, 6); op(2, 8); op(2, 4); op(6, 8);
    op(1, 7); op(5, 11); op(5, 7); op(3, 9); op(3, 5);
    op(7, 9); op(1, 2); op(3, 4); op(5, 6); op(7, 8);
    op(9, 10); op(13, 14); op(12, 13); op(13, 14); op(16, 17);
    op(15, 16); op(16, 17); op(12, 15); op(14, 17); op(14, 15);
    op(13, 16); op(13, 14); op(15, 16); op(19, 20); op(18, 19);
    op(19, 20); op(21, 22); op(23, 24); op(21, 23); op(22, 24);
    op(22, 23); op(18, 21); op(20, 23); op(20, 21); op(19, 22);
    op(22, 24); op(19, 20); op(21, 22); op(23, 24); op(12, 18);
    op(16, 22); op(16, 18); op(14, 20); op(20, 24); op(14, 16);
    op(18, 20); op(22, 24); op(13, 19); op(17, 23); op(17, 19);
    op(15, 21); op(15, 17); op(19, 21); op(13, 14); op(15, 16);
    op(17, 18); op(19, 20); op(21, 22); op(23, 24); op(0, 12);
    op(8, 20); op(8, 12); op(4, 16); op(16, 24); op(12, 16);
    op(2, 14); op(10, 22); op(10, 14); op(6, 18); op(6, 10);
    op(10, 12); op(1, 13); op(9, 21); op(9, 13); op(5, 17);
    op(13, 17); op(3, 15); op(11, 23); op(11, 15); op(7, 19);
    op(7, 11); op(11, 13); op(11, 12);
#undef op
    return p[12];
}

#if __AVX2__
// OpenCV-style packed-row AVX2 implementation for U8 PKD (cn==3).
// Processes the row as a linear byte array and uses cn-based offsets, matching
// OpenCV `medianBlur_SortNet` vector path.
inline void rpp_median3x3_packed_u8_avx(const Rpp8u *row0,
                                       const Rpp8u *row1,
                                       const Rpp8u *row2,
                                       Rpp8u *dstRow,
                                       int widthBytes,
                                       int cn)
{
    int j = 0;
    int limit = cn;

    // scalar left edge [0..cn-1]
    for (; j < limit; j++)
    {
        int j0 = j >= cn ? j - cn : j;
        int j2 = j < widthBytes - cn ? j + cn : j;

        int p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        int p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        int p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = (Rpp8u)rpp_median_3x3_sortnet(p);
    }

    const int nlanes = 32;
    for (; j < widthBytes - cn; j += nlanes)
    {
        // handle tail like OpenCV does
        if (j > widthBytes - cn - nlanes)
        {
            if (j == cn || (const Rpp8u *)dstRow == row1) // safety for in-place
                break;
            j = widthBytes - cn - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i *)(row0 + j - cn));
        __m256i p1 = _mm256_loadu_si256((const __m256i *)(row0 + j));
        __m256i p2 = _mm256_loadu_si256((const __m256i *)(row0 + j + cn));
        __m256i p3 = _mm256_loadu_si256((const __m256i *)(row1 + j - cn));
        __m256i p4 = _mm256_loadu_si256((const __m256i *)(row1 + j));
        __m256i p5 = _mm256_loadu_si256((const __m256i *)(row1 + j + cn));
        __m256i p6 = _mm256_loadu_si256((const __m256i *)(row2 + j - cn));
        __m256i p7 = _mm256_loadu_si256((const __m256i *)(row2 + j));
        __m256i p8 = _mm256_loadu_si256((const __m256i *)(row2 + j + cn));

#define OP(a, b)         \
    {                    \
        __m256i t = a;   \
        a = _mm256_min_epu8(a, b); \
        b = _mm256_max_epu8(b, t); \
    }

        // OpenCV m==3 vector sequence (same as scalar, but vectorized)
        OP(p1, p2); OP(p4, p5); OP(p7, p8); OP(p0, p1);
        OP(p3, p4); OP(p6, p7); OP(p1, p2); OP(p4, p5);
        OP(p7, p8); OP(p0, p3); OP(p5, p8); OP(p4, p7);
        OP(p3, p6); OP(p1, p4); OP(p2, p5); OP(p4, p7);
        OP(p4, p2); OP(p6, p4); OP(p4, p2);

#undef OP

        _mm256_storeu_si256((__m256i *)(dstRow + j), p4);
    }

    // scalar tail/right edge
    for (; j < widthBytes; j++)
    {
        int j0 = j >= cn ? j - cn : j;
        int j2 = j < widthBytes - cn ? j + cn : j;

        int p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        int p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        int p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

        int p[9] = {p0, p1, p2, p3, p4, p5, p6, p7, p8};
        dstRow[j] = (Rpp8u)rpp_median_3x3_sortnet(p);
    }
}

inline void rpp_median5x5_packed_u8_avx(const Rpp8u *row0,
                                       const Rpp8u *row1,
                                       const Rpp8u *row2,
                                       const Rpp8u *row3,
                                       const Rpp8u *row4,
                                       Rpp8u *dstRow,
                                       int widthBytes,
                                       int cn)
{
    int j = 0;
    int limit = cn * 2;

    // scalar left edge [0..2*cn-1]
    for (; j < limit; j++)
    {
        int j1 = j >= cn ? j - cn : j;
        int j0 = j >= cn * 2 ? j - cn * 2 : j1;
        int j3 = j < widthBytes - cn ? j + cn : j;
        int j4 = j < widthBytes - cn * 2 ? j + cn * 2 : j3;

        int p[25] = {
            row0[j0], row0[j1], row0[j], row0[j3], row0[j4],
            row1[j0], row1[j1], row1[j], row1[j3], row1[j4],
            row2[j0], row2[j1], row2[j], row2[j3], row2[j4],
            row3[j0], row3[j1], row3[j], row3[j3], row3[j4],
            row4[j0], row4[j1], row4[j], row4[j3], row4[j4]};
        dstRow[j] = (Rpp8u)rpp_median_5x5_sortnet(p);
    }

    const int nlanes = 32;
    for (; j < widthBytes - cn * 2; j += nlanes)
    {
        if (j > widthBytes - cn * 2 - nlanes)
        {
            if (j == cn * 2 || (const Rpp8u *)dstRow == row2) // safety for in-place
                break;
            j = widthBytes - cn * 2 - nlanes;
        }

        __m256i p0 = _mm256_loadu_si256((const __m256i *)(row0 + j - cn * 2));
        __m256i p5 = _mm256_loadu_si256((const __m256i *)(row1 + j - cn * 2));
        __m256i p10 = _mm256_loadu_si256((const __m256i *)(row2 + j - cn * 2));
        __m256i p15 = _mm256_loadu_si256((const __m256i *)(row3 + j - cn * 2));
        __m256i p20 = _mm256_loadu_si256((const __m256i *)(row4 + j - cn * 2));

        __m256i p1 = _mm256_loadu_si256((const __m256i *)(row0 + j - cn));
        __m256i p6 = _mm256_loadu_si256((const __m256i *)(row1 + j - cn));
        __m256i p11 = _mm256_loadu_si256((const __m256i *)(row2 + j - cn));
        __m256i p16 = _mm256_loadu_si256((const __m256i *)(row3 + j - cn));
        __m256i p21 = _mm256_loadu_si256((const __m256i *)(row4 + j - cn));

        __m256i p2 = _mm256_loadu_si256((const __m256i *)(row0 + j));
        __m256i p7 = _mm256_loadu_si256((const __m256i *)(row1 + j));
        __m256i p12 = _mm256_loadu_si256((const __m256i *)(row2 + j));
        __m256i p17 = _mm256_loadu_si256((const __m256i *)(row3 + j));
        __m256i p22 = _mm256_loadu_si256((const __m256i *)(row4 + j));

        __m256i p3 = _mm256_loadu_si256((const __m256i *)(row0 + j + cn));
        __m256i p8 = _mm256_loadu_si256((const __m256i *)(row1 + j + cn));
        __m256i p13 = _mm256_loadu_si256((const __m256i *)(row2 + j + cn));
        __m256i p18 = _mm256_loadu_si256((const __m256i *)(row3 + j + cn));
        __m256i p23 = _mm256_loadu_si256((const __m256i *)(row4 + j + cn));

        __m256i p4 = _mm256_loadu_si256((const __m256i *)(row0 + j + cn * 2));
        __m256i p9 = _mm256_loadu_si256((const __m256i *)(row1 + j + cn * 2));
        __m256i p14 = _mm256_loadu_si256((const __m256i *)(row2 + j + cn * 2));
        __m256i p19 = _mm256_loadu_si256((const __m256i *)(row3 + j + cn * 2));
        __m256i p24 = _mm256_loadu_si256((const __m256i *)(row4 + j + cn * 2));

#define OP(a, b)         \
    {                    \
        __m256i t = a;   \
        a = _mm256_min_epu8(a, b); \
        b = _mm256_max_epu8(b, t); \
    }

        // OpenCV m==5 vector sequence
        OP(p1, p2); OP(p0, p1); OP(p1, p2); OP(p4, p5); OP(p3, p4);
        OP(p4, p5); OP(p0, p3); OP(p2, p5); OP(p2, p3); OP(p1, p4);
        OP(p1, p2); OP(p3, p4); OP(p7, p8); OP(p6, p7); OP(p7, p8);
        OP(p10, p11); OP(p9, p10); OP(p10, p11); OP(p6, p9); OP(p8, p11);
        OP(p8, p9); OP(p7, p10); OP(p7, p8); OP(p9, p10); OP(p0, p6);
        OP(p4, p10); OP(p4, p6); OP(p2, p8); OP(p2, p4); OP(p6, p8);
        OP(p1, p7); OP(p5, p11); OP(p5, p7); OP(p3, p9); OP(p3, p5);
        OP(p7, p9); OP(p1, p2); OP(p3, p4); OP(p5, p6); OP(p7, p8);
        OP(p9, p10); OP(p13, p14); OP(p12, p13); OP(p13, p14); OP(p16, p17);
        OP(p15, p16); OP(p16, p17); OP(p12, p15); OP(p14, p17); OP(p14, p15);
        OP(p13, p16); OP(p13, p14); OP(p15, p16); OP(p19, p20); OP(p18, p19);
        OP(p19, p20); OP(p21, p22); OP(p23, p24); OP(p21, p23); OP(p22, p24);
        OP(p22, p23); OP(p18, p21); OP(p20, p23); OP(p20, p21); OP(p19, p22);
        OP(p22, p24); OP(p19, p20); OP(p21, p22); OP(p23, p24); OP(p12, p18);
        OP(p16, p22); OP(p16, p18); OP(p14, p20); OP(p20, p24); OP(p14, p16);
        OP(p18, p20); OP(p22, p24); OP(p13, p19); OP(p17, p23); OP(p17, p19);
        OP(p15, p21); OP(p15, p17); OP(p19, p21); OP(p13, p14); OP(p15, p16);
        OP(p17, p18); OP(p19, p20); OP(p21, p22); OP(p23, p24); OP(p0, p12);
        OP(p8, p20); OP(p8, p12); OP(p4, p16); OP(p16, p24); OP(p12, p16);
        OP(p2, p14); OP(p10, p22); OP(p10, p14); OP(p6, p18); OP(p6, p10);
        OP(p10, p12); OP(p1, p13); OP(p9, p21); OP(p9, p13); OP(p5, p17);
        OP(p13, p17); OP(p3, p15); OP(p11, p23); OP(p11, p15); OP(p7, p19);
        OP(p7, p11); OP(p11, p13); OP(p11, p12);

#undef OP

        _mm256_storeu_si256((__m256i *)(dstRow + j), p12);
    }

    // scalar tail/right edge
    for (; j < widthBytes; j++)
    {
        int j1 = j >= cn ? j - cn : j;
        int j0 = j >= cn * 2 ? j - cn * 2 : j1;
        int j3 = j < widthBytes - cn ? j + cn : j;
        int j4 = j < widthBytes - cn * 2 ? j + cn * 2 : j3;

        int p[25] = {
            row0[j0], row0[j1], row0[j], row0[j3], row0[j4],
            row1[j0], row1[j1], row1[j], row1[j3], row1[j4],
            row2[j0], row2[j1], row2[j], row2[j3], row2[j4],
            row3[j0], row3[j1], row3[j], row3[j3], row3[j4],
            row4[j0], row4[j1], row4[j], row4[j3], row4[j4]};
        dstRow[j] = (Rpp8u)rpp_median_5x5_sortnet(p);
    }
}
#endif // __AVX2__

template <typename T>
inline void median_filter_3x3_sortnet_tensor(T *srcPtrTemp,
                                            T *dstPtrTemp,
                                            Rpp32s rowIdx,
                                            Rpp32s colIdx,
                                            Rpp32s heightLimit,
                                            Rpp32s widthLimit,
                                            Rpp32s channels,
                                            RpptDescPtr srcDescPtr)
{
    using WT = std::conditional_t<std::is_integral<T>::value, int, T>;

    T blockData[9 * 4]; // channels <= 4
    Rpp32s index = 0;

    for (Rpp32s i = -1; i <= 1; i++)
    {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -1; j <= 1; j++)
        {
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
            else
            {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], channels * sizeof(T));
                index += channels;
            }
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++)
    {
        WT p[9];
        for (Rpp32s k = 0; k < 9; k++)
            p[k] = (WT)blockData[k * channels + ch];

        WT med = rpp_median_3x3_sortnet(p);
        dstPtrTemp[ch] = (T)med;
    }
}

template <typename T>
inline void median_filter_5x5_sortnet_tensor(T *srcPtrTemp,
                                            T *dstPtrTemp,
                                            Rpp32s rowIdx,
                                            Rpp32s colIdx,
                                            Rpp32s heightLimit,
                                            Rpp32s widthLimit,
                                            Rpp32s channels,
                                            RpptDescPtr srcDescPtr)
{
    using WT = std::conditional_t<std::is_integral<T>::value, int, T>;

    T blockData[25 * 4]; // channels <= 4
    Rpp32s index = 0;

    for (Rpp32s i = -2; i <= 2; i++)
    {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -2; j <= 2; j++)
        {
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
            else
            {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], channels * sizeof(T));
                index += channels;
            }
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++)
    {
        WT p[25];
        for (Rpp32s k = 0; k < 25; k++)
            p[k] = (WT)blockData[k * channels + ch];

        WT med = rpp_median_5x5_sortnet(p);
        dstPtrTemp[ch] = (T)med;
    }
}

// Generic median filter implementation
template<typename T>
inline void median_filter_generic_tensor(T *srcPtrTemp, T *dstPtrTemp, Rpp32s rowIdx, Rpp32s colIdx, Rpp32s kernelSizeSquared, Rpp32s padLength, Rpp32s heightLimit, Rpp32s widthLimit, Rpp32s channels, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
   // Temporary buffer to hold kernel window data for all channels
    T blockData[kernelSizeSquared * channels];
    Rpp32s index = 0, medianIndex = kernelSizeSquared / 2;

    // Fill blockData with padded values from the source image using nearest neighbor padding
    for (Rpp32s i = -padLength; i <= padLength; i++)
    {
        Rpp32s row = std::max(0, std::min(rowIdx + i, heightLimit));
        for (Rpp32s j = -padLength; j <= padLength; j++)
        {
            // Clamp the row and column to image boundaries (nearest-neighbor padding)
            Rpp32s col = std::max(0, std::min(colIdx + j, widthLimit));

            // Compute the index for the pixel in the input tensor
            Rpp32u srcIdx = row * srcDescPtr->strides.hStride + col * srcDescPtr->strides.wStride;

            // Copy pixel values for all channels
            if (channels == 3)
            {
                memcpy(&blockData[index], &srcPtrTemp[srcIdx], 3 * sizeof(T));
                index += 3;
            }
            else if (channels == 1)
                blockData[index++] = srcPtrTemp[srcIdx];
        }
    }

    for (Rpp32s ch = 0; ch < channels; ch++)
    {
        // Temporary buffer for the current channel's data in the kernel window
        T channelBlock[kernelSizeSquared];

        // Extract channel data from interleaved blockData
        for (Rpp32s i = 0; i < kernelSizeSquared; i++)
            channelBlock[i] = blockData[i * channels + ch];

        // Sort the data to compute median
        std::nth_element(channelBlock, channelBlock + medianIndex, channelBlock + kernelSizeSquared);
        // Assign the median value to the destination tensor
        dstPtrTemp[ch] = channelBlock[medianIndex];
    }
}

// Host function for median filter
template<typename T>
RppStatus median_filter_generic_host_tensor(T *srcPtr,
                                            RpptDescPtr srcDescPtr,
                                            T *dstPtr,
                                            RpptDescPtr dstDescPtr,
                                            Rpp32u kernelSize,
                                            RpptROIPtr roiTensorPtrSrc,
                                            RpptRoiType roiType,
                                            RppLayoutParams layoutParams,
                                            rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(Rpp32s batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        T *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s kernelSizeSquared = kernelSize * kernelSize;
        Rpp32s padLength = kernelSize / 2;
        bool useSortNet3 = (kernelSize == 3);
        bool useSortNet5 = (kernelSize == 5);

        if((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for(Rpp32s c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for(Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for(Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        if (useSortNet3)
                            median_filter_3x3_sortnet_tensor(srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                        else if (useSortNet5)
                            median_filter_5x5_sortnet_tensor(srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                        else
                            median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                        dstPtrTemp++;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
#if __AVX2__
            // OpenCV-style packed-row AVX2 for U8 PKD3
            if (std::is_same<T, Rpp8u>::value && srcDescPtr->c == 3 && (useSortNet3 || useSortNet5))
            {
                const int cn = 3;
                const int widthBytes = roi.xywhROI.roiWidth * cn;
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    const Rpp8u *row0 = (const Rpp8u *)srcPtrChannel + std::max(i - (useSortNet5 ? 2 : 1), 0) * srcDescPtr->strides.hStride;
                    const Rpp8u *row1 = (const Rpp8u *)srcPtrChannel + std::max(i - (useSortNet5 ? 1 : 0), 0) * srcDescPtr->strides.hStride;
                    const Rpp8u *row2 = (const Rpp8u *)srcPtrChannel + i * srcDescPtr->strides.hStride;

                    Rpp8u *dstRow = (Rpp8u *)dstPtrChannel + i * dstDescPtr->strides.hStride;

                    if (useSortNet3)
                    {
                        const Rpp8u *r0 = (const Rpp8u *)srcPtrChannel + std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                        const Rpp8u *r1 = (const Rpp8u *)srcPtrChannel + i * srcDescPtr->strides.hStride;
                        const Rpp8u *r2 = (const Rpp8u *)srcPtrChannel + std::min(i + 1, roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
                        rpp_median3x3_packed_u8_avx(r0, r1, r2, dstRow, widthBytes, cn);
                    }
                    else
                    {
                        const Rpp8u *r0 = (const Rpp8u *)srcPtrChannel + std::max(i - 2, 0) * srcDescPtr->strides.hStride;
                        const Rpp8u *r1 = (const Rpp8u *)srcPtrChannel + std::max(i - 1, 0) * srcDescPtr->strides.hStride;
                        const Rpp8u *r2 = (const Rpp8u *)srcPtrChannel + i * srcDescPtr->strides.hStride;
                        const Rpp8u *r3 = (const Rpp8u *)srcPtrChannel + std::min(i + 1, roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
                        const Rpp8u *r4 = (const Rpp8u *)srcPtrChannel + std::min(i + 2, roi.xywhROI.roiHeight - 1) * srcDescPtr->strides.hStride;
                        rpp_median5x5_packed_u8_avx(r0, r1, r2, r3, r4, dstRow, widthBytes, cn);
                    }
                }
            }
            else
#endif
            {
                // Scalar fallback (all types and layouts)
                T *dstPtrRow = dstPtrChannel;
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        if (useSortNet3)
                            median_filter_3x3_sortnet_tensor(srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr);
                        else if (useSortNet5)
                            median_filter_5x5_sortnet_tensor(srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr);
                        else
                            median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, srcDescPtr->c, srcDescPtr, dstDescPtr);
                        dstPtrTemp += dstDescPtr->c;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            T *dstPtrRow = dstPtrChannel;
            for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                T *dstPtrTemp = dstPtrRow;
                for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                {
                    T *dstPtrTempChn = dstPtrTemp;
                    T *srcPtrTempChn = srcPtrChannel;
                    for (Rpp32s c = 0; c < srcDescPtr->c; c++)
                    {
                        if (useSortNet3)
                            median_filter_3x3_sortnet_tensor(srcPtrTempChn, dstPtrTempChn, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                        else if (useSortNet5)
                            median_filter_5x5_sortnet_tensor(srcPtrTempChn, dstPtrTempChn, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                        else
                            median_filter_generic_tensor(srcPtrTempChn, dstPtrTempChn, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn++;
                    }
                    dstPtrTemp += dstDescPtr->c;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            for (Rpp32s c = 0; c < srcDescPtr->c; c++)
            {
                T *dstPtrRow = dstPtrChannel;
                for (Rpp32s i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    T *dstPtrTemp = dstPtrRow;
                    for (Rpp32s j = 0; j < roi.xywhROI.roiWidth; j++)
                    {
                        if (useSortNet3)
                            median_filter_3x3_sortnet_tensor(srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                        else if (useSortNet5)
                            median_filter_5x5_sortnet_tensor(srcPtrChannel, dstPtrTemp, i, j, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr);
                        else
                            median_filter_generic_tensor(srcPtrChannel, dstPtrTemp, i, j, kernelSizeSquared, padLength, roi.xywhROI.roiHeight - 1, roi.xywhROI.roiWidth - 1, 1, srcDescPtr, dstDescPtr);
                        dstPtrTemp ++;
                    }
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }
    return RPP_SUCCESS;
}

template RppStatus median_filter_generic_host_tensor<Rpp8u>(Rpp8u*,
                                                            RpptDescPtr,
                                                            Rpp8u*,
                                                            RpptDescPtr,
                                                            Rpp32u,
                                                            RpptROIPtr,
                                                            RpptRoiType,
                                                            RppLayoutParams,
                                                            rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp8s>(Rpp8s*,
                                                            RpptDescPtr,
                                                            Rpp8s*,
                                                            RpptDescPtr,
                                                            Rpp32u,
                                                            RpptROIPtr,
                                                            RpptRoiType,
                                                            RppLayoutParams,
                                                            rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp32f>(Rpp32f*,
                                                             RpptDescPtr,
                                                             Rpp32f*,
                                                             RpptDescPtr,
                                                             Rpp32u,
                                                             RpptROIPtr,
                                                             RpptRoiType,
                                                             RppLayoutParams,
                                                             rpp::Handle&);

template RppStatus median_filter_generic_host_tensor<Rpp16f>(Rpp16f*,
                                                             RpptDescPtr,
                                                             Rpp16f*,
                                                             RpptDescPtr,
                                                             Rpp32u,
                                                             RpptROIPtr,
                                                             RpptRoiType,
                                                             RppLayoutParams,
                                                             rpp::Handle&);
