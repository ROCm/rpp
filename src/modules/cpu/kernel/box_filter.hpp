/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

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

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

const __m128i xmm_pxMaskRotate0To1 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
const __m128i xmm_pxMaskRotate0To3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
const __m128i xmm_pxMaskRotate0To5 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5);
const __m128i xmm_pxMaskRotate0To7 = _mm_setr_epi8(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
const __m128i xmm_pxMaskRotate0To9 = _mm_setr_epi8(10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
const __m128i xmm_pxMaskRotate0To11 = _mm_setr_epi8(12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
const __m128i xmm_pxMaskRotate0To13 = _mm_setr_epi8(14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);

const __m256i avx_pxMaskRotate0To1 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
const __m256i avx_pxMaskRotate0To2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
const __m256i avx_pxMaskRotate0To3 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);
const __m256i avx_pxMaskRotate0To4 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
const __m256i avx_pxMaskRotate0To5 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);
const __m256i avx_pxMaskRotate0To6 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
const __m256i avx_pxMaskRotate0To7 = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);

template<typename T>
inline void increment_row_ptrs(T **srcPtrTemp, Rpp32u kernelSize, Rpp32s increment)
{
    for (int i = 0; i < kernelSize; i++)
        srcPtrTemp[i] += increment;
}

inline void get_kernel_loop_limit(Rpp32s index, Rpp32s &loopLimit, Rpp32u kernelSize, Rpp32u padLength, Rpp32u length)
{
    if ((index < padLength) || (index >= (length - padLength)))
    {  
        Rpp32u factor = (index < padLength) ? index : (length - 1 - index);
        loopLimit = kernelSize - padLength + factor;
    }
}

template<typename T>
inline void box_filter_generic_host_tensor(T **srcPtrTemp, T *dstPtrTemp, Rpp32u columnIndex,
                                           Rpp32u kernelSize, Rpp32u padLength, Rpp32u width, Rpp32s rowKernelLoopLimit,
                                           Rpp32f kernelSizeInverseSquare, Rpp32u channels = 1)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on rowIndex, columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, kernelSize, padLength, width);
    for (int i = 0; i < rowKernelLoopLimit; i++)
        for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            accum += static_cast<Rpp32f>(srcPtrTemp[i][k]);

    accum *= kernelSizeInverseSquare;
    rpp_pixel_check_and_store(accum, dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 box, process them separately
template<typename T>
inline void process_left_border_columns_pln_pln(T **srcPtrTemp, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength, 
                                                           Rpp32u width, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int k = 0; k < padLength; k++)
    {
        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, width, rowKernelLoopLimit, kernelSizeInverseSquare);
        dstPtrTemp++;
    }
}

template<typename T>
inline void process_left_border_columns_pkd_pkd(T **srcPtrTemp, T **srcPtrRow, T *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength, 
                                                           Rpp32u width, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int c = 0; c < 3; c++)
    {
        Rpp32f *dstPtrTempChannel = dstPtrTemp + c;
        for (int k = 0; k < padLength; k++)
        {
            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength, width, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
            dstPtrTempChannel += 3;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }
    
    // reset source to initial position
    for (int k = 0; k < kernelSize; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}

template<typename T>
inline void process_left_border_columns_pkd_pln(T **srcPtrTemp, T **srcPtrRow, T **dstPtrTempChannels, Rpp32u kernelSize, Rpp32u padLength, 
                                                           Rpp32u width, Rpp32s rowKernelLoopLimit, Rpp32f kernelSizeInverseSquare)
{
    for (int c = 0; c < 3; c++)
    {
        for (int k = 0; k < padLength; k++)
        {
            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength, width, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
            dstPtrTempChannels[c] += 1;
        }
        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
    }

    // reset source to initial position
    for (int k = 0; k < 9; k++)
        srcPtrTemp[k] = srcPtrRow[k];
}


// -------------------- Set 0 box_filter compute functions --------------------

// -------------------- kernel size 3x3 - U8 bitdepth compute functions --------------------

inline void compute_box_filter_u8_u8_3x3_24_host_pln(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // unpack lower half of each of 3 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));

    // unpack higher half of each of 3 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 =  _mm256_castsi256_si128(pxLower);
    pxLower2 =  _mm256_castsi256_si128(pxUpper);
    pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 =  _mm256_extracti128_si256(pxUpper, 1);

    // perform blend and shuffle operations for the first 8 output values to get required order and add them
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskRotate0To3);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 3), xmm_pxMaskRotate0To3);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // pack 16 bit values to 8 bit values using unsigned saturation and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

inline void compute_box_filter_u8_u8_3x3_24_host_pkd(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // unpack lower half of each of 3 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));

    // unpack higher half of each of 3 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 =  _mm256_castsi256_si128(pxLower);
    pxLower2 =  _mm256_castsi256_si128(pxUpper);
    pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 =  _mm256_extracti128_si256(pxUpper, 1);

    // perform blend and shuffle operations for the first 8 output values to get required order and add them
    __m128i pxTemp[2];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskRotate0To11);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);

    // perform blend and shuffle operations for the next 8 output values to get required order and add them
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 63), xmm_pxMaskRotate0To11);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // pack 16 bit values to 8 bit values using unsigned saturation and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

// -------------------- 5x5 kernel size - U8 bitdepth compute functions --------------------

inline void compute_box_filter_u8_u8_5x5_24_host_pln(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // unpack lower half of each of 7 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[4], avx_px0));

    // unpack higher half of each of 7 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[4], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 = _mm256_castsi256_si128(pxLower);
    pxLower2 = _mm256_castsi256_si128(pxUpper);
    pxUpper1 = _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 = _mm256_extracti128_si256(pxUpper, 1);

    // get the final accumalated result for first 8 elements
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 15), xmm_pxMaskRotate0To7);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[2]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[3]);

    // get the final accumalated result for next 8 elements
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 15), xmm_pxMaskRotate0To7);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[2]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[3]);

    // get the final accumalated result for last 8 elements
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 15), xmm_pxMaskRotate0To7);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[2]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[3]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // pack 16 bit values to 8 bit values using unsigned saturation and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

inline void compute_box_filter_u8_u8_5x5_18_host_pkd(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // unpack lower half of each of 7 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[4], avx_px0));

    // unpack higher half of each of 7 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[4], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 =  _mm256_castsi256_si128(pxLower);
    pxLower2 =  _mm256_castsi256_si128(pxUpper);
    pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 =  _mm256_extracti128_si256(pxUpper, 1);

    // perform blend and shuffle operations for the first 8 output values to get required order and add them
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 15), xmm_pxMaskRotate0To7);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[2]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[3]);

    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 1), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 15), xmm_pxMaskRotate0To7);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[2]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[3]);

    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 63), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 127), xmm_pxMaskRotate0To7);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[2]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[3]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // pack 16 bit values to 8 bit values using unsigned saturation and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

// -------------------- 7x7 kernel size - U8 bitdepth compute functions --------------------

inline void compute_box_filter_u8_u8_7x7_24_host_pln(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    // unpack lower half of each of 7 loaded row values from 8 bit to 16 bit and add
    __m256i pxLower, pxUpper;
    pxLower = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[4], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[5], avx_px0));
    pxLower = _mm256_add_epi16(pxLower, _mm256_unpacklo_epi8(pxRow[6], avx_px0));

    // unpack higher half of each of 7 loaded row values from 8 bit to 16 bit and add
    pxUpper = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[4], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[5], avx_px0));
    pxUpper = _mm256_add_epi16(pxUpper, _mm256_unpackhi_epi8(pxRow[6], avx_px0));

    // get 4 SSE registers from above 2 AVX registers to arrange as per required order
    __m128i pxLower1, pxLower2, pxUpper1, pxUpper2;
    pxLower1 = _mm256_castsi256_si128(pxLower);
    pxLower2 = _mm256_castsi256_si128(pxUpper);
    pxUpper1 = _mm256_extracti128_si256(pxLower, 1);
    pxUpper2 = _mm256_extracti128_si256(pxUpper, 1);

    // get the final accumalated result for first 8 elements
    __m128i pxTemp[6];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[2]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[3]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[4]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[5]);

    // get the final accumalated result for next 8 elements
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskRotate0To11);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[2]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[3]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[4]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[5]);

    // get the final accumalated result for last 8 elements
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxUpper1, pxUpper2, 63), xmm_pxMaskRotate0To11);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[0]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[1]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[2]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[3]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[4]);
    pxUpper1 = _mm_add_epi16(pxUpper1, pxTemp[5]);

    // multiply with convolution factor
    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxUpper1 = _mm_mulhi_epi16(pxUpper1, pxConvolutionFactor);

    // pack 16 bit values to 8 bit values using unsigned saturation and store in resultant registers
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
    pxDst[1] = _mm_packus_epi16(pxUpper1, xmm_px0);
}

// -------------------- 9x9 kernel size - U8 bitdepth compute functions --------------------

inline void unpacklo_and_add_9x9_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpacklo_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[4], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[5], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[6], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[7], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpacklo_epi8(pxRow[8], avx_px0));
}

inline void unpackhi_and_add_9x9_host(__m256i *pxRow, __m256i *pxDst)
{
    pxDst[0] = _mm256_unpackhi_epi8(pxRow[0], avx_px0);
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[1], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[2], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[3], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[4], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[5], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[6], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[7], avx_px0));
    pxDst[0] = _mm256_add_epi16(pxDst[0], _mm256_unpackhi_epi8(pxRow[8], avx_px0));
}

inline void blend_shuffle_add_9x9_pkd_host(__m128i *px128)
{
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 7), xmm_pxMaskRotate0To5);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(px128[0], px128[1], 63), xmm_pxMaskRotate0To11);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 1), xmm_pxMaskRotate0To1);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(px128[1], px128[2], 127), xmm_pxMaskRotate0To13);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(px128[2], px128[3], 3), xmm_pxMaskRotate0To3);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(px128[2], px128[3], 31), xmm_pxMaskRotate0To9);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[0]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[1]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[2]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[3]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[4]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[5]);
    px128[0] = _mm_add_epi16(px128[0], pxTemp[6]);
    px128[0] = _mm_add_epi16(px128[0], px128[3]);
}

inline void compute_box_filter_9x9_16_host_pln(__m256i *pxRow, __m128i *pxDst, const __m128i &pxConvolutionFactor)
{
    __m256i pxLower, pxUpper;
    unpacklo_and_add_9x9_host(pxRow, &pxLower);
    unpackhi_and_add_9x9_host(pxRow, &pxUpper);

    __m128i pxLower1, pxLower2, pxUpper1;
    pxLower1 =  _mm256_castsi256_si128(pxLower);
    pxLower2 =  _mm256_castsi256_si128(pxUpper);
    pxUpper1 =  _mm256_extracti128_si256(pxLower, 1);

    // get the final accumalated result for first 8 elements
    __m128i pxTemp[7];
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 63), xmm_pxMaskRotate0To11);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower1, pxLower2, 127), xmm_pxMaskRotate0To13);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[0]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[1]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[2]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[3]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[4]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[5]);
    pxLower1 = _mm_add_epi16(pxLower1, pxTemp[6]);
    pxLower1 = _mm_add_epi16(pxLower1, pxLower2);

    // get the final accumalated result for next 8 elements
    pxTemp[0] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 1), xmm_pxMaskRotate0To1);
    pxTemp[1] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 3), xmm_pxMaskRotate0To3);
    pxTemp[2] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 7), xmm_pxMaskRotate0To5);
    pxTemp[3] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 15), xmm_pxMaskRotate0To7);
    pxTemp[4] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 31), xmm_pxMaskRotate0To9);
    pxTemp[5] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 63), xmm_pxMaskRotate0To11);
    pxTemp[6] = _mm_shuffle_epi8(_mm_blend_epi16(pxLower2, pxUpper1, 127), xmm_pxMaskRotate0To13);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[0]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[1]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[2]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[3]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[4]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[5]);
    pxLower2 = _mm_add_epi16(pxLower2, pxTemp[6]);
    pxLower2 = _mm_add_epi16(pxLower2, pxUpper1);

    pxLower1 = _mm_mulhi_epi16(pxLower1, pxConvolutionFactor);
    pxLower2 = _mm_mulhi_epi16(pxLower2, pxConvolutionFactor);
    pxDst[0] = _mm_packus_epi16(pxLower1, pxLower2);
}

// -------------------- 3x3 kernel size - F32 bitdepth compute functions --------------------

inline void add_rows_3x3(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(pRow[0], pRow[1]);
    pDst[0] = _mm256_add_ps(pDst[0], pRow[2]);
}

inline void blend_permute_add_3x3_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));  
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_3x3_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));  
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6)); 
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// -------------------- 5x5 kernel size - F32 bitdepth compute functions --------------------

inline void add_rows_5x5(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(pRow[3], pRow[4]));
}

inline void blend_permute_add_5x5_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));  
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_5x5_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));  
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 1), avx_pxMaskRotate0To1));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// -------------------- 7x7 kernel size - F32 bitdepth compute functions --------------------

inline void add_rows_7x7(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pRow[3], pRow[4]), pRow[5]));
    pDst[0] = _mm256_add_ps(pDst[0], pRow[6]);
}

inline void blend_permute_add_7x7_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2));  
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 15), avx_pxMaskRotate0To4));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 31), avx_pxMaskRotate0To5));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6));
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

inline void blend_permute_add_7x7_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    pDst[0] = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));  
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 1), avx_pxMaskRotate0To1)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 15), avx_pxMaskRotate0To4)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 127), avx_pxMaskRotate0To7)); 
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[2], pSrc[3], 3), avx_pxMaskRotate0To2)); 
    pDst[0] = _mm256_mul_ps(pDst[0], pConvolutionFactor);
}

// -------------------- 9x9 kernel size - F32 bitdepth compute functions --------------------

inline void add_rows_9x9(__m256 *pRow, __m256 *pDst)
{
    pDst[0] = _mm256_add_ps(_mm256_add_ps(pRow[0], pRow[1]), pRow[2]);
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pRow[3], pRow[4]), pRow[5]));
    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pRow[6], pRow[7]), pRow[8]));
}

inline void blend_permute_add_9x9_pln(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    __m256 pTemp;
    pTemp = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 1), avx_pxMaskRotate0To1));  
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 3), avx_pxMaskRotate0To2)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 15), avx_pxMaskRotate0To4)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 31), avx_pxMaskRotate0To5)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 127), avx_pxMaskRotate0To7));
    pTemp = _mm256_add_ps(pTemp, pSrc[1]);
    pDst[0] = _mm256_mul_ps(pTemp, pConvolutionFactor);
}

inline void blend_permute_add_9x9_pkd(__m256 *pSrc, __m256 *pDst, __m256 pConvolutionFactor)
{
    __m256 pTemp;
    pTemp = _mm256_add_ps(pSrc[0], _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 7), avx_pxMaskRotate0To3));  
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[0], pSrc[1], 63), avx_pxMaskRotate0To6)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 1), avx_pxMaskRotate0To1)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 15), avx_pxMaskRotate0To4)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[1], pSrc[2], 127), avx_pxMaskRotate0To7)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[2], pSrc[3], 3), avx_pxMaskRotate0To2)); 
    pTemp = _mm256_add_ps(pTemp, _mm256_permutevar8x32_ps(_mm256_blend_ps(pSrc[2], pSrc[3], 31), avx_pxMaskRotate0To5));
    pTemp = _mm256_add_ps(pTemp, pSrc[3]);
    pDst[0] = _mm256_mul_ps(pTemp, pConvolutionFactor);
}

// -------------------- Set 1 box_filter load functions --------------------

// 3x3 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_u8_u8_3x3_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    else
        pxRow[2] = avx_px0;
}

// 5x5 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_u8_u8_5x5_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
        pxRow[k] = avx_px0;
}

// 7x7 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_u8_u8_7x7_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    pxRow[3] = _mm256_loadu_si256((__m256i *)srcPtrTemp[3]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
        pxRow[k] = avx_px0;
}

// 9x9 kernel loads for U8 bitdepth
inline void rpp_load_box_filter_u8_u8_9x9_host(__m256i *pxRow, Rpp8u **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pxRow[0] = _mm256_loadu_si256((__m256i *)srcPtrTemp[0]);
    pxRow[1] = _mm256_loadu_si256((__m256i *)srcPtrTemp[1]);
    pxRow[2] = _mm256_loadu_si256((__m256i *)srcPtrTemp[2]);
    pxRow[3] = _mm256_loadu_si256((__m256i *)srcPtrTemp[3]);
    pxRow[4] = _mm256_loadu_si256((__m256i *)srcPtrTemp[4]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pxRow[k] = _mm256_loadu_si256((__m256i *)srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
        pxRow[k] = avx_px0;
}

// 3x3 kernel loads for F32 bitdepth
inline void rpp_load_box_filter_f32_f32_3x3_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 2 rows for 3x3 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);

    // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
    if (rowKernelLoopLimit == 3)
        pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    else
        pRow[2] = avx_px0;
        
}

inline void rpp_load_box_filter_f32_f32_5x5_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 3 rows for 5x5 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    for (int k = 3; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 5; k++)
        pRow[k] = avx_p0;
}

inline void rpp_load_box_filter_f32_f32_7x7_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 4 rows for 7x7 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[3]);
    for (int k = 4; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 7; k++)
        pRow[k] = avx_p0;
}

inline void rpp_load_box_filter_f32_f32_9x9_host(__m256 *pRow, Rpp32f **srcPtrTemp, Rpp32s rowKernelLoopLimit)
{
    // irrespective of row location, we need to load 5 rows for 9x9 kernel
    pRow[0] = _mm256_loadu_ps(srcPtrTemp[0]);
    pRow[1] = _mm256_loadu_ps(srcPtrTemp[1]);
    pRow[2] = _mm256_loadu_ps(srcPtrTemp[2]);
    pRow[3] = _mm256_loadu_ps(srcPtrTemp[3]);
    pRow[4] = _mm256_loadu_ps(srcPtrTemp[4]);
    for (int k = 5; k < rowKernelLoopLimit; k++)
        pRow[k] = _mm256_loadu_ps(srcPtrTemp[k]);
    for (int k = rowKernelLoopLimit; k < 9; k++)
        pRow[k] = avx_p0;
}

RppStatus box_filter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
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
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f kernelSizeInverseSquare = 1.0 / (kernelSize * kernelSize);
        Rpp16s convolutionFactor = (Rpp16s) std::ceil(65536 * kernelSizeInverseSquare);
        const __m128i pxConvolutionFactor = _mm_set1_epi16(convolutionFactor);

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            Rpp8u *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        Rpp8u *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                        // process padLength number of columns in each row
                        // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[3];
                            rpp_load_box_filter_u8_u8_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_u8_u8_3x3_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));
                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength * 3; k++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3];
                        rpp_load_box_filter_u8_u8_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m128i pxDst[2];
                        compute_box_filter_u8_u8_3x3_24_host_pkd(pxRow, pxDst, pxConvolutionFactor);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        dstPtrTemp += 24;
                    }
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                Rpp8u *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp8u *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength * 3; k++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[k], k / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    }
                    increment_row_ptrs(dstPtrTempChannels, kernelSize, 1);

                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxRow[3];
                        rpp_load_box_filter_u8_u8_3x3_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m128i pxDst[2];
                        compute_box_filter_u8_u8_3x3_24_host_pkd(pxRow, pxDst, pxConvolutionFactor);

                        // convert from PKD3 to PLN3 and store channelwise
                        __m128i pxDstChn[3];
                        rpp_convert24_u8pkd3_to_u8pln3(pxDst[0], pxDst[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 8);
                    }
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3][3] = {
                                                {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                                {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
                                                {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
                                              };

                    Rpp8u *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[3];
                            rpp_load_box_filter_u8_u8_3x3_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_u8_u8_3x3_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_u8pln3_to_u8pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if (kernelSize == 5)
        {
            Rpp8u *srcPtrRow[5], *dstPtrRow;
            for (int i = 0; i < 5; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    srcPtrRow[3] = srcPtrRow[2] + srcDescPtr->strides.hStride;
                    srcPtrRow[4] = srcPtrRow[3] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for (int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1 : 0;
                        Rpp8u *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                        Rpp8u *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                        // process padLength number of columns in each row
                        // left border pixels in image which does not have required pixels in 5x5 box, process them separately
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }

                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[5];
                            rpp_load_box_filter_u8_u8_5x5_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_u8_u8_5x5_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));
                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
                        vectorLoopCount += padLength;
                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 18) * 18;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength * 3; k++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];
                    srcPtrTemp[3] = srcPtrRow[3];
                    srcPtrTemp[4] = srcPtrRow[4];

                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 18)
                    {
                        __m256i pxRow[3];
                        rpp_load_box_filter_u8_u8_5x5_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m128i pxDst[2];
                        compute_box_filter_u8_u8_5x5_18_host_pkd(pxRow, pxDst, pxConvolutionFactor);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));
                        increment_row_ptrs(srcPtrTemp, kernelSize, 18);
                        dstPtrTemp += 18;
                    }
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 18) * 18;
                Rpp8u *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                    Rpp8u *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0, c = 0; k < padLength * 3; k++, c++)
                    {
                        int channel = c % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }

                    // reset source to initial position
                    srcPtrTemp[0] = srcPtrRow[0];
                    srcPtrTemp[1] = srcPtrRow[1];
                    srcPtrTemp[2] = srcPtrRow[2];
                    srcPtrTemp[3] = srcPtrRow[3];
                    srcPtrTemp[4] = srcPtrRow[4];

                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 18)
                    {
                        __m256i pxRow[5];
                        rpp_load_box_filter_u8_u8_5x5_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m128i pxDst[2];
                        compute_box_filter_u8_u8_5x5_18_host_pkd(pxRow, pxDst, pxConvolutionFactor);

                        // convert from PKD3 to PLN3 and store channelwise
                        __m128i pxDstChn[3];
                        rpp_convert24_u8pkd3_to_u8pln3(pxDst[0], pxDst[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 18);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 6);
                    }
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3][5];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 5; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    Rpp8u *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 5x5 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[5];
                            rpp_load_box_filter_u8_u8_5x5_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_u8_u8_5x5_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_u8pln3_to_u8pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if (kernelSize == 7)
        {
            Rpp8u *srcPtrRow[7], *dstPtrRow;
            for (int i = 0; i < 7; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 7; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for (int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1 : 0;
                        Rpp8u *srcPtrTemp[7];
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        Rpp8u *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                        // process padLength number of columns in each row
                        // left border pixels in image which does not have required pixels in 7x7 box, process them separately
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }

                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                        {
                            __m256i pxRow[7];
                            rpp_load_box_filter_u8_u8_7x7_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_u8_u8_7x7_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1]));
                            increment_row_ptrs(srcPtrTemp, kernelSize, 24);
                            dstPtrTemp += 24;
                        }
                        vectorLoopCount += padLength;
                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3][7];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    Rpp8u *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 7x7 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        __m256i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[7];
                            rpp_load_box_filter_u8_u8_7x7_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);

                            __m128i pxDst[2];
                            compute_box_filter_u8_u8_7x7_24_host_pln(pxRow, pxDst, pxConvolutionFactor);
                            pxResultPln[c] = _mm256_setr_m128i(pxDst[0], pxDst[1]);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 24);
                        }
                        __m128i pxResultPkd[6];
                        // convert result from pln to pkd format and store in output buffer
                        rpp_convert72_u8pln3_to_u8pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)dstPtrTemp, pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 48), pxResultPkd[4]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 60), pxResultPkd[5]);
                        dstPtrTemp += 72;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if (kernelSize == 9)
        {
            Rpp8u *srcPtrRow[9], *dstPtrRow;
            for (int i = 0; i < 9; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 9; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp8u *srcPtrTemp[9];
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        Rpp8u *dstPtrTemp = dstPtrRow;

                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                        // process padLength number of columns in each row
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                        {
                            __m256i pxRow[9];
                            rpp_load_box_filter_u8_u8_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);
                            
                            __m128i pxDst;
                            compute_box_filter_9x9_16_host_pln(pxRow, &pxDst, pxConvolutionFactor);
                            _mm_storeu_si128((__m128i *)dstPtrTemp, pxDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 16);
                            dstPtrTemp += 16;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 64) * 64;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp8u *dstPtrTempChannel = dstPtrTemp + c;
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannel, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                            dstPtrTempChannel += 3;
                        }
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    }
                    dstPtrTemp += padLength * 3;

                    // reset source to initial position
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];

                    // load first 32 elements elements
                    __m256i pxRow[9];
                    if (alignedLength)
                        rpp_load_box_filter_u8_u8_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 32)
                    {
                        __m256i pxLower, pxUpper;
                        unpacklo_and_add_9x9_host(pxRow, &pxLower);
                        unpackhi_and_add_9x9_host(pxRow, &pxUpper);

                        __m128i px128[8];
                        px128[0] =  _mm256_castsi256_si128(pxLower);
                        px128[1] =  _mm256_castsi256_si128(pxUpper);
                        px128[2] =  _mm256_extracti128_si256(pxLower, 1);
                        px128[3] =  _mm256_extracti128_si256(pxUpper, 1);

                        // get the final accumalated result for first 8 elements
                        __m128i pxTemp[7], pxDst[4];
                        blend_shuffle_add_9x9_pkd_host(&px128[0]);
                        pxDst[0] = _mm_mulhi_epi16(px128[0], pxConvolutionFactor);

                        // compute for next 8 elements
                        increment_row_ptrs(srcPtrTemp, kernelSize, 32);
                        rpp_load_box_filter_u8_u8_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);
                        unpacklo_and_add_9x9_host(pxRow, &pxLower);
                        unpackhi_and_add_9x9_host(pxRow, &pxUpper);
                        
                        px128[4] =  _mm256_castsi256_si128(pxLower);
                        px128[5] =  _mm256_castsi256_si128(pxUpper);
                        px128[6] =  _mm256_extracti128_si256(pxLower, 1);
                        px128[7] =  _mm256_extracti128_si256(pxUpper, 1);

                        // get the final accumalated result for next 24 elements
                        blend_shuffle_add_9x9_pkd_host(&px128[1]);
                        blend_shuffle_add_9x9_pkd_host(&px128[2]);
                        blend_shuffle_add_9x9_pkd_host(&px128[3]);
                        pxDst[1] = _mm_mulhi_epi16(px128[1], pxConvolutionFactor);
                        pxDst[2] = _mm_mulhi_epi16(px128[2], pxConvolutionFactor);
                        pxDst[3] = _mm_mulhi_epi16(px128[3], pxConvolutionFactor);
                        pxDst[0] = _mm_packus_epi16(pxDst[0], pxDst[1]);
                        pxDst[1] = _mm_packus_epi16(pxDst[2], pxDst[3]);
                        _mm256_storeu_si256((__m256i *)dstPtrTemp, _mm256_setr_m128i(pxDst[0], pxDst[1])); 
                        dstPtrTemp += 32;
                    }
                    vectorLoopCount += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[3][9];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    Rpp8u *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m128i pxResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256i pxRow[9];
                            rpp_load_box_filter_u8_u8_9x9_host(pxRow, srcPtrTemp[c], rowKernelLoopLimit);
                            compute_box_filter_9x9_16_host_pln(pxRow, &pxResultPln[c], pxConvolutionFactor);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 16);
                        }
                        __m128i pxResultPkd[4];
                        rpp_convert48_u8pln3_to_u8pkd3(pxResultPln, pxResultPkd);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp), pxResultPkd[0]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 12), pxResultPkd[1]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 24), pxResultPkd[2]);
                        _mm_storeu_si128((__m128i *)(dstPtrTemp + 36), pxResultPkd[3]);
                        dstPtrTemp += 48;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 64) * 64;
                Rpp8u *dstPtrChannels[3];
                for (int c = 0; c < 3; c++)
                    dstPtrChannels[c] = dstPtrChannel + c * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp8u *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp8u *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int c = 0; c < 3; c++)
                    {
                        for (int k = 0; k < padLength; k++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[c], k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                            dstPtrTempChannels[c] += 1;
                        }
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                    }

                    // reset source to initial position
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 24)
                    {
                        // load first 32 elements elements
                        __m256i pxRow[9];
                        rpp_load_box_filter_u8_u8_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);

                        __m256i pxLower, pxUpper;
                        unpacklo_and_add_9x9_host(pxRow, &pxLower);
                        unpackhi_and_add_9x9_host(pxRow, &pxUpper);

                        __m128i px128[8];
                        px128[0] =  _mm256_castsi256_si128(pxLower);
                        px128[1] =  _mm256_castsi256_si128(pxUpper);
                        px128[2] =  _mm256_extracti128_si256(pxLower, 1);
                        px128[3] =  _mm256_extracti128_si256(pxUpper, 1);

                        // get the final accumalated result for first 8 elements
                        __m128i pxTemp[7], pxDst[4];
                        blend_shuffle_add_9x9_pkd_host(&px128[0]);
                        pxDst[0] = _mm_mulhi_epi16(px128[0], pxConvolutionFactor);

                        // compute for next 8 elements
                        increment_row_ptrs(srcPtrTemp, kernelSize, 32);
                        rpp_load_box_filter_u8_u8_9x9_host(pxRow, srcPtrTemp, rowKernelLoopLimit);
                        unpacklo_and_add_9x9_host(pxRow, &pxLower);
                        unpackhi_and_add_9x9_host(pxRow, &pxUpper);
                        
                        px128[4] =  _mm256_castsi256_si128(pxLower);
                        px128[5] =  _mm256_castsi256_si128(pxUpper);
                        px128[6] =  _mm256_extracti128_si256(pxLower, 1);
                        px128[7] =  _mm256_extracti128_si256(pxUpper, 1);

                        // get the final accumalated result for next 24 elements
                        blend_shuffle_add_9x9_pkd_host(&px128[1]);
                        blend_shuffle_add_9x9_pkd_host(&px128[2]);
                        blend_shuffle_add_9x9_pkd_host(&px128[3]);
                        pxDst[1] = _mm_mulhi_epi16(px128[1], pxConvolutionFactor);
                        pxDst[2] = _mm_mulhi_epi16(px128[2], pxConvolutionFactor);
                        pxDst[3] = _mm_mulhi_epi16(px128[3], pxConvolutionFactor);
                        pxDst[0] = _mm_packus_epi16(pxDst[0], pxDst[1]);
                        pxDst[1] = _mm_packus_epi16(pxDst[2], pxDst[3]);

                        // convert from PKD3 to PLN3 and store
                        __m128i pxDstChn[3];
                        rpp_convert24_u8pkd3_to_u8pln3(pxDst[0], pxDst[1], pxDstChn);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[0]), pxDstChn[0]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[1]), pxDstChn[1]);
                        rpp_storeu_si64((__m128i *)(dstPtrTempChannels[2]), pxDstChn[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, -8);
                        increment_row_ptrs(dstPtrTempChannels, 3, 8);
                    }
                    vectorLoopCount += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus box_filter_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
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
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp32f *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp32f kernelSizeInverseSquare = 1.0 / (kernelSize * kernelSize);
        const __m128 pConvolutionFactor = _mm_set1_ps(kernelSizeInverseSquare);
        const __m256 pConvolutionFactorAVX = _mm256_set1_ps(kernelSizeInverseSquare);

        Rpp32f *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            Rpp32f *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp32f *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        Rpp32f *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                        {
                            __m256 pRow[3], pTemp[3], pDst[2];
                            rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_3x3(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_3x3(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;

                            blend_permute_add_3x3_pln(&pTemp[0], &pDst[0], pConvolutionFactorAVX);
                            blend_permute_add_3x3_pln(&pTemp[1], &pDst[1], pConvolutionFactorAVX);

                            _mm256_storeu_ps(dstPtrTemp, pDst[0]);
                            _mm256_storeu_ps(dstPtrTemp + 8, pDst[1]);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 6);
                            dstPtrTemp += 14;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;

                    // process remaining columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                    {
                        __m256 pRow[3], pTemp[3], pDst[2];
                        rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_3x3(pRow, &pTemp[0]);
                        
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_3x3(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_3x3(pRow, &pTemp[2]);

                        blend_permute_add_3x3_pkd(&pTemp[0], &pDst[0], pConvolutionFactorAVX);
                        blend_permute_add_3x3_pkd(&pTemp[1], &pDst[1], pConvolutionFactorAVX);

                        _mm256_storeu_ps(dstPtrTemp, pDst[0]);
                        _mm256_storeu_ps(dstPtrTemp + 8,  pDst[1]);
                        dstPtrTemp += 16;
                    }
                    vectorLoopCount += padLength * 3;
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                Rpp32f *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                    Rpp32f *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);

                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        __m256 pRow[3], pTemp[3], pDst[2];
                        rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_3x3(pRow, &pTemp[0]);
                        
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_3x3(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_3x3(pRow, &pTemp[2]);

                        blend_permute_add_3x3_pkd(&pTemp[0], &pDst[0], pConvolutionFactorAVX);
                        blend_permute_add_3x3_pkd(&pTemp[1], &pDst[1], pConvolutionFactorAVX);

                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        _mm_storeu_ps(dstPtrTempChannels[0], pDstPln[0]);
                        _mm_storeu_ps(dstPtrTempChannels[1], pDstPln[1]);
                        _mm_storeu_ps(dstPtrTempChannels[2], pDstPln[2]); 

                        increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                        increment_row_ptrs(dstPtrTempChannels, kernelSize, 4);
                    }
                    vectorLoopCount += padLength * 3;
                    for (int c = 0; vectorLoopCount < bufferLength; vectorLoopCount++, c++)
                    {
                        int channel = c % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, kernelSize, dstDescPtr->strides.hStride);
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3][3] = {
                                                {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]},
                                                {srcPtrRow[0] + srcDescPtr->strides.cStride, srcPtrRow[1] + srcDescPtr->strides.cStride, srcPtrRow[2] + srcDescPtr->strides.cStride},
                                                {srcPtrRow[0] + 2 * srcDescPtr->strides.cStride, srcPtrRow[1] + 2 * srcDescPtr->strides.cStride, srcPtrRow[2] + 2 * srcDescPtr->strides.cStride}
                                              };

                    Rpp32f *dstPtrTemp = dstPtrRow;
                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    // left border pixels in image which does not have required pixels in 3x3 box, process them separately
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 14)
                    {
                        __m256 pResult[6];
                        for (int c = 0; c < 3; c++)
                        {
                            int channelStride = c * 2;
                            __m256 pRow[3], pTemp[3];
                            rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_3x3(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_box_filter_f32_f32_3x3_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_3x3(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;

                            blend_permute_add_3x3_pln(&pTemp[0], &pResult[channelStride], pConvolutionFactorAVX);
                            blend_permute_add_3x3_pln(&pTemp[1], &pResult[channelStride + 1], pConvolutionFactorAVX);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 6);
                        }

                        // convert result from pln to pkd format and store in output buffer
                        rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, pResult);                       
                        dstPtrTemp += 42;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
        }
        else if (kernelSize == 5)
        {
            Rpp32f *srcPtrRow[5], *dstPtrRow;
            for (int i = 0; i < 5; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            
            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 5; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp32f *srcPtrTemp[5] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4]};
                        Rpp32f *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                        {
                            __m256 pRow[5], pDst[2], pTemp[3];
                            rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_5x5(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_5x5(pRow, &pTemp[1]);
                            pTemp[2] = avx_p0;
                            
                            blend_permute_add_5x5_pln(&pTemp[0], &pDst[0], pConvolutionFactorAVX);
                            blend_permute_add_5x5_pln(&pTemp[1], &pDst[1], pConvolutionFactorAVX);

                            _mm256_storeu_ps(dstPtrTemp, pDst[0]);
                            _mm256_storeu_ps(dstPtrTemp + 8, pDst[1]);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 4);
                            dstPtrTemp += 12;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength * 3)) / 24) * 24;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[5];
                    for (int k = 0; k < 5; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        // add loaded values from 9 rows
                        __m256 pRow[5], pDst[2], pTemp[4];
                        rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_5x5(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_5x5(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_5x5(pRow, &pTemp[2]);
                        pTemp[3] = avx_p0;

                        blend_permute_add_5x5_pkd(&pTemp[0], &pDst[0], pConvolutionFactorAVX);
                        blend_permute_add_5x5_pkd(&pTemp[1], &pDst[1], pConvolutionFactorAVX);
                            
                        _mm256_storeu_ps(dstPtrTemp, pDst[0]);
                        _mm256_storeu_ps(dstPtrTemp + 8, pDst[1]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                        dstPtrTemp += 12;
                    }
                    vectorLoopCount += padLength * 3;
                    
                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3][5];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 5; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256 pRow[5], pTemp[2];
                            rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_5x5(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_5x5(pRow, &pTemp[1]);
                            blend_permute_add_5x5_pln(pTemp, &pResultPln[c], pConvolutionFactorAVX);
                        }
                        rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                        dstPtrTemp += 24;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 24) * 24;
                Rpp32f *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[5];
                    for (int k = 0; k < 5; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp32f *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 12)
                    {
                        // add loaded values from 9 rows
                        __m256 pRow[5], pDst[2], pTemp[4];
                        rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_5x5(pRow, &pTemp[0]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_5x5(pRow, &pTemp[1]);

                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_5x5_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_5x5(pRow, &pTemp[2]);
                        pTemp[3] = avx_p0;

                        blend_permute_add_5x5_pkd(&pTemp[0], &pDst[0], pConvolutionFactorAVX);
                        blend_permute_add_5x5_pkd(&pTemp[1], &pDst[1], pConvolutionFactorAVX);
                        
                        __m128 pDstPln[3];
                        rpp_convert12_f32pkd3_to_f32pln3(pDst, pDstPln);
                        _mm_storeu_ps(dstPtrTempChannels[0], pDstPln[0]);
                        _mm_storeu_ps(dstPtrTempChannels[1], pDstPln[1]);
                        _mm_storeu_ps(dstPtrTempChannels[2], pDstPln[2]);
                        increment_row_ptrs(srcPtrTemp, kernelSize, -4);
                        increment_row_ptrs(dstPtrTempChannels, 3, 4);
                    }

                    vectorLoopCount += padLength * 3;
                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
        else if (kernelSize == 7)
        {
            Rpp32f *srcPtrRow[7], *dstPtrRow;
            for (int i = 0; i < 7; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;
            
            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 7; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp32f *srcPtrTemp[7] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2], srcPtrRow[3], srcPtrRow[4], srcPtrRow[5], srcPtrRow[6]}; 
                        Rpp32f *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[7], pTemp[2], pDst;
                            rpp_load_box_filter_f32_f32_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_7x7(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            rpp_load_box_filter_f32_f32_7x7_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_7x7(pRow, &pTemp[1]);
                            blend_permute_add_7x7_pln(&pTemp[0], &pDst, pConvolutionFactorAVX);
                            
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                            dstPtrTemp += 8;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;

                    __m256 pRow1[7], pRow2[7];
                    rpp_load_box_filter_f32_f32_7x7_host(pRow1, srcPtrTemp, rowKernelLoopLimit);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_load_box_filter_f32_f32_7x7_host(pRow2, srcPtrTemp, rowKernelLoopLimit);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    
                    __m256 pTemp[4];
                    add_rows_7x7(pRow1, &pTemp[0]);
                    add_rows_7x7(pRow2, &pTemp[1]);
                    rpp_load_box_filter_f32_f32_7x7_host(pRow1, srcPtrTemp, rowKernelLoopLimit);
                    add_rows_7x7(pRow1, &pTemp[2]);
                    
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        // add loaded values from 7 rows
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_7x7_host(pRow2, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_7x7(pRow2, &pTemp[3]);

                        __m256 pDst;
                        blend_permute_add_7x7_pkd(pTemp, &pDst, pConvolutionFactorAVX);
                        _mm256_storeu_ps(dstPtrTemp, pDst);
                        dstPtrTemp += 8;

                        pTemp[0] = pTemp[1];
                        pTemp[1] = pTemp[2];
                        pTemp[2] = pTemp[3];
                    }
                    vectorLoopCount += padLength * 3;
                    increment_row_ptrs(srcPtrTemp, kernelSize, -16);
                    
                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3][7];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 7; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);

                    // process padLength number of columns in each row
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            __m256 pRow[7], pTemp[2];
                            rpp_load_box_filter_f32_f32_7x7_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_7x7(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_box_filter_f32_f32_7x7_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_7x7(pRow, &pTemp[1]);
                            blend_permute_add_7x7_pln(pTemp, &pResultPln[c], pConvolutionFactorAVX);
                        }
                        rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                        dstPtrTemp += 24;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
                Rpp32f *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[7];
                    for (int k = 0; k < 7; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp32f *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    vectorLoopCount += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
        else if (kernelSize == 9)
        {
            Rpp32f *srcPtrRow[9], *dstPtrRow;
            for (int i = 0; i < 9; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    for (int k = 1; k < 9; k++)
                        srcPtrRow[k] = srcPtrRow[k - 1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp32f *srcPtrTemp[9];
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[k] = srcPtrRow[k];
                        Rpp32f *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                        process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                        dstPtrTemp += padLength;
                        
                        __m256 pRow[9];
                        rpp_load_box_filter_f32_f32_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit);

                        // process alignedLength number of columns in eacn row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            // add loaded values from 9 rows
                            __m256 pTemp[2], pDst;
                            add_rows_9x9(pRow, &pTemp[0]);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            
                            rpp_load_box_filter_f32_f32_9x9_host(pRow, srcPtrTemp, rowKernelLoopLimit);
                            add_rows_9x9(pRow, &pTemp[1]);
                            blend_permute_add_9x9_pln(pTemp, &pDst, pConvolutionFactorAVX);
                            
                            _mm256_storeu_ps(dstPtrTemp, pDst);
                            dstPtrTemp += 8;
                        }
                        vectorLoopCount += padLength;

                        // process remaining columns in each row
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pkd(srcPtrTemp, srcPtrRow, dstPtrTemp, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    dstPtrTemp += padLength * 3;

                    __m256 pRow1[9], pRow2[9];
                    rpp_load_box_filter_f32_f32_9x9_host(pRow1, srcPtrTemp, rowKernelLoopLimit);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    rpp_load_box_filter_f32_f32_9x9_host(pRow2, srcPtrTemp, rowKernelLoopLimit);
                    increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                    
                    __m256 pTemp[4];
                    add_rows_9x9(pRow1, &pTemp[0]);
                    add_rows_9x9(pRow2, &pTemp[1]);
                    rpp_load_box_filter_f32_f32_9x9_host(pRow1, srcPtrTemp, rowKernelLoopLimit);
                    add_rows_9x9(pRow1, &pTemp[2]);
                    
                    // process remaining columns in each row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        // add loaded values from 9 rows
                        increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                        rpp_load_box_filter_f32_f32_9x9_host(pRow2, srcPtrTemp, rowKernelLoopLimit);
                        add_rows_9x9(pRow2, &pTemp[3]);

                        __m256 pDst;
                        blend_permute_add_9x9_pkd(pTemp, &pDst, pConvolutionFactorAVX);
                        _mm256_storeu_ps(dstPtrTemp, pDst);
                        dstPtrTemp += 8;

                        pTemp[0] = pTemp[1];
                        pTemp[1] = pTemp[2];
                        pTemp[2] = pTemp[3];
                    }
                    vectorLoopCount += padLength * 3;
                    increment_row_ptrs(srcPtrTemp, kernelSize, -16);
                    
                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTemp++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            // box filter with fused output-layout toggle (NCHW -> NHWC)
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 16) * 16;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[3][9];
                    for (int c = 0; c < 3; c++)
                    {
                        Rpp32u channelStride = c * srcDescPtr->strides.cStride;
                        for (int k = 0; k < 9; k++)
                            srcPtrTemp[c][k] = srcPtrRow[k] + channelStride;
                    }
                    Rpp32f *dstPtrTemp = dstPtrRow;

                    // get the number of rows needs to be loaded for the corresponding row
                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    for (int k = 0; k < padLength; k++)
                    {
                        for (int c = 0; c < 3; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, k, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            dstPtrTemp++;
                        }
                    }

                    // process alignedLength number of columns in eacn row
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                    {
                        __m256 pResultPln[3];
                        for (int c = 0; c < 3; c++)
                        {
                            // add loaded values from 9 rows
                            __m256 pRow[9], pTemp[2];
                            rpp_load_box_filter_f32_f32_9x9_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_9x9(pRow, &pTemp[0]);

                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 8);
                            rpp_load_box_filter_f32_f32_9x9_host(pRow, srcPtrTemp[c], rowKernelLoopLimit);
                            add_rows_9x9(pRow, &pTemp[1]);

                            blend_permute_add_9x9_pln(pTemp, &pResultPln[c], pConvolutionFactorAVX);
                        }
                        rpp_store24_f32pln3_to_f32pkd3_avx(dstPtrTemp, pResultPln);
                        dstPtrTemp += 24;
                    }
                    vectorLoopCount += padLength;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        for (int c = 0; c < srcDescPtr->c; c++)
                        {
                            box_filter_generic_host_tensor(srcPtrTemp[c], dstPtrTemp, vectorLoopCount, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                            increment_row_ptrs(srcPtrTemp[c], kernelSize, 1);
                            dstPtrTemp++;
                        }
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
            }
            else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                Rpp32u alignedLength = ((bufferLength - (2 * padLength) * 3) / 32) * 32;
                Rpp32f *dstPtrChannels[3];
                for (int i = 0; i < 3; i++)
                    dstPtrChannels[i] = dstPtrChannel + i * dstDescPtr->strides.cStride;
                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    int vectorLoopCount = 0;
                    bool padLengthRows = (i < padLength) ? 1: 0;
                    Rpp32f *srcPtrTemp[9];
                    for (int k = 0; k < 9; k++)
                        srcPtrTemp[k] = srcPtrRow[k];
                    Rpp32f *dstPtrTempChannels[3] = {dstPtrChannels[0], dstPtrChannels[1], dstPtrChannels[2]};

                    Rpp32s rowKernelLoopLimit = kernelSize;
                    get_kernel_loop_limit(i, rowKernelLoopLimit, kernelSize, padLength, roi.xywhROI.roiHeight);
                    process_left_border_columns_pkd_pln(srcPtrTemp, srcPtrRow, dstPtrTempChannels, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare);
                    vectorLoopCount += padLength * 3;

                    // process remaining columns in each row
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        int channel = vectorLoopCount % 3;
                        box_filter_generic_host_tensor(srcPtrTemp, dstPtrTempChannels[channel], vectorLoopCount / 3, kernelSize, padLength, roi.xywhROI.roiWidth, rowKernelLoopLimit, kernelSizeInverseSquare, 3);
                        increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                        dstPtrTempChannels[channel]++;
                    }
                    increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                    increment_row_ptrs(dstPtrChannels, 3, dstDescPtr->strides.hStride);
                }
            }
        }
    }

    return RPP_SUCCESS;
}