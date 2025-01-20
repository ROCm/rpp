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
#include "rpp_cpu_common.hpp"
#include "rpp_cpu_filter.hpp"

/* box filter algorithm explanation for U8 PLN1 3x3 kernel size variant
Lets take an example input of 3x32 image
x x x x x x x x x x  .. x x
x 1 2 3 4 5 6 7 8 9 .. 32 x
x 1 2 3 4 5 6 7 8 9 .. 32 x
x 1 2 3 4 5 6 7 8 9 .. 32 x
x x x x x x x x x x  .. x x
padLength = 1 (kernelSize / 2)

Below steps are followed for getting outputs for the first 0-16 locations in 1st row
1. Process padLength number of columns in each row using raw c code (outputs for 0th location)
2. Process remaining alignedLength number of columns in each row using SSE/AVX code (outputs for 1-16 locations)
    - load kernel size number of rows
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 .. | 32
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 .. | 32
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 .. | 32

    - unpack lower half to 16 bits
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16
    1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16

    - unpack higher half to 16 bits
    16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32
    16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32
    16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32

    - add the unpacked values for both lower and higher half for all the unpacked 3 rows
    1+1+1 | 2+2+2 | 3+3+3 | ... | 16+16..16
    17+17+17 | 18+18+18 | 19+19+19 | ... |32+32+32

    - blend and shuffle and above accumalted lower half and higher values to get below outputs
    2+2+2 | 3+3+3 | 4+4+4 | ... | 17+17..17
    3+3+3 | 4+4+4 | 5+5+5 | ... | 18+18..18

    - add 3 registers for getting outputs desired outputs
    1+1+1 | 2+2+2 | 3+3+3 | ... | 16+16..16
    2+2+2 | 3+3+3 | 4+4+4 | ... | 17+17..17
    3+3+3 | 4+4+4 | 5+5+5 | ... | 18+18..18
    result = ((1+1+1)+(2+2+2)+(3+3+3)) | ((2+2+2)+(3+3+3)+(4+4+4)) | ... | ((16+16+16)+(17+17+17)+(18+18+18))

    - multiply with convolution factor
    (1/9)*((1+1+1)+(2+2+2)+(3+3+3)) | (1/9)*((2+2+2)+(3+3+3)+(4+4+4)) | ... | (1/9)*((16+16+16)+(17+17+17)+(18+18+18))

    - convert back to 8 bit and store in output
    2 | 3 | 4 | ... | 17

    Repeat the same process for remaining alignedLength columns and store in output
3. Process remaining non aligned columns in each row again using raw c code*/

template<typename T>
RppStatus box_filter_char_host_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32u kernelSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams layoutParams,
                                      rpp::Handle& handle);

// F32 and F16 bitdepth
template<typename T>
RppStatus box_filter_float_host_tensor(T *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       T *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams,
                                       rpp::Handle& handle);

template<typename T>
RppStatus box_filter_generic_host_tensor(T *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         T *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32u kernelSize,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams layoutParams,
                                         rpp::Handle& handle);