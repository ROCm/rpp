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

#include "rpp.h"
#include "handle.hpp"
#include <math.h>
#include <algorithm>
#include <omp.h>
#include "rpp_cpu_simd_load_store.hpp"
#include "rpp_cpu_simd_math.hpp"


// -------------------- fisheye --------------------

template <typename T>
RppStatus fisheye_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr, RppiROI *roiPoints,
                             Rpp32u nbatchSize,
                             RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle);

// -------------------- snow --------------------

template <typename T>
RppStatus snow_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                          Rpp32f *batch_strength,
                          RppiROI *roiPoints, Rpp32u nbatchSize,
                          RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle);

// -------------------- hueRGB --------------------

template <typename T>
RppStatus hueRGB_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                            Rpp32f *batch_hueShift,
                            RppiROI *roiPoints, Rpp32u nbatchSize,
                            RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle);

// -------------------- saturationRGB --------------------

template <typename T>
RppStatus saturationRGB_host_batch(T* srcPtr, RppiSize *batch_srcSize, RppiSize *batch_srcSizeMax, T* dstPtr,
                                   Rpp32f *batch_saturationFactor,
                                   RppiROI *roiPoints, Rpp32u nbatchSize,
                                   RppiChnFormat chnFormat, Rpp32u channel, rpp::Handle& handle);
