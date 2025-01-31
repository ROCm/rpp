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

#ifndef RPP_HIP_COMMON_MISC_H
#define RPP_HIP_COMMON_MISC_H

// -------------------- Set 0 - Range checks and Range adjustment --------------------

// float pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, uchar* dst)
{
    pixel = fmax(fminf(pixel, 255), 0);
    *dst = (uchar)pixel;
}

// float pixel check for -128-127 range

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, schar* dst)
{
    pixel = fmax(fminf(pixel, 127), -128);
    *dst = (schar)pixel;
}

// float pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, float* dst)
{
    pixel = fmax(fminf(pixel, 1), 0);
    *dst = pixel;
}

__device__ __forceinline__ void rpp_hip_pixel_check_and_store(float pixel, half* dst)
{
    pixel = fmax(fminf(pixel, 1), 0);
    *dst = (half)pixel;
}

__device__ __forceinline__ float rpp_hip_pixel_check_0to255(float src_f1)
{
    return fminf(fmaxf(src_f1, 0), 255);
}

// float4 pixel check for 0-1 range

__device__ __forceinline__ float4 rpp_hip_pixel_check_0to1(float4 src_f4)
{
    return make_float4(fminf(fmaxf(src_f4.x, 0), 1),
                       fminf(fmaxf(src_f4.y, 0), 1),
                       fminf(fmaxf(src_f4.z, 0), 1),
                       fminf(fmaxf(src_f4.w, 0), 1));
}

// d_float8 pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to255(d_float8 *pix_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to255(pix_f8->f4[0]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to255(pix_f8->f4[1]);
}

// d_float8 pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to1(d_float8 *pix_f8)
{
    pix_f8->f4[0] = rpp_hip_pixel_check_0to1(pix_f8->f4[0]);
    pix_f8->f4[1] = rpp_hip_pixel_check_0to1(pix_f8->f4[1]);
}

// d_float24 pixel check for 0-255 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to255(d_float24 *pix_f24)
{
    pix_f24->f4[0] = rpp_hip_pixel_check_0to255(pix_f24->f4[0]);
    pix_f24->f4[1] = rpp_hip_pixel_check_0to255(pix_f24->f4[1]);
    pix_f24->f4[2] = rpp_hip_pixel_check_0to255(pix_f24->f4[2]);
    pix_f24->f4[3] = rpp_hip_pixel_check_0to255(pix_f24->f4[3]);
    pix_f24->f4[4] = rpp_hip_pixel_check_0to255(pix_f24->f4[4]);
    pix_f24->f4[5] = rpp_hip_pixel_check_0to255(pix_f24->f4[5]);
}

// d_float24 pixel check for 0-1 range

__device__ __forceinline__ void rpp_hip_pixel_check_0to1(d_float24 *pix_f24)
{
    pix_f24->f4[0] = rpp_hip_pixel_check_0to1(pix_f24->f4[0]);
    pix_f24->f4[1] = rpp_hip_pixel_check_0to1(pix_f24->f4[1]);
    pix_f24->f4[2] = rpp_hip_pixel_check_0to1(pix_f24->f4[2]);
    pix_f24->f4[3] = rpp_hip_pixel_check_0to1(pix_f24->f4[3]);
    pix_f24->f4[4] = rpp_hip_pixel_check_0to1(pix_f24->f4[4]);
    pix_f24->f4[5] = rpp_hip_pixel_check_0to1(pix_f24->f4[5]);
}

// d_float8 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float8 *sum_f8){}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float8 *sum_f8)
{
    sum_f8->f4[0] = sum_f8->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f8->f4[1] = sum_f8->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(schar *dstPtr, d_float8 *sum_f8)
{
    sum_f8->f4[0] = sum_f8->f4[0] - (float4) 128;    // Subtract 128 for schar image data
    sum_f8->f4[1] = sum_f8->f4[1] - (float4) 128;    // Subtract 128 for schar image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float8 *sum_f8)
{
    sum_f8->f4[0] = sum_f8->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f8->f4[1] = sum_f8->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
}

// d_float24 adjust pixel range for different bit depths

__device__ __forceinline__ void rpp_hip_adjust_range(uchar *dstPtr, d_float24 *sum_f24){}

__device__ __forceinline__ void rpp_hip_adjust_range(float *dstPtr, d_float24 *sum_f24)
{
    sum_f24->f4[0] = sum_f24->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[1] = sum_f24->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[2] = sum_f24->f4[2] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[3] = sum_f24->f4[3] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[4] = sum_f24->f4[4] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
    sum_f24->f4[5] = sum_f24->f4[5] * (float4) ONE_OVER_255;    // Divide by 255 for float image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(schar *dstPtr, d_float24 *sum_f24)
{
    sum_f24->f4[0] = sum_f24->f4[0] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[1] = sum_f24->f4[1] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[2] = sum_f24->f4[2] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[3] = sum_f24->f4[3] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[4] = sum_f24->f4[4] - (float4) 128;    // Subtract 128 for schar image data
    sum_f24->f4[5] = sum_f24->f4[5] - (float4) 128;    // Subtract 128 for schar image data
}

__device__ __forceinline__ void rpp_hip_adjust_range(half *dstPtr, d_float24 *sum_f24)
{
    sum_f24->f4[0] = sum_f24->f4[0] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[1] = sum_f24->f4[1] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[2] = sum_f24->f4[2] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[3] = sum_f24->f4[3] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[4] = sum_f24->f4[4] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
    sum_f24->f4[5] = sum_f24->f4[5] * (float4) ONE_OVER_255;    // Divide by 255 for half image data
}

#endif