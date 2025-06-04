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

#include "hip_legacy_executors.hpp"

#define amd_max3_hip(a,b,c) ((a > b) && (a > c) ? a : ((b > c) ? b : c))
#define amd_min3_hip(a,b,c) ((a < b) && (a < c) ? a : ((b < c) ? b : c))

inline void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width)
{
    int i;
    *max_height  = 0;
    *max_width =0;
    for (i=0; i<batch_size; i++){
        if(*max_height < height[i])
            *max_height = height[i];
        if(*max_width < width[i])
            *max_width = width[i];
    }
}

__device__ uchar4 convert_one_pixel_to_rgb(float4 pixel)
{
    float r, g, b;
    float h, s, v;

    h = pixel.x;
    s = pixel.y;
    v = pixel.z;

    float f = h / 60.0f;
    float hi = floor(f);
    f = f - hi;
    float p = v * (1 - s);
    float q = v * (1 - s * f);
    float t = v * (1 - s * (1 - f));

    if (hi == 0.0f || hi == 6.0f)
    {
        r = v;
        g = t;
        b = p;
    }
    else if (hi == 1.0f)
    {
        r = q;
        g = v;
        b = p;
    }
    else if (hi == 2.0f)
    {
        r = p;
        g = v;
        b = t;
    }
    else if (hi == 3.0f)
    {
        r = p;
        g = q;
        b = v;
    }
    else if (hi == 4.0f)
    {
        r = t;
        g = p;
        b = v;
    }
    else
    {
        r = v;
        g = p;
        b = q;
    }

    unsigned char red = (unsigned char) (255.0f * r);
    unsigned char green = (unsigned char) (255.0f * g);
    unsigned char blue = (unsigned char) (255.0f * b);
    unsigned char alpha = 0.0; //(unsigned char)(pixel.w);
    return (uchar4){red, green, blue, alpha};
}

__device__ float4 convert_one_pixel_to_hsv(uchar4 pixel)
{
    float r, g, b, a;
    float h, s, v;

    r = pixel.x / 255.0f;
    g = pixel.y / 255.0f;
    b = pixel.z / 255.0f;
    a = pixel.w;

    float max = amd_max3_hip(r, g, b);
    float min = amd_min3_hip(r, g, b);
    float diff = max - min;

    v = max;

    if (v == 0.0f)
    {
        // black
        h = s = 0.0f;
    }
    else
    {
        s = diff / v;
        if (diff < 0.001f)
        {
            // grey
            h = 0.0f;
        }
        else
        {
            // color
            if (max == r)
            {
                h = 60.0f * (g - b) / diff;
                if (h < 0.0f)
                {
                    h += 360.0f;
                }
            }
            else if (max == g)
            {
                h = 60.0f * (2 + (b - r) / diff);
            }
            else
            {
                h = 60.0f * (4 + (r - g) / diff);
            }
        }
    }

    return (float4){h, s, v, a};
}

extern "C" __global__ void hue_batch(unsigned char *input,
                                     unsigned char *output,
                                     float *hue,
                                     unsigned int *xroi_begin,
                                     unsigned int *xroi_end,
                                     unsigned int *yroi_begin,
                                     unsigned int *yroi_end,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long long *batch_index,
                                     unsigned int *inc, // use width * height for pln and 1 for pkd
                                     const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel; float4 hsv;
    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + inc[id_z]];
    pixel.z = input[pixIdx + 2 * inc[id_z]];
    pixel.w = 0.0;

    if((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
        hsv.x += hue[id_z];

        if(hsv.x > 360.0)
        {
            hsv.x = hsv.x - 360.0;
        }
        else if(hsv.x < 0)
        {
            hsv.x = hsv.x + 360.0;
        }

        pixel = convert_one_pixel_to_rgb(hsv); // Converting to RGB back with hue modification
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] = pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }
    else
    {
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] =  pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }

}


RppStatus hip_exec_hueRGB_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(hue_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}


RppStatus hueRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_hueRGB_batch(srcPtr, dstPtr, handle, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

extern "C" __global__ void saturation_batch(unsigned char *input,
                                            unsigned char *output,
                                            float *sat,
                                            unsigned int *xroi_begin,
                                            unsigned int *xroi_end,
                                            unsigned int *yroi_begin,
                                            unsigned int *yroi_end,
                                            unsigned int *height,
                                            unsigned int *width,
                                            unsigned int *max_width,
                                            unsigned long long *batch_index,
                                            unsigned int *inc, // use width * height for pln and 1 for pkd
                                            const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width[id_z] || id_y >= height[id_z])
    {
        return;
    }

    uchar4 pixel; float4 hsv;
    int pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * plnpkdindex;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + inc[id_z]];
    pixel.z = input[pixIdx + 2 * inc[id_z]];
    pixel.w = 0.0;

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        hsv = convert_one_pixel_to_hsv(pixel);
        hsv.y *= sat[id_z];

        hsv.y = fmaxf(fminf(hsv.y, 1.0), 0.0);

        pixel = convert_one_pixel_to_rgb(hsv); // Converting to RGB back with saturation modification
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] = pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }
    else
    {
        output[pixIdx] = pixel.x;
        output[pixIdx + inc[id_z]] = pixel.y;
        output[pixIdx + 2 * inc[id_z]] = pixel.z;
    }
}

RppStatus hip_exec_saturationRGB_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(saturation_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus saturationRGB_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_saturationRGB_batch(srcPtr, dstPtr, handle, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}