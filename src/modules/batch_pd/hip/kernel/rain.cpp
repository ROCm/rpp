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

#include <hip/hip_runtime.h>
#include "rpp_hip_host_decls.hpp"

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

__device__ unsigned int rain_xorshift(int pixid)
{
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
    return res;
}

extern "C" __global__ void rain(unsigned char *input,
                                unsigned char *output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = id_y * width * channel + id_x * channel + id_z;
    int pixel = input[pixIdx] + output[pixIdx];
    output[pixIdx] = saturate_8u(pixel);
}

extern "C" __global__ void rain_pkd(unsigned char *output,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel,
                                    const unsigned int pixelDistance,
                                    const unsigned int rainWidth,
                                    const unsigned int rainHeight,
                                    const float transparency)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = id_y * width * channel + id_x * channel;

    output[pixIdx] = 0;
    output[pixIdx + 1] = 0;
    output[pixIdx + 2] = 0;
    int rand;

    if (pixIdx % pixelDistance == 0)
    {
        int rand_id = rain_xorshift(pixIdx) % 997;
        rand_id -= rand_id % 3;

        for (int i = 0; i < rainWidth; i++)
        {
            for (int j = 0; j < rainHeight; j++)
            {
                if (id_x + i + rainWidth <= width && id_y + j < height)
                {
                    int id = i * channel + j * width * channel;
                    for(int k = 0; k < channel; k++)
                    {
                        if(channel == 0)
                        {
                            output[pixIdx + rand_id + id + k] = transparency * 196;
                        }
                        else if(channel == 1)
                        {
                            output[pixIdx + rand_id + id + k] = transparency * 226;
                        }
                        else
                        {
                            output[pixIdx + rand_id + id + k] = transparency * 255;
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ void rain_pln(unsigned char *output,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel,
                                    const unsigned int pixelDistance,
                                    const unsigned int rainWidth,
                                    const unsigned int rainHeight,
                                    const float transparency)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    int pixIdx = id_y * width + id_x;
    int channelSize = width * height;
    output[pixIdx] = 0;
    if (channel > 1)
    {
        output[pixIdx + channelSize] = 0;
        output[pixIdx + 2 * channelSize] = 0;
    }
    int rand;

    if (pixIdx % pixelDistance == 0)
    {
        int rand_id = rain_xorshift(pixIdx) % 997;
        for (int i = 0; i < rainWidth; i++)
        {
            for (int j = 0; j < rainHeight; j++)
            {
                if (id_x + i + rainWidth <= width && id_y + j < height)
                {
                    int id = i + j * width;
                    for(int k = 0; k < channel; k++)
                    {
                        if(channel == 0)
                        {
                            output[pixIdx + rand_id + id + (k * channelSize)] = transparency * 196;
                        }
                        else if(channel == 1)
                        {
                            output[pixIdx + rand_id + id + (k * channelSize)] = transparency * 226;
                        }
                        else
                        {
                            output[pixIdx + rand_id + id + (k * channelSize)] = transparency * 255;
                        }
                    }
                }
            }
        }
    }
}

extern "C" __global__ void rain_batch(unsigned char *input,
                                      unsigned char *output,
                                      float *rainPercentage,
                                      unsigned int *rainWidth,
                                      unsigned int *rainHeight,
                                      float *transparency,
                                      unsigned int *height,
                                      unsigned int *width,
                                      unsigned int *max_width,
                                      unsigned long long *batch_index,
                                      const unsigned int channel,
                                      unsigned int *inc, // use width * height for pln and 1 for pkd
                                      const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float rainProbTemp = rainPercentage[id_z];
    float rainTransTemp = transparency[id_z];
    unsigned int rainHeightTemp = rainHeight[id_z];
    unsigned int rainWidthTemp = rainWidth[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int rand;

    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex ;
        if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
        {
            float pixelDistance = 1.0;
            pixelDistance /= (rainProbTemp / 100);

            if((pixIdx - batch_index[id_z]) % (int)pixelDistance == 0)
            {
                int rand_id = rain_xorshift(pixIdx) % (9973);
                rand_id = rand_id % (int)pixelDistance;
                rand_id -= rand_id % 3;
                if(rand_id + id_x > width[id_z])
                {
                    return;
                }
                rand_id = rand_id * plnpkdindex;
                for(int i = 0; i < rainHeightTemp; i++)
                {
                    for(int j = 0; j < rainWidthTemp; j++)
                    {
                        if (id_x + i + rainWidthTemp <= width[id_z] && id_y + j + rainHeightTemp < height[id_z])
                        {
                            int id = (i * max_width[id_z] + j) * plnpkdindex;
                            int pixValue = (int)(rainTransTemp * 196);
                            output[pixIdx + rand_id + id] = saturate_8u(pixValue + output[pixIdx + rand_id + id]);
                            if(channel == 3)
                            {
                                pixValue = (int)(rainTransTemp * 226);
                                output[pixIdx + rand_id + inc[id_z] + id] = saturate_8u(pixValue + output[pixIdx + rand_id + inc[id_z] + id]);
                                pixValue = (int)(rainTransTemp * 255);
                                output[pixIdx + rand_id + inc[id_z] * 2 + id] = saturate_8u(pixValue + output[pixIdx + rand_id + inc[id_z] * 2 + id]);
                            }
                        }
                    }
                }
            }
        }
    }
}

RppStatus hip_exec_rain_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(rain_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                       handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}
