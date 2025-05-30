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

#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

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

__device__ inline unsigned int xorshift(int pixid)
{
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^ (t >> 8));
    return res;
}

extern "C" __global__ void snow_batch(unsigned char *input,
                                      unsigned char *output,
                                      const float *snowPercentage,
                                      const unsigned int *height,
                                      const unsigned int *width,
                                      const unsigned int *max_width,
                                      const unsigned long long *batch_index,
                                      const unsigned int channel,
                                      const unsigned int *inc, // use width * height for pln and 1 for pkd
                                      const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    float rainProbTemp = snowPercentage[id_z];
    unsigned int snowHeightTemp = 5;
    unsigned int snowWidthTemp = 5;
    int indextmp = 0;
    long pixIdx = 0;
    int rand;

    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
        if((id_y >= 0 ) && (id_y < height[id_z]) && (id_x >= 0) && (id_x < width[id_z]))
        {
            float pixelDistance = 1.0;
            pixelDistance /= (rainProbTemp / 100);
            if((pixIdx - batch_index[id_z]) % (int)pixelDistance == 0)
            {
                int snow_mat[5][5] = {{0,50,75,50,0},
                                      {40,80,120,80,40},
                                      {75,120,255,120,75},
                                      {40,80,120,80,40},
                                      {0,50,75,50,0}};

                int rand_id = xorshift(pixIdx) % (9973);
                rand_id = rand_id % (int)pixelDistance;
                rand_id -= rand_id % 3;
                if(rand_id + id_x > width[id_z])
                {
                    return;
                }
                rand_id = rand_id * plnpkdindex;
                for(int i = 0; i < snowHeightTemp; i++)
                {
                    for(int j = 0; j < snowWidthTemp; j++)
                    {
                        if (id_x + i + snowWidthTemp <= width[id_z] && id_y + j + snowHeightTemp < height[id_z])
                        {
                            int id = (i * max_width[id_z] + j) * plnpkdindex;
                            output[pixIdx + rand_id + id] = saturate_8u(output[pixIdx + rand_id + id] + snow_mat[i][j]);
                            if(channel == 3)
                            {
                                output[pixIdx + rand_id + inc[id_z] + id] = saturate_8u(output[pixIdx + rand_id + inc[id_z] + id] + snow_mat[i][j]);
                                output[pixIdx + rand_id + inc[id_z] * 2 + id] = saturate_8u(output[pixIdx + rand_id + inc[id_z] * 2 + id] + snow_mat[i][j]);
                            }
                        }
                    }
                }
            }
        }
    }
}

RppStatus hip_exec_snow_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = (max_width + 31) & ~31;
    int globalThreads_y = (max_height + 31) & ~31;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(snow_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus snow_hip_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nbatchSize = handle.GetBatchSize();
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    hipMemcpy(dstPtr, srcPtr, nbatchSize * max_height * max_width * channel * sizeof(unsigned char), hipMemcpyDeviceToDevice);

    hip_exec_snow_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}