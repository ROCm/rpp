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

extern "C" __global__ void random_shadow(const unsigned char *input,
                                         unsigned char *output,
                                         const unsigned int height,
                                         const unsigned int width,
                                         const unsigned int channel)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= width || id_y >= height || id_z >= channel)
    {
        return;
    }

    int pixIdx = (width * height * id_z) + (width * id_y) + id_x;

    output[pixIdx] = input[pixIdx];
}

extern "C" __global__ void random_shadow_planar(unsigned char *input,
                                                unsigned char *output,
                                                const unsigned int srcheight,
                                                const unsigned int srcwidth,
                                                const unsigned int channel,
                                                const unsigned int x1,
                                                const unsigned int y1,
                                                const unsigned int x2,
                                                const unsigned int y2)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel)
    {
        return;
    }

    int pixIdx = ((y1 - 1 + id_y) * srcwidth) + (x1 + id_x) + (id_z * srcheight * srcwidth);

    if(output[pixIdx] != input[pixIdx] / 2)
    {
        output[pixIdx] = input[pixIdx] / 2;
    }
}

extern "C" __global__ void random_shadow_packed(unsigned char *input,
                                                unsigned char *output,
                                                const unsigned int srcheight,
                                                const unsigned int srcwidth,
                                                const unsigned int channel,
                                                const unsigned int x1,
                                                const unsigned int y1,
                                                const unsigned int x2,
                                                const unsigned int y2)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel)
    {
        return;
    }

    int width = x2 - x1;

    int pixIdx = ((y1 - 1 + id_y) * channel * srcwidth) + ((x1 + id_x) * channel) + (id_z);

    if(output[pixIdx] != input[pixIdx] / 2)
    {
        output[pixIdx] = input[pixIdx] / 2;
    }
}

RppStatus hip_exec_random_shadow_packed(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u channel, Rpp32u column1, Rpp32u row1, Rpp32u column2, Rpp32u row2, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = column2 - column1;
    int globalThreads_y = row2 - row1;
    int globalThreads_z = channel;

    hipLaunchKernelGGL(random_shadow_packed,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                       handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                       channel,
                       column1,
                       row1,
                       column2,
                       row2);

    return RPP_SUCCESS;
}

RppStatus hip_exec_random_shadow_planar(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, Rpp32u channel, Rpp32u column1, Rpp32u row1, Rpp32u column2, Rpp32u row2, Rpp32s i)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = column2 - column1;
    int globalThreads_y = row2 - row1;
    int globalThreads_z = channel;

    hipLaunchKernelGGL(random_shadow_planar,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                       handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                       channel,
                       column1,
                       row1,
                       column2,
                       row2);

    return RPP_SUCCESS;
}
