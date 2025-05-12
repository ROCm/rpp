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

extern "C" __global__ void fisheye_batch(unsigned char *input,
                                         unsigned char *output,
                                         unsigned int *height,
                                         unsigned int *width,
                                         unsigned int *max_width,
                                         unsigned int *xroi_begin,
                                         unsigned int *xroi_end,
                                         unsigned int *yroi_begin,
                                         unsigned int *yroi_end,
                                         unsigned long long *batch_index,
                                         const unsigned int channel,
                                         unsigned int *inc, // use width * height for pln and 1 for pkd
                                         const int plnpkdindex) // use 1 pln 3 for pkd
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int dstpixIdx = 0;
    dstpixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z]) * plnpkdindex;

    float normY = ((float)(2 * id_y) / (float) (height[id_z])) - 1;
    float normX = ((float)(2 * id_x) / (float) (width[id_z])) - 1;
    float dist = sqrt((normX * normX) + (normY * normY));

    if ((0.0 <= dist) && (dist <= 1.0))
    {
        float newDist = sqrt(1.0 - dist * dist);
        newDist = (dist + (1.0 - newDist)) / 2.0;
        if (newDist <= 1.0)
        {
            float theta = atan2(normY, normX);
            float newX = newDist * cos(theta);
            float newY = newDist * sin(theta);
            int srcX = (int)(((newX + 1) * width[id_z]) / 2.0);
            int srcY = (int)(((newY + 1) * height[id_z]) / 2.0);
            int srcpixIdx = batch_index[id_z] + (srcX + srcY * max_width[id_z]) * plnpkdindex;

            if(srcY < yroi_end[id_z] && (srcY >=yroi_begin[id_z]) && srcX < xroi_end[id_z] && (srcX >=xroi_begin[id_z]))
            {
                if(srcpixIdx >= batch_index[id_z] && srcpixIdx <= batch_index[id_z+1])
                {
                    for(int indextmp = 0; indextmp < channel; indextmp++)
                    {
                        output[dstpixIdx] = input[srcpixIdx];
                        dstpixIdx += inc[id_z];
                        srcpixIdx += inc[id_z];
                    }
                }
            }
        }
    }
    else
    {
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dstpixIdx] = (unsigned char) 0;
            dstpixIdx += inc[id_z];
        }
    }
}

RppStatus hip_exec_fisheye_batch(Rpp8u *srcPtr, Rpp8u *dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, Rpp32u channel, Rpp32s plnpkdind, Rpp32u max_height, Rpp32u max_width)
{
    int localThreads_x = 32;
    int localThreads_y = 32;
    int localThreads_z = 1;
    int globalThreads_x = max_width;
    int globalThreads_y = max_height;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(fisheye_batch,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       dstPtr,
                       handle.GetInitHandle()->mem.mgpu.srcSize.height,
                       handle.GetInitHandle()->mem.mgpu.srcSize.width,
                       handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                       handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                       handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                       channel,
                       handle.GetInitHandle()->mem.mgpu.inc,
                       plnpkdind);

    return RPP_SUCCESS;
}

RppStatus fisheye_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    hip_exec_fisheye_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}