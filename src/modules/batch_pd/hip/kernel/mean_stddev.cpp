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
extern "C" __global__ void sum (  const unsigned char* input,
                     long *partialSums)
{
    uint local_id = hipThreadIdx_x;
    uint group_size = hipBlockDim_x;
    __shared__ int localSums[256];

    localSums[local_id] = (long)input[hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x];

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        __syncthreads();
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }
    if (local_id == 0)
    partialSums[hipBlockIdx_x] = localSums[0];
}

extern "C" __global__ void mean_stddev (  const unsigned char* input,
                     float *partial_mean_sum,
                    const float mean)
{
    uint local_id = hipThreadIdx_x;
    uint group_size = hipBlockDim_x;
    __shared__ float localSums[256];

    localSums[local_id] = (float)input[hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x];

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            __syncthreads();
            if(stride == group_size/2)
                localSums[local_id] = sqrt((localSums[local_id] - mean) * (localSums[local_id] - mean)) + sqrt((localSums[local_id + stride] - mean ) * (localSums[local_id + stride] - mean));
            else
                localSums[local_id] += localSums[local_id + stride];
        }
        __syncthreads();
    }
    if (local_id == 0)
    partial_mean_sum[hipBlockIdx_x] = localSums[0];
}
