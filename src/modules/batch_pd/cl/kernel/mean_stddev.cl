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

__kernel void sum ( __global const unsigned char* input,
                    __global long *partialSums)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    local int localSums[256];

    localSums[local_id] = (long)input[get_global_id(0)];

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }
    if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
}

__kernel void mean_stddev ( __global const unsigned char* input,
                    __global float *partial_mean_sum,
                    const float mean)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    local float localSums[256];

    localSums[local_id] = (float)input[get_global_id(0)];

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(stride == group_size/2)
                localSums[local_id] = sqrt((localSums[local_id] - mean) * (localSums[local_id] - mean)) + sqrt((localSums[local_id + stride] - mean ) * (localSums[local_id + stride] - mean));
            else
                localSums[local_id] += localSums[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    partial_mean_sum[get_group_id(0)] = localSums[0];
}
