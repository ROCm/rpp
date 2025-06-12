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

__kernel void min_loc ( __global const unsigned char* input,
                    __global unsigned char *partial_min,
                    __global unsigned char *partial_min_location)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    local unsigned char localMins[256];
    local unsigned char localMinsLocation[256];

    localMins[local_id] = input[get_global_id(0)];
    localMinsLocation[local_id] = get_global_id(0);

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(localMins[local_id] > localMins[local_id + stride])
            {
                localMins[local_id] = localMins[local_id + stride];
                localMinsLocation[local_id] = get_global_id(0) + stride;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        partial_min[get_group_id(0)] = localMins[0];
        partial_min_location[get_group_id(0)] = localMinsLocation[0];
    }
}                        
__kernel void max_loc ( __global const unsigned char* input,
                    __global unsigned char *partial_max,
                    __global unsigned char *partial_max_location)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    local unsigned char localmaxs[256];
    local unsigned char localmaxsLocation[256];

    localmaxs[local_id] = input[get_global_id(0)];
    localmaxsLocation[local_id] = get_global_id(0);

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(localmaxs[local_id] > localmaxs[local_id + stride])
            {
                localmaxs[local_id] = localmaxs[local_id + stride];
                localmaxsLocation[local_id] = get_global_id(0) + stride;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        partial_max[get_group_id(0)] = localmaxs[0];
        partial_max_location[get_group_id(0)] = localmaxsLocation[0];
    }
}    
