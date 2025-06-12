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

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

kernel
void partial_histogram_pln(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * channel;
    unsigned int pixId;
    local uint tmp_histogram [768];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 * channel;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + height * width];
        unsigned char pixelB = input[pixId + 2 * height * width];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[256+pixelG]);
        atomic_inc(&tmp_histogram[512+pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 * channel))
    {
        if (tid < (256 * channel)){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256 * channel;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}

kernel
void partial_histogram_pln1(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * channel;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 * channel;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        atomic_inc(&tmp_histogram[pixelR]);

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 * channel))
    {
        if (tid < (256 * channel)){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256 * channel;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}


kernel
void partial_histogram_pkd(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * channel;
    unsigned int pixId;
    local uint tmp_histogram [768];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 * channel;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + 1];
        unsigned char pixelB = input[pixId + 2];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[256+pixelG]);
        atomic_inc(&tmp_histogram[512+pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 * channel))
    {
        if (tid < (256 * channel)){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256 * channel;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}


kernel void
histogram_sum_partial(global unsigned int *histogramPartial,
                      global unsigned int *histogram,
                      const unsigned int num_groups,
                      const unsigned int channel)
{
    int  tid = (int)get_global_id(0);
    int  group_indx;
    int  n = num_groups;
    local uint tmp_histogram[256 * 3];
     
    tmp_histogram[tid] = histogramPartial[tid];    
    group_indx = 256*channel;
    while (--n > 1)
    {
        tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[group_indx + tid];
        group_indx += 256*channel; 
    }
    histogram[tid] = tmp_histogram[tid];
}
