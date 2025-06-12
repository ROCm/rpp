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

__kernel void integral_pkd_col( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_x * channel + id_z;
    
    output[pixIdx] = 0;

    for(int i = id_z; i <= id_x * channel + id_z ; i += channel)
    {
        output[pixIdx] += input[i];
    }
}

__kernel void integral_pln_col( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_z * height * width + id_x;

    output[pixIdx] = 0;

    for(int i = (id_z * height * width) ; i <= (id_z * height * width + id_x) ; i++)
    {
        output[pixIdx] += input[i];
    }
}

__kernel void integral_pkd_row( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_x * channel * width + id_z;
    
    output[pixIdx] = 0;

    for(int i = id_z; i <= id_x * channel * width + id_z ; i += width * channel)
    {
        output[pixIdx] += input[i];
    }
}

__kernel void integral_pln_row( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= width || id_z >= channel) return;

    int pixIdx = id_z * height * width + id_x * width;

    output[pixIdx] = 0;

    for(int i = (id_z * height * width) ; i <= (id_z * height * width + id_x * width) ; i += width)
    {
        output[pixIdx] += input[i];
    }
}

__kernel void integral_up_pln( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = ((width * (loop + 1 )) - (id_x * width) + (id_x + 1)) + (id_z * height * width);
    int A, B, C;
    A = pixIdx - width - 1;
    B = pixIdx - width;
    C = pixIdx - 1;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}

__kernel void integral_low_pln( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = ((width * (height - 1)) + (id_x + loop + 1) - (id_x * width) + 1) + (id_z * height * width);
    int A, B, C;
    A = pixIdx - width - 1;
    B = pixIdx - width;
    C = pixIdx - 1;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}

__kernel void integral_up_pkd( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = (width * channel * (loop + 1)) - (id_x * channel * width) + (id_x * channel + channel) + id_z;
    int A, B, C;
    A = pixIdx - width * channel - channel;
    B = pixIdx - width * channel;
    C = pixIdx - channel;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}

__kernel void integral_low_pkd( __global unsigned char* input,
                            __global unsigned int* output,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel,
                            const int loop,
                            const int diag
)
{
    int id_x = get_global_id(0);
    int id_z = get_global_id(2);
    if (id_x >= diag || id_z >= channel) return;
    
    int pixIdx = (width * channel * (height - 1)) - (id_x * channel * width) + (id_x * channel + (loop + 1) * channel) + channel + id_z;
    int A, B, C;
    A = pixIdx - width * channel - channel;
    B = pixIdx - width * channel;
    C = pixIdx - channel;
    output[pixIdx] = 0;
    output[pixIdx] = input[pixIdx] + output[B] + output[C] - output[A];
}
