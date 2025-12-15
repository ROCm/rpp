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

__kernel void random_shadow(
    const __global unsigned char* input,
    __global  unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) 
        return;
     int pixIdx = (width * height * id_z) + (width * id_y) + id_x;
    output[pixIdx] = input[pixIdx];
}
__kernel void random_shadow_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int pixIdx = ((y1 - 1 + id_y) * srcwidth) + (x1 + id_x) + (id_z * srcheight * srcwidth);
    if(output[pixIdx] != input[pixIdx] / 2)
    {    
        output[pixIdx] = input[pixIdx] / 2;
    }
}

__kernel void random_shadow_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);

    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int width = x2 - x1;
    int pixIdx = ((y1 - 1 + id_y) * channel * srcwidth) + ((x1 + id_x) * channel) + (id_z);
    if(output[pixIdx] != input[pixIdx] / 2)
    {    
        output[pixIdx] = input[pixIdx] / 2;
    }

}
