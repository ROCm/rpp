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

__kernel void remap_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    __global unsigned int* row,
                    __global unsigned int* column,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    
    int IPpixIdx = id_y * channel * width + id_x * channel + id_z;
    
    int pixId = id_y * width + id_x;

    int OPpixIdx = row[pixId] * channel * width + column[pixId] * channel + id_z;
    
    output[OPpixIdx] = input[IPpixIdx];
}

__kernel void remap_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    __global unsigned int* row,
                    __global unsigned int* column,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int IPpixIdx = id_y * width + id_x + id_z * width * height;
    
    int pixId = id_y * width + id_x;

    int OPpixIdx = row[pixId] * width + column[pixId] + id_z * width * height;
    
    output[OPpixIdx] = input[IPpixIdx];  
}

__kernel void remap_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    __global unsigned int* row,
                    __global unsigned int* column,
                    __global unsigned int* height,
                    __global unsigned int* width,
                    __global unsigned long *batch_index,
                    const unsigned int channel,
                    const unsigned int batch_size,
                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                    const int plnpkdindex // use 1 pln 3 for pkd
)
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long IpixIdx = 0, OPpixIdx = 0;
    int pixId = 0;

    if(id_x < width[id_z] && id_y < height[id_z] && id_z < batch_size){
        for(indextmp = 0; indextmp < channel; indextmp++){
            IpixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
            pixId = id_y * width + id_x;
            OPpixIdx = batch_index[id_z] + (row[batch_index[id_z] + pixId] * max_width[id_z] + column[batch_index[id_z] + pixId])  * plnpkdindex ;
            output[OPpixIdx] = input[IPpixIdx];
            IpixIdx += inc[id_z];
            OPpixIdx += inc[id_z];
        }     
    }
}
