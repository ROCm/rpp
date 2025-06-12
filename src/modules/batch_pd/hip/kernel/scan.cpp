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
#define SWAP(a,b) {__local int *tmp=a;a=b;b=tmp;}

extern "C" __global__ void scan_1c( int *input,
                    int *output,
                   __local  int *b,
                   __local  int *c)
{
    //printf("Inside scan");
    /*uint gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    uint lid = hipThreadIdx_x;
    uint gs = hipBlockDim_x;

    c[lid] = b[lid] = input[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = 1; s < gs; s <<= 1) {
        if(lid < (s-1)) {
            c[lid] = b[lid]+b[lid-s];
            c[lid + 256] = b[lid+ 256]+b[lid+ 256-s];
            c[lid + 512] = b[lid+ 512]+b[lid + 512-s];
        } else {
            c[lid] = b[lid];
            c[lid + 256] = b[lid + 256];
            c[lid + 512] = b[lid + 512];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(b,c);
    }
    output[gid] = b[lid];
    output[gid + 256] = b[lid + 256];
    output[gid + 512] = b[lid + 512];*/

    uint gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int i;
    if (gid == 0){
        output[0]= input[0];
        output[256]= input[256];
        output[512]= output[512];
    for(i =1; i<256; i++){
        output[i] = output[i-1] + input[i];
        output[i+256] = output[256+i-1] + input[256+i];
        output[i+ 512] = output[512+i-1] + input[512+i];
    }
   }
}

extern "C" __global__ void scan( int *input,
                    int *output)
{

    uint gid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int i;
    if (gid == 0){
        output[0]= input[0];
    for(i =1; i<256; i++){
        output[i] = output[i-1] + input[i];
    }
    }
}
extern "C" __global__ void scan_batch( int *input,
                    int *output,
                   const unsigned int batch_size,
                   __local  int *b,
                   __local  int *c)
{

    uint gid_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    uint gid_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    unsigned int start_index = 256 * gid_y;
    int i;
    if (gid_x == 0){
        output[start_index]= input[start_index];
        for(i =1; i<256; i++){
        output[start_index+i] = output[start_index+ i-1] + input[start_index + i];
       // printf("scan %d",output[start_index+ i]);

        }
    }
    
}
