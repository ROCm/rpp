#include <hip/hip_runtime.h>
extern "C" __global__ void random_shadow(
    const  unsigned char* input,
      unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) 
        return;
     int pixIdx = (width * height * id_z) + (width * id_y) + id_x;
    output[pixIdx] = input[pixIdx];
}
extern "C" __global__ void random_shadow_planar(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int pixIdx = ((y1 - 1 + id_y) * srcwidth) + (x1 + id_x) + (id_z * srcheight * srcwidth);
    if(output[pixIdx] != input[pixIdx] / 2)
    {    
        output[pixIdx] = input[pixIdx] / 2;
    }
}

extern "C" __global__ void random_shadow_packed(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int width = x2 - x1;
    int pixIdx = ((y1 - 1 + id_y) * channel * srcwidth) + ((x1 + id_x) * channel) + (id_z);
    if(output[pixIdx] != input[pixIdx] / 2)
    {    
        output[pixIdx] = input[pixIdx] / 2;
    }

}