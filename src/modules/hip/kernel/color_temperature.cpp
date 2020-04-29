#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
extern "C" __global__ void temperature_planar(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const int modificationValue
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;
    int pixIdx = id_x + id_y * width;
    int c = width * height;
    
    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);
    if( channel > 1)
    {
        output[pixIdx + c] = input[pixIdx + c];
        res = input[pixIdx + c * 2] - modificationValue;
        output[pixIdx + c * 2] = saturate_8u(res);
    }
}
extern "C" __global__ void temperature_packed(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const int modificationValue
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;
    
    int pixIdx = id_y * width * channel + id_x * channel;
    
    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);

    output[pixIdx + 1] = input[pixIdx + 1];

    res = input[pixIdx+2] - modificationValue;
    output[pixIdx+2] = saturate_8u(res);
}

__device__ unsigned char temperature(unsigned char input, float value, unsigned char RGB){
    if(RGB == 0)
        return saturate_8u(input + value);
    else if(RGB == 1)
        return (input);
    else
        return saturate_8u(input - value);
}

extern "C" __global__ void color_temperature_batch(   unsigned char* input,
                                     unsigned char* output,
                                     int *modificationValue,
                                     int *xroi_begin,
                                     int *xroi_end,
                                     int *yroi_begin,
                                     int *yroi_end,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long *batch_index,
                                    const unsigned int channel,
                                     unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    unsigned char valuergb;
    unsigned char modificationValuetmp = modificationValue[id_z];
    int indextmp=0;
    long pixIdx = 0;
    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb = input[pixIdx];
            output[pixIdx] = temperature(valuergb , modificationValuetmp, indextmp);
            pixIdx += inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(indextmp = 0; indextmp < channel; indextmp++){
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
}