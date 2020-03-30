#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__device__  unsigned char brightness( unsigned char input_pixel, float alpha, float beta){
    return saturate_8u(alpha * input_pixel + beta);
}
__device__ unsigned int get_pln_index(unsigned int id_x, unsigned int id_y, unsigned int id_z, unsigned int width, 
                        unsigned int height, unsigned channel){
 return ( id_x + id_y * width + id_z * width * height);
}
extern "C" __global__  void brightness_contrast( unsigned char* input,
                                    unsigned char* output,
                                    const float alpha,
                                    const int beta,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = get_pln_index(id_x, id_y, id_z, width, height, channel);

    int res = input[pixIdx] * alpha + beta;
    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void brightness_batch(   unsigned char* input,
                                     unsigned char* output,
                                     float *alpha,
                                     float *beta,
                                     unsigned int *xroi_begin,
                                     unsigned int *xroi_end,
                                     unsigned int *yroi_begin,
                                     unsigned int *yroi_end,
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
    float alphatmp = alpha[id_z], betatmp = beta[id_z];
    int indextmp=0;
    long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb = input[pixIdx];
            output[pixIdx] = brightness(valuergb , alphatmp , betatmp);
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
