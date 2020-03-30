#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void contrast_stretch(   unsigned char *input,
                                 unsigned char *output,
                                   const unsigned int min,
                                   const unsigned int max,
                               const unsigned int new_min,
                               const unsigned int new_max,
                               const unsigned int height,
                               const unsigned int width,
                               const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y* width + id_z * width * height;

    int res;
    res = (input[pixIdx] - min) * (new_max - new_min)/((max - min) * 1.0) + new_min ;

    output[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void contrast_batch(   unsigned char* input,
                                     unsigned char* output,
                                     unsigned int  min,
                                     unsigned int  max,
                                     unsigned int  *new_min,
                                     unsigned int  *new_max,
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
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    unsigned char valuergb;
    float tmpmin = min, tmpmax = max, tmpnew_max = new_max[id_z], tmpnew_min = new_min[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb = input[pixIdx];
            output[pixIdx] =  saturate_8u((input[pixIdx] - tmpmin) * (tmpnew_max - tmpnew_min)/((tmpmax - tmpmin) * 1.0) + tmpnew_min );
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