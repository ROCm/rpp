#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
extern "C" __global__ void accumulate(   unsigned char* input1,
                             unsigned char* input2,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int res = input1[pixIdx] + input2[pixIdx];
    input1[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void accumulate_weighted(   unsigned char* input1,
                             unsigned char* input2,
                            const double alpha,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int res = (1 - alpha) * input1[pixIdx] + alpha * input2[pixIdx];
    input1[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void accumulate_squared(   unsigned char* input,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int res = (int)pow((float)input[pixIdx],(float)2);
    input[pixIdx] = saturate_8u(res);
}

extern "C" __global__ void accumulate_batch(   unsigned char* input1,
                                     unsigned char* input2,
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
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            input1[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
            pixIdx += inc[id_z];
        }
    }
}

__device__ unsigned char accumulate_weight_formula( unsigned char input_pixel1, unsigned char input_pixel2, float alpha){
    return saturate_8u(alpha * input_pixel1 + (1 - alpha) * input_pixel2);
}


extern "C" __global__ void accumulate_weighted_batch(   unsigned char* input1,
                                     unsigned char* input2,
                                     float *alpha,
                                     int *xroi_begin,
                                     int *xroi_end,
                                     int *yroi_begin,
                                     int *yroi_end,
                                     unsigned int *height,
                                     unsigned int *width,
                                     unsigned int *max_width,
                                     unsigned long *batch_index,
                                    const unsigned int channel,
                                     unsigned int  *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    unsigned char valuergb1, valuergb2;
    float alphatmp = alpha[id_z];
    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            valuergb1 = input1[pixIdx];
            valuergb2 = input2[pixIdx];
            input1[pixIdx] = accumulate_weight_formula(valuergb1, valuergb2, alphatmp);
            pixIdx += inc[id_z];
        }
    }
}

extern "C" __global__ void accumulate_squared_batch(   unsigned char* input,
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
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    int indextmp=0;
    unsigned long pixIdx = 0;

    pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(indextmp = 0; indextmp < channel; indextmp++){
            int pixel = saturate_8u(input[pixIdx] * input[pixIdx]);
            input[pixIdx] = saturate_8u(pixel);
            pixIdx += inc[id_z];
        }
    }
}