#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void water_batch(  unsigned char* input,
                            unsigned char* output,
                            float *ampl_x,
                            float *ampl_y,
                            float *freq_x,
                            float *freq_y,
                            float *phase_x,
                            float *phase_y,
                            int *xroi_begin,
                            int *xroi_end,
                            int *yroi_begin,
                            int *yroi_end,
                            unsigned int *height,
                            unsigned int *width,
                            unsigned int *max_height,
                            unsigned int *max_width,
                            unsigned long *batch_index,
                            const unsigned int channel,
                            unsigned int *src_inc,
                            unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                output[dst_pix_id] = input[src_pix_id];
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(int indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}

extern "C" __global__ void water_batch_fp32(  float* input,
                            float* output,
                            float *ampl_x,
                            float *ampl_y,
                            float *freq_x,
                            float *freq_y,
                            float *phase_x,
                            float *phase_y,
                            int *xroi_begin,
                            int *xroi_end,
                            int *yroi_begin,
                            int *yroi_end,
                            unsigned int *height,
                            unsigned int *width,
                            unsigned int *max_height,
                            unsigned int *max_width,
                            unsigned long *batch_index,
                            const unsigned int channel,
                            unsigned int *src_inc,
                            unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                output[dst_pix_id] = input[src_pix_id];
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(int indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}

// extern "C" __global__ void water_batch_fp16(  half* input,
//                             half* output,
//                             float *ampl_x,
//                             float *ampl_y,
//                             float *freq_x,
//                             float *freq_y,
//                             float *phase_x,
//                             float *phase_y,
//                             int *xroi_begin,
//                             int *xroi_end,
//                             int *yroi_begin,
//                             int *yroi_end,
//                             unsigned int *height,
//                             unsigned int *width,
//                             unsigned int *max_height,
//                             unsigned int *max_width,
//                             unsigned long *batch_index,
//                             const unsigned int channel,
//                             unsigned int *src_inc,
//                             unsigned int *dst_inc, // use width * height for pln and 1 for pkd
//                             const int in_plnpkdind,
//                             const int out_plnpkdind // use 1 pln 3 for pkd
//                             )
// {
//     int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//     half valuergb;
//     float water_wave_x, water_wave_y;
//     float ampl_x_temp = ampl_x[id_z];
//     float ampl_y_temp = ampl_y[id_z];
//     float freq_x_temp = freq_x[id_z];
//     float freq_y_temp = freq_y[id_z];
//     float phase_x_temp = phase_x[id_z];
//     float phase_y_temp = phase_y[id_z];
//     int img_width = width[id_z];

//     int img_height = height[id_z];
//     int indextmp=0;
//     long dst_pix_id = 0;
//     dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
//     long src_pix_id = 0;
//     water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
//     water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
//     src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
//     if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
//     {   
//         for(indextmp = 0; indextmp < channel; indextmp++)
//         {
//             if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
//             {
//                 valuergb = input[src_pix_id];
//                 output[dst_pix_id] = valuergb;
//             }
//             dst_pix_id += dst_inc[id_z];
//             src_pix_id += src_inc[id_z];
//         }
//     }
//     else if((id_x < width[id_z] ) && (id_y < height[id_z])){
//             for(indextmp = 0; indextmp < channel; indextmp++){
//                 output[dst_pix_id] = input[dst_pix_id];
//                 dst_pix_id += dst_inc[id_z];
//             }
//         }
// }

extern "C" __global__ void water_batch_int8(  char* input,
                            char* output,
                            float *ampl_x,
                            float *ampl_y,
                            float *freq_x,
                            float *freq_y,
                            float *phase_x,
                            float *phase_y,
                            int *xroi_begin,
                            int *xroi_end,
                            int *yroi_begin,
                            int *yroi_end,
                            unsigned int *height,
                            unsigned int *width,
                            unsigned int *max_height,
                            unsigned int *max_width,
                            unsigned long *batch_index,
                            const unsigned int channel,
                            unsigned int *src_inc,
                            unsigned int *dst_inc, // use width * height for pln and 1 for pkd
                            const int in_plnpkdind,
                            const int out_plnpkdind // use 1 pln 3 for pkd
                            )
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    float water_wave_x, water_wave_y;
    float ampl_x_temp = ampl_x[id_z];
    float ampl_y_temp = ampl_y[id_z];
    float freq_x_temp = freq_x[id_z];
    float freq_y_temp = freq_y[id_z];
    float phase_x_temp = phase_x[id_z];
    float phase_y_temp = phase_y[id_z];
    long dst_pix_id = 0;
    dst_pix_id = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * out_plnpkdind ;
    long src_pix_id = 0;
    water_wave_x = id_x + ampl_x_temp * sin((freq_x_temp * id_y) + phase_x_temp);
    water_wave_y = id_y + ampl_y_temp * cos((freq_y_temp * id_x) + phase_y_temp);
    src_pix_id = batch_index[id_z] + ((int)water_wave_x + (int)water_wave_y * max_width[id_z]) * in_plnpkdind;
    if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
    {   
        for(int indextmp = 0; indextmp < channel; indextmp++)
        {
            if(src_pix_id >= 0 && src_pix_id < (batch_index[id_z] + (max_width[id_z] * max_height[id_z] * channel)))
            {
                output[dst_pix_id] = input[src_pix_id];
            }
            dst_pix_id += dst_inc[id_z];
            src_pix_id += src_inc[id_z];
        }
    }
    else if((id_x < width[id_z] ) && (id_y < height[id_z])){
            for(int indextmp = 0; indextmp < channel; indextmp++){
                output[dst_pix_id] = input[dst_pix_id];
                dst_pix_id += dst_inc[id_z];
            }
        }
}