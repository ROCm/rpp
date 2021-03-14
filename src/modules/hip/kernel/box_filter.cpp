#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))


extern "C" __global__ void box_filter_batch(   unsigned char* input,
                                     unsigned char* output,
                                     unsigned int *kernelSize,
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
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    unsigned char valuer,valuer1,valueg,valueg1,valueb,valueb1;
    int kernelSizeTemp = kernelSize[id_z];

    int bound = (kernelSizeTemp - 1) / 2;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        long pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {
            int r = 0, g = 0, b = 0;
            for(int i = -bound ; i <= bound ; i++)
            {
                for(int j = -bound ; j <= bound ; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        r += input[index];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            g += input[index];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            b += input[index];
                        }
                    }
                    else
                    {
                        r = 0;
                        if(channel == 3)
                        {
                            g = 0;
                            b = 0;
                        }
                        break;
                    }
                }
            }

            if(id_x >= bound && id_x <= width[id_z] - bound - 1 && id_y >= bound && id_y <= height[id_z] - bound - 1 )
            {
                int temp = (int)(r / (kernelSizeTemp * kernelSizeTemp));
                output[pixIdx] = saturate_8u(temp);
                if(channel == 3)
                {
                    temp = (int)(g / (kernelSizeTemp * kernelSizeTemp));
                    output[pixIdx + inc[id_z]] = saturate_8u(temp);
                    temp = (int)(b / (kernelSizeTemp * kernelSizeTemp));
                    output[pixIdx + inc[id_z] * 2] = saturate_8u(temp);
                }
            }
            else
            {
                for(int indextmp = 0; indextmp < channel; indextmp++)
                {
                    output[pixIdx] = input[pixIdx];
                    pixIdx += inc[id_z];
                }
            }
        }
        else if((id_x < width[id_z] ) && (id_y < height[id_z]))
        {
            for(int indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}