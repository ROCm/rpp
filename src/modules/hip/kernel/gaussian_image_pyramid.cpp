#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void gaussian_image_pyramid_pkd(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                     float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel || id_x % 2 != 0 || id_y % 2 != 0) return;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int outPixIdx = (id_y / 2) * channel * width + (id_x / 2) * channel + id_z;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = saturate_8u(sum);
}

extern "C" __global__ void gaussian_image_pyramid_pln(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                     float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel || id_x % 2 != 0 || id_y % 2 != 0) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int outPixIdx = (id_y / 2) * width + (id_x / 2) + id_z * width * height;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[outPixIdx] = saturate_8u(sum); 
}

__device__ float gaussian_generate(int x,int y, float stdDev)
{
    float res,pi=3.14;
    res= 1 / (2 * pi * stdDev * stdDev);
    float exp1,exp2;
    exp1= - (x*x) / (2*stdDev*stdDev);
    exp2= - (y*y) / (2*stdDev*stdDev);
    exp1= exp1+exp2;
    exp1=exp(exp1);
    res*=exp1;
	return res;
}


extern "C" __global__ void gaussian_image_pyramid_batch(   unsigned char* input,
                                     unsigned char* output,
                                     unsigned int *kernelSize,
                                     float *stdDev,
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
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;    int kernelSizeTemp = kernelSize[id_z];
    int indextmp=0;
    long pixIdx = 0;
    long outPixIdx = 0;
    int bound = (kernelSizeTemp - 1) / 2;
    float r = 0, g = 0, b = 0;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        outPixIdx = batch_index[id_z] + ((id_x/2)  + (id_y/2) * max_width[id_z] ) * plnpkdindex ;
            for(int i = -bound ; i <= bound ; i++)
            {
                for(int j = -bound ; j <= bound ; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        float gaussianvalue=gaussian_generate(j, i, stdDev[id_z]) / gaussian_generate(0.0, 0.0, stdDev[id_z]);
                        r += ((float)input[index]) * gaussianvalue ;
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            g += ((float)input[index]) * gaussianvalue ;
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            b += ((float)input[index]) * gaussianvalue ;
                        }
                    }
                }
            }
            r /= (kernelSize[id_z] * kernelSize[id_z]);
            g /= (kernelSize[id_z] * kernelSize[id_z]);
            b /= (kernelSize[id_z] * kernelSize[id_z]);
            output[outPixIdx] = saturate_8u(r);
            if(channel == 3)
            {
                output[outPixIdx + inc[id_z]] = saturate_8u(g);
                output[outPixIdx + inc[id_z] * 2] = saturate_8u(b);
            }
    }
}