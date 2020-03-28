#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void ced_pln3_to_pln1_batch(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;

    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    int ch = height * width;
    float value = ((input[IPpixIdx] + input[IPpixIdx + ch] + input[IPpixIdx + ch * 2]) / 3);
    output[OPpixIdx] = (unsigned char)value ;
}

extern "C" __global__ void ced_pkd3_to_pln1_batch(   unsigned char* input,
                     unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    if (id_x >= width || id_y >= height) return;

    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x * (unsigned long)channel + (unsigned long)id_y * (unsigned long)width * (unsigned long)channel;
    float value = (input[IPpixIdx] + input[IPpixIdx + 1] + input[IPpixIdx + 2]) / 3;
    output[OPpixIdx] = (unsigned char)value ;
}

extern "C" __global__ void gaussian_pln_batch(    unsigned char* input,
                                     unsigned char* output,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel,
                                     float* kernal,
                                    const unsigned int kernalheight,
                                    const unsigned int kernalwidth,
                                    const unsigned long batchIndex,
                                    const unsigned int originalChannel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    unsigned long pixIdx, OPpixIdx;
    if(originalChannel == 1)
        pixIdx = batchIndex + (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
    else
        pixIdx = (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
    OPpixIdx = (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;

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
                unsigned long index = (unsigned long)pixIdx + (unsigned long)j + ((unsigned long)i * (unsigned long)width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[OPpixIdx] = saturate_8u(sum); 
}

__device__ unsigned int hcd_power(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
        sum += sum * a;
    return sum;
}

__device__ int hcd_calcSobelx(int a[3][3])
{
    int gx[3][3]={-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sum = 0;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += a[i][j] * gx[i][j];
        }
    }
    return sum;
}

__device__ int hcd_calcSobely(int a[3][3])
{
    int gy[3][3]={-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int sum = 0;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += a[i][j] * gy[i][j];
        }
    }
    return sum;
}

extern "C" __global__ void sobel_pln_batch(   unsigned char* input,
                                 unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int sobelType
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x;
    int value = 0;
    int value1 =0;
    int a[3][3];
    for(int i = -1 ; i <= 1 ; i++)
    {
        for(int j = -1 ; j <= 1 ; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = hcd_calcSobelx(a);
        value1 = hcd_calcSobely(a);
        value = hcd_power(value,2);
        value1 = hcd_power(value1,2);
        value = sqrt( (float)(value + value1));
        output[pixIdx] = saturate_8u(value);
        
    }
    if(sobelType == 1)
    {
        value = hcd_calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = hcd_calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}

extern "C" __global__ void harris_corner_detector_strength(   unsigned char* sobelX,
                     unsigned char* sobelY,
                     float* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize,
                    const float kValue,
                    const float threshold
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    float sumXX = 0, sumYY = 0, sumXY = 0, valX = 0, valY = 0, det = 0, trace = 0, pixel = 0;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                valX = sobelX[index];
                valY = sobelY[index];
                sumXX += (valX * valX);
                sumYY += (valY * valY);
                sumXY += (valX * valY);
            }
        }
    }
    det = (sumXX * sumYY) - (sumXY * sumXY);
    trace = sumXX + sumYY;
    pixel = (det) - (kValue * trace * trace);
    if (pixel > threshold)
    {
        output[pixIdx] = pixel;
    }
    else
    {
        output[pixIdx] = 0;
    }
}

extern "C" __global__ void harris_corner_detector_nonmax_supression(   float* input,
                     float* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int bound = (kernelSize - 1) / 2;
    float pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                if(input[index] > pixel)
                {
                    return;
                }
            }
        }
    }
    output[pixIdx] = input[pixIdx];  
}

extern "C" __global__ void harris_corner_detector_pln(   unsigned char* input,
                     float* inputFloat,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    int pixIdx = id_y * width + id_x;
    if (id_x >= width || id_y >= height || id_z >= channel || inputFloat[pixIdx] == 0) return;

    unsigned int kernelSize = 5;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                input[index] = 255;
                if(channel == 3)
                {
                    input[index + height * width] = 0;
                    input[index + height * width * 2] = 0;
                }
            }
        }
    } 
}

extern "C" __global__ void harris_corner_detector_pkd(   unsigned char* input,
                     float* inputFloat,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    int pixIdx = id_y * width + id_x;
    if (id_x >= width || id_y >= height || id_z >= channel || inputFloat[pixIdx] == 0) return;
    pixIdx = id_y * channel * width + id_x * channel;

    unsigned int kernelSize = 5;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                input[index] = 255;
                input[index+1] = 0;
                input[index+2] = 0;
            }
        }
    }
}

// extern "C" __global__ void harris_corner_detector_strength_batch(   unsigned char* sobelX,
//                      unsigned char* sobelY,
//                      float* output,
//                     const unsigned int height,
//                     const unsigned int width,
//                     const unsigned int channel,
//                     const unsigned int kernelSize,
//                     const float kValue,
//                     const float threshold
// )
// {
//     int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//     if (id_x >= width || id_y >= height || id_z >= channel) return;

//     float sumXX = 0, sumYY = 0, sumXY = 0, valX = 0, valY = 0, det = 0, trace = 0, pixel = 0;

//     int pixIdx = id_y * channel * width + id_x * channel + id_z;
//     int bound = (kernelSize - 1) / 2;
//     for(int i = -bound ; i <= bound ; i++)
//     {
//         for(int j = -bound ; j <= bound ; j++)
//         {
//             if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
//             {
//                 unsigned int index = pixIdx + (j * channel) + (i * width * channel);
//                 valX = sobelX[index];
//                 valY = sobelY[index];
//                 sumXX += (valX * valX);
//                 sumYY += (valY * valY);
//                 sumXY += (valX * valY);
//             }
//         }
//     }
//     det = (sumXX * sumYY) - (sumXY * sumXY);
//     trace = sumXX + sumYY;
//     pixel = (det) - (kValue * trace * trace);
//     if (pixel > threshold)
//     {
//         output[pixIdx] = pixel;
//     }
//     else
//     {
//         output[pixIdx] = 0;
//     }
// }

// extern "C" __global__ void harris_corner_detector_nonmax_supression_batch(   float* input,
//                      float* output,
//                     const unsigned int height,
//                     const unsigned int width,
//                     const unsigned int channel,
//                     const unsigned int kernelSize
// )
// {
//     int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//     if (id_x >= width || id_y >= height || id_z >= channel) return;

    
//     int pixIdx = id_y * width + id_x + id_z * width * height;
//     int bound = (kernelSize - 1) / 2;
//     float pixel = input[pixIdx];
//     for(int i = -bound ; i <= bound ; i++)
//     {
//         for(int j = -bound ; j <= bound ; j++)
//         {
//             if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
//             {
//                 unsigned int index = pixIdx + j + (i * width);
//                 if(input[index] > pixel)
//                 {
//                     return;
//                 }
//             }
//         }
//     }
//     output[pixIdx] = input[pixIdx];  
// }

// extern "C" __global__ void harris_corner_detector_pln_batch(  unsigned char* input,
//                                                  float* inputFloat,
//                                                 const unsigned int height,
//                                                 const unsigned int width,
//                                                 const unsigned int channel,
//                                                 const unsigned long batchIndex
// )
// {
//     int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//     unsigned long pixIdx = (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
//     if (id_x >= width || id_y >= height || id_z >= channel || inputFloat[pixIdx] == 0) return;
    
//     pixIdx = (unsigned long)batchIndex + (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
    
//     unsigned int kernelSize = 5;
//     int bound = (kernelSize - 1) / 2;
//     for(int i = -bound ; i <= bound ; i++)
//     {
//         for(int j = -bound ; j <= bound ; j++)
//         {
//             if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
//             {
//                 unsigned long index = (unsigned long)pixIdx + (unsigned long)j + ((unsigned long)i * (unsigned long)width);
//                 input[index] = 255;
//                 if(channel == 3)
//                 {
//                     input[index + (unsigned long)height * (unsigned long)width] = 0;
//                     input[index + (unsigned long)height * (unsigned long)width * 2] = 0;
//                 }
//             }
//         }
//     } 
// }

// extern "C" __global__ void harris_corner_detector_pkd_batch(  unsigned char* input,
//                                                  float* inputFloat,
//                                                 const unsigned int height,
//                                                 const unsigned int width,
//                                                 const unsigned int channel,
//                                                 const unsigned long batchIndex
// )
// {
//     int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//     unsigned long pixIdx = (unsigned long)id_y * (unsigned long)width + (unsigned long)id_x;
//     if (id_x >= width || id_y >= height || id_z >= channel || inputFloat[pixIdx] == 0) return;
    
//     pixIdx = (unsigned long)batchIndex + (unsigned long)id_y * (unsigned long)channel * (unsigned long)width + (unsigned long)id_x * (unsigned long)channel;

//     unsigned int kernelSize = 5;
//     int bound = (kernelSize - 1) / 2;
//     for(int i = -bound ; i <= bound ; i++)
//     {
//         for(int j = -bound ; j <= bound ; j++)
//         {
//             if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
//             {
//                 unsigned long index = (unsigned long)pixIdx + ((unsigned long)j * (unsigned long)channel) + ((unsigned long)i * (unsigned long)width * (unsigned long)channel);
//                 input[index] = 255;
//                 input[index+1] = 0;
//                 input[index+2] = 0;
//             }
//         }
//     }
// }