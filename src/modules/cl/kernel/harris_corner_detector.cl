#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void ced_pln3_to_pln1_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    int ch = height * width;
    float value = ((input[IPpixIdx] + input[IPpixIdx + ch] + input[IPpixIdx + ch * 2]) / 3);
    output[OPpixIdx] = (unsigned char)value ;
}

__kernel void ced_pkd3_to_pln1_batch(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned long batchIndex
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    unsigned long OPpixIdx = (unsigned long)id_x + (unsigned long)id_y * (unsigned long)width;
    unsigned long IPpixIdx = (unsigned long)batchIndex + (unsigned long)id_x * (unsigned long)channel + (unsigned long)id_y * (unsigned long)width * (unsigned long)channel;
    float value = (input[IPpixIdx] + input[IPpixIdx + 1] + input[IPpixIdx + 2]) / 3;
    output[OPpixIdx] = (unsigned char)value ;
}

__kernel void gaussian_pln_batch(   __global unsigned char* input,
                                    __global unsigned char* output,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel,
                                    __global float* kernal,
                                    const unsigned int kernalheight,
                                    const unsigned int kernalwidth,
                                    const unsigned long batchIndex,
                                    const unsigned int originalChannel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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

unsigned int power(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
        sum += sum * a;
    return sum;
}

int calcSobelx(int a[3][3])
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

int calcSobely(int a[3][3])
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

__kernel void sobel_pln_batch(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int sobelType
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power(value,2);
        value1 = power(value1,2);
        value = sqrt( (float)(value + value1));
        output[pixIdx] = saturate_8u(value);
        
    }
    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}

__kernel void harris_corner_detector_strength(  __global unsigned char* sobelX,
                    __global unsigned char* sobelY,
                    __global float* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize,
                    const float kValue,
                    const float threshold
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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

__kernel void harris_corner_detector_nonmax_supression(  __global float* input,
                    __global float* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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

__kernel void harris_corner_detector_pln(  __global unsigned char* input,
                    __global float* inputFloat,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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

__kernel void harris_corner_detector_pkd(  __global unsigned char* input,
                    __global float* inputFloat,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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
// __kernel void harris_corner_detector_strength_batch(  __global unsigned char* sobelX,
//                     __global unsigned char* sobelY,
//                     __global float* output,
//                     const unsigned int height,
//                     const unsigned int width,
//                     const unsigned int channel,
//                     const unsigned int kernelSize,
//                     const float kValue,
//                     const float threshold
// )
// {
//     int id_x = get_global_id(0);
//     int id_y = get_global_id(1);
//     int id_z = get_global_id(2);
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

// __kernel void harris_corner_detector_nonmax_supression_batch(  __global float* input,
//                     __global float* output,
//                     const unsigned int height,
//                     const unsigned int width,
//                     const unsigned int channel,
//                     const unsigned int kernelSize
// )
// {
//     int id_x = get_global_id(0);
//     int id_y = get_global_id(1);
//     int id_z = get_global_id(2);
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

// __kernel void harris_corner_detector_pln_batch( __global unsigned char* input,
//                                                 __global float* inputFloat,
//                                                 const unsigned int height,
//                                                 const unsigned int width,
//                                                 const unsigned int channel,
//                                                 const unsigned long batchIndex
// )
// {
//     int id_x = get_global_id(0);
//     int id_y = get_global_id(1);
//     int id_z = get_global_id(2);
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

// __kernel void harris_corner_detector_pkd_batch( __global unsigned char* input,
//                                                 __global float* inputFloat,
//                                                 const unsigned int height,
//                                                 const unsigned int width,
//                                                 const unsigned int channel,
//                                                 const unsigned long batchIndex
// )
// {
//     int id_x = get_global_id(0);
//     int id_y = get_global_id(1);
//     int id_z = get_global_id(2);
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