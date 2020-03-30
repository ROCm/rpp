#include <hip/hip_runtime.h>
#define saturate_16u(value) ( (value) > 65535 ? 65535 : ((value) < 0 ? 0 : (value) ))
extern "C" __global__ void match_template_pkd(   unsigned char* input,
                     unsigned short* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                     unsigned char* template_img,
                    const unsigned int tHeight,
                    const unsigned int tWidth
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width * channel + id_x * channel;

    if(id_x + tWidth > width || id_y + tHeight > height)
    {
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + i] = 65535;
        }
        return;
    }      
    float up = 0;
    float down = 0;
    for(int i = 0 ; i < tHeight ; i++)
    {
        for(int j = 0 ; j < tWidth ; j++)
        {
            int tPixID = i * tWidth * channel + j * channel;
            int tPixel = (template_img[tPixID] + template_img[tPixID + 1] + template_img[tPixID + 2]) / 3;
            int index = pixIdx + i * width * channel + j * channel; 
            int pixel = (input[index] + input[index + 1] + input[index + 2]) / 3;
            up += (tPixel - pixel) * (tPixel - pixel);
            down += (pixel * pixel) * (tPixel * tPixel);
        }
    }
    down = sqrt(down);
    up = up / down;
    for(int i = 0 ; i < channel ; i++)
    {
        output[pixIdx + i] = (unsigned short)saturate_16u((65535 / 253) * up);
    }
}

extern "C" __global__ void match_template_pln(   unsigned char* input,
                     unsigned short* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                     unsigned char* template_img,
                    const unsigned int tHeight,
                    const unsigned int tWidth
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x ;
    unsigned int c = height * width;
    unsigned int tc = tHeight * tWidth;

    int tPixel = 0;
    int pixel = 0;

    if(id_x + tWidth > width || id_y + tHeight > height)
    {
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + i * c] = 65535;
        }
        return;
    }
    float up = 0;
    float down = 0;
    for(int i = 0 ; i < tHeight ; i++)
    {
        for(int j = 0 ; j < tWidth ; j++)
        {
            int tPixID = i * tWidth + j;
            tPixel = 0;
            for(int i = 0 ; i < channel ; i++)
            {    
                tPixel += template_img[tPixID + i * tc];
            }
            if(channel == 3)
            {
                tPixel = tPixel / 3;
            }
            int index = pixIdx + i * width * channel + j * channel; 
            for(int i = 0 ; i < channel ; i++)
            {   
                pixel = input[index + i * c];
            }
            if(channel == 3)
            {
                pixel = pixel / 3;
            }
            up += (tPixel - pixel) * (tPixel - pixel);
            down += (pixel * pixel) * (tPixel * tPixel);
        }
    }
    down = sqrt(down);
    up = up / down;
    for(int i = 0 ; i < channel ; i++)
    {
        output[pixIdx + i * c] = (unsigned short)saturate_16u((65535 / 253) * up);
    }
}