#define saturate_16u(value) ( (value) > 65535 ? 65535 : ((value) < 0 ? 0 : (value) ))
__kernel void match_template_pkd(  __global unsigned char* input,
                    __global unsigned short* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global unsigned char* template,
                    const unsigned int tHeight,
                    const unsigned int tWidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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
            int tPixel = (template[tPixID] + template[tPixID + 1] + template[tPixID + 2]) / 3;
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

__kernel void match_template_pln(  __global unsigned char* input,
                    __global unsigned short* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global unsigned char* template,
                    const unsigned int tHeight,
                    const unsigned int tWidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
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
                tPixel += template[tPixID + i * tc];
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