#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

unsigned int power(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
        sum += sum * a;
    return sum;
}

__kernel void local_binary_pattern_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * channel * width + id_x * channel + id_z;

    unsigned int pixel = 0;
    unsigned char neighborhood[9];

    if((id_x - 1) >= 0 && (id_y - 1) >= 0)
        neighborhood[0] = input [pixIdx - width * channel - channel];
    else
        neighborhood[0] = 0;
    
    if((id_y - 1) >= 0)
        neighborhood[1] = input [pixIdx - width * channel];
    else
        neighborhood[1] = 0;
    
    if((id_x + 1) <= width && (id_y - 1) >= 0)
        neighborhood[2] = input [pixIdx - width * channel + channel];
    else
        neighborhood[2] = 0;
    
    if((id_x + 1) <= width)
        neighborhood[3] = input [pixIdx + channel];
    else
        neighborhood[3] = 0;
    
    if((id_x + 1) <= width && (id_y + 1) <= height)
        neighborhood[4] = input [pixIdx + width * channel + channel];
    else
        neighborhood[4] = 0;

    if((id_y + 1) <= height)
        neighborhood[5] = input [pixIdx + width * channel];
    else
        neighborhood[5] = 0;
    
    if((id_x - 1) >= 0 && (id_y + 1) <= height)
        neighborhood[6] = input [pixIdx + width * channel -channel];
    else
        neighborhood[6] = 0;

    if((id_x - 1) >= 0)
        neighborhood[7] = input [pixIdx - channel];
    else
        neighborhood[7] = 0;

    neighborhood[8] = input[pixIdx];

    for(int i = 0 ; i < 8 ; i++)
    {
        if(neighborhood[i] - input[pixIdx] >= 0)
        {
            pixel += power(2, i);
        }
    }
    output[pixIdx] = saturate_8u(pixel);

}

__kernel void local_binary_pattern_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;
    
    unsigned char pixel = 0;
    unsigned char neighborhood[9];

    if((id_x - 1) >= 0 && (id_y - 1) >= 0)
        neighborhood[0] = input [pixIdx - width - 1];
    else
        neighborhood[0] = 0;
    
    if((id_y - 1) >= 0)
        neighborhood[1] = input [pixIdx - width];
    else
        neighborhood[1] = 0;
    
    if((id_x + 1) <= width && (id_y - 1) >= 0)
        neighborhood[2] = input [pixIdx - width + 1];
    else
        neighborhood[2] = 0;
    
    if((id_x + 1) <= width)
        neighborhood[3] = input [pixIdx + 1];
    else
        neighborhood[3] = 0;
    
    if((id_x + 1) <= width && (id_y + 1) <= height)
        neighborhood[4] = input [pixIdx + width + 1];
    else
        neighborhood[4] = 0;

    if((id_y + 1) <= height)
        neighborhood[5] = input [pixIdx + width];
    else
        neighborhood[5] = 0;
    
    if((id_x - 1) >= 0 && (id_y + 1) <= height)
        neighborhood[6] = input [pixIdx + width -1];
    else
        neighborhood[6] = 0;

    if((id_x - 1) >= 0)
        neighborhood[7] = input [pixIdx - 1];
    else
        neighborhood[7] = 0;

    neighborhood[8] = input[pixIdx];

    for(int i = 0 ; i < 8 ; i++)
    {
        if(neighborhood[i] - input[pixIdx] >= 0)
        {
            pixel += power(2, i);
        }
    }
    output[pixIdx] = saturate_8u(pixel);
}