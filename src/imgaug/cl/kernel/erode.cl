#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void erode_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
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

    
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(!(((id_x + j) < bound) || ((id_x + j) > width - bound) || ((id_y + i) < bound) || ((id_y + i) > height - bound)))
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                if(input[index] < pixel)
                {
                    pixel = input[index];
                }
            }
        }
    }
    output[pixIdx] = pixel;
}

__kernel void erode_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    unsigned char pixel = 0;
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int bound = (kernelSize - 1) / 2;
    
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(!(((id_x + j) < bound) || ((id_x + j) > width - bound) || ((id_y + i) < bound) || ((id_y + i) > height - bound)))
            {
                unsigned int index = pixIdx + j + (i * width);
                if(input[index] < pixel)
                {
                    pixel = input[index];
                }
            }
        }
    }
    output[pixIdx] = pixel;    
}