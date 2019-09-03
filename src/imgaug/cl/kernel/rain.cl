#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
unsigned int xorshift(int pixid) {
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
	return res;
}

__kernel void rain(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y*width*channel + id_x*channel + id_z;
    int pixel=input[pixIdx]+output[pixIdx];
    output[pixIdx]=saturate_8u(pixel);
}

__kernel void rain_pkd(__global unsigned char *output,
                       const unsigned int height,
                       const unsigned int width,
                       const unsigned int channel,
                       const unsigned int pixelDistance,
                        const unsigned int rainWidth,
                        const unsigned int rainHeight,
                        const float transparency)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height)
        return;
    int pixIdx = id_y * width * channel + id_x * channel;
    output[pixIdx] = 0;
    output[pixIdx + 1] = 0;
    output[pixIdx + 2] = 0;
    int rand;

    if (pixIdx % pixelDistance == 0)
    {
        int rand_id = xorshift(pixIdx) % 997;
        rand_id -= rand_id % 3;

        for (int i = 0; i < rainWidth; i++)
        {
            for (int j = 0; j < rainHeight; j++)
            {
                if (id_x + i + rainWidth <= width && id_y + j < height)
                {
                    int id = i * channel + j * width * channel;
                    for(int k = 0;k < channel ; k++)
                    {
                        if(channel == 0)
                            output[pixIdx + rand_id + id + k] = transparency * 196;
                        else if(channel == 1)
                            output[pixIdx + rand_id + id + k] = transparency * 226;
                        else
                            output[pixIdx + rand_id + id + k] = transparency * 255;
                    }
                }
            }
        }
    }
}

__kernel void rain_pln(__global unsigned char *output,
                       const unsigned int height,
                       const unsigned int width,
                       const unsigned int channel,
                       const unsigned int pixelDistance,
                        const unsigned int rainWidth,
                        const unsigned int rainHeight,
                        const float transparency)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height)
        return;
    int pixIdx = id_y * width + id_x;
    int channelSize = width * height;
    output[pixIdx] = 0;
    if (channel > 1)
    {
        output[pixIdx + channelSize] = 0;
        output[pixIdx + 2 * channelSize] = 0;
    }
    int rand;

    if (pixIdx % pixelDistance == 0)
    {
        int rand_id = xorshift(pixIdx) % 997;
        for (int i = 0; i < rainWidth; i++)
        {
            for (int j = 0; j < rainHeight; j++)
            {
                if (id_x + i + rainWidth <= width && id_y + j < height)
                {
                    int id = i + j * width;
                    for(int k = 0;k < channel ; k++)
                    {
                        if(channel == 0)
                            output[pixIdx + rand_id + id + ( k * channelSize)] = transparency * 196;
                        else if(channel == 1)
                            output[pixIdx + rand_id + id + ( k * channelSize)] = transparency * 226;
                        else
                            output[pixIdx + rand_id + id + ( k * channelSize)] = transparency * 255;

                    }
                }
            }
        }
        
    }
}