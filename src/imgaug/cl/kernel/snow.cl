#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
unsigned int xorshift(int pixid)
{
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^ (t >> 8));
    return res;
}

__kernel void snow(__global unsigned char *input,
                   __global unsigned char *output,
                   const unsigned int height,
                   const unsigned int width,
                   const unsigned int channel)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height)
        return;

    int pixIdx = id_y * width * channel + id_x * channel + id_z;
    int pixel = input[pixIdx] + output[pixIdx];
    output[pixIdx] = saturate_8u(pixel);
}

__kernel void snow_pkd(__global unsigned char *output,
                       const unsigned int height,
                       const unsigned int width,
                       const unsigned int channel,
                       const unsigned int pixelDistance)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height)
        return;
    int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};
    const unsigned int snowWidth = width / 200 + 1;
    float transparency = 0.5;
    int pixIdx = id_y * width * channel + id_x * channel;
    output[pixIdx] = 0;
    output[pixIdx + 1] = 0;
    output[pixIdx + 2] = 0;
    int rand;

    if (pixIdx % pixelDistance == 0)
    {
        int rand_id = xorshift(pixIdx) % 100;
        rand_id -= rand_id % 3;

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                if (id_x + i + 5 <= width && id_y + j < height)
                {
                    int id = i * channel + j * width * channel;
                    for(int k = 0;k < channel ; k++)
                        output[pixIdx + rand_id + id + k] = snow_mat[i][j];
                }
            }
        }
    }
}

__kernel void snow_pln(__global unsigned char *output,
                       const unsigned int height,
                       const unsigned int width,
                       const unsigned int channel,
                       const unsigned int pixelDistance)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height)
        return;
    int snow_mat[5][5] = {{0,50,75,50,0}, {40,80,120,80,40}, {75,120,255,120,75}, {40,80,120,80,40}, {0,50,75,50,0}};
    const unsigned int snowWidth = width / 200 + 1;
    float transparency = 0.5;
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
        int rand_id = xorshift(pixIdx) % 60;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                if (id_x + i + 5 <= width && id_y + j < height)
                {
                    int id = i + j * width;
                    for(int k = 0;k < channel ; k++)
                        output[pixIdx + rand_id + id + ( k * channelSize)] = snow_mat[i][j];
                }
            }
        }
        
    }
}