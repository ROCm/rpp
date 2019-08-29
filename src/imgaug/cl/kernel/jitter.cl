#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
unsigned int xorshift(int pixid) {
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
	return res;
}
__kernel void jitter_pkd(  __global unsigned char* input,
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

    
    int pixIdx = id_y * channel * width + id_x * channel;
    int nhx = xorshift(pixIdx) % (kernelSize);
    int nhy = xorshift(pixIdx) % (kernelSize);
    int bound = (kernelSize - 1) / 2;
    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
    {
        int index = ((id_y - bound) * channel * width) + ((id_x - bound) * channel) + (nhy * channel * width) + (nhx * channel);
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + i] = input[index + i];  
        }
    }
    else 
    {
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + i] = input[pixIdx + i];  
        }
    }
}

__kernel void jitter_pln(  __global unsigned char* input,
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

    
    int pixIdx = id_y * width + id_x;
    int channelPixel = height * width;
    int nhx = xorshift(pixIdx) % (kernelSize);
    int nhy = xorshift(pixIdx) % (kernelSize);
    int bound = (kernelSize - 1) / 2;
    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= height - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= width - 1)
    {
        int index = ((id_y - bound) * width) + (id_x - bound) + (nhy * width) + (nhx);
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + (height * width * i)] = input[index + (height * width * i)];  
        }
    }
    else 
    {
        for(int i = 0 ; i < channel ; i++)
        {
            output[pixIdx + (height * width * i)] = input[pixIdx + (height * width * i)];  
        }
    }
}