#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
unsigned int xorshift(int pixid) {
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + pixid;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
	return res;
}
__kernel void gaussian(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const float mean,
                    const float sigma,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    float res = input1[pixIdx] + input2[pixIdx];
    output[pixIdx] = saturate_8u(res);
}

__kernel void snp_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int initialPixel,
                    const unsigned int pixelDistance,
                    __global unsigned char* randPtr
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y*width*channel + id_x*channel;
    int rand;
    
    if(pixIdx % pixelDistance == 0 )
    {
        for(int i=0;i<channel;i++)
            output[pixIdx + i] = input[pixIdx+i];
        int rand_id = xorshift(pixIdx) % 60;
        rand_id-=rand_id%3;
        rand=(rand_id%2)?0:255;
        for(int i=0;i<channel;i++)
            output[pixIdx + i + rand_id] = rand;
    }
    else
        for(int i=0;i<channel;i++)
            output[pixIdx+i] = input[pixIdx+i];
}




__kernel void snp_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int initialPixel,
                    const unsigned int pixelDistance,
                    __global unsigned char* randPtr
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height) return;

    int pixIdx = id_y*width + id_x;
    int channelSize= width*height;
    
    int rand;
    
    if(pixIdx % pixelDistance == 0 )
    {
        for(int i=0;i<channel;i++)
            output[pixIdx+channelSize*i] = input[pixIdx+channelSize*i];
        int rand_id = xorshift(pixIdx) % 60;
        rand_id-=rand_id%3;
        rand=(rand_id%2)?0:255;
        for(int i=0;i<channel;i++)
            output[pixIdx+channelSize*i + rand_id] = rand;
    }
    else
        for(int i=0;i<channel;i++)
            output[pixIdx+channelSize*i] = input[pixIdx+channelSize*i];
}