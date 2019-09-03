#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void pixelate_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    if (x * 7 >= width || y * 7 >= height || c >= channel) return;

    y = y * 7;
    x = x * 7;
    int sum = 0;
    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width && y + i >= 0 && x + j >= 0)
            {    
                sum += input[((y + i) * width * channel + (x + j) * channel + c)];
            }
        }
    }
    sum /= 49;

    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width)
            {    
                output[((y + i) * width * channel + (x + j) * channel + c)] = saturate_8u(sum);
            }
        }
    }
    
}

__kernel void pixelate_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    if (x * 7 >= width || y * 7 >= height || c >= channel) return;

    y = y * 7;
    x = x * 7;
    int sum = 0;

    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width && y + i >= 0 && x + j >= 0)
            {    
                sum += input[(y + i) * width + (x + j) + c * height * width];
            }
        }
    }
    sum /= 49;

    for(int i = 0 ; i < 7 ; i++)
    {
        for(int j = 0 ; j < 7 ; j++)
        {
            if(y + i < height && x + j < width)
            {    
                output[(y + i) * width + (x + j) + c * height * width] = saturate_8u(sum);
            }
        }
    }
}