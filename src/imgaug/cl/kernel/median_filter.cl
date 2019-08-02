#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define SIZE 7*7
__kernel void median_filter_pkd(  __global unsigned char* input,
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

    int c[SIZE];
    int counter = 0;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int bound = (kernelSize - 1) / 2;
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                c[counter] = input[index];
            }
            else
                c[counter] = 0;
            counter++;
        }
    }
    for (int i = 0; i < counter - 1; i++)          
    {
        for (int j = 0; j < counter - i - 1; j++)  
        {
            if (c[j] > c[j+1]) 
            {
                int temp = c[i];
                c[i] = c[j];
                c[j] = temp;
            }
        }
    }
    counter = kernelSize * bound + bound;
    output[pixIdx] = c[counter];
}

__kernel void median_filter_pln(  __global unsigned char* input,
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

    int c[SIZE];
    int counter = 0;
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                c[counter] = input[index];
            }
            else
                c[counter] = 0;
            counter++;
        }
    }
    for (int i = 0; i < counter - 1; i++)          
    {
        for (int j = 0; j < counter - i - 1; j++)  
        {
            if (c[j] > c[j+1]) 
            {
                int temp = c[i];
                c[i] = c[j];
                c[j] = temp;
            }
        }
    }
    counter = kernelSize * bound + bound;
    output[pixIdx] = c[counter];
    output[pixIdx] = pixel; 
}