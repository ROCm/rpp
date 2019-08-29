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
    int pos;
    int max = 0;
    for (int i = 0; i < counter; i++)          
    {
        for (int j = i; j < counter; j++)  
        {
            if (max < c[j]) 
            {
                max = c[j];
                pos = j;
            }
        }
        max = 0;
        int temp = c[pos];
        c[pos] = c[i];
        c[i] = temp;
    }
    counter = kernelSize * bound + bound + 1;
    output[pixIdx] = c[counter];
}

__kernel void median_filter_pln(  __global unsigned char* input,
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
    int pos;
    int max = 0;
    for (int i = 0; i < counter; i++)          
    {
        for (int j = i; j < counter; j++)  
        {
            if (max < c[j]) 
            {
                max = c[j];
                pos = j;
            }
        }
        max = 0;
        int temp = c[pos];
        c[pos] = c[i];
        c[i] = temp;
    }
    counter = kernelSize * bound + bound + 1;
    output[pixIdx] = c[counter];
}