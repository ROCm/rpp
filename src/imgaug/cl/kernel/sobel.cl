#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

unsigned int power(unsigned int a, unsigned int b)
{
    unsigned int sum = 1;
    for(int i = 0; i < b; i++)
        sum += sum * a;
    return sum;
}

int calcSobelx(int a[3][3])
{
    int gx[3][3]={-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sum = 0;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += a[i][j] * gx[i][j];
        }
    }
    return sum;
}
int calcSobely(int a[3][3])
{
    int gy[3][3]={-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int sum = 0;
    for(int i = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += a[i][j] * gy[i][j];
        }
    }
    return sum;
}

__kernel void sobel_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int sobelType
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int value = 0;
    int value1 =0;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int a[3][3];
    for(int i = -1 ; i <= 1 ; i++)
    {
        for(int j = -1 ; j <= 1 ; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power(value,2);
        value1 = power(value1,2);
        value = sqrt( (float)(value + value1));
        output[pixIdx] = saturate_8u(value);
        
    }
    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}

__kernel void sobel_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int sobelType
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int value = 0;
    int value1 =0;
    int a[3][3];
    for(int i = -1 ; i <= 1 ; i++)
    {
        for(int j = -1 ; j <= 1 ; j++)
        {
            if(id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                a[i+1][j+1] = input[index];
            }
            else
            {
                a[i+1][j+1] = 0;
            }
        }
    }
    if(sobelType == 2)
    {
        value = calcSobelx(a);
        value1 = calcSobely(a);
        value = power(value,2);
        value1 = power(value1,2);
        value = sqrt( (float)(value + value1));
        output[pixIdx] = saturate_8u(value);
        
    }
    if(sobelType == 1)
    {
        value = calcSobely(a);
        output[pixIdx] = saturate_8u(value);
    }
    if(sobelType == 0)
    {
        value = calcSobelx(a);
        output[pixIdx] = saturate_8u(value);
    }
}