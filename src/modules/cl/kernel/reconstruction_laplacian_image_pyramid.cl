#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

__kernel void resize_pln (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/dest_width ;
    float y_ratio = ((float)(source_height -1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    A = srcPtr[x + y * source_width + id_z * source_height * source_width];
    B = srcPtr[x + 1  + y * source_width + id_z * source_height * source_width];
    C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+1) + (y+1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}

__kernel void resize_pkd (  __global unsigned char* srcPtr,
                            __global unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/dest_width ;
    float y_ratio = ((float)(source_height -1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[x * channel + y * source_width * channel + id_z];
    B = srcPtr[(x +1) * channel + y * source_width * channel + id_z];
    C = srcPtr[x * channel + (y+1) * source_width * channel + id_z];
    D = srcPtr[(x+1) * channel + (y+1) * source_width * channel + id_z];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                  C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
    dstPtr[pixId] =  saturate_8u(pixVal);

}

__kernel void gaussian_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + (j * channel) + (i * width * channel);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = saturate_8u(sum);
}

__kernel void gaussian_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    __global float* kernal,
                    const unsigned int kernalheight,
                    const unsigned int kernalwidth
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_y * width + id_x + id_z * width * height;
    int boundx = (kernalwidth - 1) / 2;
    int boundy = (kernalheight - 1) / 2;
    int sum = 0;
    int counter = 0;
    for(int i = -boundy ; i <= boundy ; i++)
    {
        for(int j = -boundx ; j <= boundx ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                sum += input[index] * kernal[counter];
            }
            counter++;
        }
    }
    output[pixIdx] = saturate_8u(sum); 
}

__kernel void reconstruction_laplacian_image_pyramid_pkd(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int height2,
                    const unsigned int width2,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x * channel + id_y * width * channel + id_z;
    output[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
}

__kernel void reconstruction_laplacian_image_pyramid_pln(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int height2,
                    const unsigned int width2,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * height * width;
    output[pixIdx] = saturate_8u(input1[pixIdx] + input2[pixIdx]);
}