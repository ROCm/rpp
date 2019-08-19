#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void look_up_table_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    __global unsigned char* lutPtr,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int index = input[pixIdx] * channel + id_z;
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;
}

__kernel void look_up_table_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    __global unsigned char* lutPtr,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int index = input[pixIdx] + id_z * 256;
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;   
}