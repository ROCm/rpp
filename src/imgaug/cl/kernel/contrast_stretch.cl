#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void contrast_stretch(  __global unsigned char *input,
                                __global unsigned char *output,
                                   const unsigned int min,
                                   const unsigned int max,
                               const unsigned int new_min,
                               const unsigned int new_max,
                               const unsigned int height,
                               const unsigned int width,
                               const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y* width + id_z * width * height;

    int res;
    res = (input[pixIdx] - min) * (new_max - new_min)/((max - min) * 1.0) + new_min ;

    output[pixIdx] = saturate_8u(res);
}