#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void contrast_stretch(  __global unsigned char *a,
                                __global unsigned char *c,
                                   const unsigned int min,
                                   const unsigned int max,
                               const unsigned int new_min,
                               const unsigned int new_max,
                               const unsigned short height,
                               const unsigned short width,
                               const unsigned short channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y* width + id_z * width * height;

    int res = a[pixIdx] - min) * (new_max - new_min)/((max - min) * 1.0) + new_min ;

    c[pixIdx] = saturate_8u(res);
}