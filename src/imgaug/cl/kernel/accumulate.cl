#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void accumulate(  __global unsigned char* input1,
                            __global unsigned char* input2,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int res = input1[pixIdx] + input2[pixIdx];
    input1[pixIdx] = saturate_8u(res);
}

__kernel void accumulate_weighted(  __global unsigned char* input1,
                            __global unsigned char* input2,
                            const double alpha,
                            const unsigned int height,
                            const unsigned int width,
                            const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    int res = (1 - alpha) * input1[pixIdx] + alpha * input2[pixIdx];
    input1[pixIdx] = saturate_8u(res);
}
