__kernel void blend(    __global unsigned char* input1,
                        __global unsigned char* input2,
                        __global unsigned char* output,
                        const unsigned int height,
                        const unsigned int width,
                        const float alpha,
                        const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;

    output[pixIdx] = ((1-alpha) * input1[pixIdx]) + (alpha * input2[pixIdx]);
}