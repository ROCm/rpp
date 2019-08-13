__kernel void channel_combine_pln(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* input3,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx1 = IPpixIdx;
    int OPpixIdx2 = IPpixIdx + width * height;
    int OPpixIdx3 = IPpixIdx + 2 * width * height;

    output[OPpixIdx1] = input1[IPpixIdx];
    output[OPpixIdx2] = input2[IPpixIdx];
    output[OPpixIdx3] = input3[IPpixIdx];
}
__kernel void channel_combine_pkd(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* input3,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int IPpixIdx = id_x + id_y * width;
    int OPpixIdx = IPpixIdx * channel;
    output[OPpixIdx] = input1[IPpixIdx];
    output[OPpixIdx + 1] = input2[IPpixIdx];
    output[OPpixIdx + 2] = input3[IPpixIdx];
}