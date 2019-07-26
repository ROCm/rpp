__kernel void channel_extract_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int extractChannelNumber
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    int OPpixIdx = id_x + id_y * width;
    int IPpixIdx = OPpixIdx + extractChannelNumber * width * height;
    output[OPpixIdx] = input[IPpixIdx];
}
__kernel void channel_extract_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int extractChannelNumber
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;

    // int OPpixIdx = id_y * channel * width + id_x * channel;
    int OPpixIdx = id_y * width + id_x ;
    // int IPpixIdx = OPpixIdx + extractChannelNumber;
    int IPpixIdx = id_y * width * channel + id_x * channel + extractChannelNumber;
    output[OPpixIdx] = input[IPpixIdx];
    // output[OPpixIdx] = input[IPpixIdx];
    // output[OPpixIdx+1] = input[IPpixIdx];
    // output[OPpixIdx+2] = input[IPpixIdx];
}