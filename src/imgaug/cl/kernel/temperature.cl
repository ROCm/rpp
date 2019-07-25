#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void temperature_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float modificationValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;
    int pixIdx = id_x + id_y * width;
    int c = width * height;
    
    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);
    if( channel > 1)
    {
        output[pixIdx + c] = input[pixIdx + c];
        res = input[pixIdx + c * 2] - modificationValue;
        output[pixIdx + c * 2] = saturate_8u(res);
    }
}
__kernel void temperature_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float modificationValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    if (id_x >= width || id_y >= height) return;
    
    int pixIdx = id_y * width * channel + id_x * channel;
    
    int res = input[pixIdx] + modificationValue;
    output[pixIdx] = saturate_8u(res);

    output[pixIdx + 1] = input[pixIdx + 1];

    res = input[pixIdx+2] - modificationValue;
    output[pixIdx+2] = saturate_8u(res);
}