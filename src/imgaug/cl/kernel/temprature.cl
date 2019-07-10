#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void temprature_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float modificationValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int pixIdx = id_x + id_y * width + id_z * width * height;
    
    if(pixIdx >= 0 && pixIdx <= width*height)
    {
        int res = input[pixIdx] + modificationValue;
        output[pixIdx] = saturate_8u(res);
    }
    
    else if(pixIdx >= width*height*(channel-1) && pixIdx <= width*height*channel)
    {
        int res = input[pixIdx] - modificationValue;
        output[pixIdx] = saturate_8u(res);        
    }
    else
    {
        output[pixIdx] = input[pixIdx];
    }
}
__kernel void temprature_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const float modificationValue
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    
    int pixIdx = id_x + id_y * width + id_z * width * height;
    
    if(pixIdx % 3 == 0)
    {
        int res = input[pixIdx] + modificationValue;
        output[pixIdx] = saturate_8u(res);
    }
    
    else if(pixIdx % 3 == 2)
    {
        int res = input[pixIdx] - modificationValue;
        output[pixIdx] = saturate_8u(res);        
    }

    else
    {
        output[pixIdx] = input[pixIdx];
    }
}