#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void random_shadow_pkd(
    const __global unsigned char* input,
    __global  unsigned char* output,
        const unsigned int height,
        const unsigned int width,
        const unsigned int channel
){
__kernel void random_shadow(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int pixIdx=(srcwidth*srcheight*id_z) + (srcwidth * id_y) + id_x;
    output[pixIdx] = input[pixIdx];
}
__kernel void random_shadow_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int pixIdx=(srcwidth*srcheight*id_z) + (srcwidth * id_y) + id_x;
    if(id_x >= x1 && id_x <= x2 && id_y >= y1 && id_y <=y2)
    {
        if(output[pixIdx] != input[pixIdx]/2)
        {    
            output[pixIdx] = input[pixIdx]/2;
        }
    }
}

__kernel void random_shadow_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int channel,
                    const unsigned int x1,
                    const unsigned int y1,
                    const unsigned int x2,
                    const unsigned int y2
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);

    if (id_x >= srcwidth || id_y >= srcheight || id_z >= channel) 
        return;
    int pixIdx = (channel * srcwidth * id_y) + (channel * id_x) + (id_z);
    
    if(id_x >= x1 && id_x <= x2 && id_y >= y1 && id_y <=y2)
    {
        if(output[pixIdx] != input[pixIdx]/2)
        {    
            output[pixIdx] = input[pixIdx]/2;
        }
    }

}