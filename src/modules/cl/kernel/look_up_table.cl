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

__kernel void look_up_table_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned char* lutPtr,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    long pixIdx = 0;
    if(id_x < width[id_z] && id_y < height[id_z])
    {
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        int luptrIndex = id_z * plnpkdindex * 256;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {   
            for(indextmp = 0; indextmp < channel; indextmp++)
            {
                luptrIndex = (id_z * plnpkdindex * 256) + (input[pixIdx] * plnpkdindex);
                output[pixIdx] = saturate_8u(lutPtr[luptrIndex]);
                pixIdx += inc[id_z];
            }
        }
        else if((id_x < width[id_z] ) && (id_y < height[id_z]))
        {
            for(indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}