#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void randomcropletterbox_planar(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int dstheight,
                    const unsigned int dstwidth,
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
    int width=x2-x1+6;
    int height=y2-y1+6;
    int displacement=((dstwidth-width)*id_y)+(channel*srcheight*srcwidth);
    int OPpixIdx = (id_x) + (id_y * srcwidth) + (id_z * srcwidth * srcheight) + displacement;
    int IPpixIdx = ((x1) + (y1 *srcwidth) + (id_z * srcwidth * srcheight)) + ((id_x) + (id_y * srcwidth));
    output[OPpixIdx] = input[IPpixIdx];

    if(id_y <=5 || id_y >= dstheight-5 || id_x <= 5 || id_x >= dstwidth-5)
        output[OPpixIdx]=0;
}

__kernel void randomcropletterbox_packed(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int srcheight,
                    const unsigned int srcwidth,
                    const unsigned int dstheight,
                    const unsigned int dstwidth,
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

    int width=x2-x1+6;
    int height=y2-y1+6;
    int displacement=((dstwidth-width)*channel*id_y);
    int OPpixIdx = (channel * id_y * width) + (id_x * channel) + (id_z) + displacement;
    int IPpixIdx = (srcwidth * channel * y1 + x1 * channel) + (channel * id_y * srcwidth) + (id_x * channel) + (id_z);
    output[OPpixIdx] = input[IPpixIdx];

    if(id_y <=5 || id_y >= height-5 || id_x <= 5 || id_x >= width-5)
        output[OPpixIdx]=0;

}