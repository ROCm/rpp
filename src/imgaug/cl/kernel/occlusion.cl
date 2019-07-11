#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void occlude(  __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int x1,
                    const unsigned int x2,
                    const unsigned int y1,
                    const unsigned int y2,
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
    
    if(idx >= x1 && idx <=x2 && idx >=y1 && idx<=y2){
        int res = input1[pixIdx] + input2[pixIdx];
        output[pixIdx] = saturate_8u(res);
    }
    else
        output[pixIdx] = input1[pixIdx];
}