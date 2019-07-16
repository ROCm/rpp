#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

__kernel void gamma_correction(  __global unsigned char* a,
                                    __global unsigned char* b,
                                    const float gamma,
                                    const unsigned int height,
                                    const unsigned int width,
                                    const unsigned int channel
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    float temp; // for storing intermediate float converted value
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    int pixIdx = id_x + id_y * width + id_z * width * height;
    temp = a[pixIdx]/ 255.0;
    temp = pow(temp, gamma);
    temp = temp * 255;
    b[pixIdx] = saturate_8u(temp);
}
