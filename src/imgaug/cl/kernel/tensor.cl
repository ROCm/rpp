#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
__kernel void tensor_add(   const unsigned int tensorDimension,
                    __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;
    
    unsigned int value = input1[pixIdx] + input2[pixIdx];
    output[pixIdx] = saturate_8u(value);
}

__kernel void tensor_subtract(   const unsigned int tensorDimension,
                    __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;
    
    unsigned int value = input1[pixIdx] - input2[pixIdx];
    output[pixIdx] = saturate_8u(value);
}

__kernel void tensor_multiply(   const unsigned int tensorDimension,
                    __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;
    
    unsigned int value = input1[pixIdx] * input2[pixIdx];
    output[pixIdx] = saturate_8u(value);
}

