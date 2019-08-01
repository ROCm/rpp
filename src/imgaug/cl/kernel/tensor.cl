__kernel void tensor_add(   const unsigned int tensorDimension,
                    __global unsigned int* tensorDimensionValues,
                    __global unsigned char* input1,
                    __global unsigned char* input2,
                    __global unsigned char* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    printf("Hello");
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    int pixIdx;
    if(tensorDimension == 1)
        pixIdx = id_x;
    else if(tensorDimension == 2)
        pixIdx = id_x + id_y * a ;
    else
        pixIdx = id_y * c * a + id_x * c + id_z;
    output[pixIdx] = input1[pixIdx] + input2[pixIdx];
}

__kernel void tensor_subtract(   const unsigned int tensorDimension,
                    __global unsigned int* tensorDimensionValues,
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
    int pixIdx;
    if(tensorDimension == 1)
        pixIdx = id_x;
    else if(tensorDimension == 2)
        pixIdx = id_x + id_y * a ;
    else
        pixIdx = id_y * c * a + id_x * c + id_z;

    output[pixIdx] = input1[pixIdx] - input2[pixIdx];
}

__kernel void tensor_multiply(   const unsigned int tensorDimension,
                    __global unsigned int* tensorDimensionValues,
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
    int pixIdx;
    if(tensorDimension == 1)
        pixIdx = id_x;
    else if(tensorDimension == 2)
        pixIdx = id_x + id_y * a ;
    else
        pixIdx = id_y * c * a + id_x * c + id_z;

    output[pixIdx] = input1[pixIdx] * input2[pixIdx];
}

