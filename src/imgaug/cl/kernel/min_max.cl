__kernel void reduction_max(
	const __global unsigned int* input,
    __global  unsigned int* partial_sum,
    __global unsigned int* shared_data,
    const unsigned int size
)
{
    unsigned int id_x = get_global_id(0);
    unsigned int locId_x = get_local_id(0);
    unsigned int grpDim_x = get_local_size(0);
    unsigned int grpId_x = get_group_id(0);

    shared_data[locId_x] = (id_x < size)? input[id_x] : 0;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if (id_x >= size) return;

    for (int stride=grpDim_x/2 ; stride > 0; stride=stride>>1 )
    {
        if (locId_x < stride)
        {
            // shared_data[id_x] += shared_data[id_x + s];
            shared_data[locId_x] = shared_data[locId_x + stride] > shared_data[locId_x]
                                ? shared_data[locId_x + stride] : shared_data[locId_x];

        }
       barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    if (locId_x == 0) partial_sum[grpId_x] = shared_data[locId_x];

}
