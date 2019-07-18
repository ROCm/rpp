#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

kernel
void histogram_pln(__global unsigned char *input, 
                             const int num_pixels_per_workitem,
                             const unsigned int src_height,
                             const unsigned int src_width,
                             const unsigned int channel,
                             __global uint *histogram){
    int     id_x = get_global_id(0);
    int     id_y = get_global_id(1);
    int     id_z = get_global_id(2);

    __local uint tmp_histogram[256 * 3];
    //int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * 3;
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    int pixIdx = id_x + id_y * src_width + id_z * src_height * src_height;

    atom_inc(&tmp_histogram[input[pixIdx] * id_z]);
    barrier(CLK_LOCAL_MEM_FENCE);

    barrier(CLK_LOCAL_MEM_FENCE);




    }