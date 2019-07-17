#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

kernel
void histogram_pkd(__global unsigned char *img, 
                             const int num_pixels_per_workitem,
                             const unsigned int src_height,
                             const unsigned int src_width,
                             __global uint *histogram){
    int     id_x = get_global_id(0);
    int     id_y = get_global_id(1);
    __local uint tmp_histogram[257 * 3];



                             }