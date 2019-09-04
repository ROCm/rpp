#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#define round(value) ( (value - (int)(value)) >=0.5 ? (value + 1) : (value))

kernel
void partial_histogram_pln(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 ;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 ;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x * width + id_y * width * height;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + width * height];
        unsigned char pixelB = input[pixId + 2 * width * height];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[pixelG]);
        atomic_inc(&tmp_histogram[pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}


kernel
void partial_histogram_pln1(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * channel;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 * channel;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        atomic_inc(&tmp_histogram[pixelR]);

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 * channel))
    {
        if (tid < (256 * channel)){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256 * channel;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}


kernel
void partial_histogram_pkd(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 ;
    unsigned int pixId;
    local uint tmp_histogram [256];
    int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int j = 256 ;
    int indx = 0;
    do
    {
        if (tid < j)
        tmp_histogram[indx+tid] = 0;
        j -= local_size;
        indx += local_size;
    } while (j > 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((id_x < width) && (id_y < height))
    {
        pixId = id_x * channel + id_y * width * channel;
        unsigned char pixelR = input[pixId];
        unsigned char pixelG = input[pixId + 1];
        unsigned char pixelB = input[pixId + 2];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[pixelG]);
        atomic_inc(&tmp_histogram[pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 ))
    {
        if (tid < (256 )){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
        }
    }
    else
    {
        j = 256;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
            }
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}


kernel void
histogram_sum_partial(global unsigned int *histogramPartial,
                      global unsigned int *histogram,
                      const unsigned int num_groups,
                      const unsigned int channel)
{
    int  tid = (int)get_global_id(0);
    int  group_indx;
    int  n = num_groups;
    local uint tmp_histogram[256];
     
    tmp_histogram[tid] = histogramPartial[tid];    
    group_indx = 256;
    while (--n > 1)
    {
        tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[group_indx + tid];
        group_indx += 256; 
    }
    histogram[tid] = tmp_histogram[tid];
}

kernel void
histogram_equalize_pln(global unsigned char *input,
                   global unsigned char *output,
                   global unsigned int *cum_histogram,
                   const unsigned int width,
                   const unsigned int height,
                   const unsigned int channel
                   )
{
    float normalize_factor = 255.0 / (height * width * channel);
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int id_z = get_global_id(2);
    unsigned pixId;
    pixId = id_x  + id_y * width + id_z * height * width;
    output[pixId] = cum_histogram[input[pixId]] * (normalize_factor);
}

kernel void
histogram_equalize_pkd(global unsigned char *input,
                   global unsigned char *output,
                   global unsigned int *cum_histogram,
                   const unsigned int width,
                   const unsigned int height,
                   const unsigned int channel
                   )
{
    float normalize_factor = 255.0 / (height * width * channel);
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int id_z = get_global_id(2);
    unsigned pixId;
    pixId = id_x * channel + id_y * width * channel + id_z;
    output[pixId] = round(cum_histogram[input[pixId]] * (normalize_factor));
}


