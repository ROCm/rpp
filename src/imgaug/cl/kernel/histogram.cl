#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

/*kernel
void partial_histogram_pln(__global unsigned char *input,
                            __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    unsigned int pixId;
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * channel;
    if (channel == 3)
    local uint tmp_histogram[ 768];
    if (channel ==1)
    local uint tmp_histogram[ 256 ];
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
        if(channel == 3) {
            unsigned char pixelR = input[id_x + id_y * width];
            unsigned char pixelG = input[id_x + id_y * width + (width * height)];
            unsigned char pixelB = input[id_x + id_y * width + (width * height * 2)];
            atomic_inc(&tmp_histogram[pixelR]);
            atomic_inc(&tmp_histogram[256+pixelG]);
            atomic_inc(&tmp_histogram[512+pixelB]);
        } else {
            unsigned char pixel = input[id_x];
            atomic_inc(&tmp_histogram[pixel]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 * channel))
    {
        if (tid < (256 * channel))
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
    }
    else
    {
        j = 256 * channel;
        indx = 0;
        do
        {
            if (tid < j)
            histogramPartial[group_indx + indx + tid] =
            tmp_histogram[indx + tid];
            j -= local_size;
            indx += local_size;
        } while (j > 0);
    }
}*/

kernel
void partial_histogram_pkd(__global unsigned char *input,
                           __global unsigned int *histogramPartial,
                           const unsigned int width,
                           const unsigned int height,
                           const unsigned int channel){

    //printf("inside");
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int local_size = (int)get_local_size(0) * (int)get_local_size(1);
    int group_indx = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * 256 * channel;
    //printf("%lu - %lu - %lu ",get_group_id(0), get_group_id(1), get_num_groups(0));
    unsigned int pixId;
    local uint tmp_histogram [768];
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
        unsigned char pixelG = input[pixId + 1];
        unsigned char pixelB = input[pixId + 2];
        atomic_inc(&tmp_histogram[pixelR]);
        atomic_inc(&tmp_histogram[256+pixelG]);
        atomic_inc(&tmp_histogram[512+pixelB]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_size >= (256 * channel))
    {
        //printf(" hist");
        if (tid < (256 * channel)){
            histogramPartial[group_indx + tid] = tmp_histogram[tid];
            //printf("%d",histogramPartial[tid]);
        }
    }
    else
    {
        //printf("g_idx %d",group_indx);
        j = 256 * channel;
        indx = 0;
        do
        {
            if (tid < j)
            {
                histogramPartial[group_indx + indx + tid] = tmp_histogram[ indx + tid];
                //printf("%d",histogramPartial[group_indx + indx + tid]);
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
    local uint tmp_histogram[256 * 3];
     
    tmp_histogram[tid] = histogramPartial[tid];
   // printf("inside %d", tmp_histogram[ tid]);
    
    group_indx = 256*channel;
    while (--n > 1)
    {
        tmp_histogram[tid] = tmp_histogram[tid] +  histogramPartial[group_indx + tid];
        //printf("inside  histopartial%d", histogramPartial[ group_indx + tid]);
        printf("inside %d", tmp_histogram[ tid]);
          
          //printf("inside %d", group_indx + tid);
        group_indx += 256*channel;
         
       
       // n = n-1;
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
    float normalize_factor = 255.0 / (height * width);
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int id_z = get_global_id(2);

    unsigned pixId;
    pixId = id_x + id_y * width + id_z * width * height;
    output[pixId] = cum_histogram[input[pixId] + id_z * 256 ] * normalize_factor;
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
    //printf("Inside histogram_equalize_pkd");
    float normalize_factor = 255.0 / (height * width);
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int id_z = get_global_id(2);
    unsigned pixId;
    pixId = id_x * channel + id_y * width * channel + id_z;
    //printf("%d", cum_histogram[input[pixId]]);
    output[pixId] = cum_histogram[input[pixId] + id_z * 256] * (normalize_factor);
    //output[pixId] = input[pixId];
}
