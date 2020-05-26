#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

kernel void
crop_batch( __global unsigned char* input, // Input Tensor NHCW or NCHW (Depending on Planar or Packed)
            __global unsigned char* output, // Output Tensor (For now of type RPP8U), FLOAT32 will be given depending on necessity
            __global unsigned int *dst_height,
            __global unsigned int *dst_width,
            __global unsigned int *src_width,
            __global unsigned int *start_x,
            __global unsigned int *start_y,
            __global unsigned int *max_src_width,
            __global unsigned int *max_dst_width,
            __global unsigned long *src_batch_index,
            __global unsigned long *dst_batch_index,
            const unsigned int channel,
            //const unsigned int batch_size,
            __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
            __global unsigned int *dst_inc,
            const int plnpkdindex // use 1 pln 3 for pkd
        )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y  * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = input[src_pixIdx];
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }     
    }
    else
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0; 
            dst_pixIdx += dst_inc[id_z];
        }   
    }
}

kernel void
crop_fp32_batch(__global float* input, // Input Tensor NHCW or NCHW (Depending on Planar or Packed)
                __global float* output, // Output Tensor (For now of type RPP8U), FLOAT32 will be given depending on necessity
                __global unsigned int *dst_height,
                __global unsigned int *dst_width,
                __global unsigned int *src_width,
                __global unsigned int *start_x,
                __global unsigned int *start_y,
                __global unsigned int *max_src_width,
                __global unsigned int *max_dst_width,
                __global unsigned long *src_batch_index,
                __global unsigned long *dst_batch_index,
                const unsigned int channel,
                //const unsigned int batch_size,
                __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
                __global unsigned int *dst_inc,
                const int plnpkdindex // use 1 pln 3 for pkd
            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y  * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = input[src_pixIdx];
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }     
    }
    else
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0; 
            dst_pixIdx += dst_inc[id_z];
        }   
    }
}

kernel void
crop_fp16_batch(__global half* input, // Input Tensor NHCW or NCHW (Depending on Planar or Packed)
                __global half* output, // Output Tensor (For now of type RPP8U), FLOAT32 will be given depending on necessity
                __global unsigned int *dst_height,
                __global unsigned int *dst_width,
                __global unsigned int *src_width,
                __global unsigned int *start_x,
                __global unsigned int *start_y,
                __global unsigned int *max_src_width,
                __global unsigned int *max_dst_width,
                __global unsigned long *src_batch_index,
                __global unsigned long *dst_batch_index,
                const unsigned int channel,
                //const unsigned int batch_size,
                __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
                __global unsigned int *dst_inc,
                const int plnpkdindex // use 1 pln 3 for pkd
            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y  * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = input[src_pixIdx];
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }     
    }
    else
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = 0; 
            dst_pixIdx += dst_inc[id_z];
        }   
    }
}
