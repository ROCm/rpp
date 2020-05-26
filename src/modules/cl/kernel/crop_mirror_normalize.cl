#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))


// kernel void
// mirror_normalize_pln(global unsigned char *input,
//                    global unsigned char *output, // Can be made to FP32 based on the requirement
//                    const float mean,
//                    const float std_dev,
//                    const uint flip,
//                    //const uint format_toggle,// Can be done outside to avoid branch
//                    const unsigned int width,
//                    const unsigned int height,
//                    const unsigned int channel
//                    )
// {
//     unsigned int id_x = get_global_id(0);
//     unsigned int id_y = get_global_id(1);
//     unsigned int id_z = get_global_id(2);
//         //Condition should be put depending on the flip or no-flip for this
//     unsigned int pixId =  (width-1 - id_x) + id_y * width + id_z * width * height;
//     // 
//     unsigned int oPixId =  id_x + id_y * width + id_z * width * height; // output pix id will change according to format as well //TODO
//     if(flip)
//     {
//         output[oPixId] = (unsigned char)saturate_8u(255.0 * ((input[pixId] - mean) / std_dev));
//     }
//     else
//     {
//         output[oPixId] = (unsigned char)saturate_8u(255.0 * ((input[oPixId] - mean) / std_dev));

//     }
    
// }

// kernel void
// mirror_normalize_pkd(global unsigned char *input,
//                    global unsigned char *output, // Can be made to FP32 based on the requirement
//                    const float mean,
//                    const float std_dev,
//                    const uint flip,
//                    //const uint format_toggle,// Can be done outside to avoid branch
//                    const unsigned int width,
//                    const unsigned int height,
//                    const unsigned int channel
//                    )
// {
//     unsigned int id_x = get_global_id(0);
//     unsigned int id_y = get_global_id(1);
//     unsigned int id_z = get_global_id(2);
//     unsigned int pixId =  ((width-1 - id_x) + id_y * width) * channel + id_z; 
//     unsigned int oPixId =  (id_x + id_y * width) * channel + id_z; // output pix id will change according to format as well //TODO
//     if(flip)
//     {
//         output[oPixId] = (unsigned char)saturate_8u(255.0 * ((input[pixId] - mean) / std_dev));
//     }
//     else
//     {
//         output[oPixId] = (unsigned char)saturate_8u(255.0 * ((input[oPixId] - mean) / std_dev));

//     }
// }

// uchar4 normalize(uchar4 in_pixel, float mean, float std_dev)
// {
//     uchar4 out_pixel;
//     out_pix.x = saturate_8u((255.0 * ((in_pixel.x - mean) / std_dev)));
//     out_pix.y = saturate_8u((255.0 * ((in_pixel.y - mean) / std_dev)));
//     out_pix.z = saturate_8u((255.0 * ((in_pixel.z - mean) / std_dev)));
//     out_pix.x =  0.0; //(255.0 * ((in_pixel.x - mean) / std_dev));
// }


kernel void
crop_mirror_normalize_batch (   __global unsigned char* input, // Input Tensor NHCW or NCHW (Depending on Planar or Packed)
                                __global unsigned char* output, // Output Tensor (For now of type RPP8U), FLOAT32 will be given depending on necessity
                                __global unsigned int *dst_height,
                                __global unsigned int *dst_width,
                                __global unsigned int *src_width,
                                __global unsigned int *start_x,
                                __global unsigned int *start_y,
                                __global float *mean,
                                __global float *std_dev,
                                __global unsigned int *flip,
                                __global unsigned int *max_src_width,
                                __global unsigned int *max_dst_width,
                                __global unsigned long *src_batch_index,
                                __global unsigned long *dst_batch_index,
                                const unsigned int channel,
                               // const unsigned int batch_size,
                                __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                __global unsigned int *dst_inc,
                                const int plnpkdindex // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    const float local_mean        = mean[id_z];
    const float local_std_dev     = std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    if(local_flip == 1) {src_pixIdx =  src_batch_index[id_z] + ((max_src_width[id_z] -1 -(id_x + start_x[id_z])) +  (id_y + start_y[id_z]) * max_src_width[id_z]) * plnpkdindex; }
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = input[src_pixIdx];//(unsigned char)saturate_8u(255.0 * ((input[src_pixIdx] - local_mean) / local_std_dev)); 
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
crop_mirror_normalize_batch_fp16(__global half *input, // Input Tensor NHCW or NCHW (Depending on Planar or Packed)
                                __global half *output, // Output Tensor (For now of type RPP8U), FLOAT32 will be given depending on necessity
                                __global unsigned int *dst_height,
                                __global unsigned int *dst_width,
                                __global unsigned int *src_width,
                                __global unsigned int *start_x,
                                __global unsigned int *start_y,
                                __global float *mean,
                                __global float *std_dev,
                                __global unsigned int *flip,
                                __global unsigned int *max_src_width,
                                __global unsigned int *max_dst_width,
                                __global unsigned long *src_batch_index,
                                __global unsigned long *dst_batch_index,
                                const unsigned int channel,
                               // const unsigned int batch_size,
                                __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                __global unsigned int *dst_inc,
                                const int plnpkdindex // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    const half local_mean        = (half)mean[id_z];
    const half local_std_dev     = (half)std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    if(local_flip == 1) {src_pixIdx =  src_batch_index[id_z] + ((max_src_width[id_z] -1 -(id_x + start_x[id_z])) +  (id_y + start_y[id_z]) * max_src_width[id_z]) * plnpkdindex; }
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (input[src_pixIdx] - local_mean) / local_std_dev; 
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
crop_mirror_normalize_batch_fp32(__global float *input, // Input Tensor NHCW or NCHW (Depending on Planar or Packed)
                                __global float *output, // Output Tensor (For now of type RPP8U), FLOAT32 will be given depending on necessity
                                __global unsigned int *dst_height,
                                __global unsigned int *dst_width,
                                __global unsigned int *src_width,
                                __global unsigned int *start_x,
                                __global unsigned int *start_y,
                                __global float *mean,
                                __global float *std_dev,
                                __global unsigned int *flip,
                                __global unsigned int *max_src_width,
                                __global unsigned int *max_dst_width,
                                __global unsigned long *src_batch_index,
                                __global unsigned long *dst_batch_index,
                                const unsigned int channel,
                               // const unsigned int batch_size,
                                __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                __global unsigned int *dst_inc,
                                const int plnpkdindex // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    const float local_mean        =  mean[id_z];
    const float local_std_dev     =  std_dev[id_z];
    const unsigned int local_flip = flip[id_z];
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    if(local_flip == 1) {src_pixIdx =  src_batch_index[id_z] + ((max_src_width[id_z] -1 -(id_x + start_x[id_z])) +  (id_y + start_y[id_z]) * max_src_width[id_z]) * plnpkdindex; }
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = (input[src_pixIdx] - local_mean) / local_std_dev; 
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
crop_batch( __global unsigned char* input, 
                                __global unsigned char* output, 
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
                               // const unsigned int batch_size,
                                __global unsigned int *src_inc, // use width * height for pln and 1 for pkd
                                __global unsigned int *dst_inc,
                                const int plnpkdindex // use 1 pln 3 for pkd
                            )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int indextmp=0;
    unsigned long src_pixIdx   =  src_batch_index[id_z] + (id_x + start_x[id_z]  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex;//((id_x + start_x[id_z])  + (id_y + start_y[id_z]) * max_src_width[id_z] ) * plnpkdindex ;
    unsigned long dst_pixIdx   =  dst_batch_index[id_z] + (id_x  + id_y * max_dst_width[id_z] ) * plnpkdindex;   // output pix id will change according to format as well //TODO
    if((id_x < dst_width[id_z]) && (id_y < dst_height[id_z]))
    {
        for(indextmp = 0; indextmp < channel; indextmp++)
        {
            output[dst_pixIdx] = input[src_pixIdx];//(unsigned char)saturate_8u(255.0 * ((input[src_pixIdx] - local_mean) / local_std_dev)); 
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