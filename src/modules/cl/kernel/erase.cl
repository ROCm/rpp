#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void erase_batch(
    __global unsigned char *input,
    __global unsigned char *output, 
    __global unsigned int *src_width, 
    __global unsigned int *roi_index,
    __global unsigned int *num_of_boxex,
    __global unsigned int *start_x, __global unsigned int *start_y, 
    __global unsigned int *end_x, __global unsigned int *end_y,
    __global unsigned char *color, // set of 3 colours for color images 
    __global unsigned int *max_src_width,
    __global unsigned int *max_dst_width,
    __global unsigned long *src_batch_index,
    __global unsigned long *dst_batch_index, const unsigned int channel,
    __global unsigned int *src_inc, 
    __global unsigned int *dst_inc,
    const int in_plnpkdind, const int out_plnpkdind 
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  int in_roi = 0;
  for(int i = 0; i < num_of_boxes[id_z]; i++)
  {
      //if in that specific ROI region - then make it 1
  }
  if(in_roi)
  {
      color_index = roi_index[id_z] * channel;
      for(int i = 0; i < num_of_boxes[id_z]; i++)
      {
        for (indextmp = 0; indextmp < channel; indextmp++) 
        {
            output[dst_pixIdx] = color[roi_index[id_z] * channel];
            src_pixIdx += src_inc[id_z];
            dst_pixIdx += dst_inc[id_z];
        }
      //Fill the color associate with that specific ROI
    }
  }
 else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = input[src_pix_idx];
      dst_pixIdx += dst_inc[id_z];
    }
  }
}