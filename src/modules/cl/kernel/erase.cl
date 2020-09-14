#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void
erase_batch(__global unsigned char *input, __global unsigned char *output,
            __global unsigned int *box_info, __global unsigned char *colors,
            __global unsigned int *box_offset,
            __global unsigned int *no_of_boxes,
            __global unsigned int *src_height, __global unsigned int *src_width,
            __global unsigned int *max_width,
            __global unsigned long *batch_index, const unsigned int channel,
            __global unsigned int *src_inc, __global unsigned int *dst_inc,
            const int in_plnpkdind, const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  uchar3 pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i) * 3;
      pixel.x = colors[temp];
      pixel.y = colors[temp + 1];
      pixel.z = colors[temp + 2];
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
  if (is_erase) {
    output[dst_pix_idx] = pixel.x;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.y;
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = pixel.z;
    dst_pix_idx += dst_inc[id_z];
  } else {
    output[dst_pix_idx] = input[src_pix_idx];
    src_pix_idx += dst_inc[id_z];
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
    src_pix_idx += dst_inc[id_z];
    dst_pix_idx += dst_inc[id_z];
    output[dst_pix_idx] = input[src_pix_idx];
    src_pix_idx += dst_inc[id_z];
    dst_pix_idx += dst_inc[id_z];
  }
}

kernel void erase_batch_1(
    __global unsigned char *input, __global unsigned char *output,
    __global unsigned int *box_info, __global unsigned char *colors,
    __global unsigned int *box_offset, __global unsigned int *no_of_boxes,
    __global unsigned int *src_height, __global unsigned int *src_width,
    __global unsigned int *max_width, __global unsigned long *batch_index,
    const unsigned int channel, __global unsigned int *src_inc,
    __global unsigned int *dst_inc, const int in_plnpkdind,
    const int out_plnpkdind) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  uchar pixel;
  uint l_box_offset = box_offset[id_z];
  bool is_erase = false;
  for (int i = 0; i < no_of_boxes[id_z]; i++) {
    int temp = (l_box_offset + i) * 4;
    if (id_x >= box_info[temp] && id_x < box_info[temp + 2] &&
        id_y >= box_info[temp + 1] && id_y < box_info[temp + 3]) {
      is_erase = true;
      temp = (l_box_offset + i);
      pixel = colors[temp];
    }
  }
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
  if (is_erase) {
    output[dst_pix_idx] = pixel;
  } else {
    output[dst_pix_idx] = input[src_pix_idx];
  }
}