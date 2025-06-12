/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable


__kernel void color_cast_batch(
    __global unsigned char *input,
    __global unsigned char *output,
    __global unsigned char* user_input_r,  //which color to cast for red
    __global unsigned char* user_input_g,  //which color to cast for green
    __global unsigned char* user_input_b,  //which color to cast for blue
    __global float *alpha,
    __global int *xroi_begin,
    __global int *xroi_end,
    __global int *yroi_begin,
    __global int *yroi_end,
    __global unsigned int *height,
    __global unsigned int *width,
    __global unsigned int *max_width,
    __global unsigned long *batch_index,
    const unsigned int channel,
    __global unsigned int *inc,  // use width * height for pln and 1 for pkd
    __global unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind
) {

 int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);

  unsigned char user_input[3]={user_input_r[id_z],user_input_g[id_z],user_input_b[id_z]};
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
  unsigned char input_pixel1, input_pixel2;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {

    for (indextmp = channel-1; indextmp >= 0; indextmp--) {
      input_pixel1 = input[src_pix_idx];
      input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}



__kernel void color_cast_batch_fp16(
    __global half *input,
    __global half *output,
    __global unsigned char* user_input_r,  //which color to cast for red
    __global unsigned char* user_input_g,  //which color to cast for green
    __global unsigned char* user_input_b,  //which color to cast for blue
    __global float *alpha,
    __global int *xroi_begin,
    __global int *xroi_end,
    __global int *yroi_begin,
    __global int *yroi_end,
    __global unsigned int *height,
    __global unsigned int *width,
    __global unsigned int *max_width,
    __global unsigned long *batch_index,
    const unsigned int channel,
    __global unsigned int *inc,  // use width * height for pln and 1 for pkd
    __global unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind
) {

  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);

  half user_input[3]={(half)(user_input_r[id_z]/255.0 ),(half)(user_input_g[id_z]/255.0) ,(half)(user_input_b[id_z]/255.0) };
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
  half input_pixel1, input_pixel2;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {

    for (indextmp = channel-1; indextmp >= 0; indextmp--) {
      input_pixel1 = input[src_pix_idx];
      input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}



__kernel void color_cast_batch_fp32(
    __global float *input,
    __global float *output,
    __global unsigned char* user_input_r,  //which color to cast for red
    __global unsigned char* user_input_g,  //which color to cast for green
    __global unsigned char* user_input_b,  //which color to cast for blue
    __global float *alpha,
    __global int *xroi_begin,
    __global int *xroi_end,
    __global int *yroi_begin,
    __global int *yroi_end,
    __global unsigned int *height,
    __global unsigned int *width,
    __global unsigned int *max_width,
    __global unsigned long *batch_index,
    const unsigned int channel,
    __global unsigned int *inc,  // use width * height for pln and 1 for pkd
    __global unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind
) {

  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);

  float user_input[3]={user_input_r[id_z]/255.0 ,user_input_g[id_z]/255.0 ,user_input_b[id_z]/255.0};
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
  float input_pixel1, input_pixel2;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {

    for (indextmp = channel-1; indextmp >= 0; indextmp--) {
      input_pixel1 = input[src_pix_idx];
      input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}



__kernel void color_cast_batch_int8(
    __global char *input,
    __global char *output,
    __global unsigned char* user_input_r,  //which color to cast for red
    __global unsigned char* user_input_g,  //which color to cast for green
    __global unsigned char* user_input_b,  //which color to cast for blue
    __global float *alpha,
    __global int *xroi_begin,
    __global int *xroi_end,
    __global int *yroi_begin,
    __global int *yroi_end,
    __global unsigned int *height,
    __global unsigned int *width,
    __global unsigned int *max_width,
    __global unsigned long *batch_index,
    const unsigned int channel,
    __global unsigned int *inc,  // use width * height for pln and 1 for pkd
    __global unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind
) {

  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);

  char user_input[3]={user_input_r[id_z]-128 ,user_input_g[id_z]-128 ,user_input_b[id_z]-128};
  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
  char input_pixel1, input_pixel2;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {

    for (indextmp = channel-1; indextmp >= 0; indextmp--) {
      input_pixel1 = input[src_pix_idx];
      input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}

