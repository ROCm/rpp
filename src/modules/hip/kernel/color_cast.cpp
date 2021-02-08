#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void color_cast_batch(
    unsigned char *input,
    unsigned char *output,
    unsigned char* user_input_r,  //which color to cast for red
    unsigned char* user_input_g,  //which color to cast for green
    unsigned char* user_input_b,  //which color to cast for blue
    float *alpha,
    int *xroi_begin,
    int *xroi_end,
    int *yroi_begin,
    int *yroi_end,
    unsigned int *height,
    unsigned int *width,
    unsigned int *max_width,
    unsigned long *batch_index,
    const unsigned int channel,
    unsigned int *inc,  // use width * height for pln and 1 for pkd
    unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind 
) {

 int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

  unsigned char user_input[3]={user_input_r[id_z],user_input_g[id_z],user_input_b[id_z]};
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
 
  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
   
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      unsigned char input_pixel1 = input[src_pix_idx];
      unsigned char input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}

// extern "C" __global__ void color_cast_batch_fp16(
//     half *input,
//     half *output,
//     unsigned char* user_input_r,  //which color to cast for red
//     unsigned char* user_input_g,  //which color to cast for green
//     unsigned char* user_input_b,  //which color to cast for blue
//     float *alpha,
//     int *xroi_begin,
//     int *xroi_end,
//     int *yroi_begin,
//     int *yroi_end,
//     unsigned int *height,
//     unsigned int *width,
//     unsigned int *max_width,
//     unsigned long *batch_index,
//     const unsigned int channel,
//     unsigned int *inc,  // use width * height for pln and 1 for pkd
//     unsigned int *dstinc , // use width * height for pln and 1 for pkd
//     int in_plnpkdind,         // use 1 pln 3 for pkd
//     int out_plnpkdind 
// ) {

//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//   half user_input[3]={(half)(user_input_r[id_z]/255.0 ),(half)(user_input_g[id_z]/255.0) ,(half)(user_input_b[id_z]/255.0) };
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

//   float alphatmp=alpha[id_z];
 
//   if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
//       (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
   
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       half input_pixel1 = input[src_pix_idx];
//       half input_pixel2 = user_input[indextmp];
//       output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

//       src_pix_idx += inc[id_z];
//       dst_pix_idx += dstinc[id_z];
//     }
//   } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       output[dst_pix_idx] = 0;
//       dst_pix_idx += dstinc[id_z];
//     }
//   }
// }

extern "C" __global__ void color_cast_batch_fp32(
    float *input,
    float *output,
    unsigned char* user_input_r,  //which color to cast for red
    unsigned char* user_input_g,  //which color to cast for green
    unsigned char* user_input_b,  //which color to cast for blue
    float *alpha,
    int *xroi_begin,
    int *xroi_end,
    int *yroi_begin,
    int *yroi_end,
    unsigned int *height,
    unsigned int *width,
    unsigned int *max_width,
    unsigned long *batch_index,
    const unsigned int channel,
    unsigned int *inc,  // use width * height for pln and 1 for pkd
    unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind
) {

  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

  float divisor = 255;
  float user_input[3]={user_input_r[id_z]/divisor ,user_input_g[id_z]/divisor ,user_input_b[id_z]/divisor};
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
 
  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
   
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      float input_pixel1 = input[src_pix_idx];
      float input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);
                           
      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}

extern "C" __global__ void color_cast_batch_int8(
    char *input,
    char *output,
    unsigned char* user_input_r,  //which color to cast for red
    unsigned char* user_input_g,  //which color to cast for green
    unsigned char* user_input_b,  //which color to cast for blue
    float *alpha,
    int *xroi_begin,
    int *xroi_end,
    int *yroi_begin,
    int *yroi_end,
    unsigned int *height,
    unsigned int *width,
    unsigned int *max_width,
    unsigned long *batch_index,
    const unsigned int channel,
    unsigned int *inc,  // use width * height for pln and 1 for pkd
    unsigned int *dstinc , // use width * height for pln and 1 for pkd
    int in_plnpkdind,         // use 1 pln 3 for pkd
    int out_plnpkdind
) {

  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

  int subtrahend = 128;
  char user_input[3]={(char)(user_input_r[id_z]-subtrahend), (char)(user_input_g[id_z]-subtrahend), (char)(user_input_b[id_z]-subtrahend)};
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  float alphatmp=alpha[id_z];
 
  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
   
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      char input_pixel1 = input[src_pix_idx];
      char input_pixel2 = user_input[indextmp];
      output[dst_pix_idx] =(alphatmp * input_pixel1 + (1 - alphatmp) * input_pixel2);

      src_pix_idx += inc[id_z];
      dst_pix_idx += dstinc[id_z];
    }
  } else if((id_x < width[id_z] ) && (id_y < height[id_z])){
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      output[dst_pix_idx] = 0;
      dst_pix_idx += dstinc[id_z];
    }
  }
}