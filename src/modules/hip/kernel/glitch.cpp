#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void glitch_batch(
    unsigned char *input, unsigned char *output,
    unsigned int *x_offset_r, unsigned int *y_offset_r,
    unsigned int *x_offset_g, unsigned int *y_offset_g,
    unsigned int *x_offset_b, unsigned int *y_offset_b,
    int *xroi_begin, int *xroi_end, int *yroi_begin,
    int *yroi_end, unsigned int *height,
    unsigned int *width, unsigned int *max_width,
    unsigned long *batch_index, const unsigned int channel,
    unsigned int *inc,    // use width * height for pln and 1 for pkd
    unsigned int *dstinc, // use width * height for pln and 1 for pkd
    int in_plnpkdind,              // use 1 pln 3 for pkd
    int out_plnpkdind) {

  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
  
  output[dst_pix_idx] = input[src_pix_idx];
  output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
  output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

  unsigned char R, G, B;
  R = B = G = 0;
  int x_r, x_g, x_b, y_r, y_g, y_b;
  // R
  x_r = (id_x + x_offset_r[id_z]);
  y_r = (id_y + y_offset_r[id_z]);
 
  // G
  x_g = (id_x + x_offset_g[id_z]);
  y_g = (id_y + y_offset_g[id_z]);
 
  // B
  x_b = (id_x + x_offset_b[id_z]);
  y_b = (id_y + y_offset_b[id_z]);
  
  // R
  if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) &&
      (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
  {
    R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = R;
    dst_pix_idx += dstinc[id_z];
  }
    
  // G
  if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) &&
      (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
  {
    G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = G;
    dst_pix_idx += dstinc[id_z];
  }

  // B
  if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) &&
      (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
  {
    B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = B;
    dst_pix_idx += dstinc[id_z];
  }
}

// extern "C" __global__ void glitch_batch_fp16(
//     half *input, half *output,
//     unsigned int *x_offset_r, unsigned int *y_offset_r,
//     unsigned int *x_offset_g, unsigned int *y_offset_g,
//     unsigned int *x_offset_b, unsigned int *y_offset_b,
//     int *xroi_begin, int *xroi_end, int *yroi_begin,
//     int *yroi_end, unsigned int *height,
//     unsigned int *width, unsigned int *max_width,
//     unsigned long *batch_index, const unsigned int channel,
//     unsigned int *inc,    // use width * height for pln and 1 for pkd
//     unsigned int *dstinc, // use width * height for pln and 1 for pkd
//     int in_plnpkdind,              // use 1 pln 3 for pkd
//     int out_plnpkdind) {

//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//   int indextmp = 0;
//   unsigned long src_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
//   unsigned long dst_pix_idx =
//       batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;
  
//   output[dst_pix_idx] = input[src_pix_idx];
//   output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
//   output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

//   half R, G, B;
//   R = B = G = 0;
//   int x_r, x_g, x_b, y_r, y_g, y_b;
//   // R
//   x_r = (id_x + x_offset_r[id_z]);
//   y_r = (id_y + y_offset_r[id_z]);
 
//   // G
//   x_g = (id_x + x_offset_g[id_z]);
//   y_g = (id_y + y_offset_g[id_z]);
 
//   // B
//   x_b = (id_x + x_offset_b[id_z]);
//   y_b = (id_y + y_offset_b[id_z]);
  
//   // R
//   if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) &&
//       (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
//   {
//     R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind +
//               indextmp * inc[id_z]];
//     indextmp = indextmp + 1;
//     output[dst_pix_idx] = R;
//     dst_pix_idx += dstinc[id_z];
//   }
    
//   // G
//   if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) &&
//       (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
//   {
//     G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind +
//               indextmp * inc[id_z]];
//     indextmp = indextmp + 1;
//     output[dst_pix_idx] = G;
//     dst_pix_idx += dstinc[id_z];
//   }

//   // B
//   if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) &&
//       (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
//   {
//     B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind +
//               indextmp * inc[id_z]];
//     indextmp = indextmp + 1;
//     output[dst_pix_idx] = B;
//     dst_pix_idx += dstinc[id_z];
//   }
// }

extern "C" __global__ void glitch_batch_fp32(
    float *input, float *output,
    unsigned int *x_offset_r, unsigned int *y_offset_r,
    unsigned int *x_offset_g, unsigned int *y_offset_g,
    unsigned int *x_offset_b, unsigned int *y_offset_b,
    int *xroi_begin, int *xroi_end, int *yroi_begin,
    int *yroi_end, unsigned int *height,
    unsigned int *width, unsigned int *max_width,
    unsigned long *batch_index, const unsigned int channel,
    unsigned int *inc,    // use width * height for pln and 1 for pkd
    unsigned int *dstinc, // use width * height for pln and 1 for pkd
    int in_plnpkdind,              // use 1 pln 3 for pkd
    int out_plnpkdind) {

  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  output[dst_pix_idx] = input[src_pix_idx];
  output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
  output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];
  
  float R, G, B;
  R = B = G = 0;
  int x_r, x_g, x_b, y_r, y_g, y_b;
  // R
  x_r = (id_x + x_offset_r[id_z]);
  y_r = (id_y + y_offset_r[id_z]);
 
  // G
  x_g = (id_x + x_offset_g[id_z]);
  y_g = (id_y + y_offset_g[id_z]);
 
  // B
  x_b = (id_x + x_offset_b[id_z]);
  y_b = (id_y + y_offset_b[id_z]);
  
  // R
  if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) &&
      (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
  {
    R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = R;
    dst_pix_idx += dstinc[id_z];
  }
    
  // G
  if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) &&
      (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
  {
    G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = G;
    dst_pix_idx += dstinc[id_z];
  }

  // B
  if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) &&
      (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
  {
    B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = B;
    dst_pix_idx += dstinc[id_z];
  }
}

extern "C" __global__ void glitch_batch_int8(
    char *input, char *output,
    unsigned int *x_offset_r, unsigned int *y_offset_r,
    unsigned int *x_offset_g, unsigned int *y_offset_g,
    unsigned int *x_offset_b, unsigned int *y_offset_b,
    int *xroi_begin, int *xroi_end, int *yroi_begin,
    int *yroi_end, unsigned int *height,
    unsigned int *width, unsigned int *max_width,
    unsigned long *batch_index, const unsigned int channel,
    unsigned int *inc,    // use width * height for pln and 1 for pkd
    unsigned int *dstinc, // use width * height for pln and 1 for pkd
    int in_plnpkdind,              // use 1 pln 3 for pkd
    int out_plnpkdind) {

  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

  int indextmp = 0;
  unsigned long src_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * in_plnpkdind;
  unsigned long dst_pix_idx =
      batch_index[id_z] + (id_x + id_y * max_width[id_z]) * out_plnpkdind;

  output[dst_pix_idx] = input[src_pix_idx];
  output[dst_pix_idx + dstinc[id_z]] = input[src_pix_idx + inc[id_z]];
  output[dst_pix_idx + dstinc[id_z] + dstinc[id_z]] = input[src_pix_idx + inc[id_z] + inc[id_z]];

  char R, G, B;
  R = B = G = 0;
  int x_r, x_g, x_b, y_r, y_g, y_b;
  // R
  x_r = (id_x + x_offset_r[id_z]);
  y_r = (id_y + y_offset_r[id_z]);
 
  // G
  x_g = (id_x + x_offset_g[id_z]);
  y_g = (id_y + y_offset_g[id_z]);
 
  // B
  x_b = (id_x + x_offset_b[id_z]);
  y_b = (id_y + y_offset_b[id_z]);
  

  // R
  if ((y_r >= yroi_begin[id_z]) && (y_r <= yroi_end[id_z]) &&
      (x_r >= xroi_begin[id_z]) && (x_r <= xroi_end[id_z]))
  {
    R = input[batch_index[id_z] + (x_r + y_r * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = R;
    dst_pix_idx += dstinc[id_z];
  }
    
  // G
  if ((y_g >= yroi_begin[id_z]) && (y_g <= yroi_end[id_z]) &&
      (x_g >= xroi_begin[id_z]) && (x_g <= xroi_end[id_z]))
  {
    G = input[batch_index[id_z] + (x_g + y_g * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = G;
    dst_pix_idx += dstinc[id_z];
  }

  // B
  if ((y_b >= yroi_begin[id_z]) && (y_b <= yroi_end[id_z]) &&
      (x_b >= xroi_begin[id_z]) && (x_b <= xroi_end[id_z]))
  {
    B = input[batch_index[id_z] + (x_b + y_b * max_width[id_z]) * in_plnpkdind +
              indextmp * inc[id_z]];
    indextmp = indextmp + 1;
    output[dst_pix_idx] = B;
    dst_pix_idx += dstinc[id_z];
  }
}