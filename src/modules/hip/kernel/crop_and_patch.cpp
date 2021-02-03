#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void crop_and_patch_batch(
    unsigned char *srcPtr1, unsigned char *srcPtr2,
    unsigned char *dstPtr, unsigned int *source_height,
    unsigned int *source_width, unsigned int *dest_height,
    unsigned int *dest_width, unsigned int *x11,
    unsigned int *y11, unsigned int *x12,
    unsigned int *y12, unsigned int *x21,
    unsigned int *y21, unsigned int *x22,
    unsigned int *y22, unsigned int *max_source_width,
    unsigned int *max_dest_width,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}

// extern "C" __global__ void crop_and_patch_batch_fp16(
//     half *srcPtr1, half *srcPtr2,
//     half *dstPtr, unsigned int *source_height,
//     unsigned int *source_width, unsigned int *dest_height,
//     unsigned int *dest_width, unsigned int *x11,
//     unsigned int *y11, unsigned int *x12,
//     unsigned int *y12, unsigned int *x21,
//     unsigned int *y21, unsigned int *x22,
//     unsigned int *y22, unsigned int *max_source_width,
//     unsigned int *max_dest_width,
//     unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int *source_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dest_inc,
//     int in_plnpkdind, // use 1 pln 3 for pkd
//     int out_plnpkdind) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   half A, B, C, D;
//   int x, y;
//   half pixVal;
//   float x_ratio =
//       ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
//   float y_ratio =
//       ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
//   float x_diff, y_diff;
//   A = B = C = D = 0;

//   int indextmp = 0;
//   unsigned long dst_pixIdx = 0, src_pixIdx = 0;

//   dst_pixIdx = dest_batch_index[id_z] +
//                (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//   src_pixIdx = source_batch_index[id_z] +
//                (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
//       (id_y <= y22[id_z])) {
//     x = (int)(x_ratio * (id_x - x21[id_z]));
//     y = (int)(y_ratio * (id_y - y21[id_z]));

//     x_diff = (x_ratio * (id_x - x21[id_z])) - x;
//     y_diff = (y_ratio * (id_y - y21[id_z])) - y;

//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       A = srcPtr2[source_batch_index[id_z] +
//                   ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
//                       in_plnpkdind +
//                   indextmp * source_inc[id_z]];
//       B = srcPtr2[source_batch_index[id_z] +
//                   ((x + x11[id_z] + 1) +
//                    (y + y11[id_z]) * max_source_width[id_z]) *
//                       in_plnpkdind +
//                   indextmp * source_inc[id_z]];
//       C = srcPtr2[source_batch_index[id_z] +
//                   ((x + x11[id_z]) +
//                    (y + y11[id_z] + 1) * max_source_width[id_z]) *
//                       in_plnpkdind +
//                   indextmp * source_inc[id_z]];
//       D = srcPtr2[source_batch_index[id_z] +
//                   ((x + x11[id_z] + 1) +
//                    (y + y11[id_z] + 1) * max_source_width[id_z]) *
//                       in_plnpkdind +
//                   indextmp * source_inc[id_z]];

//       pixVal =
//           (half)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                 C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
//       dstPtr[dst_pixIdx] = (pixVal);
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
//       dst_pixIdx += dest_inc[id_z];
//       src_pixIdx += source_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void crop_and_patch_batch_fp32(
    float *srcPtr1, float *srcPtr2,
    float *dstPtr, unsigned int *source_height,
    unsigned int *source_width, unsigned int *dest_height,
    unsigned int *dest_width, unsigned int *x11,
    unsigned int *y11, unsigned int *x12,
    unsigned int *y12, unsigned int *x21,
    unsigned int *y21, unsigned int *x22,
    unsigned int *y22, unsigned int *max_source_width,
    unsigned int *max_dest_width,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
  float A, B, C, D;
  int x, y;
  float pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (float)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}

extern "C" __global__ void crop_and_patch_batch_int8(
     char *srcPtr1,  char *srcPtr2,
     char *dstPtr, unsigned int *source_height,
    unsigned int *source_width, unsigned int *dest_height,
    unsigned int *dest_width, unsigned int *x11,
    unsigned int *y11, unsigned int *x12,
    unsigned int *y12, unsigned int *x21,
    unsigned int *y21, unsigned int *x22,
    unsigned int *y22, unsigned int *max_source_width,
    unsigned int *max_dest_width,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc,
    int in_plnpkdind, // use 1 pln 3 for pkd
    int out_plnpkdind) {
  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
  char A, B, C, D;
  int x, y;
   char pixVal;
  float x_ratio =
      ((float)(x12[id_z] - x11[id_z] + 1) / (x22[id_z] - x21[id_z] + 1));
  float y_ratio =
      ((float)(y12[id_z] - y11[id_z] + 1) / (y22[id_z] - y21[id_z] + 1));
  float x_diff, y_diff;
  A = B = C = D = 0;

  int indextmp = 0;
  unsigned long dst_pixIdx = 0, src_pixIdx = 0;

  dst_pixIdx = dest_batch_index[id_z] +
               (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
  src_pixIdx = source_batch_index[id_z] +
               (id_x + id_y * max_source_width[id_z]) * in_plnpkdind;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  if ((id_x >= x21[id_z]) && (id_x <= x22[id_z]) && (id_y >= y21[id_z]) &&
      (id_y <= y22[id_z])) {
    x = (int)(x_ratio * (id_x - x21[id_z]));
    y = (int)(y_ratio * (id_y - y21[id_z]));

    x_diff = (x_ratio * (id_x - x21[id_z])) - x;
    y_diff = (y_ratio * (id_y - y21[id_z])) - y;

    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) + (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      B = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z]) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      C = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z]) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];
      D = srcPtr2[source_batch_index[id_z] +
                  ((x + x11[id_z] + 1) +
                   (y + y11[id_z] + 1) * max_source_width[id_z]) *
                      in_plnpkdind +
                  indextmp * source_inc[id_z]];

      pixVal =
          (char)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = (pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr1[src_pixIdx];
      dst_pixIdx += dest_inc[id_z];
      src_pixIdx += source_inc[id_z];
    }
  }
}