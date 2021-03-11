#include <hip/hip_runtime.h>
# include <half.hpp>
using half_float::half;
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value)   ))

extern "C" __global__ void resize_pln (   unsigned char* srcPtr,
                             unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/dest_width ;
    float y_ratio = ((float)(source_height -1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    A = srcPtr[x + y * source_width + id_z * source_height * source_width];
    B = srcPtr[x + 1  + y * source_width + id_z * source_height * source_width];
    C = srcPtr[x + (y + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+1) + (y+1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}

extern "C" __global__ void resize_pkd (   unsigned char* srcPtr,
                             unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(source_width -1 ))/dest_width ;
    float y_ratio = ((float)(source_height -1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;

    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x = (int)(x_ratio * id_x) ;
    y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    pixId = id_x * channel + id_y * dest_width * channel + id_z;

    A = srcPtr[x * channel + y * source_width * channel + id_z];
    B = srcPtr[(x +1) * channel + y * source_width * channel + id_z];
    C = srcPtr[x * channel + (y+1) * source_width * channel + id_z];
    D = srcPtr[(x+1) * channel + (y+1) * source_width * channel + id_z];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                  C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
    dstPtr[pixId] =  saturate_8u(pixVal);

}



extern "C" __global__ void resize_crop_pln (   unsigned char* srcPtr,
                             unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int x1,
                            const unsigned int y1,
                            const unsigned int x2,
                            const unsigned int y2,
                            const unsigned int padding,
                            const unsigned int type,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x2 - x1 ))/dest_width ;
    float y_ratio = ((float)(y2 - y1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;
    A = B = C = D = 0;
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x =  (int)(x_ratio * id_x) ;
    y =  (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    if(type == 0)
        pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height;
    else
        pixId = id_x + id_y * dest_width + id_z * dest_width * dest_height + ((dest_width + padding * 2) * padding) + (id_y * padding * 2) + (padding);
    A = srcPtr[(x + x1) + (y+y1) * source_width + id_z * source_height * source_width];
    B = srcPtr[(x + x1 + 1)  + (y+y1) * source_width + id_z * source_height * source_width];
    C = srcPtr[(x + x1)+ (y + y1 + 1) * source_width + id_z * source_height * source_width];
    D = srcPtr[(x+ x1 + 1) + (y+ y1 +1) * source_width + id_z * source_height * source_width];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}

extern "C" __global__ void resize_crop_pkd (   unsigned char* srcPtr,
                             unsigned char* dstPtr,
                            const unsigned int source_height,
                            const unsigned int source_width,
                            const unsigned int dest_height,
                            const unsigned int dest_width,
                            const unsigned int x1,
                            const unsigned int y1,
                            const unsigned int x2,
                            const unsigned int y2,
                            const unsigned int padding,
                            const unsigned int type,
                            const unsigned int channel
)
{
    int A, B, C, D, x, y, index, pixVal ;
    float x_ratio = ((float)(x2 - x1 ))/dest_width ;
    float y_ratio = ((float)(y2 - y1 ))/dest_height;
    float x_diff, y_diff, ya, yb ;
    A = B = C = D = 0;
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;

    x =  (int)(x_ratio * id_x) ;
    y =  (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x;
    y_diff = (y_ratio * id_y) - y ;

    unsigned int pixId;
    if(type == 0)
        pixId = id_x * channel + id_y * dest_width * channel + id_z;
    else
        pixId = id_x * channel + id_y * dest_width * channel + id_z + ((dest_width + padding * 2) * channel * padding) + (id_y * padding * 2 * channel) + (padding * channel);
    A = srcPtr[(x+x1) * channel + (y+y1) * source_width * channel + id_z];
    B = srcPtr[(x +x1 +1) * channel + (y+y1) * source_width * channel + id_z];
    C = srcPtr[(x+x1) * channel + (y+ y1+ 1) * source_width * channel + id_z];
    D = srcPtr[(x+x1+1) * channel + (y+y1+1) * source_width * channel + id_z];

    pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)
                    ) ;

    dstPtr[pixId] =  saturate_8u(pixVal);

}


extern "C" __global__ void resize_batch( unsigned char* srcPtr,
                                     unsigned char* dstPtr,
                                     unsigned int *source_height,
                                     unsigned int *source_width,
                                     unsigned int *dest_height,
                                     unsigned int *dest_width,
                                     unsigned int *max_source_width,
                                     unsigned int *max_dest_width,
                                     unsigned long *source_batch_index,
                                     unsigned long *dest_batch_index,
                                    const unsigned int channel,
                                     unsigned int *source_inc, // use width * height for pln and 1 for pkd
                                     unsigned int *dest_inc,
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    float x_ratio = ((float)(source_width[id_z] -1 ))/dest_width[id_z] ;
    float y_ratio = ((float)(source_height[id_z] -1 ))/dest_height[id_z];
    float x_diff, y_diff, ya, yb ;

    unsigned long src_pixIdx = 0, dst_pixIdx = 0;


    if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z]) return;


    int x = (int)(x_ratio * id_x) ;
    int y = (int)(y_ratio * id_y) ;

    x_diff = (x_ratio * id_x) - x ;
    y_diff = (y_ratio * id_y) - y ;


    dst_pixIdx = dest_batch_index[id_z] + (id_x  + id_y * max_dest_width[id_z] ) * plnpkdindex;
    for(int indextmp = 0; indextmp < channel; indextmp++){
        int A = srcPtr[source_batch_index[id_z] + (x  + y * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];
        int B = srcPtr[source_batch_index[id_z] + ((x + 1)   + y * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];
        int C = srcPtr[source_batch_index[id_z] + (x  + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];
        int D = srcPtr[source_batch_index[id_z] + ((x + 1)  + (y + 1) * max_source_width[id_z]) * plnpkdindex + indextmp*source_inc[id_z]];

        int pixVal = (int)(  A*(1-x_diff)*(1-y_diff) +  B*(x_diff)*(1-y_diff) +
                    C*(y_diff)*(1-x_diff)   +  D*(x_diff*y_diff)) ;
        dstPtr[dst_pixIdx] =  saturate_8u(pixVal);
        dst_pixIdx += dest_inc[id_z];
    }
}

extern "C" __global__ void resize_crop_batch(
    unsigned char *srcPtr, unsigned char *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc, const unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      int A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      int pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = saturate_8u(pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

extern "C" __global__ void resize_crop_batch_int8(
    char *srcPtr, char *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc, const unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      int A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      int pixVal =
          (char)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}


// extern "C" __global__ void resize_crop_batch_fp16(
//     half *srcPtr, half *dstPtr,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width, int *xroi_begin,
//     int *xroi_end, int *yroi_begin, int *yroi_end,
//     unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int
//         *source_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dest_inc, const unsigned int padding,
//     const unsigned int type,
//     const int in_plnpkdind, const int out_plnpkdind
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   int x, y;
//   float x_ratio =
//       ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
//   float y_ratio =
//       ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
//   float x_diff, y_diff, ya, yb;

//   unsigned long src_pixIdx = 0, dst_pixIdx = 0;

//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   x = (int)(x_ratio * id_x);
//   y = (int)(y_ratio * id_y);

//   x_diff = (x_ratio * id_x) - x;
//   y_diff = (y_ratio * id_y) - y;

//   x = xroi_begin[id_z] + x;
//   y = yroi_begin[id_z] + y;

//   if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       float A = srcPtr[source_batch_index[id_z] +
//                  (x + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float B = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float C = srcPtr[source_batch_index[id_z] +
//                  (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float D = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];

//       float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
//       dstPtr[dst_pixIdx] = (half)pixVal;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void resize_crop_batch_fp32(
    float *srcPtr, float *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      float A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

extern "C" __global__ void resize_crop_batch_u8_fp32(
    unsigned char *srcPtr, float *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      float A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = pixVal / 255.0;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

// extern "C" __global__ void resize_crop_batch_u8_fp16(
//     unsigned char *srcPtr, half *dstPtr,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width, int *xroi_begin,
//     int *xroi_end, int *yroi_begin, int *yroi_end,
//     unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int *source_inc, unsigned int *dest_inc,
//     const unsigned int padding, const unsigned int type,
//     const int in_plnpkdind, const int out_plnpkdind
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   int x, y, index;
//   float x_ratio =
//       ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
//   float y_ratio =
//       ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
//   float x_diff, y_diff, ya, yb;

//   unsigned long src_pixIdx = 0, dst_pixIdx = 0;

//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   x = (int)(x_ratio * id_x);
//   y = (int)(y_ratio * id_y);

//   x_diff = (x_ratio * id_x) - x;
//   y_diff = (y_ratio * id_y) - y;

//   x = xroi_begin[id_z] + x;
//   y = yroi_begin[id_z] + y;

//   if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       half A = srcPtr[source_batch_index[id_z] +
//                  (x + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       half B = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       half C = srcPtr[source_batch_index[id_z] +
//                  (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       half D = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];

//       half pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
//       dstPtr[dst_pixIdx] = (half)(pixVal/255.0);
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void resize_crop_batch_u8_int8(
    unsigned char *srcPtr, char *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, unsigned int *dest_inc,
    const unsigned int padding, const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      float A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = (char)(pixVal - 128);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = -128;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

extern "C" __global__ void resize_crop_mirror_batch(
    unsigned char *srcPtr, unsigned char *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    int *mirror, unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                     out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      int A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      int pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = saturate_8u(pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

extern "C" __global__ void resize_crop_mirror_batch_int8(
    char *srcPtr, char *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    int *mirror, unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                     out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      int A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      int D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      int pixVal =
          (char)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = -128;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

// extern "C" __global__ void resize_crop_mirror_batch_fp16(
//     half *srcPtr, half *dstPtr,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width, int *xroi_begin,
//     int *xroi_end, int *yroi_begin, int *yroi_end,
//     int *mirror, unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int
//         *source_inc, // use width * height for pln and 1 for pkd
//     unsigned int *dest_inc,
//     const int in_plnpkdind, const int out_plnpkdind
// ) {
//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
//   int x, y, index;
//   float x_ratio =
//       ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
//   float y_ratio =
//       ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
//   float x_diff, y_diff, ya, yb;

//   unsigned long src_pixIdx = 0, dst_pixIdx = 0;

//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   x = (int)(x_ratio * id_x);
//   y = (int)(y_ratio * id_y);

//   x_diff = (x_ratio * id_x) - x;
//   y_diff = (y_ratio * id_y) - y;

//   x = xroi_begin[id_z] + x;
//   y = yroi_begin[id_z] + y;

//   if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
//                      out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       float A = srcPtr[source_batch_index[id_z] +
//                  (x + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float B = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float C = srcPtr[source_batch_index[id_z] +
//                  (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];
//       float D = srcPtr[source_batch_index[id_z] +
//                  ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
//                  indextmp * source_inc[id_z]];

//       float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
//                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
//       dstPtr[dst_pixIdx] = (half)pixVal;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   } else {
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (int indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void resize_crop_mirror_batch_fp32(
    float *srcPtr, float *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    int *mirror, unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int x, y, index;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  unsigned long src_pixIdx = 0, dst_pixIdx = 0;

  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 ((dest_width[id_z] - 1 - id_x) + id_y * max_dest_width[id_z]) *
                    out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      float A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      float D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      float pixVal = A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
               C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff);
      dstPtr[dst_pixIdx] = pixVal;
      dst_pixIdx += dest_inc[id_z];
    }
  } else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (int indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

extern "C" __global__ void random_crop_letterbox_batch(
    unsigned char *srcPtr, unsigned char *dstPtr,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *max_source_width,
    unsigned int *max_dest_width, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc, unsigned int padding,
    const unsigned int type,
    const int in_plnpkdind, const int out_plnpkdind
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x, id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y, id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int A, B, C, D, x, y, index, pixVal;
  float x_ratio =
      ((float)(xroi_end[id_z] - xroi_begin[id_z] - 1)) / dest_width[id_z];
  float y_ratio =
      ((float)(yroi_end[id_z] - yroi_begin[id_z] - 1)) / dest_height[id_z];
  float x_diff, y_diff, ya, yb;

  int indextmp = 0;
  unsigned long src_pixIdx = 0, dst_pixIdx = 0;
  unsigned int minVal = ((dest_height[id_z] < dest_width[id_z]) ? dest_height[id_z] : dest_width[id_z]);
  padding = (5 * minVal / 100);

  if (id_x >= dest_width[id_z] - padding || id_y >= dest_height[id_z] - padding || id_x < padding || id_y < padding)
    return;

  x = (int)(x_ratio * id_x);
  y = (int)(y_ratio * id_y);

  x_diff = (x_ratio * id_x) - x;
  y_diff = (y_ratio * id_y) - y;

  x = xroi_begin[id_z] + x;
  y = yroi_begin[id_z] + y;

  if ((x + 1) < source_width[id_z] && (y + 1) < source_height[id_z]) {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      A = srcPtr[source_batch_index[id_z] +
                 (x + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      B = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + y * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      C = srcPtr[source_batch_index[id_z] +
                 (x + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];
      D = srcPtr[source_batch_index[id_z] +
                 ((x + 1) + (y + 1) * max_source_width[id_z]) * in_plnpkdind +
                 indextmp * source_inc[id_z]];

      pixVal =
          (int)(A * (1 - x_diff) * (1 - y_diff) + B * (x_diff) * (1 - y_diff) +
                C * (y_diff) * (1 - x_diff) + D * (x_diff * y_diff));
      dstPtr[dst_pixIdx] = saturate_8u(pixVal);
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}