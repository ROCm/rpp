#include <hip/hip_runtime.h>

#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define PI 3.14159265
#define RAD(deg) (deg * PI / 180)

extern "C" __global__ void rotate_pln (   unsigned char* srcPtr,
							 unsigned char* dstPtr,
							const float angleDeg,
							const unsigned int source_height,
							const unsigned int source_width,
							const unsigned int dest_height,
							const unsigned int dest_width,
							const unsigned int channel
)
{
	float angleRad = RAD(angleDeg);
	float rotate[4];
	rotate[0] = cos(angleRad);
	rotate[1] = -1 * sin( angleRad);
	rotate[2] = sin( angleRad);
	rotate[3] = cos( angleRad);   

	int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

	int xc = id_x - dest_width/2;
	int yc = id_y - dest_height/2;
	
	int k ;
	int l ;
   
	k = (int)((rotate[0] * xc )+ (rotate[1] * yc));
	l = (int)((rotate[2] * xc) + (rotate[3] * yc));
	k = k + source_width/2;
	l = l + source_height/2;
	if (l < source_height && l >=0 && k < source_width && k >=0 )
	dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] =
							srcPtr[(id_z * source_height * source_width) + (l * source_width) + k];
	else
	dstPtr[(id_z * dest_height * dest_width) + (id_y * dest_width) + id_x] = 0;
	

}

extern "C" __global__ void rotate_pkd (   unsigned char* srcPtr,
							 unsigned char* dstPtr,
							const float angleDeg,
							const unsigned int source_height,
							const unsigned int source_width,
							const unsigned int dest_height,
							const unsigned int dest_width,
							const unsigned int channel
)
{
	float angleRad = RAD(angleDeg);
	float rotate[4];
	rotate[0] = cos(angleRad);
	rotate[1] = -1 * sin( angleRad);
	rotate[2] = sin( angleRad);
	rotate[3] = cos( angleRad);

	int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

	if (id_x >= dest_width || id_y >= dest_height || id_z >= channel) return;
	int xc = id_x - dest_width/2;
	int yc = id_y - dest_height/2;
	
	int k ;
	int l ;
   
	k = (int)((rotate[0] * xc )+ (rotate[1] * yc));
	l = (int)((rotate[2] * xc) + (rotate[3] * yc));
	k = k + (int)(source_width/2);
	l = l + (int)(source_height/2);
	
	
	if (l < source_height && l >=0 && k < source_width && k >=0 ){
		dstPtr[id_z + (channel * id_y * dest_width) + (channel * id_x)] =
								 srcPtr[id_z + (channel * l * source_width) + (channel * k)];
	}
	
}

extern "C" __global__ void rotate_batch(
    unsigned char *srcPtr, unsigned char *dstPtr,
    float *angleDeg, unsigned int *source_height,
    unsigned int *source_width, unsigned int *dest_height,
    unsigned int *dest_width, unsigned int *xroi_begin,
    unsigned int *xroi_end, unsigned int *yroi_begin,
    unsigned int *yroi_end, unsigned int *max_source_width,
    unsigned int *max_dest_width,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int
        *source_inc, // use width * height for pln and 1 for pkd
    unsigned int *dest_inc, const int in_plnpkdind,
    const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  float angleRad = RAD(angleDeg[id_z]);
  float rotate[4];
  rotate[0] = cos(angleRad);
  rotate[1] = -1 * sin(angleRad);
  rotate[2] = sin(angleRad);
  rotate[3] = cos(angleRad);
  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);

  int k = (int)((rotate[0] * xc) + (rotate[1] * yc));
  int l = (int)((rotate[2] * xc) + (rotate[3] * yc));
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    unsigned long src_pixIdx, dst_pixIdx;
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    unsigned long dst_pixIdx;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}

extern "C" __global__ void rotate_batch_int8(
    char *srcPtr, char *dstPtr, float *angleDeg,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *xroi_begin, unsigned int *xroi_end,
    unsigned int *yroi_begin, unsigned int *yroi_end,
    unsigned int *max_source_width,
    unsigned int *max_dest_width,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  float angleRad = RAD(angleDeg[id_z]);
  float rotate[4];
  rotate[0] = cos(angleRad);
  rotate[1] = -1 * sin(angleRad);
  rotate[2] = sin(angleRad);
  rotate[3] = cos(angleRad);

  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);

  int k = (int)((rotate[0] * xc) + (rotate[1] * yc));
  int l = (int)((rotate[2] * xc) + (rotate[3] * yc));
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    unsigned long src_pixIdx, dst_pixIdx;
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    unsigned long dst_pixIdx;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = -128;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}
// extern "C" __global__ void rotate_batch_fp16(
//     half *srcPtr, half *dstPtr, float *angleDeg,
//     unsigned int *source_height, unsigned int *source_width,
//     unsigned int *dest_height, unsigned int *dest_width,
//     unsigned int *xroi_begin, unsigned int *xroi_end,
//     unsigned int *yroi_begin, unsigned int *yroi_end,
//     unsigned int *max_source_width,
//     unsigned int *max_dest_width,
//     unsigned long *source_batch_index,
//     unsigned long *dest_batch_index, const unsigned int channel,
//     unsigned int *source_inc, unsigned int *dest_inc,
//     const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
// ) {
//   int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
// 	int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
// 	int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
//   if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
//     return;

//   int indextmp = 0;
//   float angleRad = RAD(angleDeg[id_z]);
//   float rotate[4];
//   rotate[0] = cos(angleRad);
//   rotate[1] = -1 * sin(angleRad);
//   rotate[2] = sin(angleRad);
//   rotate[3] = cos(angleRad);

//   int xc = id_x - (dest_width[id_z] >> 1);
//   int yc = id_y - (dest_height[id_z] >> 1);

//   int k = (int)((rotate[0] * xc) + (rotate[1] * yc));
//   int l = (int)((rotate[2] * xc) + (rotate[3] * yc));
//   k = k + (source_width[id_z] >> 1);
//   l = l + (source_height[id_z] >> 1);

//   if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
//       (k >= xroi_begin[id_z])) {
//     unsigned long src_pixIdx, dst_pixIdx;
//     src_pixIdx = source_batch_index[id_z] +
//                  (k + l * max_source_width[id_z]) * in_plnpkdind;
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
//       src_pixIdx += source_inc[id_z];
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }

//   else {
//     unsigned long dst_pixIdx;
//     dst_pixIdx = dest_batch_index[id_z] +
//                  (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
//     for (indextmp = 0; indextmp < channel; indextmp++) {
//       dstPtr[dst_pixIdx] = 0;
//       dst_pixIdx += dest_inc[id_z];
//     }
//   }
// }

extern "C" __global__ void rotate_batch_fp32(
    float *srcPtr, float *dstPtr, float *angleDeg,
    unsigned int *source_height, unsigned int *source_width,
    unsigned int *dest_height, unsigned int *dest_width,
    unsigned int *xroi_begin, unsigned int *xroi_end,
    unsigned int *yroi_begin, unsigned int *yroi_end,
    unsigned int *max_source_width,
    unsigned int *max_dest_width,
    unsigned long *source_batch_index,
    unsigned long *dest_batch_index, const unsigned int channel,
    unsigned int *source_inc, unsigned int *dest_inc,
    const int in_plnpkdind, const int out_plnpkdind // use 1 pln 3 for pkd
) {
  int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
	int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
  if (id_x >= dest_width[id_z] || id_y >= dest_height[id_z])
    return;

  int indextmp = 0;
  float angleRad = RAD(angleDeg[id_z]);
  float rotate[4];
  rotate[0] = cos(angleRad);
  rotate[1] = -1 * sin(angleRad);
  rotate[2] = sin(angleRad);
  rotate[3] = cos(angleRad);

  int xc = id_x - (dest_width[id_z] >> 1);
  int yc = id_y - (dest_height[id_z] >> 1);

  int k = (int)((rotate[0] * xc) + (rotate[1] * yc));
  int l = (int)((rotate[2] * xc) + (rotate[3] * yc));
  k = k + (source_width[id_z] >> 1);
  l = l + (source_height[id_z] >> 1);

  if (l < yroi_end[id_z] && (l >= yroi_begin[id_z]) && k < xroi_end[id_z] &&
      (k >= xroi_begin[id_z])) {
    unsigned long src_pixIdx, dst_pixIdx;
    src_pixIdx = source_batch_index[id_z] +
                 (k + l * max_source_width[id_z]) * in_plnpkdind;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = srcPtr[src_pixIdx];
      src_pixIdx += source_inc[id_z];
      dst_pixIdx += dest_inc[id_z];
    }
  }

  else {
    unsigned long dst_pixIdx;
    dst_pixIdx = dest_batch_index[id_z] +
                 (id_x + id_y * max_dest_width[id_z]) * out_plnpkdind;
    for (indextmp = 0; indextmp < channel; indextmp++) {
      dstPtr[dst_pixIdx] = 0;
      dst_pixIdx += dest_inc[id_z];
    }
  }
}