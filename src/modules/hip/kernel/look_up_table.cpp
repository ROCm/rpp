#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))

extern "C" __global__ void look_up_table_pkd(   unsigned char* input,
                     unsigned char* output,
                     unsigned char* lutPtr,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int index = input[pixIdx] * channel + id_z;
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;
}

extern "C" __global__ void look_up_table_pln(   unsigned char* input,
                     unsigned char* output,
                     unsigned char* lutPtr,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int index = input[pixIdx] + id_z * 256;
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;   
}

extern "C" __global__ void look_up_table_batch(
    unsigned char *input, unsigned char *output,
    unsigned char *lutPtr, int *xroi_begin,
    int *xroi_end, int *yroi_begin, int *yroi_end,
    unsigned int *height, unsigned int *width,
    unsigned int *max_width, unsigned long *batch_index,
    const unsigned int channel,
    unsigned int *inc, // use width * height for pln and 1 for pkd
    const int plnpkdindex       // use 1 pln 3 for pkd
) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  int indextmp = 0;
  long pixIdx = 0;
  if (id_x < width[id_z] && id_y < height[id_z]) {
    pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;
    int luptrIndex = id_z * plnpkdindex * 256;
    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
        (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
      for (indextmp = 0; indextmp < channel; indextmp++) {
        luptrIndex = (id_z * channel * 256) + (input[pixIdx] * plnpkdindex);
        output[pixIdx] = saturate_8u(lutPtr[luptrIndex]);
        pixIdx += inc[id_z];
      }
    } else if ((id_x < width[id_z]) && (id_y < height[id_z])) {
      for (indextmp = 0; indextmp < channel; indextmp++) {
        output[pixIdx] = input[pixIdx];
        pixIdx += inc[id_z];
      }
    }
  }
}