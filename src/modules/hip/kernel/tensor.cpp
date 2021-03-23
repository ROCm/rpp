#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define saturate_8u_unsigned(value) ( (value) > 255 ? 255 : value)

extern "C" __global__ void tensor_add(const unsigned int tensorDimension,
                         unsigned char *input1,
                         unsigned char *input2,
                         unsigned char *output, const unsigned int a,
                         const unsigned int b, const unsigned int c) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  unsigned int value = input1[pixIdx] + input2[pixIdx];
  output[pixIdx] = saturate_8u_unsigned(value);
}

extern "C" __global__ void tensor_subtract(const unsigned int tensorDimension,
                              unsigned char *input1,
                              unsigned char *input2,
                              unsigned char *output,
                              const unsigned int a, const unsigned int b,
                              const unsigned int c) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  unsigned int value = input1[pixIdx] - input2[pixIdx];
  output[pixIdx] = saturate_8u_unsigned(value);
}

extern "C" __global__ void tensor_multiply(const unsigned int tensorDimension,
                              unsigned char *input1,
                              unsigned char *input2,
                              unsigned char *output,
                              const unsigned int a, const unsigned int b,
                              const unsigned int c) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  unsigned int value = input1[pixIdx] * input2[pixIdx];
  output[pixIdx] = saturate_8u_unsigned(value);
}
extern "C" __global__ void tensor_matrix_multiply(
    unsigned char *input1, unsigned char *input2,
    unsigned char *output, const unsigned int r1,
    const unsigned int c1, const unsigned int r2, const unsigned int c2) {
  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
  if (id_x >= c2 || id_y >= r1 || id_z >= 1)
    return;
  unsigned int OpPixIdx = id_y * c2 + id_x;
  output[OpPixIdx] = 0;
  for (int j = 0; j < c1; j++) {
    unsigned int pixIdx1 = id_y * c1 + j;
    unsigned int pixIdx2 = j * c2 + id_x;
    int value = input1[pixIdx1] * input2[pixIdx2];
    output[OpPixIdx] = saturate_8u(output[OpPixIdx] + value);
  }
}

extern "C" __global__ void tensor_transpose(unsigned char *input,
                               unsigned char *output,
                               unsigned int *out_dims,
                               unsigned int *perm,
                               unsigned int *dst_strides,
                               unsigned int *src_strides) {

  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

  if (id_x >= out_dims[0] || id_y >= out_dims[1] ||
      id_z >= (out_dims[2] * out_dims[3]))
    return;

  int dst_idx = id_x * dst_strides[0] + id_y * dst_strides[1] +
                (id_z / out_dims[2]) * dst_strides[2] +
                (id_z % out_dims[2]) * dst_strides[3];
  int src_idx = id_x * src_strides[perm[0]] + id_y * src_strides[perm[1]] +
                (id_z / out_dims[2]) * src_strides[perm[2]] +
                (id_z % out_dims[2]) * src_strides[perm[3]];
  output[dst_idx] = input[src_idx];
}

// extern "C" __global__ void tensor_transpose_fp16(half *input, half *output,
//                                     unsigned int *out_dims,
//                                     unsigned int *perm,
//                                     unsigned int *dst_strides,
//                                     unsigned int *src_strides) {

//   int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
//   int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
//   int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

//   if (id_x >= out_dims[0] || id_y >= out_dims[1] ||
//       id_z >= (out_dims[2] * out_dims[3]))
//     return;

//   int dst_idx = id_x * dst_strides[0] + id_y * dst_strides[1] +
//                 (id_z / out_dims[2]) * dst_strides[2] +
//                 (id_z % out_dims[2]) * dst_strides[3];
//   int src_idx = id_x * src_strides[perm[0]] + id_y * src_strides[perm[1]] +
//                 (id_z / out_dims[2]) * src_strides[perm[2]] +
//                 (id_z % out_dims[2]) * src_strides[perm[3]];
//   output[dst_idx] = input[src_idx];
// }

extern "C" __global__ void tensor_transpose_fp32(float *input,
                                    float *output,
                                    unsigned int *out_dims,
                                    unsigned int *perm,
                                    unsigned int *dst_strides,
                                    unsigned int *src_strides) {

  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

  if (id_x >= out_dims[0] || id_y >= out_dims[1] ||
      id_z >= (out_dims[2] * out_dims[3]))
    return;

  int dst_idx = id_x * dst_strides[0] + id_y * dst_strides[1] +
                (id_z / out_dims[2]) * dst_strides[2] +
                (id_z % out_dims[2]) * dst_strides[3];
  int src_idx = id_x * src_strides[perm[0]] + id_y * src_strides[perm[1]] +
                (id_z / out_dims[2]) * src_strides[perm[2]] +
                (id_z % out_dims[2]) * src_strides[perm[3]];
  output[dst_idx] = input[src_idx];
}

extern "C" __global__ void tensor_transpose_int8(char *input, char *output,
                                    unsigned int *out_dims,
                                    unsigned int *perm,
                                    unsigned int *dst_strides,
                                    unsigned int *src_strides) {

  int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
  int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
  int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;

  if (id_x >= out_dims[0] || id_y >= out_dims[1] ||
      id_z >= (out_dims[2] * out_dims[3]))
    return;

  int dst_idx = id_x * dst_strides[0] + id_y * dst_strides[1] +
                (id_z / out_dims[2]) * dst_strides[2] +
                (id_z % out_dims[2]) * dst_strides[3];
  int src_idx = id_x * src_strides[perm[0]] + id_y * src_strides[perm[1]] +
                (id_z / out_dims[2]) * src_strides[perm[2]] +
                (id_z % out_dims[2]) * src_strides[perm[3]];
  output[dst_idx] = input[src_idx];
}

extern "C" __global__ void tensor_convert_bit_depth_u8s8(   const unsigned int tensorDimension,
                     unsigned char* input,
                     char* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;
    
    output[pixIdx] = (char)(input[pixIdx] - 128);
}

extern "C" __global__ void tensor_convert_bit_depth_u8u16(   const unsigned int tensorDimension,
                     unsigned char* input,
                     unsigned short* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;
    
    output[pixIdx] = (unsigned short)(input[pixIdx] * 257);
}

extern "C" __global__ void tensor_convert_bit_depth_u8s16(   const unsigned int tensorDimension,
                     unsigned char* input,
                     short* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;
    
    output[pixIdx] = (short)((input[pixIdx] * 257) - 32768);
}

extern "C" __global__ void tensor_look_up_table(   const unsigned int tensorDimension,
                     unsigned char* input,
                     unsigned char* output,
                    const unsigned int a,
                    const unsigned int b,
                    const unsigned int c,
                     unsigned char* lutPtr
)
{
    int id_x = hipBlockIdx_x *hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y *hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z *hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= a || id_y >= b || id_z >= c) return;
    
    int pixIdx = id_y * c * a + id_x * c + id_z;

    int index = input[pixIdx];
    unsigned char pixel = lutPtr[index];
    output[pixIdx] = pixel;   
}
