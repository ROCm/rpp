#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
__kernel void tensor_add(const unsigned int tensorDimension,
                         __global unsigned char *input1,
                         __global unsigned char *input2,
                         __global unsigned char *output, const unsigned int a,
                         const unsigned int b, const unsigned int c) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  unsigned int value = input1[pixIdx] + input2[pixIdx];
  output[pixIdx] = saturate_8u(value);
}

__kernel void tensor_subtract(const unsigned int tensorDimension,
                              __global unsigned char *input1,
                              __global unsigned char *input2,
                              __global unsigned char *output,
                              const unsigned int a, const unsigned int b,
                              const unsigned int c) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  unsigned int value = input1[pixIdx] - input2[pixIdx];
  output[pixIdx] = saturate_8u(value);
}

__kernel void tensor_multiply(const unsigned int tensorDimension,
                              __global unsigned char *input1,
                              __global unsigned char *input2,
                              __global unsigned char *output,
                              const unsigned int a, const unsigned int b,
                              const unsigned int c) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  unsigned int value = input1[pixIdx] * input2[pixIdx];
  output[pixIdx] = saturate_8u(value);
}
__kernel void tensor_matrix_multiply(
    __global unsigned char *input1, __global unsigned char *input2,
    __global unsigned char *output, const unsigned int r1,
    const unsigned int c1, const unsigned int r2, const unsigned int c2) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= c2 || id_y >= r1 || id_z >= 1)
    return;
  unsigned int OpPixIdx = id_y * c2 + id_x;
  unsigned int pixIdx1, pixIdx2;
  output[OpPixIdx] = 0;
  for (int j = 0; j < c1; j++) {
    pixIdx1 = id_y * c1 + j;
    pixIdx2 = j * c2 + id_x;
    int value = input1[pixIdx1] * input2[pixIdx2];
    output[OpPixIdx] = saturate_8u(output[OpPixIdx] + value);
  }
}

__kernel void tensor_convert_bit_depth_u8s8(const unsigned int tensorDimension,
                                            __global unsigned char *input,
                                            __global char *output,
                                            const unsigned int a,
                                            const unsigned int b,
                                            const unsigned int c) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  output[pixIdx] = (char)(input[pixIdx] - 128);
}

__kernel void tensor_convert_bit_depth_u8u16(const unsigned int tensorDimension,
                                             __global unsigned char *input,
                                             __global unsigned short *output,
                                             const unsigned int a,
                                             const unsigned int b,
                                             const unsigned int c) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  output[pixIdx] = (unsigned short)(input[pixIdx] * 257);
}

__kernel void tensor_convert_bit_depth_u8s16(const unsigned int tensorDimension,
                                             __global unsigned char *input,
                                             __global short *output,
                                             const unsigned int a,
                                             const unsigned int b,
                                             const unsigned int c) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  output[pixIdx] = (short)((input[pixIdx] * 257) - 32768);
}

__kernel void tensor_look_up_table(const unsigned int tensorDimension,
                                   __global unsigned char *input,
                                   __global unsigned char *output,
                                   const unsigned int a, const unsigned int b,
                                   const unsigned int c,
                                   __global unsigned char *lutPtr) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= a || id_y >= b || id_z >= c)
    return;

  int pixIdx = id_y * c * a + id_x * c + id_z;

  int index = input[pixIdx];
  unsigned char pixel = lutPtr[index];
  output[pixIdx] = pixel;
}

__kernel void tensor_transopose(__global unsigned char *input,
                                __global unsigned char *output,
                                __global unsigned int *src_strides,
                                __global unsigned int *dst_strides,
                                __global unsigned int *in_dims,
                                __global unsigned int *out_dims,
                                __global unsigned int *perm) {

  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  int dst_idx = id_x * dst_strides[0] + id_y * dst_strides[1] +
                (id_z / out_dims[2]) * dst_strides[2] +
                (id_z % out_dims[2]) * dst_strides[3];
  int src_idx = id_x * src_strides[perm[0]] + id_y * src_strides[perm[1]] +
                (id_z / in_dims[perm[2]]) * src_strides[perm[2]] +
                (id_z % in_dims[perm[2]]) * src_strides[perm[3]];
  output[dst_idx] = input[src_idx];
}