#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))

float gaussian(int x, int y, float std_dev) {
  float res, pi = 3.14;
  res = 1 / (2 * pi * stdDev * std_dev);
  float exp1, exp2;
  exp1 = -(x * x) / (2 * std_dev * std_dev);
  exp2 = -(y * y) / (2 * std_dev * std_dev);
  exp1 = exp1 + exp2;
  exp1 = exp(exp1);
  res *= exp1;
  return res;
}

__kernel void non_linear_blend_batch(
    __global unsigned char *input1, __global unsigned char *input2,
    __global unsigned char *output, __global float *std_dev,
    __global int *xroi_begin, __global int *xroi_end, __global int *yroi_begin,
    __global int *yroi_end, __global unsigned int *height,
    __global unsigned int *width, __global unsigned int *max_width,
    __global unsigned long *batch_index, const unsigned int channel,
    __global unsigned int *inc, // use width * height for pln and 1 for pkd
    const int plnpkdindex       // use 1 pln 3 for pkd
) {
  int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
  unsigned char valuergb1, valuergb2;
  float tempstd_dev = width[id_z] / 8;
  int indextmp = 0;
  unsigned long pixIdx = 0;

  pixIdx = batch_index[id_z] + (id_x + id_y * max_width[id_z]) * plnpkdindex;

  if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
      (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    int x = (id_x - (width[id_z] / 2));
    int y = (id_y - (height[id_z] / 2));
    float gaussianvalue =
        gaussian(x, y, tempstdDev) / gaussian(0.0, 0.0, tempstdDev);
    for (indextmp = 0; indextmp < channel; indextmp++) {
      valuergb1 = input1[pixIdx];
      valuergb2 = input2[pixIdx];
      output[pixIdx] =
          gaussianvalue * input1[pixIdx] + (1 - gaussianvalue) * input2[pixIdx];
      pixIdx += inc[id_z];
    }
  } else if ((id_x < width[id_z]) && (id_y < height[id_z])) {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[pixIdx] = input1[pixIdx];
      pixIdx += inc[id_z];
    }
  } else {
    for (indextmp = 0; indextmp < channel; indextmp++) {
      output[pixIdx] = 0;
      pixIdx += inc[id_z];
    }
  }
}