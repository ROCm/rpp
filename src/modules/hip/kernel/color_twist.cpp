#include <hip/hip_runtime.h>
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
// #define RPPMIN2(a,b)            ((a < b) ? a : b)
#define RPPMIN3(a,b,c)          ((a < b) && (a < c) ?  a : ((b < c) ? b : c))
// #define RPPMAX2(a,b)            ((a > b) ? a : b)
#define RPPMAX3(a,b,c)          ((a > b) && (a > c) ?  a : ((b > c) ? b : c))

__device__ uchar4 convert_one_pixel_to_rgb(float4 pixel) {
  float r, g, b;
  float h, s, v;

  h = pixel.x;
  s = pixel.y;
  v = pixel.z;

  float f = h / 60.0f;
  float hi = floor(f);
  f = f - hi;
  float p = v * (1 - s);
  float q = v * (1 - s * f);
  float t = v * (1 - s * (1 - f));

  if (hi == 0.0f || hi == 6.0f) {
    r = v;
    g = t;
    b = p;
  } else if (hi == 1.0f) {
    r = q;
    g = v;
    b = p;
  } else if (hi == 2.0f) {
    r = p;
    g = v;
    b = t;
  } else if (hi == 3.0f) {
    r = p;
    g = q;
    b = v;
  } else if (hi == 4.0f) {
    r = t;
    g = p;
    b = v;
  } else {
    r = v;
    g = p;
    b = q;
  }

  unsigned char red = (unsigned char)(255.0f * r);
  unsigned char green = (unsigned char)(255.0f * g);
  unsigned char blue = (unsigned char)(255.0f * b);
  unsigned char alpha = 0.0; //(unsigned char)(pixel.w);
  return (uchar4){red, green, blue, alpha};
}

__device__ float4 convert_one_pixel_to_hsv(uchar4 pixel) {
  float r, g, b, a;
  float h, s, v;

  r = pixel.x / 255.0f;
  g = pixel.y / 255.0f;
  b = pixel.z / 255.0f;
  a = pixel.w;

  float max = RPPMAX3(r, g, b);
  float min = RPPMIN3(r, g, b);
  float diff = max - min;

  v = max;

  if (v == 0.0f) { // black
    h = s = 0.0f;
  } else {
    s = diff / v;
    if (diff < 0.001f) { // grey
      h = 0.0f;
    } else { // color
      if ((max > r - 0.001) && (max < r + 0.001)) {
        h = 60.0f * (g - b) / diff;
        if (h < 0.0f) {
          h += 360.0f;
        }
      } else if (max == g) {
        h = 60.0f * (2 + (b - r) / diff);
      } else {
        h = 60.0f * (4 + (r - g) / diff);
      }
    }
  }

  return (float4){h, s, v, a};
}

extern "C" __global__ void color_twist_batch(
    unsigned char *input, 
    unsigned char *output, 
    float *alpha, 
    float *beta, 
    float *hue,
    float *sat, 
    int *xroi_begin, 
    int *xroi_end,
    int *yroi_begin, 
    int *yroi_end,
    unsigned int *height, 
    unsigned int *width,
    unsigned int *max_width, 
    unsigned long *batch_index,
    unsigned int *inc, 
    unsigned int *dst_inc,
    const int in_plnpkdind, 
    const int out_plnpkdind
)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    if (id_x >= width[id_z] || id_y >= height[id_z])
    return;
    uchar4 pixel;
    float4 hsv;

    unsigned int l_inc = inc[id_z]; 
    unsigned int d_inc  = dst_inc[id_z];
    int pixIdx =
        batch_index[id_z] + (id_y * max_width[id_z] + id_x) * in_plnpkdind;
    int out_pixIdx = batch_index[id_z] + (id_y * max_width[id_z] + id_x) * out_plnpkdind;
    pixel.x = input[pixIdx];
    pixel.y = input[pixIdx + l_inc];
    pixel.z = input[pixIdx + 2 * l_inc];
    pixel.w = 0.0;
    float alpha1 = alpha[id_z], beta1 = beta[id_z];

    if ((id_y >= yroi_begin[id_z]) && (id_y <= yroi_end[id_z]) &&
        (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z])) {
    hsv = convert_one_pixel_to_hsv(pixel); // Converting to HSV
    hsv.x += hue[id_z];
    if (hsv.x > 360.0) {
        hsv.x = hsv.x - 360.0;
    } else if (hsv.x < 0) {
        hsv.x = hsv.x + 360.0;
    }
    hsv.y *= sat[id_z];
    if (hsv.y > 1.0) {
        hsv.y = 1.0;
    } else if (hsv.y < 0.0) {
        hsv.y = 0.0;
    }
    pixel = convert_one_pixel_to_rgb(
        hsv); // Converting to RGB back with hue modification
    output[out_pixIdx] = saturate_8u(alpha1 * pixel.x + beta1);
    output[out_pixIdx + d_inc] = saturate_8u(alpha1 * pixel.y + beta1);
    output[out_pixIdx + 2 * d_inc] = saturate_8u(alpha1 * pixel.z + beta1);
    } else {
    output[out_pixIdx] = pixel.x;
    output[out_pixIdx + d_inc] = pixel.y;
    output[out_pixIdx + 2 * d_inc] = pixel.z;
    }
}