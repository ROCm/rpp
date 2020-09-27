#define saturate_8u(value) ((value) > 255 ? 255 : ((value) < 0 ? 0 : (value)))
int calcIx(int a[3][3]) {
  int gx[3][3] = {3, 0, -3, 10, 0, -10, 3, 0, -3};
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sum += a[i][j] * gx[i][j];
    }
  }
  return sum;
}

int calcIy(int a[3][3]) {
  int gy[3][3] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
  int sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      sum += a[i][j] * gy[i][j];
    }
  }
  return sum;
}

__kernel void optical_flow_pyramid_grad(__global unsigned char *input,
                                        __global float *output,
                                        const unsigned int height,
                                        const unsigned int width,
                                        const unsigned int channel,
                                        const unsigned int sobelType) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);
  if (id_x >= width || id_y >= height || id_z >= channel)
    return;

  int pixIdx = id_y * width + id_x;
  int value = 0;
  int value1 = 0;
  int a[3][3];
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      if (id_x != 0 && id_x != width - 1 && id_y != 0 && id_y != height - 1) {
        unsigned int index = pixIdx + j + (i * width);
        a[i + 1][j + 1] = input[index];
      } else {
        a[i + 1][j + 1] = 0;
      }
    }
  }
  if (sobelType == 1) {
    value = calcIy(a);
    output[pixIdx] = (float)value;
  }
  if (sobelType == 0) {
    value = calcIx(a);
    output[pixIdx] = (float)value;
  }
}

__kernel void optical_flow_pyramid(
    __global unsigned char *input1, __global unsigned char *input2,
    __global float *Ix, __global float *Iy, __global float *It,
    const unsigned int height, const unsigned int width,
    __global unsigned int *OldPoints, __global unsigned int *NewPointsEstimates,
    __global float *NewPoints, const unsigned int numPoints,
    const float threshold, const unsigned int numIterations,
    const unsigned int kernelSize) {
  int x = get_global_id(0);

  if (x >= numPoints) {
    return;
  }

  int id_x, id_y, pixIdx, pixIdx1;

  id_x = NewPoints[x * 2];
  id_y = NewPoints[x * 2 + 1];

  pixIdx = id_y * width + id_x;

  int bound = (kernelSize - 1) / 2;

  float G[2][2] = {0, 0, 0, 0};
  float b[2] = {0, 0};

  float Vx = NewPointsEstimates[x * 2], Vy = NewPointsEstimates[x * 2 + 1];
  float residual = height + width;

  pixIdx = (unsigned int)(id_y * width) + (unsigned int)id_x;
  pixIdx1 = (unsigned int)(Vy * width) + (unsigned int)Vx;

  for (int loop = 0; loop < numIterations; loop++) {
    if (residual <= threshold) {
      return;
    }
    for (int i = -bound; i <= bound; i++) {
      for (int j = -bound; j <= bound; j++) {
        if (id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 &&
            id_y + i <= height - 1 && Vx + j >= 0 && Vx + j <= width - 1 &&
            Vy + i >= 0 && Vy + i <= height - 1) {
          unsigned int index = pixIdx + j + (i * width);
          unsigned int index1 = pixIdx1 + j + (i * width);

          It[index] = (input1[index] - input2[index1]);

          G[0][0] += Ix[index] * Ix[index];
          G[0][1] += Ix[index] * Iy[index];
          G[1][0] += Ix[index] * Iy[index];
          G[1][1] += Iy[index] * Iy[index];

          b[0] += Ix[index] * It[index];
          b[1] += Iy[index] * It[index];
        }
      }
    }
    float detG = (G[0][0] * G[1][1]) - (G[0][1] * G[1][0]);
    if (detG == 0)
      return;
    float Ginv[2][2] = {0, 0, 0, 0};
    Ginv[0][0] = G[1][1] / detG;
    Ginv[0][1] = -G[0][1] / detG;
    Ginv[1][0] = -G[1][0] / detG;
    Ginv[1][1] = G[0][0] / detG;

    NewPoints[x * 2] += Ginv[0][0] * b[0] + Ginv[0][1] * b[1];
    NewPoints[x * 2 + 1] += Ginv[1][0] * b[0] + Ginv[1][1] * b[1];

    Vx = NewPoints[x * 2];
    Vy = NewPoints[x * 2 + 1];

    residual = Ginv[0][0] * b[0] + Ginv[0][1] * b[1] - Ginv[1][0] * b[0] +
               Ginv[1][1] * b[1];
    residual = (residual < 0) ? -residual : residual;

    pixIdx1 = (unsigned int)(Vy * width) + (unsigned int)Vx;
  }
}