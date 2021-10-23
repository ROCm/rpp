#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// Case 0,1,2
// #define BLOCK_SIZE (16)
// // #define FILTER_SIZE (5)
// // #define TILE_SIZE (12) // BLOCK_SIZE - (2 * (FILTER_SIZE / 2))
// #define FILTER_SIZE (3)
// #define TILE_SIZE (14) // BLOCK_SIZE - (2 * (FILTER_SIZE / 2))

// Case 3,4
// #define BLOCK_SIZE (16)
// // #define FILTER_SIZE (5)
// // #define TILE_SIZE_Y (12) // BLOCK_SIZE - (2 * (FILTER_SIZE / 2))
// // #define TILE_SIZE_X (15) // ((BLOCK_SIZE * 8) - (2 * (FILTER_SIZE / 2))) / 8
// #define FILTER_SIZE (3)
// #define TILE_SIZE_Y (14) // BLOCK_SIZE - (2 * (FILTER_SIZE / 2))
// #define TILE_SIZE_X (15) // ((BLOCK_SIZE * 8) - (2 * (FILTER_SIZE / 2))) / 8




// __device__ void brightness_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *alpha_f4, float4 *beta_f4)
// {
//     dst_f8->x = src_f8->x * *alpha_f4 + *beta_f4;
//     dst_f8->y = src_f8->y * *alpha_f4 + *beta_f4;
// }

// __device__ void brightness_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *alpha_f4, float4 *beta_f4)
// {
//     dst_f8->x = src_f8->x * *alpha_f4 + *beta_f4 * (float4)0.0039216;
//     dst_f8->y = src_f8->y * *alpha_f4 + *beta_f4 * (float4)0.0039216;
// }

// __device__ void brightness_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *alpha_f4, float4 *beta_f4)
// {
//     dst_f8->x = rpp_hip_pixel_check((src_f8->x + (float4)128) * *alpha_f4 + *beta_f4) - (float4)128;
//     dst_f8->y = rpp_hip_pixel_check((src_f8->y + (float4)128) * *alpha_f4 + *beta_f4) - (float4)128;
// }

// __device__ void brightness_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *alpha_f4, float4 *beta_f4)
// {
//     dst_f8->x = src_f8->x * *alpha_f4 + *beta_f4 * (float4)0.0039216;
//     dst_f8->y = src_f8->y * *alpha_f4 + *beta_f4 * (float4)0.0039216;
// }

// template <typename T>
// __global__ void brightness_pkd_tensor(T *srcPtr,
//                                       int nStrideSrc,
//                                       int hStrideSrc,
//                                       T *dstPtr,
//                                       int nStrideDst,
//                                       int hStrideDst,
//                                       float *alpha,
//                                       float *beta,
//                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
//     uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

//     float4 alpha_f4 = (float4)alpha[id_z];
//     float4 beta_f4 = (float4)beta[id_z];

//     d_float8 src_f8, dst_f8;

//     rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
//     brightness_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
// }

template <typename T>
__global__ void box_filter_pln_tensor(T *srcPtr,
                                      int nStrideSrc,
                                      int cStrideSrc,
                                      int hStrideSrc,
                                      T *dstPtr,
                                      int nStrideDst,
                                      int cStrideDst,
                                      int hStrideDst,
                                      int channelsDst,
                                    //   uint *kernelSize, // Case 0,1,2,3,4
                                      uint kernelSize, // Case 5,6,7,8
                                      uint padLength, // Case 5,6,7,8
                                      uint2 tileSize, // Case 5,6,7,8
                                    //   float4 multiplier_f4, // Case 5,6,7
                                      RpptROIPtr roiTensorPtrSrc)
{
    // Case 0 - Kernel config from brightness

    // int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    // int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    // {
    //     return;
    // }

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

    // dstPtr[dstIdx] = srcPtr[srcIdx];





    // Case 1 - 1 pixel kernel config with dst=src

    // int id_x_o = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    // int id_y_o = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // if ((id_y_o >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x_o >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    // {
    //     return;
    // }

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_o + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_o + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // dstPtr[dstIdx] = srcPtr[srcIdx];





    // Case 2 - 1 pixel kernel config with lds processing

    // int id_x_o = hipBlockIdx_x * TILE_SIZE + hipThreadIdx_x;
    // int id_y_o = hipBlockIdx_y * TILE_SIZE + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - FILTER_SIZE / 2;
    // int id_y_i = id_y_o - FILTER_SIZE / 2;
    // int sum = 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ uchar sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    // {
    //     sBuffer[hipThreadIdx_y][hipThreadIdx_x] = srcPtr[srcIdx];
    // }
    // else
    // {
    //     sBuffer[hipThreadIdx_y][hipThreadIdx_x] = 0;
    // }

    // __syncthreads();

    // if ((hipThreadIdx_x < TILE_SIZE) && (hipThreadIdx_y < TILE_SIZE))
    // {
    //     for(int row = 0; row < FILTER_SIZE; row++)
    //     {
    //         for(int col = 0; col < FILTER_SIZE; col++)
    //         {
    //             sum += sBuffer[hipThreadIdx_y + row][hipThreadIdx_x + col];
    //         }
    //     }
    //     sum /= (FILTER_SIZE * FILTER_SIZE);

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     {
    //         dstPtr[dstIdx] = sum;
    //     }
    // }






    // Case 3 - 8 pixels kernel config with lds processing - lds in u8 - with xdir patch

    // int id_x_o = (hipBlockIdx_x * TILE_SIZE_X + hipThreadIdx_x) * 8;
    // int id_y_o = hipBlockIdx_y * TILE_SIZE_Y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - FILTER_SIZE / 2;
    // int id_y_i = id_y_o - FILTER_SIZE / 2;
    // d_float8 sum_f8;
    // sum_f8.x = (float4) 0;
    // sum_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ uchar sBuffer[BLOCK_SIZE][BLOCK_SIZE * 8];

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    // {
    //     *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x * 8] = *(uint2 *)&srcPtr[srcIdx];
    // }
    // else
    // {
    //     *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x * 8] = make_uint2(0, 0);
    // }

    // __syncthreads();

    // if ((hipThreadIdx_x < TILE_SIZE_X) && (hipThreadIdx_y < TILE_SIZE_Y))
    // {
    //     for(int row = 0; row < FILTER_SIZE; row++)
    //     {
    //         for(int col = 0; col < FILTER_SIZE; col++)
    //         {
    //             // sum += sBuffer[hipThreadIdx_y + row][hipThreadIdx_x + col];
    //             sum_f8.x = sum_f8.x + make_float4(sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col],
    //                                               sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 1],
    //                                               sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 2],
    //                                               sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 3]);
    //             sum_f8.y = sum_f8.y + make_float4(sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 4],
    //                                               sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 5],
    //                                               sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 6],
    //                                               sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col + 7]);
    //         }
    //     }
    //     sum_f8.x = sum_f8.x / (float4)(FILTER_SIZE * FILTER_SIZE);
    //     sum_f8.y = sum_f8.y / (float4)(FILTER_SIZE * FILTER_SIZE);

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     {
    //         rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    //     }
    // }





    // Case 4 - 8 pixels kernel config with lds processing - lds in f32 - with xdir patch

    // int id_x_o = (hipBlockIdx_x * TILE_SIZE_X + hipThreadIdx_x) * 8;
    // int id_y_o = hipBlockIdx_y * TILE_SIZE_Y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - FILTER_SIZE / 2;
    // int id_y_i = id_y_o - FILTER_SIZE / 2;
    // d_float8 sum_f8;
    // sum_f8.x = (float4) 0;
    // sum_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ float sBuffer[BLOCK_SIZE][BLOCK_SIZE * 8];
    // d_float8 src_f8;

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    // else
    //     src_f8 = sum_f8;

    // *(d_float8 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x * 8] = src_f8;

    // __syncthreads();

    // if ((hipThreadIdx_x < TILE_SIZE_X) && (hipThreadIdx_y < TILE_SIZE_Y))
    // {
    //     for(int row = 0; row < FILTER_SIZE; row++)
    //     {
    //         for(int col = 0; col < FILTER_SIZE; col++)
    //         {
    //             src_f8 = *(d_float8 *)&(sBuffer[hipThreadIdx_y + row][hipThreadIdx_x * 8 + col]);
    //             sum_f8.x = sum_f8.x + src_f8.x;
    //             sum_f8.y = sum_f8.y + src_f8.y;
    //         }
    //     }
    //     sum_f8.x = sum_f8.x / (float4)(FILTER_SIZE * FILTER_SIZE);
    //     sum_f8.y = sum_f8.y / (float4)(FILTER_SIZE * FILTER_SIZE);

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     {
    //         rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    //     }
    // }




    // // Case 5 - 8 pixels kernel config with lds processing - lds in f32 - with xdir patch - remove divides, reduce multiplies

    // int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    // int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    // int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - padLength;
    // int id_y_i = id_y_o - padLength;
    // d_float8 sum_f8;
    // sum_f8.x = (float4) 0;
    // sum_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ float sBuffer[16][128];
    // d_float8 src_f8;

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    // else
    //     src_f8 = sum_f8;

    // *(d_float8 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = src_f8;

    // __syncthreads();

    // if ((hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    // {
    //     for(int row = 0; row < kernelSize; row++)
    //     {
    //         for(int col = 0; col < kernelSize; col++)
    //         {
    //             src_f8 = *(d_float8 *)&(sBuffer[hipThreadIdx_y + row][hipThreadIdx_x8 + col]);
    //             sum_f8.x = sum_f8.x + src_f8.x;
    //             sum_f8.y = sum_f8.y + src_f8.y;
    //         }
    //     }
    //     sum_f8.x = sum_f8.x * multiplier_f4;
    //     sum_f8.y = sum_f8.y * multiplier_f4;

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     {
    //         rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    //     }
    // }






    // Case 6 - 8 pixels kernel config with lds processing - lds in f32 - with xdir patch - remove divides, reduce multiplies, adds padding

    // int id_x_o = (hipBlockIdx_x * tileSize.x + hipThreadIdx_x) * 8;
    // int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - padLength;
    // int id_y_i = id_y_o - padLength;
    // d_float8 sum_f8;
    // sum_f8.x = (float4) 0;
    // sum_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ float sBuffer[16][144];
    // d_float8_padded src_f9;

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f9.data);
    // else
    //     src_f9.data = sum_f8;

    // int hipThreadIdx_x9 = hipThreadIdx_x * 9;
    // *(d_float8_padded *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x9] = src_f9;

    // __syncthreads();

    // if ((hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    // {
    //     for(int row = 0; row < kernelSize; row++)
    //     {
    //         for(int col = 0; col < kernelSize; col++)
    //         {
    //             src_f9 = *(d_float8_padded *)&(sBuffer[hipThreadIdx_y + row][hipThreadIdx_x9 + col]);
    //             sum_f8.x = sum_f8.x + src_f9.data.x;
    //             sum_f8.y = sum_f8.y + src_f9.data.y;
    //         }
    //     }
    //     sum_f8.x = sum_f8.x * multiplier_f4;
    //     sum_f8.y = sum_f8.y * multiplier_f4;

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     {
    //         rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    //     }
    // }






    // Case 7 - 8 pixels kernel config with lds processing - lds in u8 - with xdir patch - remove divides, reduce multiplies, without padding

    // int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    // int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    // int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - padLength;
    // int id_y_i = id_y_o - padLength;
    // d_float8 sum_f8;
    // sum_f8.x = (float4) 0;
    // sum_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ uchar sBuffer[16][128];

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = *(uint2 *)&srcPtr[srcIdx]; // write separate func
    // else
    //     *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);

    // __syncthreads();

    // if ((hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    // {
    //     for(int row = 0; row < kernelSize; row++)
    //     {
    //         int hipThreadIdx_y_row = hipThreadIdx_y + row;
    //         for(int col = 0; col < kernelSize; col++)
    //         {
    //             int hipThreadIdx_x8_col = hipThreadIdx_x8 + col;
    //             sum_f8.x = sum_f8.x + make_float4(sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col],
    //                                               sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 1],
    //                                               sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 2],
    //                                               sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 3]);
    //             sum_f8.y = sum_f8.y + make_float4(sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 4],
    //                                               sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 5],
    //                                               sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 6],
    //                                               sBuffer[hipThreadIdx_y_row][hipThreadIdx_x8_col + 7]);
    //         }
    //     }
    //     sum_f8.x = sum_f8.x * multiplier_f4;
    //     sum_f8.y = sum_f8.y * multiplier_f4;

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) // move if up
    //     {
    //         rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    //     }
    // }









    // Case 8 - 8 pixels kernel config with lds processing - lds in u8 - with xdir patch - remove divides, reduce multiplies, without padding, individual pixel processing with fmaf

    // int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    // int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    // int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - padLength;
    // int id_y_i = id_y_o - padLength;
    // d_float8 sum_f8;
    // sum_f8.x = (float4) 0;
    // sum_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ uchar sBuffer[16][128];

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //     *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = *(uint2 *)&srcPtr[srcIdx]; // write separate func
    // else
    //     *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);

    // __syncthreads();

    // if ((hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    // {
    //     for(int row = 0; row < kernelSize; row++)
    //     {
    //         uint src;
    //         float src_f;
    //         uint *sBufferPtr = (uint *)&sBuffer[hipThreadIdx_y + row][hipThreadIdx_x8];
    //         src = sBufferPtr[0];
    //         src_f = rpp_hip_unpack0(src);
    //         sum_f8.x.x = fmaf(src_f, 0.1111111f, sum_f8.x.x);
    //         src_f = rpp_hip_unpack1(src);
    //         sum_f8.x.x = fmaf(src_f, 0.1111111f, sum_f8.x.x);
    //         sum_f8.x.y = fmaf(src_f, 0.1111111f, sum_f8.x.y);
    //         src_f = rpp_hip_unpack2(src);
    //         sum_f8.x.x = fmaf(src_f, 0.1111111f, sum_f8.x.x);
    //         sum_f8.x.y = fmaf(src_f, 0.1111111f, sum_f8.x.y);
    //         sum_f8.x.z = fmaf(src_f, 0.1111111f, sum_f8.x.z);
    //         src_f = rpp_hip_unpack3(src);
    //         sum_f8.x.y = fmaf(src_f, 0.1111111f, sum_f8.x.y);
    //         sum_f8.x.z = fmaf(src_f, 0.1111111f, sum_f8.x.z);
    //         sum_f8.x.w = fmaf(src_f, 0.1111111f, sum_f8.x.w);
    //         src = sBufferPtr[1];
    //         src_f = rpp_hip_unpack0(src);
    //         sum_f8.x.z = fmaf(src_f, 0.1111111f, sum_f8.x.z);
    //         sum_f8.x.w = fmaf(src_f, 0.1111111f, sum_f8.x.w);
    //         sum_f8.y.x = fmaf(src_f, 0.1111111f, sum_f8.y.x);
    //         src_f = rpp_hip_unpack1(src);
    //         sum_f8.x.w = fmaf(src_f, 0.1111111f, sum_f8.x.w);
    //         sum_f8.y.x = fmaf(src_f, 0.1111111f, sum_f8.y.x);
    //         sum_f8.y.y = fmaf(src_f, 0.1111111f, sum_f8.y.y);
    //         src_f = rpp_hip_unpack2(src);
    //         sum_f8.y.x = fmaf(src_f, 0.1111111f, sum_f8.y.x);
    //         sum_f8.y.y = fmaf(src_f, 0.1111111f, sum_f8.y.y);
    //         sum_f8.y.z = fmaf(src_f, 0.1111111f, sum_f8.y.z);
    //         src_f = rpp_hip_unpack3(src);
    //         sum_f8.y.y = fmaf(src_f, 0.1111111f, sum_f8.y.y);
    //         sum_f8.y.z = fmaf(src_f, 0.1111111f, sum_f8.y.z);
    //         sum_f8.y.w = fmaf(src_f, 0.1111111f, sum_f8.y.w);
    //         src = sBufferPtr[2];
    //         src_f = rpp_hip_unpack0(src);
    //         sum_f8.y.z = fmaf(src_f, 0.1111111f, sum_f8.y.z);
    //         sum_f8.y.w = fmaf(src_f, 0.1111111f, sum_f8.y.w);
    //         src_f = rpp_hip_unpack1(src);
    //         sum_f8.y.w = fmaf(src_f, 0.1111111f, sum_f8.y.w);
    //     }

    //     if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) && (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight)) // move if up
    //     {
    //         rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    //     }
    // }










    // Case 9 - 8 pixels kernel config with lds processing - lds in u8 - with xdir patch - remove divides, reduce multiplies, without padding, individual pixel processing with fmaf, dual row processing

    // int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    // int hipThreadIdx_y2 = hipThreadIdx_y * 2;
    // int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    // int id_y_o = (hipBlockIdx_y * tileSize.y * 2) + hipThreadIdx_y2;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // int id_x_i = id_x_o - padLength;
    // int id_y_i = id_y_o - padLength;
    // d_float8 sum1_f8, sum2_f8;
    // sum1_f8.x = (float4) 0;
    // sum1_f8.y = (float4) 0;
    // sum2_f8.x = (float4) 0;
    // sum2_f8.y = (float4) 0;

    // uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    // uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    // __shared__ uchar sBuffer[32][128];

    // if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    // {
    //     if ((id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //         *(uint2 *)&sBuffer[hipThreadIdx_y2][hipThreadIdx_x8] = *(uint2 *)&srcPtr[srcIdx]; // write separate func
    //     else
    //         *(uint2 *)&sBuffer[hipThreadIdx_y2][hipThreadIdx_x8] = make_uint2(0, 0);

    //     id_y_i++;
    //     srcIdx += hStrideSrc;

    //     if ((id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    //         *(uint2 *)&sBuffer[hipThreadIdx_y2 + 1][hipThreadIdx_x8] = *(uint2 *)&srcPtr[srcIdx]; // write separate func
    //     else
    //         *(uint2 *)&sBuffer[hipThreadIdx_y2 + 1][hipThreadIdx_x8] = make_uint2(0, 0);
    // }

    // __syncthreads();

    // if ((hipThreadIdx_x < tileSize.x) && (hipThreadIdx_y < tileSize.y))
    // {
    //     uint src;
    //     float src_f;
    //     uint *sBufferPtr;

    //     // Row 0
    //     sBufferPtr = (uint *)&sBuffer[hipThreadIdx_y2][hipThreadIdx_x8];
    //     src = sBufferPtr[0];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     src = sBufferPtr[1];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     src = sBufferPtr[2];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);

    //     // Row 1
    //     sBufferPtr = (uint *)&sBuffer[hipThreadIdx_y2 + 1][hipThreadIdx_x8];
    //     src = sBufferPtr[0];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     src = sBufferPtr[1];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);
    //     src = sBufferPtr[2];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);

    //     // Row 2
    //     sBufferPtr = (uint *)&sBuffer[hipThreadIdx_y2 + 2][hipThreadIdx_x8];
    //     src = sBufferPtr[0];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum1_f8.x.x = fmaf(src_f, 0.1111111f, sum1_f8.x.x);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum1_f8.x.y = fmaf(src_f, 0.1111111f, sum1_f8.x.y);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     src = sBufferPtr[1];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.x.z = fmaf(src_f, 0.1111111f, sum1_f8.x.z);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.x.w = fmaf(src_f, 0.1111111f, sum1_f8.x.w);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum1_f8.y.x = fmaf(src_f, 0.1111111f, sum1_f8.y.x);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum1_f8.y.y = fmaf(src_f, 0.1111111f, sum1_f8.y.y);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);
    //     src = sBufferPtr[2];
    //     src_f = rpp_hip_unpack0(src);
    //     sum1_f8.y.z = fmaf(src_f, 0.1111111f, sum1_f8.y.z);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);
    //     src_f = rpp_hip_unpack1(src);
    //     sum1_f8.y.w = fmaf(src_f, 0.1111111f, sum1_f8.y.w);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);

    //     // Row 3
    //     sBufferPtr = (uint *)&sBuffer[hipThreadIdx_y2 + 3][hipThreadIdx_x8];
    //     src = sBufferPtr[0];
    //     src_f = rpp_hip_unpack0(src);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum2_f8.x.x = fmaf(src_f, 0.1111111f, sum2_f8.x.x);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum2_f8.x.y = fmaf(src_f, 0.1111111f, sum2_f8.x.y);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     src = sBufferPtr[1];
    //     src_f = rpp_hip_unpack0(src);
    //     sum2_f8.x.z = fmaf(src_f, 0.1111111f, sum2_f8.x.z);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     src_f = rpp_hip_unpack1(src);
    //     sum2_f8.x.w = fmaf(src_f, 0.1111111f, sum2_f8.x.w);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     src_f = rpp_hip_unpack2(src);
    //     sum2_f8.y.x = fmaf(src_f, 0.1111111f, sum2_f8.y.x);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     src_f = rpp_hip_unpack3(src);
    //     sum2_f8.y.y = fmaf(src_f, 0.1111111f, sum2_f8.y.y);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);
    //     src = sBufferPtr[2];
    //     src_f = rpp_hip_unpack0(src);
    //     sum2_f8.y.z = fmaf(src_f, 0.1111111f, sum2_f8.y.z);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);
    //     src_f = rpp_hip_unpack1(src);
    //     sum2_f8.y.w = fmaf(src_f, 0.1111111f, sum2_f8.y.w);

    //     if (id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) // move if up
    //     {
    //         if (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
    //             rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum1_f8);

    //         id_y_o++;
    //         dstIdx += hStrideDst;

    //         if (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
    //             rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum2_f8);
    //     }
    // }










    // Case 10 - 8 pixels kernel config with lds processing - lds in u8 - with xdir patch - remove divides, reduce multiplies, without padding, individual pixel processing with fmaf

    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 sum_f8;
    sum_f8.x = (float4) 0;
    sum_f8.y = (float4) 0;

    uint srcIdx = (id_z * nStrideSrc) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y_o * hStrideDst) + id_x_o;

    __shared__ uchar sBuffer[16][128];

    if ((id_x_i >= 0) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= 0) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = *(uint2 *)&srcPtr[srcIdx]; // write separate func
    else
        *(uint2 *)&sBuffer[hipThreadIdx_y][hipThreadIdx_x8] = make_uint2(0, 0);

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        for(int row = 0; row < kernelSize; row++)
        {
            uint src;
            float src_f;
            uint *sBufferPtr = (uint *)&sBuffer[hipThreadIdx_y + row][hipThreadIdx_x8];
            src = sBufferPtr[0];
            src_f = rpp_hip_unpack0(src);
            sum_f8.x.x = fmaf(src_f, 0.1111111f, sum_f8.x.x);
            src_f = rpp_hip_unpack1(src);
            sum_f8.x.x = fmaf(src_f, 0.1111111f, sum_f8.x.x);
            sum_f8.x.y = fmaf(src_f, 0.1111111f, sum_f8.x.y);
            src_f = rpp_hip_unpack2(src);
            sum_f8.x.x = fmaf(src_f, 0.1111111f, sum_f8.x.x);
            sum_f8.x.y = fmaf(src_f, 0.1111111f, sum_f8.x.y);
            sum_f8.x.z = fmaf(src_f, 0.1111111f, sum_f8.x.z);
            src_f = rpp_hip_unpack3(src);
            sum_f8.x.y = fmaf(src_f, 0.1111111f, sum_f8.x.y);
            sum_f8.x.z = fmaf(src_f, 0.1111111f, sum_f8.x.z);
            sum_f8.x.w = fmaf(src_f, 0.1111111f, sum_f8.x.w);
            src = sBufferPtr[1];
            src_f = rpp_hip_unpack0(src);
            sum_f8.x.z = fmaf(src_f, 0.1111111f, sum_f8.x.z);
            sum_f8.x.w = fmaf(src_f, 0.1111111f, sum_f8.x.w);
            sum_f8.y.x = fmaf(src_f, 0.1111111f, sum_f8.y.x);
            src_f = rpp_hip_unpack1(src);
            sum_f8.x.w = fmaf(src_f, 0.1111111f, sum_f8.x.w);
            sum_f8.y.x = fmaf(src_f, 0.1111111f, sum_f8.y.x);
            sum_f8.y.y = fmaf(src_f, 0.1111111f, sum_f8.y.y);
            src_f = rpp_hip_unpack2(src);
            sum_f8.y.x = fmaf(src_f, 0.1111111f, sum_f8.y.x);
            sum_f8.y.y = fmaf(src_f, 0.1111111f, sum_f8.y.y);
            sum_f8.y.z = fmaf(src_f, 0.1111111f, sum_f8.y.z);
            src_f = rpp_hip_unpack3(src);
            sum_f8.y.y = fmaf(src_f, 0.1111111f, sum_f8.y.y);
            sum_f8.y.z = fmaf(src_f, 0.1111111f, sum_f8.y.z);
            sum_f8.y.w = fmaf(src_f, 0.1111111f, sum_f8.y.w);
            src = sBufferPtr[2];
            src_f = rpp_hip_unpack0(src);
            sum_f8.y.z = fmaf(src_f, 0.1111111f, sum_f8.y.z);
            sum_f8.y.w = fmaf(src_f, 0.1111111f, sum_f8.y.w);
            src_f = rpp_hip_unpack1(src);
            sum_f8.y.w = fmaf(src_f, 0.1111111f, sum_f8.y.w);
        }

        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &sum_f8);
    }
}

// template <typename T>
// __global__ void brightness_pkd3_pln3_tensor(T *srcPtr,
//                                             int nStrideSrc,
//                                             int hStrideSrc,
//                                             T *dstPtr,
//                                             int nStrideDst,
//                                             int cStrideDst,
//                                             int hStrideDst,
//                                             float *alpha,
//                                             float *beta,
//                                             RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;

//     float4 alpha_f4 = (float4)alpha[id_z];
//     float4 beta_f4 = (float4)beta[id_z];

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
//     brightness_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.x);

//     dstIdx += cStrideDst;

//     brightness_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.y);

//     dstIdx += cStrideDst;

//     brightness_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.z);
// }

// template <typename T>
// __global__ void brightness_pln3_pkd3_tensor(T *srcPtr,
//                                             int nStrideSrc,
//                                             int cStrideSrc,
//                                             int hStrideSrc,
//                                             T *dstPtr,
//                                             int nStrideDst,
//                                             int hStrideDst,
//                                             float *alpha,
//                                             float *beta,
//                                             RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }

//     uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
//     uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;

//     float4 alpha_f4 = (float4)(alpha[id_z]);
//     float4 beta_f4 = (float4)(beta[id_z]);

//     d_float24 src_f24, dst_f24;

//     rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr, srcIdx, cStrideSrc, &src_f24);
//     brightness_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &alpha_f4, &beta_f4);
//     brightness_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &alpha_f4, &beta_f4);
//     brightness_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &alpha_f4, &beta_f4);
//     rpp_hip_pack_float24_and_store24(dstPtr, dstIdx, &dst_f24);
// }

template <typename T>
RppStatus hip_exec_box_filter_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     Rpp32u kernelSize,
                                     RpptROIPtr roiTensorPtrSrc,
                                     rpp::Handle& handle)
{
    // Case 0 - Kernel config from brightness
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    // int globalThreads_y = dstDescPtr->h;
    // int globalThreads_z = handle.GetBatchSize();


    // Case 1 - 1 pixel kernel config with dst=src
    // int localThreads_x = BLOCK_SIZE;
    // int localThreads_y = BLOCK_SIZE;
    // int localThreads_z = 1;
    // int globalThreads_x = dstDescPtr->strides.hStride;
    // int globalThreads_y = dstDescPtr->h;
    // int globalThreads_z = handle.GetBatchSize();



    // Case 2 - 1 pixel kernel config with lds processing
    // int localThreads_x = BLOCK_SIZE;
    // int localThreads_y = BLOCK_SIZE;
    // int localThreads_z = 1;
    // int globalThreads_x = dstDescPtr->strides.hStride;
    // int globalThreads_y = dstDescPtr->h;
    // int globalThreads_z = handle.GetBatchSize();



    // Case 3,4 - 8 pixels kernel config with lds processing
    // int localThreads_x = BLOCK_SIZE;
    // int localThreads_y = BLOCK_SIZE;
    // int localThreads_z = 1;
    // int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    // int globalThreads_y = dstDescPtr->h;
    // int globalThreads_z = handle.GetBatchSize();



    // Case 5,6,7 - 8 pixels kernel config with lds processing, remove divides and reduce multiplies
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    // int globalThreads_y = dstDescPtr->h;
    // int globalThreads_z = handle.GetBatchSize();

    // float4 multiplier_f4 = (float4) (1.0f / (kernelSize * kernelSize));
    // uint padLength = kernelSize / 2;
    // uint padLengthTwice = padLength * 2;
    // uint2 tileSize;
    // tileSize.x = (128 - padLengthTwice) / 8;
    // tileSize.y = 16 - padLengthTwice;




    // Case 8,10 - 8 pixels kernel config with lds processing, remove divides and reduce multiplies
    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;




    // Case 9 - 8 pixels kernel config with lds processing, remove divides and reduce multiplies
    // int localThreads_x = 16;
    // int localThreads_y = 16;
    // int localThreads_z = 1;
    // int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    // int globalThreads_y = dstDescPtr->h >> 1;
    // int globalThreads_z = handle.GetBatchSize();

    // uint padLength = kernelSize / 2;
    // uint padLengthTwice = padLength * 2;
    // uint2 tileSize;
    // tileSize.x = (128 - padLengthTwice) / 8;
    // tileSize.y = (32 - padLengthTwice) / 2;








    // if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    // {
    //     hipLaunchKernelGGL(brightness_pkd_tensor,
    //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.hStride,
    //                        handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                        handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                        roiTensorPtrSrc);
    // }
    // else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))





    // Case 0 - Kernel config from brightness

    // if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    // {
    //     hipLaunchKernelGGL(box_filter_pln_tensor,
    //                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.cStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.cStride,
    //                        dstDescPtr->strides.hStride,
    //                        dstDescPtr->c,
    //                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                        roiTensorPtrSrc);
    // }

    // Case 1 - 1 pixel kernel config with dst=src

    // if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    // {
    //     hipLaunchKernelGGL(box_filter_pln_tensor,
    //                        dim3(ceil((float)globalThreads_x/BLOCK_SIZE), ceil((float)globalThreads_y/BLOCK_SIZE), ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.cStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.cStride,
    //                        dstDescPtr->strides.hStride,
    //                        dstDescPtr->c,
    //                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                        roiTensorPtrSrc);
    // }

    // Case 2 - 1 pixel kernel config with lds processing

    // if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    // {
    //     hipLaunchKernelGGL(box_filter_pln_tensor,
    //                        dim3(ceil((float)globalThreads_x/TILE_SIZE), ceil((float)globalThreads_y/TILE_SIZE), ceil((float)globalThreads_z/localThreads_z)),
    //                     //    dim3((globalThreads_x + TILE_SIZE - 1) / TILE_SIZE, (globalThreads_y + TILE_SIZE - 1) / TILE_SIZE, ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.cStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.cStride,
    //                        dstDescPtr->strides.hStride,
    //                        dstDescPtr->c,
    //                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                        roiTensorPtrSrc);
    // }

    // Case 3,4 - 8 pixels kernel config with lds processing in u8/f32

    // if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    // {
    //     hipLaunchKernelGGL(box_filter_pln_tensor,
    //                        dim3(ceil((float)globalThreads_x/TILE_SIZE_X), ceil((float)globalThreads_y/TILE_SIZE_Y), ceil((float)globalThreads_z/localThreads_z)),
    //                     //    dim3((globalThreads_x + TILE_SIZE - 1) / TILE_SIZE, (globalThreads_y + TILE_SIZE - 1) / TILE_SIZE, ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.cStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.cStride,
    //                        dstDescPtr->strides.hStride,
    //                        dstDescPtr->c,
    //                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
    //                        roiTensorPtrSrc);
    // }

    // Case 5,6,7 - 8 pixels kernel config with lds processing in f32, remove divides, reduce multiplies, with/without padding

    // if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    // {
    //     hipLaunchKernelGGL(box_filter_pln_tensor,
    //                        dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
    //                        dim3(localThreads_x, localThreads_y, localThreads_z),
    //                        0,
    //                        handle.GetStream(),
    //                        srcPtr,
    //                        srcDescPtr->strides.nStride,
    //                        srcDescPtr->strides.cStride,
    //                        srcDescPtr->strides.hStride,
    //                        dstPtr,
    //                        dstDescPtr->strides.nStride,
    //                        dstDescPtr->strides.cStride,
    //                        dstDescPtr->strides.hStride,
    //                        dstDescPtr->c,
    //                        kernelSize,
    //                        padLength,
    //                        tileSize,
    //                        multiplier_f4,
    //                        roiTensorPtrSrc);
    // }

    // Case 8,9,10 - 8 pixels kernel config with lds processing in u8, remove divides, reduce multiplies, with/without padding

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(box_filter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.cStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.cStride,
                           dstDescPtr->strides.hStride,
                           dstDescPtr->c,
                           kernelSize,
                           padLength,
                           tileSize,
                           roiTensorPtrSrc);
    }





    // else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    // {
    //     if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    //     {
    //         hipLaunchKernelGGL(brightness_pkd3_pln3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.cStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    //     else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    //     {
    //         globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    //         hipLaunchKernelGGL(brightness_pln3_pkd3_tensor,
    //                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                            dim3(localThreads_x, localThreads_y, localThreads_z),
    //                            0,
    //                            handle.GetStream(),
    //                            srcPtr,
    //                            srcDescPtr->strides.nStride,
    //                            srcDescPtr->strides.cStride,
    //                            srcDescPtr->strides.hStride,
    //                            dstPtr,
    //                            dstDescPtr->strides.nStride,
    //                            dstDescPtr->strides.hStride,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
    //                            handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
    //                            roiTensorPtrSrc);
    //     }
    // }

    return RPP_SUCCESS;
}
