#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - median_filter device helpers --------------------

__device__ void median_filter_3x3_row_hip_compute(uchar *srcPtr, d_float8 *median_f8)
{
    __shared__ float smem[3][10];  // Shared memory for 3 rows

    float median_vals[8];

    // Load 3 rows into shared memory
    for (int i = 0; i < 3; i++)
    {
        uint3 src_ui3 = *(uint3 *)(srcPtr + i * SMEM_LENGTH_X);
        smem[i][0] = rpp_hip_unpack0(src_ui3.x);
        smem[i][1] = rpp_hip_unpack1(src_ui3.x);
        smem[i][2] = rpp_hip_unpack2(src_ui3.x);
        smem[i][3] = rpp_hip_unpack3(src_ui3.x);
        smem[i][4] = rpp_hip_unpack0(src_ui3.y);
        smem[i][5] = rpp_hip_unpack1(src_ui3.y);
        smem[i][6] = rpp_hip_unpack2(src_ui3.y);
        smem[i][7] = rpp_hip_unpack3(src_ui3.y);
        smem[i][8] = rpp_hip_unpack0(src_ui3.z);
        smem[i][9] = rpp_hip_unpack1(src_ui3.z);
    }

    __syncthreads(); // Sync to ensure all data is loaded

    for (int filter = 0; filter < 8; filter++)
    {
        float window[9];

        // Load 3x3 window from shared memory
        window[0] = smem[0][filter];
        window[1] = smem[0][filter + 1];
        window[2] = smem[0][filter + 2];
        window[3] = smem[1][filter];
        window[4] = smem[1][filter + 1];
        window[5] = smem[1][filter + 2];
        window[6] = smem[2][filter];
        window[7] = smem[2][filter + 1];
        window[8] = smem[2][filter + 2];

        // **Bitonic Sort (Efficient GPU Sorting)**
        #pragma unroll
        for (int i = 0; i < 9; i++)
        {
            #pragma unroll
            for (int j = 0; j < 8 - i; j++)
            {
                if (window[j] > window[j + 1])
                {
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        // Store the median (5th element after sorting)
        median_vals[filter] = window[4];
    }

    // Store results in d_float8 structure
    median_f8->f4[0] = make_float4(median_vals[0], median_vals[1], median_vals[2], median_vals[3]);
    median_f8->f4[1] = make_float4(median_vals[4], median_vals[5], median_vals[6], median_vals[7]);
}

__device__ void median_filter_5x5_row_hip_compute(uchar *srcPtr, d_float8 *median_f8)
{
    __shared__ float smem[5][12];  // Shared memory for 5 rows

    float median_vals[8];

    // Load 5 rows into shared memory
    for (int i = 0; i < 5; i++)
    {
        uint3 src_ui3 = *(uint3 *)(srcPtr + i * SMEM_LENGTH_X);
        smem[i][0] = rpp_hip_unpack0(src_ui3.x);
        smem[i][1] = rpp_hip_unpack1(src_ui3.x);
        smem[i][2] = rpp_hip_unpack2(src_ui3.x);
        smem[i][3] = rpp_hip_unpack3(src_ui3.x);
        smem[i][4] = rpp_hip_unpack0(src_ui3.y);
        smem[i][5] = rpp_hip_unpack1(src_ui3.y);
        smem[i][6] = rpp_hip_unpack2(src_ui3.y);
        smem[i][7] = rpp_hip_unpack3(src_ui3.y);
        smem[i][8] = rpp_hip_unpack0(src_ui3.z);
        smem[i][9] = rpp_hip_unpack1(src_ui3.z);
        smem[i][10] = rpp_hip_unpack2(src_ui3.z);
        smem[i][11] = rpp_hip_unpack3(src_ui3.z);
    }

    __syncthreads();

    for (int filter = 0; filter < 8; filter++)
    {
        float window[25];

        // Load 5x5 window from shared memory
        for (int i = 0; i < 25; i++)
        {
            window[i] = smem[i / 5][filter + (i % 5)];
        }

        // **Bitonic Sort**
        #pragma unroll
        for (int i = 0; i < 25; i++)
        {
            #pragma unroll
            for (int j = 0; j < 24 - i; j++)
            {
                if (window[j] > window[j + 1])
                {
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        median_vals[filter] = window[12]; // Median element
    }

    median_f8->f4[0] = make_float4(median_vals[0], median_vals[1], median_vals[2], median_vals[3]);
    median_f8->f4[1] = make_float4(median_vals[4], median_vals[5], median_vals[6], median_vals[7]);
}

__device__ void median_filter_7x7_row_hip_compute(uchar *srcPtr, d_float8 *median_f8)
{
    __shared__ float smem[7][14];  // Shared memory for 7 rows

    float median_vals[8];

    for (int i = 0; i < 7; i++)
    {
        uint4 src_ui4 = *(uint4 *)(srcPtr + i * SMEM_LENGTH_X);
        smem[i][0] = rpp_hip_unpack0(src_ui4.x);
        smem[i][1] = rpp_hip_unpack1(src_ui4.x);
        smem[i][2] = rpp_hip_unpack2(src_ui4.x);
        smem[i][3] = rpp_hip_unpack3(src_ui4.x);
        smem[i][4] = rpp_hip_unpack0(src_ui4.y);
        smem[i][5] = rpp_hip_unpack1(src_ui4.y);
        smem[i][6] = rpp_hip_unpack2(src_ui4.y);
        smem[i][7] = rpp_hip_unpack3(src_ui4.y);
        smem[i][8] = rpp_hip_unpack0(src_ui4.z);
        smem[i][9] = rpp_hip_unpack1(src_ui4.z);
        smem[i][10] = rpp_hip_unpack2(src_ui4.z);
        smem[i][11] = rpp_hip_unpack3(src_ui4.z);
        smem[i][12] = rpp_hip_unpack0(src_ui4.w);
        smem[i][13] = rpp_hip_unpack1(src_ui4.w);
    }

    __syncthreads();

    for (int filter = 0; filter < 8; filter++)
    {
        float window[49];

        for (int i = 0; i < 49; i++)
        {
            window[i] = smem[i / 7][filter + (i % 7)];
        }

        // **Bitonic Sort**
        #pragma unroll
        for (int i = 0; i < 49; i++)
        {
            #pragma unroll
            for (int j = 0; j < 48 - i; j++)
            {
                if (window[j] > window[j + 1])
                {
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        median_vals[filter] = window[24]; // Median element
    }

    median_f8->f4[0] = make_float4(median_vals[0], median_vals[1], median_vals[2], median_vals[3]);
    median_f8->f4[1] = make_float4(median_vals[4], median_vals[5], median_vals[6], median_vals[7]);
}

__device__ void median_filter_9x9_row_hip_compute(uchar *srcPtr, d_float8 *median_f8)
{
    __shared__ float smem[9][16];  // Shared memory for 9 rows

    float median_vals[8];

    for (int i = 0; i < 9; i++)
    {
        uint4 src_ui4 = *(uint4 *)(srcPtr + i * SMEM_LENGTH_X);
        smem[i][0] = rpp_hip_unpack0(src_ui4.x);
        smem[i][1] = rpp_hip_unpack1(src_ui4.x);
        smem[i][2] = rpp_hip_unpack2(src_ui4.x);
        smem[i][3] = rpp_hip_unpack3(src_ui4.x);
        smem[i][4] = rpp_hip_unpack0(src_ui4.y);
        smem[i][5] = rpp_hip_unpack1(src_ui4.y);
        smem[i][6] = rpp_hip_unpack2(src_ui4.y);
        smem[i][7] = rpp_hip_unpack3(src_ui4.y);
        smem[i][8] = rpp_hip_unpack0(src_ui4.z);
        smem[i][9] = rpp_hip_unpack1(src_ui4.z);
        smem[i][10] = rpp_hip_unpack2(src_ui4.z);
        smem[i][11] = rpp_hip_unpack3(src_ui4.z);
        smem[i][12] = rpp_hip_unpack0(src_ui4.w);
        smem[i][13] = rpp_hip_unpack1(src_ui4.w);
        smem[i][14] = rpp_hip_unpack2(src_ui4.w);
        smem[i][15] = rpp_hip_unpack3(src_ui4.w);
    }

    __syncthreads();

    for (int filter = 0; filter < 8; filter++)
    {
        float window[81];

        for (int i = 0; i < 81; i++)
        {
            window[i] = smem[i / 9][filter + (i % 9)];
        }

        // **Bitonic Sort**
        #pragma unroll
        for (int i = 0; i < 81; i++)
        {
            #pragma unroll
            for (int j = 0; j < 80 - i; j++)
            {
                if (window[j] > window[j + 1])
                {
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }

        median_vals[filter] = window[40]; // Median element
    }

    median_f8->f4[0] = make_float4(median_vals[0], median_vals[1], median_vals[2], median_vals[3]);
    median_f8->f4[1] = make_float4(median_vals[4], median_vals[5], median_vals[6], median_vals[7]);
}

template <typename T>
__global__ void median_filter_3x3_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0; i < 8; i++)  
        {
            int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clamped_y * srcStridesNH.y) + (clamped_x * 3);

            tempBuffer[i * 3] = srcPtr[clampedIdx];         // R
            tempBuffer[i * 3 + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[i * 3 + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // if(id_x_o == 0 && id_y_o == 0 && id_z == 0)
        // {
        //     for(int row = 0; row < 9 ; row++)
        //     {
        //         printf("\n");
        //         for(int col = 0; col < 9; col++)
        //         {
        //             printf(" %f ", (float)tempBuffer[row * 3 + col]);
        //         }
        //     }
        // }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0; i < 8; i++)  
        {
            int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clamped_y * srcStridesNH.y) + (clamped_x * 3);

            tempBuffer[i * 3] = srcPtr[clampedIdx];         // R
            tempBuffer[i * 3 + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[i * 3 + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0; i < 8; i++)  
        {
            int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clamped_y * srcStridesNH.y) + (clamped_x * 3);

            tempBuffer[i * 3] = srcPtr[clampedIdx];         // R
            tempBuffer[i * 3 + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[i * 3 + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pkd_hip_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i > roiTensorPtrSrc[id_z].xywhROI.xy.x) && ((id_x_i + 7 + padLength) < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
    (id_y_i > roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[24]; // Temporary storage for 8 pixels, 3 channels

        for (int i = 0; i < 8; i++)  
        {
            int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNH.x) + (clamped_y * srcStridesNH.y) + (clamped_x * 3);

            tempBuffer[i * 3] = srcPtr[clampedIdx];         // R
            tempBuffer[i * 3 + 1] = srcPtr[clampedIdx + 1]; // G
            tempBuffer[i * 3 + 2] = srcPtr[clampedIdx + 2]; // B
        }

        // Use helper function to load padded data into shared memory
        rpp_hip_load24_pkd3_to_uchar8_pln3(tempBuffer, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // Compute input pixel coordinates with edge replication
    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    if ((id_x_i >= roiTensorPtrSrc[id_z].xywhROI.xy.x) && (id_x_i + 7 + padLength < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= roiTensorPtrSrc[id_z].xywhROI.xy.y) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else
    {
        // Nearest-neighbor padding
        T tempBuffer[8]; // Temporary storage for 8 pixels
        for (int i = 0; i < 8; i++)
        {
            int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
            int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

            int clampedIdx = (id_z * srcStridesNCH.x) + (clamped_y * srcStridesNCH.z) + clamped_x;
            tempBuffer[i] = srcPtr[clampedIdx];  // Load nearest pixel
        }
        rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]); // Convert to uchar8
    }

    __syncthreads();

    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    // Process the remaining 2 channels for RGB images
    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
        {
            T tempBuffer[8];
            for (int i = 0; i < 8; i++)
            {
                int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (clamped_y * srcStridesNCH.z) + clamped_x;
                tempBuffer[i] = srcPtr[clampedIdx];
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        else
        {
            T tempBuffer[8];
            for (int i = 0; i < 8; i++)
            {
                int clamped_x = max(roiTensorPtrSrc[id_z].xywhROI.xy.x,
                                    min(id_x_i + i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1));
                int clamped_y = max(roiTensorPtrSrc[id_z].xywhROI.xy.y,
                                    min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1));

                int clampedIdx = (id_z * srcStridesNCH.x) + (clamped_y * srcStridesNCH.z) + clamped_x;
                tempBuffer[i] = srcPtr[clampedIdx];
            }
            rpp_hip_load8_to_uchar8(tempBuffer, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        }

        __syncthreads();

        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
                                          uint padLength,
                                          uint2 tileSize,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pln_hip_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 uint padLength,
                                                 uint2 tileSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float8 median_f8;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_1C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;
    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
        rpp_hip_adjust_range(dstPtr, &median_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
    }

    if (channelsDst == 3)
    {
        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }

        __syncthreads();
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;
        if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
            rpp_hip_load8_to_uchar8(srcPtr + srcIdx, &src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
        __syncthreads();
        if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
            (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
            (hipThreadIdx_x < tileSize.x) &&
            (hipThreadIdx_y < tileSize.y))
        {
            median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y    ][hipThreadIdx_x8], &median_f8);
            rpp_hip_adjust_range(dstPtr, &median_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &median_f8);
        }
    }
}

// -------------------- Set 3 - PKD3->PLN3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pkd3_pln3_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = min(max(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1);
    id_y_i = min(max(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                uint padLength,
                                                uint2 tileSize,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                uint padLength,
                                                uint2 tileSize,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pkd3_pln3_hip_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y_o * dstStridesNCH.z) + id_x_o;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    uchar *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load24_pkd3_to_uchar8_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &median_f24);
    }
}

// -------------------- Set 4 - PLN3->PKD3 for T = U8/F32/F16/I8 --------------------

// kernelSize = 3
template <typename T>
__global__ void median_filter_3x3_pln3_pkd3_hip_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       uint padLength,
                                                       uint2 tileSize,
                                                       RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = min(max(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x), roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1);
    id_y_i = min(max(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y), roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_3x3_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 5
template <typename T>
__global__ void median_filter_5x5_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint padLength,
                                                uint2 tileSize,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_5x5_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 7
template <typename T>
__global__ void median_filter_7x7_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint padLength,
                                                uint2 tileSize,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_7x7_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// kernelSize = 9
template <typename T>
__global__ void median_filter_9x9_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                uint padLength,
                                                uint2 tileSize,
                                                RpptROIPtr roiTensorPtrSrc)
{
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int id_x_o = (hipBlockIdx_x * tileSize.x * 8) + hipThreadIdx_x8;
    int id_y_o = hipBlockIdx_y * tileSize.y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int id_x_i = id_x_o - padLength;
    int id_y_i = id_y_o - padLength;
    id_x_i = max(min(id_x_i, roiTensorPtrSrc[id_z].xywhROI.xy.x + roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1), roiTensorPtrSrc[id_z].xywhROI.xy.x);
    id_y_i = max(min(id_y_i, roiTensorPtrSrc[id_z].xywhROI.xy.y + roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1), roiTensorPtrSrc[id_z].xywhROI.xy.y);
    d_float24 median_f24;
    __shared__ uchar src_smem[SMEM_LENGTH_Y_3C][SMEM_LENGTH_X];

    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + ((id_y_i + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x_i + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y_o * dstStridesNH.y) + id_x_o * 3;

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    if ((id_x_i >= -(int)padLength) && (id_x_i < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_i >= -(int)padLength) && (id_y_i < roiTensorPtrSrc[id_z].xywhROI.roiHeight))
    {
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.x, &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.y, &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_to_uchar8(srcPtr + srcIdx.z, &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    __syncthreads();
    if ((id_x_o < roiTensorPtrSrc[id_z].xywhROI.roiWidth) &&
        (id_y_o < roiTensorPtrSrc[id_z].xywhROI.roiHeight) &&
        (hipThreadIdx_x < tileSize.x) &&
        (hipThreadIdx_y < tileSize.y))
    {
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.x    ][hipThreadIdx_x8], &median_f24.f8[0]);
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.y    ][hipThreadIdx_x8], &median_f24.f8[1]);
        median_filter_9x9_row_hip_compute(&src_smem[hipThreadIdx_y_channel.z    ][hipThreadIdx_x8], &median_f24.f8[2]);
        rpp_hip_adjust_range(dstPtr, &median_f24);
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &median_f24);
    }
}

// -------------------- Set 5 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_median_filter_tensor(T *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        T *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        Rpp32u kernelSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + kernelSize + 7) >> 3;
    int globalThreads_y = dstDescPtr->h + kernelSize;
    int globalThreads_z = handle.GetBatchSize();

    uint padLength = kernelSize / 2;
    uint padLengthTwice = padLength * 2;
    uint2 tileSize;
    tileSize.x = (128 - padLengthTwice) / 8;
    tileSize.y = 16 - padLengthTwice;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = ((dstDescPtr->strides.hStride + kernelSize) / 3 + 7 ) >> 3;

        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(median_filter_3x3_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(median_filter_5x5_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(median_filter_7x7_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(median_filter_9x9_pkd_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        if (kernelSize == 3)
        {
            hipLaunchKernelGGL(median_filter_3x3_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 5)
        {
            hipLaunchKernelGGL(median_filter_5x5_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 7)
        {
            hipLaunchKernelGGL(median_filter_7x7_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
        else if (kernelSize == 9)
        {
            hipLaunchKernelGGL(median_filter_9x9_pln_hip_tensor,
                               dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               padLength,
                               tileSize,
                               roiTensorPtrSrc);
        }
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(median_filter_3x3_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(median_filter_5x5_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(median_filter_7x7_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(median_filter_9x9_pkd3_pln3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;

            if (kernelSize == 3)
            {
                hipLaunchKernelGGL(median_filter_3x3_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 5)
            {
                hipLaunchKernelGGL(median_filter_5x5_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 7)
            {
                hipLaunchKernelGGL(median_filter_7x7_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
            else if (kernelSize == 9)
            {
                hipLaunchKernelGGL(median_filter_9x9_pln3_pkd3_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/tileSize.x), ceil((float)globalThreads_y/tileSize.y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   padLength,
                                   tileSize,
                                   roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}
