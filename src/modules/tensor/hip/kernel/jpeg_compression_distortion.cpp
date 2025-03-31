/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_tensor_executors.hpp"
#include "rpp_hip_math.hpp"

// DCT Constants
__device__ const float dctA = 1.387039845322148f;
__device__ const float dctB = 1.306562964876377f;
__device__ const float dctC = 1.175875602419359f;
__device__ const float dctD = 0.785694958387102f;
__device__ const float dctE = 0.541196100146197f;
__device__ const float dctF = 0.275899379282943f;
__device__ const float normCoeff = 0.3535533905932737f;

// Computing Y from R G B
template <typename T>
__device__ inline void y_hip_compute(T *src , d_float8 *r_f8, d_float8 *g_f8, d_float8 *b_f8, d_float8 *y_f8)
{
    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value)
    {
        rpp_hip_math_multiply8_const(r_f8, r_f8, (float4)255.0f);
        rpp_hip_math_multiply8_const(g_f8, g_f8, (float4)255.0f);
        rpp_hip_math_multiply8_const(b_f8, b_f8, (float4)255.0f);
    }
    else if constexpr (std::is_same<T, schar>::value)
    {
        rpp_hip_math_add8_const(r_f8, r_f8, (float4)128.0f);
        rpp_hip_math_add8_const(g_f8, g_f8, (float4)128.0f);
        rpp_hip_math_add8_const(b_f8, b_f8, (float4)128.0f);
    }

    const float4 yR_f4 = (float4)0.299f;
    const float4 yG_f4 = (float4)0.587f;
    const float4 yB_f4 = (float4)0.114f;

    // RGB to Y conversion
    y_f8->f4[0] = r_f8->f4[0] * yR_f4 + g_f8->f4[0] * yG_f4 + b_f8->f4[0] * yB_f4;
    y_f8->f4[1] = r_f8->f4[1] * yR_f4 + g_f8->f4[1] * yG_f4 + b_f8->f4[1] * yB_f4;
}

// Downsampling and computing Cb and Cr
__device__ inline void downsample_cbcr_hip_compute(d_float8 *r1_f8, d_float8 *r2_f8, d_float8 *g1_f8, d_float8 *g2_f8, d_float8 *b1_f8, d_float8 *b2_f8, float4 *cb_f4, float4 *cr_f4)
{
    // Vertical downsampling
    d_float8 avgR_f8, avgG_f8, avgB_f8;
    avgR_f8.f4[0] = (r1_f8->f4[0] + r2_f8->f4[0]) * (float4)0.5f;
    avgR_f8.f4[1] = (r1_f8->f4[1] + r2_f8->f4[1]) * (float4)0.5f;
    avgG_f8.f4[0] = (g1_f8->f4[0] + g2_f8->f4[0]) * (float4)0.5f;
    avgG_f8.f4[1] = (g1_f8->f4[1] + g2_f8->f4[1]) * (float4)0.5f;
    avgB_f8.f4[0] = (b1_f8->f4[0] + b2_f8->f4[0]) * (float4)0.5f;
    avgB_f8.f4[1] = (b1_f8->f4[1] + b2_f8->f4[1]) * (float4)0.5f;

    // Horizontal downsampling
    float4 avgR_f4 = make_float4(
        (avgR_f8.f4[0].x + avgR_f8.f4[0].y) * 0.5f,
        (avgR_f8.f4[0].z + avgR_f8.f4[0].w) * 0.5f,
        (avgR_f8.f4[1].x + avgR_f8.f4[1].y) * 0.5f,
        (avgR_f8.f4[1].z + avgR_f8.f4[1].w) * 0.5f
    );

    float4 avgG_f4 = make_float4(
        (avgG_f8.f4[0].x + avgG_f8.f4[0].y) * 0.5f,
        (avgG_f8.f4[0].z + avgG_f8.f4[0].w) * 0.5f,
        (avgG_f8.f4[1].x + avgG_f8.f4[1].y) * 0.5f,
        (avgG_f8.f4[1].z + avgG_f8.f4[1].w) * 0.5f
    );

    float4 avgB_f4 = make_float4(
        (avgB_f8.f4[0].x + avgB_f8.f4[0].y) * 0.5f,
        (avgB_f8.f4[0].z + avgB_f8.f4[0].w) * 0.5f,
        (avgB_f8.f4[1].x + avgB_f8.f4[1].y) * 0.5f,
        (avgB_f8.f4[1].z + avgB_f8.f4[1].w) * 0.5f
    );

    // Convert to CbCr
    *cb_f4 = avgR_f4 * (float4)-0.168736f + avgG_f4 * (float4)-0.331264f + avgB_f4 * (float4)0.500000f + (float4)128.0f;
    *cr_f4 = avgR_f4 * (float4)0.500000f  + avgG_f4 * (float4)-0.418688f + avgB_f4 * (float4)-0.081312f + (float4)128.0f;
}

// DCT forward 1D implementation
__device__ inline void dct_fwd_8x8_1d(float *vec, bool sub_128)
{
    int val = -128.0f * sub_128;
    float inp[8];
    for(int i = 0; i < 8 ; i ++)
       inp[i] = vec[i] + val;

    float temp0 = inp[0] + inp[7];
    float temp1 = inp[1] + inp[6];
    float temp2 = inp[2] + inp[5];
    float temp3 = inp[3] + inp[4];
    float temp4 = inp[0] - inp[7];
    float temp5 = inp[6] - inp[1];
    float temp6 = inp[2] - inp[5];
    float temp7 = inp[4] - inp[3];

    float temp8 = temp0 + temp3;
    float temp9 = temp0 - temp3;
    float temp10 = temp1 + temp2;
    float temp11 = temp1 - temp2;

    vec[0] = normCoeff * (temp8 + temp10);
    vec[2] = normCoeff * (dctB * temp9 + dctE * temp11);
    vec[4] = normCoeff * (temp8 - temp10);
    vec[6] = normCoeff * (dctE * temp9 - dctB * temp11);

    vec[1] = normCoeff * (dctA * temp4 - dctC * temp5 + dctD * temp6 - dctF * temp7);
    vec[3] = normCoeff * (dctC * temp4 + dctF * temp5 - dctA * temp6 + dctD * temp7);
    vec[5] = normCoeff * (dctD * temp4 + dctA * temp5 + dctF * temp6 - dctC * temp7);
    vec[7] = normCoeff * (dctF * temp4 + dctD * temp5 + dctC * temp6 + dctA * temp7);
}

// Inverse 1D DCT
__device__ inline void dct_inv_8x8_1d(float *vec, bool add_128)
{
    int val = 128.0f * add_128;

    float inp[8];
    for(int i = 0; i < 8 ; i ++)
       inp[i] = vec[i] ;

    float temp0 =inp[0] +inp[4];
    float temp1 = dctB *inp[2] + dctE *inp[6];

    float temp2 = temp0 + temp1;
    float temp3 = temp0 - temp1;
    float temp4 = dctF *inp[7] + dctA *inp[1] + dctC *inp[3] + dctD *inp[5];
    float temp5 = dctA *inp[7] - dctF *inp[1] + dctD *inp[3] - dctC *inp[5];

    float temp6 =inp[0] -inp[4];
    float temp7 = dctE *inp[2] - dctB *inp[6];

    float temp8 = temp6 + temp7;
    float temp9 = temp6 - temp7;
    float temp10 = dctC *inp[1] - dctD *inp[7] - dctF *inp[3] - dctA *inp[5];
    float temp11 = dctD *inp[1] + dctC *inp[7] - dctA *inp[3] + dctF *inp[5];

    vec[0] = fmaf(normCoeff, (temp2 + temp4), val);
    vec[7] = fmaf(normCoeff, (temp2 - temp4), val);
    vec[4] = fmaf(normCoeff, (temp3 + temp5), val);
    vec[3] = fmaf(normCoeff, (temp3 - temp5), val);

    vec[1] = fmaf(normCoeff, (temp8 + temp10), val);
    vec[5] = fmaf(normCoeff, (temp9 - temp11), val);
    vec[2] = fmaf(normCoeff, (temp9 + temp11), val);
    vec[6] = fmaf(normCoeff, (temp8 - temp10), val);
}

// Quantization
__device__ inline void quantize(float* value, int* coeff)
{
    for (int i = 0; i < 8; i++)
        value[i] = coeff[i] * roundf(value[i] / (coeff[i]));
}

// Horizontal Upsampling and Color conversion to RGB
__device__ inline void upsample_and_RGB_hip_compute(float4 Cb_f4, float4 Cr_f4, d_float8 *Ch1_f8, d_float8* Ch2_f8, d_float8 *Ch3_f8)
{
    d_float8 Cb_f8, Cr_f8, R_f8 , G_f8, B_f8;
    // Copy Y values
    d_float8 y_f8 = *((d_float8*)Ch1_f8);

    // Subtract 128 from Cb_f4 and Cr_f4 before expanding
    Cb_f4 = Cb_f4 - (float4)128.0f;
    Cr_f4 = Cr_f4 - (float4)128.0f;

    // Expand each value
    Cb_f8.f4[0] = make_float4(Cb_f4.x, Cb_f4.x, Cb_f4.y, Cb_f4.y);
    Cb_f8.f4[1] = make_float4(Cb_f4.z, Cb_f4.z, Cb_f4.w, Cb_f4.w);
    Cr_f8.f4[0] = make_float4(Cr_f4.x, Cr_f4.x, Cr_f4.y, Cr_f4.y);
    Cr_f8.f4[1] = make_float4(Cr_f4.z, Cr_f4.z, Cr_f4.w, Cr_f4.w);

    // Now use the offset-adjusted values in the conversion
    R_f8.f4[0] = y_f8.f4[0] + (float4)1.402f * Cr_f8.f4[0];
    R_f8.f4[1] = y_f8.f4[1] + (float4)1.402f * Cr_f8.f4[1];

    G_f8.f4[0] = y_f8.f4[0] - (float4)0.344136285f * Cb_f8.f4[0] - (float4)0.714136285f * Cr_f8.f4[0];
    G_f8.f4[1] = y_f8.f4[1] - (float4)0.344136285f * Cb_f8.f4[1] - (float4)0.714136285f * Cr_f8.f4[1];

    B_f8.f4[0] = y_f8.f4[0] + (float4)1.772f * Cb_f8.f4[0];
    B_f8.f4[1] = y_f8.f4[1] + (float4)1.772f * Cb_f8.f4[1];

    // Write back the results
    *((d_float8*)Ch1_f8) = R_f8;
    *((d_float8*)Ch2_f8) = G_f8;
    *((d_float8*)Ch3_f8) = B_f8;
}

// Clamping based on the bitDepth
template <typename T>
__device__ inline void clamp_range(T *src, float* values)
{
    int low = 0, high = 255;
    if constexpr (std::is_same<T, schar>::value)
        low = -128, high = 127;
    else if constexpr (std::is_same<T, float>::value  || std::is_same<T, half>::value)
        low = 0, high = 1;

    for (int j = 0; j < 8; j++)
        values[j] = fminf(fmaxf(values[j], low), high);
}

// Generic clamping
__device__ inline void clamp_range(float* values)
{
    for (int j = 0; j < 8; j++)
        values[j] = fminf(fmaxf(values[j], 0), 255);
}

template <typename T>
__global__ void jpeg_compression_distortion_pkd3_hip_tensor(T *srcPtr,
                                                            uint2 srcStridesNH,
                                                            T *dstPtr,
                                                            uint2 dstStridesNH,
                                                            RpptROIPtr roiTensorPtrSrc,
                                                            int *tableY,
                                                            int *tableCbCr,
                                                            float qScale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth))
        return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    int srcIdx;
    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;

    // Check if we need special handling for image edges
    if (id_y < roiHeight)
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);
    else  // All out-of-bounds threads use the last valid row
        srcIdx = (id_z * srcStridesNH.x) + ((roiHeight - 1 + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);

    bool isEdge = ((id_x + 8) > roiWidth) && (id_x < alignedWidth);
    if (!isEdge)
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    else
    {
        int validPixels = roiWidth - id_x;
        // Load valid pixels (only if id_x is within valid range)
        if (validPixels > 0)
        {
            for (int i = 0; i < validPixels; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3)];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3) + 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3) + 2];
            }
        }
        // Pad 16 pixels by duplicating the last valid pixel
        for (int i = validPixels; i < min(validPixels + 16, 8); i++)
        {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx + ((validPixels - 1) * 3)];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + ((validPixels - 1) * 3) + 1];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx + ((validPixels - 1) * 3) + 2];
        }
    }
    __syncthreads();

    d_float8 y_f8;
    int cbcrY = hipThreadIdx_y * 2;
    y_hip_compute(srcPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8)
    {
        float4 cb_f4, cr_f4;
        // Downsample RGB and convert to CbCr
        downsample_cbcr_hip_compute((d_float8*)&src_smem[cbcrY][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 1][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 16][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 17][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 32][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 33][hipThreadIdx_x8],&cb_f4, &cr_f4);
        // Store Y and downsampled CbCr
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cb_f4;
        // Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cr_f4;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = y_f8;
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Doing -128 as part of DCT,
    // 1D row wise FWD DCT for Y Cb and Cr channels
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    // 1D column wise FWD DCT for Y Cb and Cr channels
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y;
    // Process all 128 columns
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0], false);
        dct_fwd_8x8_1d(&colVec[8], false);
        dct_fwd_8x8_1d(&colVec[16], false);
        dct_fwd_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &tableY[(hipThreadIdx_y % 8) * 8]);
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &tableCbCr[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();

    //----INVERSE STEPS---
    // 1D column wise IDCT for Y Cb and Cr channels
    if((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0], false);
        dct_inv_8x8_1d(&colVec[8], false);
        dct_inv_8x8_1d(&colVec[16], false);
        dct_inv_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // 1D row wise IDCT for Y Cb and Cr channels
    // Adding back 128 as part of INV DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Vertical Upsampling
    float4 cb_f4, cr_f4;
    cbcrY = hipThreadIdx_y/2;
    cb_f4 = *(float4*)&src_smem[cbcrY + 16][hipThreadIdx_x4];
    cr_f4 = *(float4*)&src_smem[cbcrY + 24][hipThreadIdx_x4];
    __syncthreads();

    // Convert back to RGB
    upsample_and_RGB_hip_compute(cb_f4, cr_f4, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    if((id_x < roiWidth) && (id_y < roiHeight))
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, src_smem_channel);
}

template <typename T>
__global__ void jpeg_compression_distortion_pln3_hip_tensor(T *srcPtr,
                                                            uint3 srcStridesNCH,
                                                            T *dstPtr,
                                                            uint3 dstStridesNCH,
                                                            RpptROIPtr roiTensorPtrSrc,
                                                            int *tableY,
                                                            int *tableCbCr,
                                                            float qScale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth))
        return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    uint3 srcIdx, dstIdx;
    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;

    // Determine actual source coordinates based on ROI bounds
    int srcY = (id_y < roiHeight) ? (id_y + roiY) : (roiHeight - 1 + roiY);

    int srcX = id_x + roiX;
    srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    // Horizontal loading with proper padding to 16-pixel multiples
    bool isEdge = (id_x + 8 > roiWidth);
    if (!isEdge)
    {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        // Edge case: Need to pad horizontally
        int validPixels = (id_x < roiWidth) ? (roiWidth - id_x) : 0;
        if (validPixels > 0)
        {
            // Load valid pixels for each channel separately
            for (int i = 0; i < validPixels; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + i];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + i];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + i];
            }
            // Pad remaining pixels in the block by duplicating the last valid pixel for each channel
            for (int i = validPixels; i < 8; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + validPixels - 1];
            }
        }
        else
        {
            // Completely outside ROI width but within aligned width - duplicate last pixel of the row
            int srcX = (roiWidth - 1) + roiX; // Last valid pixel in the row
            srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            srcIdx.y = srcIdx.x + srcStridesNCH.y;
            srcIdx.z = srcIdx.y + srcStridesNCH.y;
            // Duplicate the last valid pixel across the entire 8-pixel block
            for (int i = 0; i < 8; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z];
            }
        }
    }
    __syncthreads();

    d_float8 y_f8;
    int cbcrY = hipThreadIdx_y * 2;
    y_hip_compute(srcPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8)
    {
        float4 cb_f4, cr_f4;
        // Downsample RGB and convert to CbCr
        downsample_cbcr_hip_compute((d_float8*)&src_smem[cbcrY][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 1][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 16][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 17][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 32][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 33][hipThreadIdx_x8],&cb_f4, &cr_f4);
        // Store Y and downsampled CbCr
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cb_f4;
        // Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cr_f4;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = y_f8;
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Doing -128 as part of DCT,
    // 1D row wise FWD DCT for Y Cb and Cr channels
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    // 1D column wise FWD DCT for Y Cb and Cr channels
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y;
    // Process all 128 columns
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0],  false);
        dct_fwd_8x8_1d(&colVec[8],  false);
        dct_fwd_8x8_1d(&colVec[16], false);
        dct_fwd_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &tableY[(hipThreadIdx_y % 8) * 8]);
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &tableCbCr[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();

    //----INVERSE STEPS---
    // 1D column wise IDCT for Y Cb and Cr channels
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0],  false);
        dct_inv_8x8_1d(&colVec[8],  false);
        dct_inv_8x8_1d(&colVec[16], false);
        dct_inv_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // 1D row wise IDCT for Y Cb and Cr channels
    // Adding back 128 as part of INV DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Vertical Upsampling
    float4 cb_f4, cr_f4;
    cbcrY = hipThreadIdx_y/2;
    cb_f4 = *(float4*)&src_smem[cbcrY + 16][hipThreadIdx_x4];
    cr_f4 = *(float4*)&src_smem[cbcrY + 24][hipThreadIdx_x4];
    __syncthreads();

    // YCbCr to RGB
    upsample_and_RGB_hip_compute(cb_f4, cr_f4, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    if((id_x < roiWidth) && (id_y < roiHeight))
    {
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.x, (d_float8 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.y, (d_float8 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.z, (d_float8 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
}

template <typename T>
__global__ void jpeg_compression_distortion_pln1_hip_tensor(T *srcPtr,
                                                            uint3 srcStridesNCH,
                                                            T *dstPtr,
                                                            uint3 dstStridesNCH,
                                                            RpptROIPtr roiTensorPtrSrc,
                                                            int *tableY,
                                                            float qScale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth))
        return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[16][128];  // Assuming 16 rows (aligned height for 1 channel)
    d_float8 zeroes_f8 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    *(d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] = zeroes_f8;
    __syncthreads();

    uint srcIdx, dstIdx;

    // Calculate destination indices first
    dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // Determine actual source coordinates based on ROI bounds
    int srcY = (id_y < roiHeight) ? (id_y + roiY) : (roiHeight - 1 + roiY);

    int srcX = id_x + roiX;
    srcIdx = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;

    // Horizontal loading with proper padding to 16-pixel multiples
    bool isEdge = (id_x + 8 > roiWidth);
    if (!isEdge)
    {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else
    {
        // Edge case: Need to pad horizontally
        int validPixels = (id_x < roiWidth) ? (roiWidth - id_x) : 0;

        if (validPixels > 0)
        {
            // Load valid pixels for each channel separately
            for (int i = 0; i < validPixels; i++)
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + i];

            // Pad remaining pixels in the block by duplicating the last valid pixel for each channel
            for (int i = validPixels; i < 8; i++)
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + validPixels - 1];
        }
        else
        {
            // Completely outside ROI width but within aligned width - duplicate last pixel of the row
            int srcX = (roiWidth - 1) + roiX; // Last valid pixel in the row
            srcIdx = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            // Duplicate the last valid pixel across the entire 8-pixel block
            for (int i = 0; i < 8; i++)
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx];
        }
    }
    __syncthreads();

    // Doing -128 as part of DCT,
    // 1D row wise DCT
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], true);
    __syncthreads();

    // Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y;
    // Process all 128 columns
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[16];

        for (int i = 0; i < 16; i++)
            colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0], false);
        dct_fwd_8x8_1d(&colVec[8], false);

        for (int i = 0; i < 16; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Quantization
    quantize(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &tableY[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();

    //----INVERSE STEPS---
    // Process all 128 columns
    // Adding back 128 as part of DCT
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[16];

        for (int i = 0; i < 16; i++)
            colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0], false);
        dct_inv_8x8_1d(&colVec[8], false);

        for (int i = 0; i < 16; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], true);
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    __syncthreads();

    if((id_x < roiWidth) && (id_y < roiHeight))
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, (d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
}

template <typename T>
__global__ void jpeg_compression_distortion_pkd3_pln3_hip_tensor( T *srcPtr,
                                                                  uint2 srcStridesNH,
                                                                  T *dstPtr,
                                                                  uint3 dstStridesNCH,
                                                                  RpptROIPtr roiTensorPtrSrc,
                                                                  int *tableY,
                                                                  int *tableCbCr,
                                                                  float qScale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth))
        return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    int srcIdx;
    uint3 dstIdx;
    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;

    // Check if we need special handling for image edges
    if (id_y < roiHeight)
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);
    else  // All out-of-bounds threads use the last valid row
        srcIdx = (id_z * srcStridesNH.x) + ((roiHeight - 1 + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);

    bool isEdge = ((id_x + 8) > roiWidth) && (id_x < alignedWidth);
    if (!isEdge)
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    else
    {
        int validPixels = roiWidth - id_x;
        // Load valid pixels (only if id_x is within valid range)
        if (validPixels > 0)
        {
            for (int i = 0, idx = srcIdx; i < validPixels; i++, idx += 3)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[idx];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[idx + 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[idx + 2];
            }
        }
        // Pad 16 pixels by duplicating the last valid pixel
        int lastValidIdx = srcIdx + ((validPixels - 1) * 3);
        for (int i = validPixels; i < min(validPixels + 16, 8); i++)
        {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx + 1];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[lastValidIdx + 2];
        }
    }
    __syncthreads();
    d_float8 y_f8;
    int cbcrY = hipThreadIdx_y * 2;
    y_hip_compute(srcPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8)
    {
        float4 cb_f4, cr_f4;
        // Downsample RGB and convert to CbCr
        downsample_cbcr_hip_compute((d_float8*)&src_smem[cbcrY][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 1][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 16][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 17][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 32][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 33][hipThreadIdx_x8],&cb_f4, &cr_f4);
        // Store Y and downsampled CbCr
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cb_f4;
        // Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cr_f4;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = y_f8;
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Doing -128 as part of DCT,
    // 1D row wise FWD DCT for Y Cb and Cr channels
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    // 1D column wise FWD DCT for Y Cb and Cr channels
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y;
    // Process all 128 columns
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0],  false);
        dct_fwd_8x8_1d(&colVec[8],  false);
        dct_fwd_8x8_1d(&colVec[16], false);
        dct_fwd_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &tableY[(hipThreadIdx_y % 8) * 8]);
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &tableCbCr[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();

    //----INVERSE STEPS---
    // 1D column wise IDCT for Y Cb and Cr channels
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0],  false);
        dct_inv_8x8_1d(&colVec[8],  false);
        dct_inv_8x8_1d(&colVec[16], false);
        dct_inv_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // 1D row wise IDCT for Y Cb and Cr channels
    // Adding back 128 as part of INV DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Vertical Upsampling
    float4 cb_f4, cr_f4;
    cbcrY = hipThreadIdx_y/2;
    cb_f4 = *(float4*)&src_smem[cbcrY + 16][hipThreadIdx_x4];
    cr_f4 = *(float4*)&src_smem[cbcrY + 24][hipThreadIdx_x4];
    __syncthreads();

    // YCbCr to RGB
    upsample_and_RGB_hip_compute(cb_f4, cr_f4, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    if((id_x < roiWidth) && (id_y < roiHeight))
    {
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.x, (d_float8 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.y, (d_float8 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.z, (d_float8 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
}

template <typename T>
__global__ void jpeg_compression_distortion_pln3_pkd3_hip_tensor( T *srcPtr,
                                                                  uint3 srcStridesNCH,
                                                                  T *dstPtr,
                                                                  uint2 dstStridesNH,
                                                                  RpptROIPtr roiTensorPtrSrc,
                                                                  int *tableY,
                                                                  int *tableCbCr,
                                                                  float qScale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = (roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) & ~15;
    int alignedHeight = (roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) & ~15;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth))
        return;

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[48][128];  // Assuming 48 rows (aligned height for 3 channels)
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    uint3 srcIdx;

    // Check if we need special handling for image edges
    int id_y_clamped;
    id_y_clamped = id_y < roiHeight ? id_y : roiHeight - 1;

    srcIdx.x = (id_z * srcStridesNCH.x) +((id_y_clamped + roiY) * srcStridesNCH.z) +(id_x + roiX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;

    // Determine actual source coordinates based on ROI bounds
    int srcY = (id_y < roiHeight) ? (id_y + roiY) : (roiHeight - 1 + roiY);

    // Horizontal loading with proper padding to 16-pixel multiples
    bool isEdge = (id_x + 8 > roiWidth);
    if (!isEdge)
    {
        // Normal case: The entire 8-pixel block fits within the ROI
        int srcX = id_x + roiX;
        srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
        srcIdx.y = srcIdx.x + srcStridesNCH.y;
        srcIdx.z = srcIdx.y + srcStridesNCH.y;

        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    }
    else
    {
        // Edge case: Need to pad horizontally
        int validPixels = (id_x < roiWidth) ? (roiWidth - id_x) : 0;

        if (validPixels > 0)
        {
            // Load valid pixels and pad the rest
            int srcX = id_x + roiX;
            srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            srcIdx.y = srcIdx.x + srcStridesNCH.y;
            srcIdx.z = srcIdx.y + srcStridesNCH.y;

            // Load valid pixels for each channel separately
            for (int i = 0; i < validPixels; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + i];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + i];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + i];
            }
            // Pad remaining pixels in the block by duplicating the last valid pixel for each channel
            for (int i = validPixels; i < 8; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + validPixels - 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + validPixels - 1];
            }
        }
        else
        {
            // Completely outside ROI width but within aligned width - duplicate last pixel of the row
            int srcX = (roiWidth - 1) + roiX; // Last valid pixel in the row
            srcIdx.x = (id_z * srcStridesNCH.x) + (srcY * srcStridesNCH.z) + srcX;
            srcIdx.y = srcIdx.x + srcStridesNCH.y;
            srcIdx.z = srcIdx.y + srcStridesNCH.y;
            // Duplicate the last valid pixel across the entire 8-pixel block
            for (int i = 0; i < 8; i++)
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z];
            }
        }
    }
    __syncthreads();

    d_float8 y_f8;
    int cbcrY = hipThreadIdx_y * 2;
    y_hip_compute(srcPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8)
    {
        float4 cb_f4, cr_f4;
        // Downsample RGB and convert to CbCr
        downsample_cbcr_hip_compute((d_float8*)&src_smem[cbcrY][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 1][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 16][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 17][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 32][hipThreadIdx_x8], (d_float8*)&src_smem[cbcrY + 33][hipThreadIdx_x8],&cb_f4, &cr_f4);
        // Store Y and downsampled CbCr
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cb_f4;
        // Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = cr_f4;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = y_f8;
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Doing -128 as part of DCT,
    // 1D row wise FWD DCT for Y Cb and Cr channels
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    // 1D column wise FWD DCT for Y Cb and Cr channels
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y;
    // Process all 128 columns
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&colVec[0],  false);
        dct_fwd_8x8_1d(&colVec[8],  false);
        dct_fwd_8x8_1d(&colVec[16], false);
        dct_fwd_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &tableY[(hipThreadIdx_y % 8) * 8]);
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &tableCbCr[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();

    //----INVERSE STEPS---
    // 1D column wise IDCT for Y Cb and Cr channels
    if ((col < 128) && (col < alignedWidth))
    {
        // Load column into temporary array
        float colVec[32];

        for (int i = 0; i < 32; i++)
            colVec[i] = src_smem[i][col];

        dct_inv_8x8_1d(&colVec[0],  false);
        dct_inv_8x8_1d(&colVec[8],  false);
        dct_inv_8x8_1d(&colVec[16], false);
        dct_inv_8x8_1d(&colVec[24], false);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = colVec[i];
    }
    __syncthreads();

    // 1D row wise IDCT for Y Cb and Cr channels
    // Adding back 128 as part of INV DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], true);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], true);
    __syncthreads();

    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    __syncthreads();

    // Vertical Upsampling
    float4 cb_f4, cr_f4;
    cbcrY = hipThreadIdx_y/2;
    cb_f4 = *(float4*)&src_smem[cbcrY + 16][hipThreadIdx_x4];
    cr_f4 = *(float4*)&src_smem[cbcrY + 24][hipThreadIdx_x4];
    __syncthreads();

    // YCbCr to RGB
    upsample_and_RGB_hip_compute(cb_f4, cr_f4, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    clamp_range(srcPtr, (float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    if((id_x < roiWidth) && (id_y < roiHeight))
        rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, src_smem_channel);
}

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion(T *srcPtr,
                                               RpptDescPtr srcDescPtr,
                                               T *dstPtr,
                                               RpptDescPtr dstDescPtr,
                                               RpptROIPtr roiTensorPtrSrc,
                                               RpptRoiType roiType,
                                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    int quality = 50;
    quality = std::clamp<int>(quality, 1, 100);
    float qScale = (quality < 50) ? (50.0f / quality) : (2.0f - (2 * quality / 100.0f));
    // Allocate pinned memory
    Rpp32s *tableY = reinterpret_cast<Rpp32s *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    Rpp32s *tableCbCr = tableY + 64;

    // Initialize and modify the tables
    int tableYInit[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };

    int tableCbCrInit[64] = {
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99
    };

    // Populate pinned memory with scaled and clamped values
    for (int i = 0; i < 64; i++)
    {
        tableY[i] = std::max<uint8_t>(static_cast<uint8_t>(std::clamp(qScale * tableYInit[i], 0.0f, 255.0f)), 1);
        tableCbCr[i] = std::max<uint8_t>(static_cast<uint8_t>(std::clamp(qScale * tableCbCrInit[i], 0.0f, 255.0f)), 1);
    }

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(jpeg_compression_distortion_pkd3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           tableY,
                           tableCbCr,
                           qScale);
    }

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && (srcDescPtr->c == 3))
    {
        hipLaunchKernelGGL(jpeg_compression_distortion_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/16), ceil((float)globalThreads_y/16), ceil((float)globalThreads_z/1)),
                           dim3(16, 16, 1),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           tableY,
                           tableCbCr,
                           qScale);
    }

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && (srcDescPtr->c == 1))
    {
        hipLaunchKernelGGL(jpeg_compression_distortion_pln1_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           tableY,
                           qScale);
    }

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(jpeg_compression_distortion_pkd3_pln3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           tableY,
                           tableCbCr,
                           qScale);
    }

    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(jpeg_compression_distortion_pln3_pkd3_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc,
                           tableY,
                           tableCbCr,
                           qScale);
    }
    return RPP_SUCCESS;
}

template RppStatus hip_exec_jpeg_compression_distortion<Rpp8u>(Rpp8u*,
                                                               RpptDescPtr,
                                                               Rpp8u*,
                                                               RpptDescPtr,
                                                               RpptROIPtr,
                                                               RpptRoiType,
                                                               rpp::Handle&);

template RppStatus hip_exec_jpeg_compression_distortion<Rpp32f>(Rpp32f*,
                                                                RpptDescPtr,
                                                                Rpp32f*,
                                                                RpptDescPtr,
                                                                RpptROIPtr,
                                                                RpptRoiType,
                                                                rpp::Handle&);

template RppStatus hip_exec_jpeg_compression_distortion<half>(half*,
                                                              RpptDescPtr,
                                                              half*,
                                                              RpptDescPtr,
                                                              RpptROIPtr,
                                                              RpptRoiType,
                                                              rpp::Handle&);

template RppStatus hip_exec_jpeg_compression_distortion<Rpp8s>(Rpp8s*,
                                                               RpptDescPtr,
                                                               Rpp8s*,
                                                               RpptDescPtr,
                                                               RpptROIPtr,
                                                               RpptRoiType,
                                                               rpp::Handle&);