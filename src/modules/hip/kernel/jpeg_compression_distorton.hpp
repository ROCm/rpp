#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// DCT Constants
__device__ const float dctA = 1.387039845322148f;        
__device__ const float dctB = 1.306562964876377f;        
__device__ const float dctC = 1.175875602419359f;        
__device__ const float dctD = 0.785694958387102f;        
__device__ const float dctE = 0.541196100146197f;        
__device__ const float dctF = 0.275899379282943f;        
__device__ const float normCoeff = 0.3535533905932737f; 

__device__ inline float clamp_float(float value, float minVal, float maxVal) {
    return fmaxf(minVal, fminf(value, maxVal));
}

// Computing Y from R G B
template <typename T>
__device__ inline void y_hip_compute(T *src , d_float8 *r_f8, d_float8 *g_f8, d_float8 *b_f8, d_float8 *y_f8)
{
    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value) {
        rpp_hip_math_multiply8_const(r_f8, r_f8, (float4)255.0f);
        rpp_hip_math_multiply8_const(g_f8, g_f8, (float4)255.0f);
        rpp_hip_math_multiply8_const(b_f8, b_f8, (float4)255.0f);
    }
    else if constexpr (std::is_same<T, schar>::value) {
        rpp_hip_math_add8_const(r_f8, r_f8, (float4)128.0f);
        rpp_hip_math_add8_const(g_f8, g_f8, (float4)128.0f);
        rpp_hip_math_add8_const(b_f8, b_f8, (float4)128.0f);
    }
    
    const float4 yR_f4 = (float4)0.299f;
    const float4 yG_f4 = (float4)0.587f;
    const float4 yB_f4 = (float4)0.114f;

    //  RGB to Y conversion
    y_f8->f4[0] = r_f8->f4[0] * yR_f4 + g_f8->f4[0] * yG_f4 + b_f8->f4[0] * yB_f4;
    y_f8->f4[1] = r_f8->f4[1] * yR_f4 + g_f8->f4[1] * yG_f4 + b_f8->f4[1] * yB_f4;
}
//Downsampling and computing Cb and Cr
__device__ inline void downsample_cbcr_hip_compute(d_float8 *r1_f8, d_float8 *r2_f8,d_float8 *g1_f8, d_float8 *g2_f8,d_float8 *b1_f8, d_float8 *b2_f8, float4 *cb_f4, float4 *cr_f4)
{   
    // Vertical downsampling
    d_float8 avgR_f8, avgG_f8, avgB_f8;
    avgR_f8.f4[0] = (r1_f8->f4[0] + r2_f8->f4[0]) * (float4)0.5f;
    avgR_f8.f4[1] = (r1_f8->f4[1] + r2_f8->f4[1]) * (float4)0.5f;
    avgG_f8.f4[0] = (g1_f8->f4[0] + g2_f8->f4[0]) * (float4)0.5f;
    avgG_f8.f4[1] = (g1_f8->f4[1] + g2_f8->f4[1]) * (float4)0.5f;
    avgB_f8.f4[0] = (b1_f8->f4[0] + b2_f8->f4[0]) * (float4)0.5f;
    avgB_f8.f4[1] = (b1_f8->f4[1] + b2_f8->f4[1]) * (float4)0.5f;
    
    // Horizontal downsampling with clamping
    float4 avgR_f4 = make_float4(
        clamp_float((avgR_f8.f4[0].x + avgR_f8.f4[0].y) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgR_f8.f4[0].z + avgR_f8.f4[0].w) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgR_f8.f4[1].x + avgR_f8.f4[1].y) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgR_f8.f4[1].z + avgR_f8.f4[1].w) * 0.5f, 0.0f, 255.0f)
    );
    float4 avgG_f4 = make_float4(
        clamp_float((avgG_f8.f4[0].x + avgG_f8.f4[0].y) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgG_f8.f4[0].z + avgG_f8.f4[0].w) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgG_f8.f4[1].x + avgG_f8.f4[1].y) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgG_f8.f4[1].z + avgG_f8.f4[1].w) * 0.5f, 0.0f, 255.0f)
    );
    float4 avgB_f4 = make_float4(
        clamp_float((avgB_f8.f4[0].x + avgB_f8.f4[0].y) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgB_f8.f4[0].z + avgB_f8.f4[0].w) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgB_f8.f4[1].x + avgB_f8.f4[1].y) * 0.5f, 0.0f, 255.0f),
        clamp_float((avgB_f8.f4[1].z + avgB_f8.f4[1].w) * 0.5f, 0.0f, 255.0f)
    );
    
    // Convert to CbCr 
    *cb_f4 = avgR_f4 * (float4)-0.168736f + avgG_f4 * (float4)-0.331264f + avgB_f4 * (float4)0.500000f + (float4)128.0f;
    *cr_f4 = avgR_f4 * (float4)0.500000f  + avgG_f4 * (float4)-0.418688f + avgB_f4 * (float4)-0.081312f + (float4)128.0f;
}
// DCT forward 1D implementation
__device__ inline void dct_fwd_8x8_1d(float *vecf8, int stride,int rowcol, bool sub_128) 
{
    int val = -128 * sub_128;
    float inp[8];
    for(int i = 0; i < 8 ; i ++)
       inp[i] = vecf8[i] + val;

    float temp0 =inp[0] +inp[7];
    float temp1 =inp[1] +inp[6];
    float temp2 =inp[2] +inp[5];
    float temp3 =inp[3] +inp[4];
    float temp4 =inp[0] -inp[7];
    float temp5 =inp[6] -inp[1];
    float temp6 =inp[2] -inp[5];
    float temp7 =inp[4] -inp[3];

    float temp8 = temp0 + temp3;
    float temp9 = temp0 - temp3;
    float temp10 = temp1 + temp2;
    float temp11 = temp1 - temp2;

    inp[0] = normCoeff * (temp8 + temp10);
    inp[2] = normCoeff * (dctB * temp9 + dctE * temp11);
    inp[4] = normCoeff * (temp8 - temp10);
    inp[6] = normCoeff * (dctE * temp9 - dctB * temp11);
 
    inp[1] = normCoeff * (dctA * temp4 - dctC * temp5 + dctD * temp6 - dctF * temp7);
    inp[3] = normCoeff * (dctC * temp4 + dctF * temp5 - dctA * temp6 + dctD * temp7);
    inp[5] = normCoeff * (dctD * temp4 + dctA * temp5 + dctF * temp6 - dctC * temp7);
    inp[7] = normCoeff * (dctF * temp4 + dctD * temp5 + dctC * temp6 + dctA * temp7);

    for(int i = 0; i < 8 ; i ++)
        vecf8[i] =inp[i];
}
//Inverse 1D DCT
__device__ inline void dct_inv_8x8_1d(float *vecf8, int stride,int rowcol, bool add_128) 
{
    int val = 128 * add_128;

    float inp[8];
    for(int i = 0; i < 8 ; i ++)
       inp[i] = vecf8[i] ;

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

    vecf8[0] = fmaf(normCoeff, (temp2 + temp4), val);
    vecf8[7] = fmaf(normCoeff, (temp2 - temp4), val);
    vecf8[4] = fmaf(normCoeff, (temp3 + temp5), val);
    vecf8[3] = fmaf(normCoeff, (temp3 - temp5), val);

    vecf8[1] = fmaf(normCoeff, (temp8 + temp10), val);
    vecf8[5] = fmaf(normCoeff, (temp9 - temp11), val);
    vecf8[2] = fmaf(normCoeff, (temp9 + temp11), val);
    vecf8[6] = fmaf(normCoeff, (temp8 - temp10), val);
}
//Quantization
__device__ inline void quantize(float* value, int* coeff) {
    for (int i = 0; i < 8; i++) {
        value[i] = coeff[i] * roundf(value[i] * __frcp_rn(coeff[i]));
    }
}
// Horizontal Upsampling and Color conversion to RGB
__device__ inline void Upsample_and_RGB_hip_compute(float4 Cb, float4 Cr, d_float8 *Ch1, d_float8* Ch2, d_float8 *Ch3)
{   
    d_float8 Cb_f8, Cr_f8, R_f8 , G_f8, B_f8;
    // Copy Y values
    d_float8 Y_f8 = *((d_float8*)Ch1);
    
    // Subtract 128 from Cb and Cr before expanding
    Cb = Cb - (float4)128.0f;
    Cr = Cr - (float4)128.0f;
    
    // Expand each value 
    Cb_f8.f4[0] = make_float4(Cb.x, Cb.x, Cb.y, Cb.y);
    Cb_f8.f4[1] = make_float4(Cb.z, Cb.z, Cb.w, Cb.w);
    Cr_f8.f4[0] = make_float4(Cr.x, Cr.x, Cr.y, Cr.y);
    Cr_f8.f4[1] = make_float4(Cr.z, Cr.z, Cr.w, Cr.w);
    
    // Now use the offset-adjusted values in the conversion
    R_f8.f4[0] = Y_f8.f4[0] + (float4)1.402f * Cr_f8.f4[0];
    R_f8.f4[1] = Y_f8.f4[1] + (float4)1.402f * Cr_f8.f4[1];
    
    G_f8.f4[0] = Y_f8.f4[0] - (float4)0.344136285f * Cb_f8.f4[0] - (float4)0.714136285f * Cr_f8.f4[0];
    G_f8.f4[1] = Y_f8.f4[1] - (float4)0.344136285f * Cb_f8.f4[1] - (float4)0.714136285f * Cr_f8.f4[1];
    
    B_f8.f4[0] = Y_f8.f4[0] + (float4)1.772f * Cb_f8.f4[0];
    B_f8.f4[1] = Y_f8.f4[1] + (float4)1.772f * Cb_f8.f4[1];
    
    // Write back the results
    *((d_float8*)Ch1) = R_f8;
    *((d_float8*)Ch2) = G_f8;
    *((d_float8*)Ch3) = B_f8;
}
template <typename T>
__device__ inline void clamp_range(T *src, float* values, int num_elements = 8) {
    int low,high;
    if constexpr (std::is_same<T, uchar>::value)
        low = 0, high = 255;
    else if constexpr (std::is_same<T, schar>::value)
        low = -128, high = 127;
    else if constexpr (std::is_same<T, float>::value  || std::is_same<T, half>::value)
        low = 0, high = 1;

    for (int j = 0; j < num_elements; j++) {
        values[j] = fminf(fmaxf(values[j], low), high);
    }
}

__device__ inline void ceil_values(float* values) {
    for (int j = 0; j < 8; j++) {
        values[j] = ceil(values[j]);
    }
}

__device__ void clamp_coordinates(int &x, int &y, int width, int height) {
    x = min(max(x, 0), width - 1);
    y = min(max(y, 0), height - 1);
}

template <typename T>
__device__ inline void clamp_range(T *dst, float* values, float lower_limit, float upper_limit, int par) {
    if (std::is_same<T, schar>::value) {
        if (par == -1) {
            lower_limit = -128;
            upper_limit = 127;
        }
    }
    for (int j = 0; j < 8; j++) {
        values[j] = fminf(fmaxf(values[j], lower_limit), upper_limit);
    }
}
//PKD3 to PKD3
template <typename T>
__global__ void jpeg_compression_distortion_pkd3_hip_tensor( T *srcPtr,
                                                             uint2 srcStridesNH,
                                                             T *dstPtr,
                                                             uint2 dstStridesNH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             int *Ytable,
                                                             int *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) / 16) * 16;
    int alignedHeight = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) / 16) * 16;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) {
        return;
    }

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

    bool isEdge = ((id_x + 8) > roiWidth) && (id_x < roiWidth);
    if (!isEdge) 
    {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    } 
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
    // Downsample Cb and Cr channels
    d_float8 Y_f8;
    int CbCry = hipThreadIdx_y * 2;
    float4 Cb, Cr;
    y_hip_compute(srcPtr,(d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&Y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8) {  
    // Downsample RGB and convert to CbCr
    downsample_cbcr_hip_compute((d_float8*)&src_smem[CbCry][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 1][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 16][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 17][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 32][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 33][hipThreadIdx_x8],&Cb, &Cr);
    
    // Store Y and downsampled CbCr.. Cr just below Cb for continutiy 
    *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
    //Storing Cr below Cb (8 x 64)
    *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = Y_f8;
    __syncthreads();

    // Doing -128 as part of DCT,
    //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&col_vec[0], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();

    // 1D row wise DCT for Y Cb and Cr channela
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 

    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);

    //----INVERSE STEPS---
    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 
    __syncthreads();
    

    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];
        
        dct_inv_8x8_1d(&col_vec[0], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[8], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[16], 1, col, true);
        dct_inv_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) 
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();
    
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] , 0.0f, 255.0f, 0);
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] , 0.0f, 255.0f,0);
    __syncthreads();
        
    // Vertical Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];
    __syncthreads();
   
    // Convert back to RGB
    Upsample_and_RGB_hip_compute(Cb,Cr,(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();
    
    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);  
    __syncthreads(); 
  
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx,src_smem_channel);
}

//PLN3 to PLN3
template <typename T>
__global__ void jpeg_compression_distortion_pln3_hip_tensor( T *srcPtr,
                                                             uint3 srcStridesNCH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             int *Ytable,
                                                             int *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) / 16) * 16;
    int alignedHeight = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) / 16) * 16;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) {
        return;
    }

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

    int3 srcIdx;
    int3 dstIdx;

    // Check if we need special handling for image edges
    int id_y_clamped;
    id_y_clamped = id_y < roiHeight ? id_y : roiHeight - 1;
    
    srcIdx.x = (id_z * srcStridesNCH.x) +((id_y_clamped + roiY) * srcStridesNCH.z) +(id_x + roiX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;

    // Loading data into shared memory each channel individually 
    bool is_edge = ((id_x + 8) > roiWidth) && (id_x < roiWidth);
    if (!is_edge) 
    {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } 
    else 
    {
        int validPixels = roiWidth - id_x;

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
            // If we're completely outside the ROI, pad with the last valid pixel
            for (int i = 0; i < 8; i++) 
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x - 1];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y - 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z - 1];
            }
        }
    }
    __syncthreads();

    d_float8 Y_f8;
    int CbCry = hipThreadIdx_y * 2;
    float4 Cb, Cr;
    y_hip_compute(srcPtr,(d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&Y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8) 
    {  
        // Downsample RGB and convert to CbCr
        downsample_cbcr_hip_compute((d_float8*)&src_smem[CbCry][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 1][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 16][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 17][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 32][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 33][hipThreadIdx_x8],&Cb, &Cr);
        
        // Store Y and downsampled CbCr.. Cr just below Cb for continutiy 
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
        //Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = Y_f8;
    __syncthreads();

    // Doing -128 as part of DCT,
    //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&col_vec[0], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();

    // 1D row wise DCT for Y Cb and Cr channela
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 

    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);

    //----INVERSE STEPS---
    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 
    __syncthreads();
    
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];
        
        dct_inv_8x8_1d(&col_vec[0], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[8], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[16], 1, col, true);
        dct_inv_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) 
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();
    
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] , 0.0f, 255.0f, 0);
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] , 0.0f, 255.0f,0);
    __syncthreads();
        
    // Vertical Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];
    __syncthreads();
   
    // Convert back to RGB
    Upsample_and_RGB_hip_compute(Cb,Cr,(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();
    
    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads(); 

    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);  
    __syncthreads(); 
  
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.x, (d_float8 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.y, (d_float8 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.z, (d_float8 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
}

//PLN1 to PLN1
template <typename T>
__global__ void jpeg_compression_distortion_pln1_hip_tensor( T *srcPtr,
                                                             uint3 srcStridesNCH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             int *Ytable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) / 16) * 16;
    int alignedHeight = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) / 16) * 16;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) {
        return;
    }

    // ROI parameters
    int roiX = roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int roiY = roiTensorPtrSrc[id_z].xywhROI.xy.y;
    int roiWidth = roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    int roiHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight;

    // Shared memory declaration
    __shared__ float src_smem[16][128];  // Assuming 48 rows (aligned height for 3 channels)

    float *src_smem_channel = &src_smem[hipThreadIdx_y][hipThreadIdx_x8];

    // Check if we need special handling for image edges
    int id_y_clamped;
    id_y_clamped = id_y < roiHeight ? id_y : roiHeight - 1;
    
    int srcIdx = (id_z * srcStridesNCH.x) +((id_y_clamped + roiY) * srcStridesNCH.z) +(id_x + roiX);
    int dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    // Loading data into shared memory each channel individually 
    bool is_edge = ((id_x + 8) > roiWidth) && (id_x < roiWidth);
    if (!is_edge) 
    {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    }
    else 
    {
        int validPixels = roiWidth - id_x;

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
            // If we're completely outside the ROI, pad with the last valid pixel
            for (int i = 0; i < 8; i++) 
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx - 1];
        }
    }
    __syncthreads();

    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value){
        rpp_hip_math_multiply8_const((d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (float4)255);
    }
    else if constexpr (std::is_same<T, schar>::value)
    {
        rpp_hip_math_add8_const((d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (float4)128);
    }
    __syncthreads();

    // Doing -128 as part of DCT,
    //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[16];
        
        for (int i = 0; i < 16; i++) 
            col_vec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&col_vec[0], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col,  true);

        for (int i = 0; i < 16; i++)
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();

    // 1D row wise DCT for Y Cb and Cr channela
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);

    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);  

    //----INVERSE STEPS---
    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    __syncthreads();
    

    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[16];
        
        for (int i = 0; i < 16; i++) 
            col_vec[i] = src_smem[i][col];
        
        dct_inv_8x8_1d(&col_vec[0], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[8], 1, col,  true);

        for (int i = 0; i < 16; i++) 
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();
    
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8] , 0.0f, 255.0f, 0);
    __syncthreads();
    
    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  
    __syncthreads(); 
  
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, (d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
}

//PKD3 to PLN3
template <typename T>
__global__ void jpeg_compression_distortion_pkd3_pln3_hip_tensor( T *srcPtr,
                                                             uint2 srcStridesNH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             int *Ytable,
                                                             int *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) / 16) * 16;
    int alignedHeight = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) / 16) * 16;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) {
        return;
    }

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
    int3 dstIdx;
    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;

    // Check if we need special handling for image edges
    if (id_y < roiHeight) 
        srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);
    else  // All out-of-bounds threads use the last valid row
        srcIdx = (id_z * srcStridesNH.x) + ((roiHeight - 1 + roiY) * srcStridesNH.y) + ((id_x + roiX) * 3);

    bool isEdge = ((id_x + 8) > roiWidth) && (id_x < roiWidth);
    if (!isEdge) 
    {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    } 
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
    // Downsample Cb and Cr channels
    d_float8 Y_f8;
    int CbCry = hipThreadIdx_y * 2;
    float4 Cb, Cr;
    y_hip_compute(srcPtr,(d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&Y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8) {  
    // Downsample RGB and convert to CbCr
    downsample_cbcr_hip_compute((d_float8*)&src_smem[CbCry][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 1][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 16][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 17][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 32][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 33][hipThreadIdx_x8],&Cb, &Cr);
    
    // Store Y and downsampled CbCr.. Cr just below Cb for continutiy 
    *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
    //Storing Cr below Cb (8 x 64)
    *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = Y_f8;
    __syncthreads();

    // Doing -128 as part of DCT,
    //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&col_vec[0], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();

    // 1D row wise DCT for Y Cb and Cr channela
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 

    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);

    //----INVERSE STEPS---
    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 
    __syncthreads();
    

    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];
        
        dct_inv_8x8_1d(&col_vec[0], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[8], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[16], 1, col, true);
        dct_inv_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) 
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();
    
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] , 0.0f, 255.0f, 0);
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] , 0.0f, 255.0f,0);
    __syncthreads();
        
    // Vertical Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];
    __syncthreads();
   
    // Convert back to RGB
    Upsample_and_RGB_hip_compute(Cb,Cr,(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();
    
    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);  
    __syncthreads(); 
  
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.x, (d_float8 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.y, (d_float8 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx.z, (d_float8 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
}

//PLN3 to PKD3
template <typename T>
__global__ void jpeg_compression_distortion_pln3_pkd3_hip_tensor( T *srcPtr,
                                                             uint3 srcStridesNCH,
                                                             T *dstPtr,
                                                             uint2 dstStridesNH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             int *Ytable,
                                                             int *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;

    int alignedWidth = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 15) / 16) * 16;
    int alignedHeight = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 15) / 16) * 16;

    // Boundary checks
    if ((id_y >= alignedHeight) || (id_x >= alignedWidth)) {
        return;
    }

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

    int3 srcIdx;

    // Check if we need special handling for image edges
    int id_y_clamped;
    id_y_clamped = id_y < roiHeight ? id_y : roiHeight - 1;
    
    srcIdx.x = (id_z * srcStridesNCH.x) +((id_y_clamped + roiY) * srcStridesNCH.z) +(id_x + roiX);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;

    // Loading data into shared memory each channel individually 
    bool is_edge = ((id_x + 8) > roiWidth) && (id_x < roiWidth);
    if (!is_edge) 
    {
        // Load 8 pixels at once for each channel
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.x, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.y, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.z, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } 
    else 
    {
        int validPixels = roiWidth - id_x;

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
            // If we're completely outside the ROI, pad with the last valid pixel
            for (int i = 0; i < 8; i++) 
            {
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x - 1];
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y - 1];
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z - 1];
            }
        }
    }
    __syncthreads();

    d_float8 Y_f8;
    int CbCry = hipThreadIdx_y * 2;
    float4 Cb, Cr;
    y_hip_compute(srcPtr,(d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 16][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y + 32][hipThreadIdx_x8],&Y_f8);
    __syncthreads();

    if(hipThreadIdx_y < 8) 
    {  
        // Downsample RGB and convert to CbCr
        downsample_cbcr_hip_compute((d_float8*)&src_smem[CbCry][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 1][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 16][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 17][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 32][hipThreadIdx_x8],(d_float8*)&src_smem[CbCry + 33][hipThreadIdx_x8],&Cb, &Cr);
        // Store Y and downsampled CbCr.. Cr just below Cb for continutiy 
        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
        //Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = Y_f8;
    __syncthreads();

    // Doing -128 as part of DCT,
    //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];

        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];

        dct_fwd_8x8_1d(&col_vec[0], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++)
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();

    // 1D row wise DCT for Y Cb and Cr channela
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 

    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);

    //----INVERSE STEPS---
    // Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false); 
    __syncthreads();
    
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < alignedWidth) 
    {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) 
            col_vec[i] = src_smem[i][col];
        
        dct_inv_8x8_1d(&col_vec[0], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[8], 1, col,  true);
        dct_inv_8x8_1d(&col_vec[16], 1, col, true);
        dct_inv_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) 
            src_smem[i][col] = col_vec[i];
    }
    __syncthreads();
    
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] , 0.0f, 255.0f, 0);
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] , 0.0f, 255.0f,0);
    __syncthreads();
        
    // Vertical Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];
    __syncthreads();
   
    // Convert back to RGB
    Upsample_and_RGB_hip_compute(Cb,Cr,(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();
    
    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);  
    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);  
    __syncthreads(); 
  
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx,src_smem_channel);
}

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion(
    T *srcPtr,
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

    int quality = 50; // should be taken as a param
    quality = std::clamp<int>(quality, 1, 100);
    float q_scale = (quality < 50) ? (50.0f / quality) : (2.0f - (2 * quality / 100.0f));
    // Allocate pinned memory
    int *Ytable, *CbCrtable;
    hipHostMalloc((void**)&Ytable, 64 * sizeof(int), hipHostMallocMapped);
    hipHostMalloc((void**)&CbCrtable, 64 * sizeof(int), hipHostMallocMapped);

    // Initialize and modify the tables
    int YtableInit[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };

    int CbCrtableInit[64] = {
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
    for (int i = 0; i < 64; i++) {
        Ytable[i] = std::max<uint8_t>(static_cast<uint8_t>(std::clamp(q_scale * YtableInit[i], 0.0f, 255.0f)), 1);
        CbCrtable[i] = std::max<uint8_t>(static_cast<uint8_t>(std::clamp(q_scale * CbCrtableInit[i], 0.0f, 255.0f)), 1);
        //printf("YTab %d CbCrTab %d\n", Ytable[i], CbCrtable[i]);
    }

    // Map pinned memory to GPU without explicit copy
    int *gpuYtable, *gpuCbCrtable;
    hipHostGetDevicePointer((void**)&gpuYtable, Ytable, 0);
    hipHostGetDevicePointer((void**)&gpuCbCrtable, CbCrtable, 0);


//PKD3 to PKD3
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
                       gpuYtable,
                       gpuCbCrtable,
                       q_scale);
    }
 
//PLN3 to PLN3
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && 
        (srcDescPtr->c == 3)) 
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
                       gpuYtable,
                       gpuCbCrtable,
                       q_scale);
    }
   
 //PLN1 to PLN1   
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && 
        (srcDescPtr->c == 1)) 
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
                       gpuYtable,
                       q_scale);
    }

//PKD3 to PLN3     
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
                       gpuYtable,
                       gpuCbCrtable,
                       q_scale);
    }

//PLN3 to PKD3
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
                       gpuYtable,
                       gpuCbCrtable,
                       q_scale);
    }
   
    return RPP_SUCCESS;
}