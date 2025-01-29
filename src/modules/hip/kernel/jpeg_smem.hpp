#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// DCT Constants
__device__ constexpr float a = 1.387039845322148f;    // sqrt(2) * cos(    pi / 16)
__device__ constexpr float b = 1.306562964876377f;    // sqrt(2) * cos(    pi /  8)
__device__ constexpr float c = 1.175875602419359f;    // sqrt(2) * cos(3 * pi / 16)
__device__ constexpr float d = 0.785694958387102f;    // sqrt(2) * cos(5 * pi / 16)
__device__ constexpr float e = 0.541196100146197f;    // sqrt(2) * cos(3 * pi /  8)
__device__ constexpr float f = 0.275899379282943f;    // sqrt(2) * cos(7 * pi / 16)
__device__ constexpr float norm_factor = 0.3535533905932737f;  // 1 / sqrt(8)
__device__ int test = 0;

__device__ void YCbCr_hip_compute(float *Ch1, float *Ch2, float *Ch3)
{
    d_float8 Y_f8, Cb_f8, Cr_f8, *Ch1_f8, *Ch2_f8, *Ch3_f8;
   // d_float8 temp_Ch1, temp_Ch2, temp_Ch3;  // Temporary storage for input values
    
    Ch1_f8 = (d_float8 *)Ch1;
    Ch2_f8 = (d_float8 *)Ch2;
    Ch3_f8 = (d_float8 *)Ch3;
    
    Y_f8.f4[0]  = Ch1_f8->f4[0] * (float4)0.299 + Ch2_f8->f4[0] * (float4)0.587 + Ch3_f8->f4[0] * float4(0.114);
    Y_f8.f4[1]  = Ch1_f8->f4[1] * (float4)0.299 + Ch2_f8->f4[1] * (float4)0.587 + Ch3_f8->f4[1] * float4(0.114);

    Cb_f8.f4[0] = Ch1_f8->f4[0] * (float4)(-0.168736) + Ch2_f8->f4[0] * (float4)(-0.331264) + Ch3_f8->f4[0] * (float4)0.5 + (float4)128;
    Cb_f8.f4[1] = Ch1_f8->f4[1] * (float4)(-0.168736) + Ch2_f8->f4[1] * (float4)(-0.331264) + Ch3_f8->f4[1] * (float4)0.5 + (float4)128;

    Cr_f8.f4[0] = Ch1_f8->f4[0] * (float4)0.5 + Ch2_f8->f4[0] * (float4)(-0.418688) + Ch3_f8->f4[0] * (float4)(-0.081312) + (float4)128;
    Cr_f8.f4[1] = Ch1_f8->f4[1] * (float4)0.5 + Ch2_f8->f4[1] * (float4)(-0.418688) + Ch3_f8->f4[1] * (float4)(-0.081312) + (float4)128;

    //Storing the results back into the shared memory (Inplace)
    *Ch1_f8 = Y_f8; 
    *Ch2_f8 = Cb_f8;  
    *Ch3_f8 = Cr_f8;  
}

__device__ void verticalDownSampling(float *CbCr1, float *CbCr2)
{
    if (!CbCr1 || !CbCr2) return;

    d_float8 *CbCr1_f8, *CbCr2_f8;
    CbCr1_f8 = (d_float8 *)CbCr1;
    CbCr2_f8 = (d_float8 *)CbCr2;


    CbCr1_f8->f4[0] = (CbCr1_f8->f4[0] + CbCr2_f8->f4[0]) * 0.5f;
    CbCr1_f8->f4[1] = (CbCr1_f8->f4[1] + CbCr2_f8->f4[1]) * 0.5f;
}


__device__ void horizontalDownSampling(float *CbCr1, float *CbCr2, d_float8 *CbCr)
{
    if (!CbCr1 || !CbCr2 || !CbCr) return; 

    d_float8 *CbCr1_f8,*CbCr2_f8;
    CbCr1_f8 = (d_float8 *)CbCr1;
    CbCr2_f8 = (d_float8 *)CbCr2;

    //Each thread carries 8 elements (float8) per channel add odd elements to even elements and * 0.5
    d_float8 odds,evens;
    evens.f4[0] = make_float4(CbCr1_f8->f4[0].x,CbCr1_f8->f4[0].z,CbCr1_f8->f4[1].x,CbCr1_f8->f4[1].z) ;
    evens.f4[1] = make_float4(CbCr2_f8->f4[0].x,CbCr2_f8->f4[0].z,CbCr2_f8->f4[1].x,CbCr2_f8->f4[1].z) ;
    odds.f4[0]  = make_float4(CbCr1_f8->f4[0].y,CbCr1_f8->f4[0].w,CbCr1_f8->f4[1].y,CbCr1_f8->f4[1].w) ;
    odds.f4[1]  = make_float4(CbCr2_f8->f4[0].y,CbCr2_f8->f4[0].w,CbCr2_f8->f4[1].y,CbCr2_f8->f4[1].w) ;

    // Horizontal average for Cb and Store the results back in the first d_float8 in  Cb
    CbCr->f4[0] = (evens.f4[0] + odds.f4[0]) * (float4)0.5;
    CbCr->f4[1] = (evens.f4[1] + odds.f4[1]) * (float4)0.5;
}

__device__ void combinedDownSampling(float *src1_row1, float *src1_row2, d_float8 *output)
{
    if (!src1_row1 || !src1_row2 || !output) return;

    d_float8 *row1_f8_first = (d_float8 *)src1_row1;
    d_float8 *row1_f8_second = (d_float8 *)(src1_row1 + 8);  // Point to second float8
    d_float8 *row2_f8_first = (d_float8 *)src1_row2;
    d_float8 *row2_f8_second = (d_float8 *)(src1_row2 + 8);  // Point to second float8
    
    // vertical downsampling 
    d_float8 vertical_first, vertical_second;
    vertical_first.f4[0] = (row1_f8_first->f4[0] + row2_f8_first->f4[0]) * 0.5f;
    vertical_first.f4[1] = (row1_f8_first->f4[1] + row2_f8_first->f4[1]) * 0.5f;
    vertical_second.f4[0] = (row1_f8_second->f4[0] + row2_f8_second->f4[0]) * 0.5f;
    vertical_second.f4[1] = (row1_f8_second->f4[1] + row2_f8_second->f4[1]) * 0.5f;

    d_float8 evens, odds;
    // First half
    evens.f4[0] = make_float4(vertical_first.f4[0].x, vertical_first.f4[0].z,
                             vertical_first.f4[1].x, vertical_first.f4[1].z);
    odds.f4[0] = make_float4(vertical_first.f4[0].y, vertical_first.f4[0].w,
                            vertical_first.f4[1].y, vertical_first.f4[1].w);
    // Second half
    evens.f4[1] = make_float4(vertical_second.f4[0].x, vertical_second.f4[0].z,
                             vertical_second.f4[1].x, vertical_second.f4[1].z);
    odds.f4[1] = make_float4(vertical_second.f4[0].y, vertical_second.f4[0].w,
                            vertical_second.f4[1].y, vertical_second.f4[1].w);
    
    // Horizontal downsampling
    output->f4[0] = (evens.f4[0] + odds.f4[0]) * (float4)0.5;
    output->f4[1] = (evens.f4[1] + odds.f4[1]) * (float4)0.5;
}

// DCT forward 1D implementation
__device__ void dct_fwd_8x8_1d(float *vecf8, int stride,int rowcol, bool row) 
{
    //Adjust for rows and columns
    float x0 = vecf8[0] ;
    float x1 = vecf8[1] ;
    float x2 = vecf8[2] ;
    float x3 = vecf8[3] ;
    float x4 = vecf8[4] ;
    float x5 = vecf8[5] ;
    float x6 = vecf8[6] ;
    float x7 = vecf8[7] ;

    float tmp0 = x0 + x7;
    float tmp1 = x1 + x6;
    float tmp2 = x2 + x5;
    float tmp3 = x3 + x4;
    float tmp4 = x0 - x7;
    float tmp5 = x6 - x1;
    float tmp6 = x2 - x5;
    float tmp7 = x4 - x3;

    float tmp8 = tmp0 + tmp3;
    float tmp9 = tmp0 - tmp3;
    float tmp10 = tmp1 + tmp2;
    float tmp11 = tmp1 - tmp2;

    x0 = norm_factor * (tmp8 + tmp10);
    x2 = norm_factor * (b * tmp9 + e * tmp11);
    x4 = norm_factor * (tmp8 - tmp10);
    x6 = norm_factor * (e * tmp9 - b * tmp11);

    x1 = norm_factor * (a * tmp4 - c * tmp5 + d * tmp6 - f * tmp7);
    x3 = norm_factor * (c * tmp4 + f * tmp5 - a * tmp6 + d * tmp7);
    x5 = norm_factor * (d * tmp4 + a * tmp5 + f * tmp6 - c * tmp7);
    x7 = norm_factor * (f * tmp4 + d * tmp5 + c * tmp6 + a * tmp7);

    vecf8[0] = x0;
    vecf8[1] = x1;
    vecf8[2] = x2;
    vecf8[3] = x3;
    vecf8[4] = x4;
    vecf8[5] = x5;
    vecf8[6] = x6;
    vecf8[7] = x7;
}
//Quantization
//coeff * round( value * 1/coeff)
__device__  void quantize(float* value, int* coeff) {

    for(int i = 0; i < 8; i++) {
        value[i] = coeff[i] * roundf(value[i] * __frcp_rn(coeff[i]));
    }

}
//Inverse DCT
__device__ void dct_inv_8x8_1d(float *vecf8, int stride,int rowcol, bool row) 
{

    //Adjust for rows and columns
    float x0 = vecf8[0];
    float x1 = vecf8[1];
    float x2 = vecf8[2];
    float x3 = vecf8[3];
    float x4 = vecf8[4];
    float x5 = vecf8[5];
    float x6 = vecf8[6];
    float x7 = vecf8[7];

    float tmp0 = x0 + x4;
    float tmp1 = b * x2 + e * x6;

    float tmp2 = tmp0 + tmp1;
    float tmp3 = tmp0 - tmp1;
    float tmp4 = f * x7 + a * x1 + c * x3 + d * x5;
    float tmp5 = a * x7 - f * x1 + d * x3 - c * x5;

    float tmp6 = x0 - x4;
    float tmp7 = e * x2 - b * x6;

    float tmp8 = tmp6 + tmp7;
    float tmp9 = tmp6 - tmp7;
    float tmp10 = c * x1 - d * x7 - f * x3 - a * x5;
    float tmp11 = d * x1 + c * x7 - a * x3 + f * x5;

    x0 = norm_factor * (tmp2 + tmp4);
    x7 = norm_factor * (tmp2 - tmp4);
    x4 = norm_factor * (tmp3 + tmp5);
    x3 = norm_factor * (tmp3 - tmp5);

    x1 = norm_factor * (tmp8 + tmp10);
    x5 = norm_factor * (tmp9 - tmp11);
    x2 = norm_factor * (tmp9 + tmp11);
    x6 = norm_factor * (tmp8 - tmp10);
  
    vecf8[0] = x0 ;
    vecf8[1] = x1 ;
    vecf8[2] = x2 ;
    vecf8[3] = x3 ;
    vecf8[4] = x4 ;
    vecf8[5] = x5 ;
    vecf8[6] = x6 ;
    vecf8[7] = x7 ;
}
// Upsampling
__device__ void Upsampling(float* Ch2, float *Ch3 , float4 Cb , float4 Cr)
{
    Ch2[0] = Cb.x;   
    Ch2[1] = Cb.x;   
    Ch2[2] = Cb.y;  
    Ch2[3] = Cb.y;  
    Ch2[4] = Cb.z;
    Ch2[5] = Cb.z; 
    Ch2[6] = Cb.w; 
    Ch2[7] = Cb.w;  

    Ch3[0] = Cr.x;   
    Ch3[1] = Cr.x;   
    Ch3[2] = Cr.y;  
    Ch3[3] = Cr.y;  
    Ch3[4] = Cr.z;
    Ch3[5] = Cr.z; 
    Ch3[6] = Cr.w; 
    Ch3[7] = Cr.w;  
}
//RGB to YCbCr
__device__ void RGB_hip_compute(float *Ch1, float* Ch2, float *Ch3)
{
    // Check for null pointers
    if (!Ch1 || !Ch2 || !Ch3) return;

    d_float8 Ch1_f8 = *((d_float8*)Ch1);
    d_float8 Ch2_f8 = *((d_float8*)Ch2);
    d_float8 Ch3_f8 = *((d_float8*)Ch3);

    d_float8 R_f8, G_f8, B_f8;

    // R = Y + 1.402 × (Cr - 128)
    R_f8.f4[0] = Ch1_f8.f4[0] + (Ch3_f8.f4[0] - 128.0f) * 1.402f;
    R_f8.f4[1] = Ch1_f8.f4[1] + (Ch3_f8.f4[1] - 128.0f) * 1.402f;

    // G = Y - 0.344136 × (Cb - 128) - 0.714136 × (Cr - 128)
    G_f8.f4[0] = Ch1_f8.f4[0] - (Ch2_f8.f4[0] - 128.0f) * 0.344136f - (Ch3_f8.f4[0] - 128.0f) * 0.714136f;
    G_f8.f4[1] = Ch1_f8.f4[1] - (Ch2_f8.f4[1] - 128.0f) * 0.344136f - (Ch3_f8.f4[1] - 128.0f) * 0.714136f;

    // B = Y + 1.772 × (Cb - 128)
    B_f8.f4[0] = Ch1_f8.f4[0] + 1.772f * (Ch2_f8.f4[0] - 128.0f);
    B_f8.f4[1] = Ch1_f8.f4[1] + 1.772f * (Ch2_f8.f4[1] - 128.0f);

    // Write back the results
    *((d_float8*)Ch1) = R_f8;
    *((d_float8*)Ch2) = G_f8;
    *((d_float8*)Ch3) = B_f8;
}
__device__ inline void clamp_range(float* values, float lower_limit, float upper_limit, int num_elements = 8) {
    for (int j = 0; j < num_elements; j++) {
        values[j] = fminf(fmaxf(values[j], lower_limit), upper_limit);
    }
}

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
    int hipThreadIdx_x8 = hipThreadIdx_x << 3;
    int hipThreadIdx_x16 = hipThreadIdx_x8 * 2;
    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth) )
    {
        return;
    }

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;
    
    d_float24 src_f24, dst_f24;
    d_float8 zeros_f8 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // 16 Rows x 3 Channels and 16 columns with each element being a float8 
    __shared__ float src_smem[16*3][16*8];

    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];

    *(d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] = zeros_f8;
    *(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = zeros_f8;
    *(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = zeros_f8;
        
    if ((id_y < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (id_x < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    }

    __syncthreads();

    YCbCr_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    // clamp_range(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], 0.0f, 255.0f);
    // clamp_range(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], 0.0f, 255.0f);
    // clamp_range(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], 0.0f, 255.0f);

    //Downsampling
    int CbCry = hipThreadIdx_y * 2;
     d_float8 CbCr;    
        combinedDownSampling(
                             &src_smem[16 + CbCry][hipThreadIdx_x16],
                             &src_smem[16 + CbCry + 1][hipThreadIdx_x16],
                             &CbCr);
        __syncthreads();
    // Rearranging Cb and Cr side by side in shared memory

    // int DownSampledXbound = ceil((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2)/8) * 8;
    int DownSampledXbound = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2 + 7) / 8) * 8;
    int halfHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight / 2;

    if (hipThreadIdx_y < halfHeight) {
        bool isValidData = (hipThreadIdx_x8 < (roiTensorPtrSrc[id_z].xywhROI.roiWidth/2));

        if (hipThreadIdx_x8 < DownSampledXbound) {
            if (hipThreadIdx_y < 8) {
                // Store Cb in first half
                *(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = 
                    isValidData ? CbCr : zeros_f8;
            } else {
                // Store Cr in second half (offset by 64)
                *(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = 
                    isValidData ? CbCr : zeros_f8;
            }
        } else {
            // Zero-pad if outside width bounds
            if (hipThreadIdx_y < 8) {
                *(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = zeros_f8;
            } else {
                *(d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = zeros_f8;
            }
        }
    }
    __syncthreads();

    if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0) {
    for (int i = 16; i < 24; i++) {
                printf("\n");
        for (int j = 0; j < 128; j++) {
            printf("%f ", src_smem[i][j]);
        }
    }
    }
__syncthreads();
// -128.0f before DCT

    *(float4 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] -= (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + 4] -= (float4)128.0f;
    if(hipThreadIdx_y < 8){
    *(float4 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] -= (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + 4] -= (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] -= (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + 4] -= (float4)128.0f;
    }
 __syncthreads();

    // clamp_range(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], -128.0f, 128.0f);
    // clamp_range(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], -128.0f, 128.0f);
    // clamp_range(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], -128.0f, 128.0f);


//Fwd DCT
 int col = hipThreadIdx_x * 8 + hipThreadIdx_y; 
    float col_Vec[8];
    // Process all 128 columns 
    if (col < 128 && id_x < (roiTensorPtrSrc[id_z].xywhROI.roiWidth) ) {
        for(int j=0; j<2;j++) {
            int y_index = j * 8;
            for(int i=0;i<8;i++)
                col_Vec[i] = src_smem[y_index + i][col];
            dct_fwd_8x8_1d(col_Vec, 128,col,false);
        }
    }
    __syncthreads();

    if (col < 128 && id_x < (DownSampledXbound) ) {
        for(int i=0;i<8;i++)
            col_Vec[i] = src_smem[16 + i][col];
        
            dct_fwd_8x8_1d(col_Vec, 128,col,false);

        for(int i=0;i<8;i++)
            col_Vec[i] = src_smem[32 + i][col];
        
            dct_fwd_8x8_1d(col_Vec, 128,col,false);
    }
    __syncthreads();
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,true);
    //1D row wise DCT for Cb and Cr channels but should only done for 8 rows but here being done for 16 rows
    if(hipThreadIdx_y < 8){
        dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,true);  
        dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],0,hipThreadIdx_x8,true);
    }
    // After row-wise DCT and synchronization
    __syncthreads();
  
    //Quantization on each layer of Y Cb Cr
    //For every 8 elements row wise, we are taking respected row in the table and doing quantization
    //For Cb Cr
    //Here we can do this for only 8 threads in Y, here and in row wise DCT
    if(hipThreadIdx_y < 8){
        quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);  
        quantize(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]); 
    }
    //For Y also
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();

    
//INVERSE STEPS
    //Inverse DCT
    //1D row wise DCT for Y channel
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,true);
    //1D row wise DCT for Cb and Cr channels but should only done for 8 rows but here being done for 16 rows
    if(hipThreadIdx_y < 8){
        dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,true);  
        dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],0,hipThreadIdx_x8,true); 
    }
    // After row-wise DCT and synchronization
    __syncthreads();
    //Now for each column we should do DCT
    //So by now in smem first [0 to 16] x (16 x 8) has Y, [16 to 24] x (8 x 8) has Cb and [16 to 24] x (8 x 8) has Cr
    //We have 16 x 16 threads on X and Y dimension and 128 elements in each row
    // Each thread takes one column (total 128 threads for 128 columns)
    // Process all 128 columns 
    if (col < 128 && id_x < (roiTensorPtrSrc[id_z].xywhROI.roiWidth) ) {
        for(int j=0; j<2;j++) {
            int y_index = j * 8;
            for(int i=0;i<8;i++)
                col_Vec[i] = src_smem[y_index + i][col];
            dct_inv_8x8_1d(col_Vec, 128,col,false);
        }
    }
    __syncthreads();

    if (col < 128 && id_x < (DownSampledXbound) ) {
        for(int i=0;i<8;i++)
            col_Vec[i] = src_smem[16 + i][col];
        
            dct_inv_8x8_1d(col_Vec, 128,col,false);

        for(int i=0;i<8;i++)
            col_Vec[i] = src_smem[32 + i][col];
        
            dct_inv_8x8_1d( col_Vec, 128,col,false);
    }
    __syncthreads();
//Adding 128 after Inv DCT
    *(float4 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]     += (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + 4] += (float4)128.0f;
    if(hipThreadIdx_y < 8){
    *(float4 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]     += (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + 4] += (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]     += (float4)128.0f;
    *(float4 *)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + 4] += (float4)128.0f;
    }

 __syncthreads();

 
    //RGB to YCbCr

// Read Cb and Cr values from rows 16 to 24
int Cby, Cry;
        Cby = 16 + (hipThreadIdx_y / 2); 
        Cry = 32 + (hipThreadIdx_y / 2);// Calculate the row index for Cb and Cr
        int hipThreadIdx_x4 = hipThreadIdx_x8 / 2;

        // Read 4 floats (packed into float4) for Cb and Cr
        float4 Cb = *(float4*)(&src_smem[Cby][hipThreadIdx_x4]);        // Cb: rows 16–24, cols 0–64
        float4 Cr = *(float4*)(&src_smem[Cry][hipThreadIdx_x4]);   // Cr: rows 16–24, cols 64–128

        __syncthreads();

        // Perform Horizontal Upsampling for both Cb and Cr
        Upsampling(
            &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],  
            &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8],
            Cb,
            Cr
        );

        __syncthreads();
        
    // YCbCr to  RGB
    RGB_hip_compute(
        &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],                    
        &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]                                           
    );
     __syncthreads(); 

    clamp_range(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], 0.0f, 255.0f);
    clamp_range(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], 0.0f, 255.0f);
    clamp_range(&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8], 0.0f, 255.0f);

    __syncthreads();

    // Store properly
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx,src_smem_channel);
}

template <typename T>
RppStatus hip_exec_jpeg_compression_distortion( T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    int quality = 100; //should be taken as a param
    printf("\n Height is %d Width is %d ",srcDescPtr->strides.hStride,dstDescPtr->h);
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride /3 + 7 ) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    // Rpp32u *Ytable = handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem;
    // Rpp32u *CbCrtable = handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem;

    quality = std::clamp<int>(quality, 1, 100);
    float q_scale = 1.0f;
    if (quality < 50) {
        q_scale = 50.0f / quality;
    } else {
        q_scale = 2.0f - (2 * quality / 100.0f);
    }

    // Ytable should keep in pinned memory for quantization
    int Ytable[64] = {
        16, 11, 10, 16, 24, 40, 51, 61 ,12, 12, 14, 19, 26, 58, 60, 55 ,14, 13, 16, 24, 40, 57, 69, 56 ,14, 17, 22, 29, 51, 87, 80, 62 ,18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99
    };

    //  CbCrtable for quant
    int CbCrtable[64] = {
        17, 18, 24, 47, 99, 99, 99, 99,18, 21, 26, 66, 99, 99, 99, 99,24, 26, 56, 99, 99, 99, 99, 99,47, 66, 99, 99, 99, 99, 99, 99,99, 99, 99, 99, 99, 99, 99, 99,99, 99, 99, 99, 99, 99, 99, 99,99, 99, 99, 99, 99, 99, 99, 99,99, 99, 99, 99, 99, 99, 99, 99
    };
    for (int i = 0; i < 64; i++) 
    {
            Ytable[i] = std::max<uint8_t>(static_cast<uint8_t>(std::clamp(q_scale * Ytable[i], 0.0f, 255.0f)), 1);
            CbCrtable[i] = std::max<uint8_t>(static_cast<uint8_t>(std::clamp(q_scale * CbCrtable[i], 0.0f, 255.0f)), 1);
    }
    int *gpuYtable, *gpuCbCrtable;
    hipMalloc((void**)&gpuYtable, 64 * sizeof(int));
    hipMalloc((void**)&gpuCbCrtable, 64 * sizeof(int));
    hipMemcpy(gpuYtable, Ytable, 64 * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(gpuCbCrtable, CbCrtable, 64 * sizeof(int), hipMemcpyHostToDevice);
    //globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;

    printf("\n Total number of threads in X %d Blocks %f", LOCAL_THREADS_X,ceil((float)globalThreads_x/LOCAL_THREADS_X) );
    printf("\n Total number of threads in Y %d Blocks %f", LOCAL_THREADS_Y,ceil((float)globalThreads_y/LOCAL_THREADS_Y));
    printf("\n Total number of threads in Z %d Blocks %f", LOCAL_THREADS_Z,ceil((float)globalThreads_z/LOCAL_THREADS_Z) );
    //PKD3 to PKD3
    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride /3 + 7 ) >> 3;
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
    return RPP_SUCCESS;
}