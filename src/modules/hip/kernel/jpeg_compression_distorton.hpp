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
__device__ constexpr float4 ONE_TWENTY_EIGHT_f4 = (float4)128.0f;

__device__ int test = 0;

//RGB to YCbCr
template <typename T>
__device__ void YCbCr_hip_compute(T *src , d_float8 *Ch1_f8, d_float8 *Ch2_f8, d_float8 *Ch3_f8)
{
    d_float8 Y_f8, Cb_f8, Cr_f8;

    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value){
        rpp_hip_math_multiply8_const(Ch1_f8, Ch1_f8, (float4)255.0f);
        rpp_hip_math_multiply8_const(Ch2_f8, Ch2_f8, (float4)255.0f);
        rpp_hip_math_multiply8_const(Ch3_f8, Ch3_f8, (float4)255.0f);
    }
    else if constexpr (std::is_same<T, schar>::value)
    {
        rpp_hip_math_add8_const(Ch1_f8, Ch1_f8, ONE_TWENTY_EIGHT_f4);
        rpp_hip_math_add8_const(Ch2_f8, Ch2_f8, ONE_TWENTY_EIGHT_f4);
        rpp_hip_math_add8_const(Ch3_f8, Ch3_f8, ONE_TWENTY_EIGHT_f4);
    }
    
    // YCbCr conversion
    Y_f8.f4[0] = Ch1_f8->f4[0] * (float4)0.299000f + Ch2_f8->f4[0] * (float4)0.587000f + Ch3_f8->f4[0] * (float4)0.114000f;
    Y_f8.f4[1] = Ch1_f8->f4[1] * (float4)0.299000f + Ch2_f8->f4[1] * (float4)0.587000f + Ch3_f8->f4[1] * (float4)0.114000f;
    
    Cb_f8.f4[0] = Ch1_f8->f4[0] * (float4)(-0.168736f) + Ch2_f8->f4[0] * (float4)(-0.331264f) + Ch3_f8->f4[0] * (float4)0.500000f + ONE_TWENTY_EIGHT_f4;
    Cb_f8.f4[1] = Ch1_f8->f4[1] * (float4)(-0.168736f) + Ch2_f8->f4[1] * (float4)(-0.331264f) + Ch3_f8->f4[1] * (float4)0.500000f + ONE_TWENTY_EIGHT_f4;
    
    Cr_f8.f4[0] = Ch1_f8->f4[0] * (float4)0.500000f + Ch2_f8->f4[0] * (float4)(-0.418688f) + Ch3_f8->f4[0] * (float4)(-0.081312f) + ONE_TWENTY_EIGHT_f4;
    Cr_f8.f4[1] = Ch1_f8->f4[1] * (float4)0.500000f + Ch2_f8->f4[1] * (float4)(-0.418688f) + Ch3_f8->f4[1] * (float4)(-0.081312f) + ONE_TWENTY_EIGHT_f4;
    
    *Ch1_f8 = Y_f8;
    *Ch2_f8 = Cb_f8;
    *Ch3_f8 = Cr_f8;
}

__device__ void combined_Downsampling(float *src_row1, float *src_row2, float4 *dst)
{
    // Cast input pointers to d_float8
    d_float8 *row1_f8 = (d_float8*)src_row1;
    d_float8 *row2_f8 = (d_float8*)src_row2;
    
    // Vertical downsampling
    d_float8 vertical;
    vertical.f4[0] = (row1_f8->f4[0] + row2_f8->f4[0]) * 0.5f;
    vertical.f4[1] = (row1_f8->f4[1] + row2_f8->f4[1]) * 0.5f;
    
    // Horizontal downsampling 
    *dst = make_float4(
        (vertical.f4[0].x + vertical.f4[0].y) * 0.5f,
        (vertical.f4[0].z + vertical.f4[0].w) * 0.5f,
        (vertical.f4[1].x + vertical.f4[1].y) * 0.5f,
        (vertical.f4[1].z + vertical.f4[1].w) * 0.5f
    );
}
// DCT forward 1D implementation
__device__ void dct_fwd_8x8_1d(float *vecf8, int stride,int rowcol, bool sub_128) 
{
    int val = -128 * sub_128;
    float x0 = vecf8[0] + val;
    float x1 = vecf8[1] + val;
    float x2 = vecf8[2] + val;
    float x3 = vecf8[3] + val;
    float x4 = vecf8[4] + val;
    float x5 = vecf8[5] + val;
    float x6 = vecf8[6] + val;
    float x7 = vecf8[7] + val;

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

    vecf8[0] = norm_factor * (tmp8 + tmp10);
    vecf8[2] = norm_factor * (b * tmp9 + e * tmp11);
    vecf8[4] = norm_factor * (tmp8 - tmp10);
    vecf8[6] = norm_factor * (e * tmp9 - b * tmp11);

    vecf8[1] = norm_factor * (a * tmp4 - c * tmp5 + d * tmp6 - f * tmp7);
    vecf8[3] = norm_factor * (c * tmp4 + f * tmp5 - a * tmp6 + d * tmp7);
    vecf8[5] = norm_factor * (d * tmp4 + a * tmp5 + f * tmp6 - c * tmp7);
    vecf8[7] = norm_factor * (f * tmp4 + d * tmp5 + c * tmp6 + a * tmp7);
}

//Quantization
//Can vectorize this
__device__ void quantize(float* value, float* coeff , bool is_edge) {
    for (int i = 0; i < 8; i++) {
        value[i] = coeff[i] * roundf(value[i] / coeff[i]);
    }
}

//Inverse DCT
__device__ void dct_inv_8x8_1d(float *vecf8, int stride,int rowcol, bool add_128) 
{
    int val = 128 * add_128;

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

    vecf8[0] = fmaf(norm_factor, (tmp2 + tmp4), val);
    vecf8[7] = fmaf(norm_factor, (tmp2 - tmp4), val);
    vecf8[4] = fmaf(norm_factor, (tmp3 + tmp5), val);
    vecf8[3] = fmaf(norm_factor, (tmp3 - tmp5), val);

    vecf8[1] = fmaf(norm_factor, (tmp8 + tmp10), val);
    vecf8[5] = fmaf(norm_factor, (tmp9 - tmp11), val);
    vecf8[2] = fmaf(norm_factor, (tmp9 + tmp11), val);
    vecf8[6] = fmaf(norm_factor, (tmp8 - tmp10), val);
}
__device__ void upsample_and_RGB_hip_compute(float4 Cb, float4 Cr , d_float8 *Ch1, d_float8* Ch2, d_float8 *Ch3)
{

    // Check for null pointers
    if (!Ch1 || !Ch2 || !Ch3) return;
    d_float8 Cb_f8, Cr_f8;

    //Copy Y values
    d_float8 Y_f8 = *((d_float8*)Ch1);
    // Expand each value into two identical values
    Cb_f8.f4[0] = make_float4(Cb.x, Cb.x, Cb.y, Cb.y);
    Cb_f8.f4[1] = make_float4(Cb.z, Cb.z, Cb.w, Cb.w);

    Cr_f8.f4[0] = make_float4(Cr.x, Cr.x, Cr.y, Cr.y);
    Cr_f8.f4[1] = make_float4(Cr.z, Cr.z, Cr.w, Cr.w); 

    d_float8 R_f8, G_f8, B_f8;

    // R = Y + 1.402 × (Cr - 128)
    R_f8.f4[0] = Y_f8.f4[0] + (float4)1.402 * (Cr_f8.f4[0] - ONE_TWENTY_EIGHT_f4);
    R_f8.f4[1] = Y_f8.f4[1] + (float4)1.402 * (Cr_f8.f4[1] - ONE_TWENTY_EIGHT_f4);

    // G = Y - 0.344136 × (Cb - 128) - 0.714136 × (Cr - 128)
    G_f8.f4[0] = Y_f8.f4[0] - (float4)0.344136 * (Cb_f8.f4[0] - ONE_TWENTY_EIGHT_f4) - (Cr_f8.f4[0] - ONE_TWENTY_EIGHT_f4) * (float4)0.714136;
    G_f8.f4[1] = Y_f8.f4[1] - (float4)0.344136 * (Cb_f8.f4[1] - ONE_TWENTY_EIGHT_f4) - (Cr_f8.f4[1] - ONE_TWENTY_EIGHT_f4) * (float4)0.714136;

    // B = Y + 1.772 × (Cb - 128)
    B_f8.f4[0] = Y_f8.f4[0] + (float4)1.772 * (Cb_f8.f4[0] - ONE_TWENTY_EIGHT_f4);
    B_f8.f4[1] = Y_f8.f4[1] + (float4)1.772 * (Cb_f8.f4[1] - ONE_TWENTY_EIGHT_f4);

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
                                                             float *Ytable,
                                                             float *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;
    
    int aligned_width = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 7) / 8) * 8;
    int aligned_height = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 7) / 8) * 8;

    if ((id_y >= aligned_height) || (id_x >= aligned_width)) {
        return;
    }

    int DownSampledXbound = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2 + 7) / 8) * 8;
    int halfHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight / 2;

    // Calculate indices
    int srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;
    
    // Shared memory declaration - 48 rows (16 x 3 rows) x 128 columns (16 x 8 float8s)
    __shared__ float src_smem[48][128];
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;
    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];
    
    bool is_edge = (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    if (!is_edge)
    {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else 
    {
        int valid_pixels = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x;
        
        // First load all valid pixels
        for (int i = 0; i < valid_pixels; i++) {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3)];     
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3) + 1]; 
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3) + 2]; 
        }
        
        // Fill remaining positions by repeating valid pixels in reverse
        for (int i = valid_pixels; i < 8; i++) {
            // Calculate position in the reverse sequence
            int reverse_idx = (valid_pixels - 1) - ((i - valid_pixels) % valid_pixels);
            
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + reverse_idx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + reverse_idx];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + reverse_idx];
        }
    }
    __syncthreads();
    // Convert to YCbCr
    YCbCr_hip_compute(srcPtr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    int CbCry = hipThreadIdx_y * 2;
    float4 Cb,Cr;
    Cb = Cr = (float4)0 ;
    // Downsample Cb and Cr channels
    if (hipThreadIdx_y < 8) {
        combined_Downsampling(
            &src_smem[CbCry + 16][hipThreadIdx_x8],
            &src_smem[CbCry + 17][hipThreadIdx_x8],
            &Cb);
            
        combined_Downsampling(
            &src_smem[CbCry + 32][hipThreadIdx_x8],
            &src_smem[CbCry + 33][hipThreadIdx_x8],
            &Cr);

        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
        //Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }
    __syncthreads();

    // ceil_values((float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
    // ceil_values((float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);

    // clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8] , 0.0f, 255.0f, 0);
    // clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] , 0.0f, 255.0f, 0);

// Doing -128 as part of DCT,
 //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_fwd_8x8_1d(&col_vec[0], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col,  true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
  
    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8],is_edge);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8],is_edge);

//INVERSE STEPS
    //Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
    // After row-wise DCT and synchronization
    __syncthreads();
    //Now for each column we should do inverse DCT
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        // Y channel - first 8x8 block
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_inv_8x8_1d(&col_vec[0], 1, col, true);
        dct_inv_8x8_1d(&col_vec[8], 1, col, true);
        dct_inv_8x8_1d(&col_vec[16], 1, col,true);
        dct_inv_8x8_1d(&col_vec[24], 1, col,true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();

//Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];

    __syncthreads();
    
    // Convert back to RGB
    upsample_and_RGB_hip_compute(
        Cb,
        Cr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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
                                                             float *Ytable,
                                                             float *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;
   
    int aligned_width = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 7) / 8) * 8;
    int aligned_height = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 7) / 8) * 8;


    if ((id_y >= aligned_height) || (id_x >= aligned_width)) {
        return;
    }

    int x_pos = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int y_pos = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;
    
    // Clamp coordinates to valid range
    clamp_coordinates(x_pos, y_pos, roiTensorPtrSrc[id_z].xywhROI.roiWidth, 
                     roiTensorPtrSrc[id_z].xywhROI.roiHeight);

    int DownSampledXbound = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2 + 7) / 8) * 8;
    int halfHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight / 2;

    // Calculate indices
    int3 srcIdx , dstIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + 
                ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + 
                (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;
    
    // Shared memory declaration - 48 rows (16 x 3 rows) x 128 columns (16 x 8 float8s)
    __shared__ float src_smem[48][128];
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;
    
    // Loading data into shared memory each channel individually 
    bool is_edge = (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    if (!is_edge) {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.x,
            (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.y,
            (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.z,
            (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Handle first valid pixels normally
        int remaining_width = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x;
        remaining_width = max(0, min(8, remaining_width));
        
        // Load valid pixels
        for (int i = 0; i < remaining_width; i++) {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + i];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + i];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + i];
        }
        
        // Fill remaining pixels by mirroring valid pixels in reverse order
        for (int i = 0; i < (8 - remaining_width); i++) {
            int source_idx = ((remaining_width - 1) - (i % remaining_width));
            
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + source_idx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + source_idx];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + source_idx];
        }
    }
    __syncthreads();

    // Convert to YCbCr
    YCbCr_hip_compute(srcPtr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    int CbCry = hipThreadIdx_y * 2;
    float4 Cb,Cr;
    Cb = Cr = (float4)0 ;
    // Downsample Cb and Cr channels
    if (hipThreadIdx_y < 8) {
        combined_Downsampling(
            &src_smem[CbCry + 16][hipThreadIdx_x8],
            &src_smem[CbCry + 17][hipThreadIdx_x8],
            &Cb);
            
        combined_Downsampling(
            &src_smem[CbCry + 32][hipThreadIdx_x8],
            &src_smem[CbCry + 33][hipThreadIdx_x8],
            &Cr);

        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
        //Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }
    __syncthreads();

// Doing -128 as part of DCT,
 //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_fwd_8x8_1d(&col_vec[0], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
  
    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8],is_edge);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8],is_edge);

//INVERSE STEPS
    //Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
    // After row-wise DCT and synchronization
    __syncthreads();
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        // Y channel - first 8x8 block
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_inv_8x8_1d(&col_vec[0], 1, col, true);
        dct_inv_8x8_1d(&col_vec[8], 1, col, true);
        dct_inv_8x8_1d(&col_vec[16], 1, col,true);
        dct_inv_8x8_1d(&col_vec[24], 1, col,true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();

//Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];

    __syncthreads();
    
    // Convert back to RGB
    upsample_and_RGB_hip_compute(
        Cb,
        Cr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();
    
    // Clamp values and store results
        rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);

        clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);  
        clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);  
        clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);  
        __syncthreads();  

    //   rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &result);
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
                                                             float *Ytable,
                                                             float *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;
   
    int aligned_width = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 7) / 8) * 8;
    int aligned_height = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 7) / 8) * 8;


    if ((id_y >= aligned_height) || (id_x >= aligned_width)) {
        return;
    }

    int x_pos = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int y_pos = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;
    
    // Clamp coordinates to valid range
    clamp_coordinates(x_pos, y_pos, roiTensorPtrSrc[id_z].xywhROI.roiWidth, 
                     roiTensorPtrSrc[id_z].xywhROI.roiHeight);
     __syncthreads(); 
    int DownSampledXbound = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2 + 7) / 8) * 8;
    int halfHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight / 2;

    // Calculate indices
    int3 srcIdx;
    srcIdx.x = (id_z * srcStridesNCH.x) + 
                ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + 
                (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdx.y = srcIdx.x + srcStridesNCH.y;
    srcIdx.z = srcIdx.y + srcStridesNCH.y;

    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;
    
    // Shared memory declaration - 48 rows (16 x 3 rows) x 128 columns (16 x 8 float8s)
    __shared__ float src_smem[48][128];
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];
    
    // Loading data into shared memory each channel individually 
    bool is_edge = (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    if (!is_edge) {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.x,
            (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.y,
            (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8]);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx.z,
            (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    } else {
        // Handle first valid pixels normally
        int remaining_width = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x;
        remaining_width = max(0, min(8, remaining_width));
        
        // Load valid pixels
        for (int i = 0; i < remaining_width; i++) {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx.x + i];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx.y + i];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx.z + i];
        }
        
        // Fill remaining pixels by mirroring valid pixels in reverse order
        for (int i = 0; i < (8 - remaining_width); i++) {
            int source_idx = ((remaining_width - 1) - (i % remaining_width));
            
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + source_idx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + source_idx];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + source_idx];
        }
    }
    __syncthreads();

    // Convert to YCbCr
    YCbCr_hip_compute(srcPtr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    int CbCry = hipThreadIdx_y * 2;
    float4 Cb,Cr;
    Cb = Cr = (float4)0 ;
    // Downsample Cb and Cr channels
    if (hipThreadIdx_y < 8) {
        combined_Downsampling(
            &src_smem[CbCry + 16][hipThreadIdx_x8],
            &src_smem[CbCry + 17][hipThreadIdx_x8],
            &Cb);
            
        combined_Downsampling(
            &src_smem[CbCry + 32][hipThreadIdx_x8],
            &src_smem[CbCry + 33][hipThreadIdx_x8],
            &Cr);

        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
        //Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }
    __syncthreads();

// Doing -128 as part of DCT,
 //Fwd DCT
 int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_fwd_8x8_1d(&col_vec[0], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
  
    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8],is_edge);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8],is_edge);

//INVERSE STEPS
    //Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
    // After row-wise DCT and synchronization
    __syncthreads();
    // Each thread takes one column (total 128 threads for 128 columns)
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        // Y channel - first 8x8 block
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_inv_8x8_1d(&col_vec[0], 1, col, true);
        dct_inv_8x8_1d(&col_vec[8], 1, col, true);
        dct_inv_8x8_1d(&col_vec[16], 1, col,true);
        dct_inv_8x8_1d(&col_vec[24], 1, col,true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();

//Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];

    __syncthreads();
    
    // Convert back to RGB
    upsample_and_RGB_hip_compute(
        Cb,
        Cr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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

    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx,src_smem_channel); 
}

//PKD3 to PLN3
template <typename T>
__global__ void jpeg_compression_distortion_pkd3_pln3_hip_tensor( T *srcPtr,
                                                             uint2 srcStridesNH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             float *Ytable,
                                                             float *CbCrtable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;
    

    int aligned_width = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 7) / 8) * 8;
    int aligned_height = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 7) / 8) * 8;


    if ((id_y >= aligned_height) || (id_x >= aligned_width)) {
        return;
    }

    int DownSampledXbound = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2 + 7) / 8) * 8;
    int halfHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight / 2;

    // Calculate indices
    int srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int3 dstIdx;
    dstIdx.x = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx.y = dstIdx.x + dstStridesNCH.y;
    dstIdx.z = dstIdx.y + dstStridesNCH.y;
    
    // Shared memory declaration - 48 rows (16 x 3 rows) x 128 columns (16 x 8 float8s)
    __shared__ float src_smem[48][128];
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;
    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];
    
    // Loading data into shared memory each channel individually 
    bool is_edge = (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    if (!is_edge)
    {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else 
    {
        int valid_pixels = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x;
        
        // First load all valid pixels
        for (int i = 0; i < valid_pixels; i++) {
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3)];     
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3) + 1]; 
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = srcPtr[srcIdx + (i * 3) + 2]; 
        }
        
        // Fill remaining positions by repeating valid pixels in reverse
        for (int i = valid_pixels; i < 8; i++) {
            // Calculate position in the reverse sequence
            int reverse_idx = (valid_pixels - 1) - ((i - valid_pixels) % valid_pixels);
            
            src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + i] = src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8 + reverse_idx];
            src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + i] = src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8 + reverse_idx];
            src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + i] = src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8 + reverse_idx];
        }
    }
    __syncthreads();
    // Convert to YCbCr
    YCbCr_hip_compute(srcPtr,
        (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
        (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();

    int CbCry = hipThreadIdx_y * 2;
    float4 Cb,Cr;
    Cb = Cr = (float4)0 ;
    // Downsample Cb and Cr channels
    if (hipThreadIdx_y < 8) {
        combined_Downsampling(
            &src_smem[CbCry + 16][hipThreadIdx_x8],
            &src_smem[CbCry + 17][hipThreadIdx_x8],
            &Cb);
            
        combined_Downsampling(
            &src_smem[CbCry + 32][hipThreadIdx_x8],
            &src_smem[CbCry + 33][hipThreadIdx_x8],
            &Cr);

        *(float4*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cb;
        //Storing Cr below Cb (8 x 64)
        *(float4*)&src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x4] = Cr;
    }
    __syncthreads();

// Doing -128 as part of DCT,
 //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_fwd_8x8_1d(&col_vec[0], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[16], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[24], 1, col, true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();
    //1D row wise DCT for Y channel and CbCr
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
  
    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8],is_edge);  
    quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8],is_edge);
//INVERSE STEPS
    //Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],0,hipThreadIdx_x8,false);
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
    __syncthreads();
    //Now for each column we should do inverse DCT
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[32];
        
        for (int i = 0; i < 32; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_inv_8x8_1d(&col_vec[0], 1, col, true);
        dct_inv_8x8_1d(&col_vec[8], 1, col, true);
        dct_inv_8x8_1d(&col_vec[16], 1, col,true);
        dct_inv_8x8_1d(&col_vec[24], 1, col,true);

        for (int i = 0; i < 32; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();

//Upsampling
    CbCry = hipThreadIdx_y/2;
    Cb = *(float4*)&src_smem[CbCry + 16][hipThreadIdx_x4];
    Cr = *(float4*)&src_smem[CbCry + 24][hipThreadIdx_x4];

    __syncthreads();
    
    // Convert back to RGB
    upsample_and_RGB_hip_compute(Cb,
                                 Cr,
                                 (d_float8*)&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],
                                 (d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],
                                 (d_float8*)&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
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
//PLN1 to PLN1

template <typename T>
__global__ void jpeg_compression_distortion_pln1_hip_tensor( T *srcPtr,
                                                             uint3 srcStridesNCH,
                                                             T *dstPtr,
                                                             uint3 dstStridesNCH,
                                                             RpptROIPtr roiTensorPtrSrc,
                                                             float *Ytable,
                                                             float q_scale)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int hipThreadIdx_x8 = hipThreadIdx_x * 8;
    int hipThreadIdx_x4 = hipThreadIdx_x * 4;
   
    int aligned_width = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth + 7) / 8) * 8;
    int aligned_height = ((roiTensorPtrSrc[id_z].xywhROI.roiHeight + 7) / 8) * 8;

    if ((id_y >= aligned_height) || (id_x >= aligned_width)) {
        return;
    }

    int x_pos = id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x;
    int y_pos = id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y;
    
    // Clamp coordinates to valid range
    clamp_coordinates(x_pos, y_pos, roiTensorPtrSrc[id_z].xywhROI.roiWidth, 
                     roiTensorPtrSrc[id_z].xywhROI.roiHeight);

    int DownSampledXbound = ((roiTensorPtrSrc[id_z].xywhROI.roiWidth/2 + 7) / 8) * 8;
    int halfHeight = roiTensorPtrSrc[id_z].xywhROI.roiHeight / 2;

    // Calculate indices
    int srcIdx , dstIdx;
    srcIdx = (id_z * srcStridesNCH.x) + 
                ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + 
                (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    
    // Shared memory declaration - 48 rows (16 x 3 rows) x 128 columns (16 x 8 float8s)
    __shared__ float src_smem[16][128];
    
    // Loading data into shared memory each channel individually 
    bool is_edge = (id_x + 8) > roiTensorPtrSrc[id_z].xywhROI.roiWidth;
    if (!is_edge) {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx,
            (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
    } else {
        // Handle first valid pixels normally
        int remaining_width = roiTensorPtrSrc[id_z].xywhROI.roiWidth - id_x;
        remaining_width = max(0, min(8, remaining_width));
        
        // Load valid pixels
        for (int i = 0; i < remaining_width; i++) {
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + i] = srcPtr[srcIdx + i];
        }
        
        // Fill remaining pixels by mirroring valid pixels in reverse order
        for (int i = 0; i < (8 - remaining_width); i++) {
            int source_idx = ((remaining_width - 1) - (i % remaining_width));
            
            src_smem[hipThreadIdx_y][hipThreadIdx_x8 + remaining_width + i] = 
                src_smem[hipThreadIdx_y][hipThreadIdx_x8 + source_idx];
        }
    }
    __syncthreads();

    if constexpr (std::is_same<T, float>::value || std::is_same<T, half>::value){
        rpp_hip_math_multiply8_const((d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (float4)255);
    }
    else if constexpr (std::is_same<T, schar>::value)
    {
        rpp_hip_math_add8_const((d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], (d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8], ONE_TWENTY_EIGHT_f4);
    }
    __syncthreads();
// Doing -128 as part of DCT,
 //Fwd DCT
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[16];
        
        for (int i = 0; i < 16; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_fwd_8x8_1d(&col_vec[0], 1, col, true);
        dct_fwd_8x8_1d(&col_vec[8], 1, col, true);
        for (int i = 0; i < 16; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
  
    //Quantization on each layer of Y Cb Cr
    quantize(&src_smem[hipThreadIdx_y][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8],is_edge);  

//INVERSE STEPS
    //Inverse DCT
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y][hipThreadIdx_x8],0,hipThreadIdx_x8,false);  
    // After row-wise DCT and synchronization
    __syncthreads();
    // Process all 128 columns 
    //Adding back 128 as part of DCT
    if (col < 128 && col < aligned_width) {
        // Load column into temporary array
        float col_vec[16];
        
        // Y channel - first 8x8 block
        for (int i = 0; i < 16; i++) {
            col_vec[i] = src_smem[i][col];
        }
        dct_inv_8x8_1d(&col_vec[0], 1, col, true);
        dct_inv_8x8_1d(&col_vec[8], 1, col, true);

        for (int i = 0; i < 16; i++) {
            src_smem[i][col] = col_vec[i];
        }
    }
    __syncthreads();
    
    // Clamp values and store results
    rpp_hip_adjust_range(dstPtr, (d_float8*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);

    clamp_range(srcPtr,(float*)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);  
    __syncthreads();  

    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, (d_float8 *)&src_smem[hipThreadIdx_y][hipThreadIdx_x8]);
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
    float *Ytable, *CbCrtable;
    hipHostMalloc((void**)&Ytable, 64 * sizeof(float), hipHostMallocMapped);
    hipHostMalloc((void**)&CbCrtable, 64 * sizeof(float), hipHostMallocMapped);

    // Initialize and modify the tables
    float YtableInit[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };

    float CbCrtableInit[64] = {
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99
    };

    for (int i = 0; i < 64; i++) {
        Ytable[i] = std::max<float>(static_cast<float>(std::clamp(q_scale * YtableInit[i], 0.0f, 255.0f)), 1);
        CbCrtable[i] = std::max<float>(static_cast<float>(std::clamp(q_scale * CbCrtableInit[i], 0.0f, 255.0f)), 1);
    }

    // Map pinned memory to GPU without explicit copy
    float *gpuYtable, *gpuCbCrtable;
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