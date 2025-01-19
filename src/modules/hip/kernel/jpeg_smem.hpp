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
    d_float8 temp_Ch1, temp_Ch2, temp_Ch3;  // Temporary storage for input values
    
    Ch1_f8 = (d_float8 *)Ch1;
    Ch2_f8 = (d_float8 *)Ch2;
    Ch3_f8 = (d_float8 *)Ch3;
    
    // Store input values
    temp_Ch1 = *Ch1_f8;
    temp_Ch2 = *Ch2_f8;
    temp_Ch3 = *Ch3_f8;

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

    // if( blockIdx.z == 0 && blockIdx.y  == 0 && blockIdx.x  == 0 )
    // {
    //     printf("\n Input RGB:"
    //            "\n R[0]: %f %f %f %f  R[1]: %f %f %f %f"
    //            "\n G[0]: %f %f %f %f  G[1]: %f %f %f %f"
    //            "\n B[0]: %f %f %f %f  B[1]: %f %f %f %f"
    //            "\n Output YCbCr:"
    //            "\n Y[0]: %f %f %f %f  Y[1]: %f %f %f %f"
    //            "\n Cb[0]: %f %f %f %f  Cb[1]: %f %f %f %f"
    //            "\n Cr[0]: %f %f %f %f  Cr[1]: %f %f %f %f"
    //            "\n Block Indices: tidX %d tidY %d tidZ %d bidX %d",
    //            // Input RGB values
    //            temp_Ch1.f4[0].x, temp_Ch1.f4[0].y, temp_Ch1.f4[0].z, temp_Ch1.f4[0].w,
    //            temp_Ch1.f4[1].x, temp_Ch1.f4[1].y, temp_Ch1.f4[1].z, temp_Ch1.f4[1].w,
    //            temp_Ch2.f4[0].x, temp_Ch2.f4[0].y, temp_Ch2.f4[0].z, temp_Ch2.f4[0].w,
    //            temp_Ch2.f4[1].x, temp_Ch2.f4[1].y, temp_Ch2.f4[1].z, temp_Ch2.f4[1].w,
    //            temp_Ch3.f4[0].x, temp_Ch3.f4[0].y, temp_Ch3.f4[0].z, temp_Ch3.f4[0].w,
    //            temp_Ch3.f4[1].x, temp_Ch3.f4[1].y, temp_Ch3.f4[1].z, temp_Ch3.f4[1].w,
    //            // Output YCbCr values
    //            Ch1_f8->f4[0].x, Ch1_f8->f4[0].y, Ch1_f8->f4[0].z, Ch1_f8->f4[0].w,
    //            Ch1_f8->f4[1].x, Ch1_f8->f4[1].y, Ch1_f8->f4[1].z, Ch1_f8->f4[1].w,
    //            Ch2_f8->f4[0].x, Ch2_f8->f4[0].y, Ch2_f8->f4[0].z, Ch2_f8->f4[0].w,
    //            Ch2_f8->f4[1].x, Ch2_f8->f4[1].y, Ch2_f8->f4[1].z, Ch2_f8->f4[1].w,
    //            Ch3_f8->f4[0].x, Ch3_f8->f4[0].y, Ch3_f8->f4[0].z, Ch3_f8->f4[0].w,
    //            Ch3_f8->f4[1].x, Ch3_f8->f4[1].y, Ch3_f8->f4[1].z, Ch3_f8->f4[1].w,
    //            hipThreadIdx_x, hipThreadIdx_y, hipThreadIdx_z,blockIdx.x);
    // }
}

__device__ void verticalDownSampling(float *CbCr1, float *CbCr2)
{
    d_float8 *CbCr1_f8, *CbCr2_f8;
    CbCr1_f8 = (d_float8 *)CbCr1;
    CbCr2_f8 = (d_float8 *)CbCr2;


    //For Debugging
    float4 tempCbCr1[2];
    tempCbCr1[0] = CbCr1_f8->f4[0];
    tempCbCr1[1] = CbCr1_f8->f4[1];

    CbCr1_f8->f4[0] = (CbCr1_f8->f4[0] + CbCr2_f8->f4[0]) * 0.5f;
    CbCr1_f8->f4[1] = (CbCr1_f8->f4[1] + CbCr2_f8->f4[1]) * 0.5f;
    // CbCr2_f8->f4[0] = (float4)1;
    // CbCr2_f8->f4[1] = (float4)1;
    // if(hipBlockIdx_y == test && hipBlockIdx_x ==  test && hipBlockIdx_z ==  test )
    // {
    //     printf("\n Vertical Downsampling : [CbCr %f %f %f %f %f %f %f %f] [CbCr2 %f %f %f %f %f %f %f %f] [Res %f %f %f %f %f %f %f %f] [bidX  %d bidY %d bidZ %d]",
    //                     tempCbCr1[0].x,tempCbCr1[0].y,tempCbCr1[0].z,tempCbCr1[0].w,
    //                     tempCbCr1[1].x,tempCbCr1[1].y,tempCbCr1[1].z,tempCbCr1[1].w,
    //                     CbCr2_f8->f4[0].x,CbCr2_f8->f4[0].y,CbCr2_f8->f4[0].z,CbCr2_f8->f4[0].w,
    //                     CbCr2_f8->f4[1].x,CbCr2_f8->f4[1].y,CbCr2_f8->f4[1].z,CbCr2_f8->f4[1].w,
    //                     CbCr1_f8->f4[0].x,CbCr1_f8->f4[0].y,CbCr1_f8->f4[0].z,CbCr1_f8->f4[0].w,
    //                     CbCr1_f8->f4[1].x,CbCr1_f8->f4[1].y,CbCr1_f8->f4[1].z,CbCr1_f8->f4[1].w,
                                                            
    //                     hipThreadIdx_x,hipThreadIdx_y,hipThreadIdx_z);

    // }
}


__device__ void horizontalDownSampling(float *CbCr1, float *CbCr2, d_float8 *CbCr)
{
    d_float8 *CbCr1_f8,*CbCr2_f8;
    CbCr1_f8 = (d_float8 *)CbCr1;
    CbCr2_f8 = (d_float8 *)CbCr2;

    // if(hipThreadIdx_x == test && hipThreadIdx_y == test && hipThreadIdx_z == test)
    // {
    //     printf("\n Horizontal Downsampling : [Cb %f %f %f %f] [Cr %f %f %f %f] bidX %d bidY %d bidZ %d",
    //                                                                Cb1_f8->f4[0].x,Cb2_f8->f4[0].y,Cb2_f8->f4[0].z,Cb2_f8->f4[0].w,
    //                                                                Cr1_f8->f4[0].x,Cr1_f8->f4[0].y,Cr1_f8->f4[0].z,Cr1_f8->f4[0].w,
    //                                                                hipBlockIdx_x,hipBlockIdx_y,hipBlockIdx_z);
    // }
    //Each thread carries 8 elements (float8) per channel add odd elements to even elements and * 0.5
    d_float8 odds,evens;
    evens.f4[0] = make_float4(CbCr1_f8->f4[0].x,CbCr1_f8->f4[0].z,CbCr1_f8->f4[1].x,CbCr1_f8->f4[1].z) ;
    evens.f4[1] = make_float4(CbCr2_f8->f4[0].x,CbCr2_f8->f4[0].z,CbCr2_f8->f4[1].x,CbCr2_f8->f4[1].z) ;
    odds.f4[0]  = make_float4(CbCr1_f8->f4[0].y,CbCr1_f8->f4[0].w,CbCr1_f8->f4[1].y,CbCr1_f8->f4[1].w) ;
    odds.f4[1]  = make_float4(CbCr2_f8->f4[0].y,CbCr2_f8->f4[0].w,CbCr2_f8->f4[1].y,CbCr2_f8->f4[1].w) ;

    // Horizontal average for Cb and Store the results back in the first d_float8 in  Cb
    CbCr->f4[0] = (evens.f4[0] + odds.f4[0]) * (float4)0.5;
    CbCr->f4[1] = (evens.f4[1] + odds.f4[1]) * (float4)0.5;
    // CbCr2_f8->f4[0] = (float4)1;
    // CbCr2_f8->f4[1] = (float4)1;

    // if(hipThreadIdx_x == test && hipThreadIdx_y ==  test && hipThreadIdx_z == test)
    // {
    //     printf("\n Horizontal Downsampling : [Evens %f %f %f %f %f %f %f %f]  [Odds %f %f %f %f %f %f %f %f]  [Res %f %f %f %f %f %f %f %f] idX %d , idY %d, idZ %d",
    //             evens.f4[0].x,evens.f4[0].y,evens.f4[0].z,evens.f4[0].w,evens.f4[1].x,evens.f4[1].y,evens.f4[1].z,evens.f4[1].w,
    //                                             odds.f4[0].x,odds.f4[0].y,odds.f4[0].z,odds.f4[0].w,odds.f4[1].x,odds.f4[1].y,odds.f4[1].z,odds.f4[1].w,
    //                                             CbCr->f4[0].x,CbCr->f4[0].y,CbCr->f4[0].z,CbCr->f4[0].w,CbCr->f4[1].x,CbCr->f4[1].y,CbCr->f4[1].z,CbCr->f4[1].w,
    //                                             hipBlockIdx_x,hipBlockIdx_y,hipBlockIdx_z);
    // }
}

// DCT forward 1D implementation
__device__ void dct_fwd_8x8_1d(float *vecf8, int stride,int rowcol, bool row) 
{
    uint4 x_idx1,x_idx2,y_idx1,y_idx2;
    if(row)
    {
       x_idx1 = make_uint4(0,1,2,3);
       x_idx2 = make_uint4(4,5,6,7);
       y_idx1 = (uint4)rowcol;
       y_idx2 = (uint4)rowcol;
    }
    else
    {
       x_idx1 = (uint4)rowcol;
       x_idx2 = (uint4)rowcol;
       y_idx1 = make_uint4(0,1,2,3);
       y_idx2 = make_uint4(4,5,6,7);
    }

    //Adjust for rows and columns
    float x0 = vecf8[y_idx1.x * stride + x_idx1.x];
    float x1 = vecf8[y_idx1.y * stride + x_idx1.y];
    float x2 = vecf8[y_idx1.z * stride + x_idx1.z];
    float x3 = vecf8[y_idx1.w * stride + x_idx1.w];
    float x4 = vecf8[y_idx2.x * stride + x_idx2.x];
    float x5 = vecf8[y_idx2.y * stride + x_idx2.y];
    float x6 = vecf8[y_idx2.z * stride + x_idx2.z];
    float x7 = vecf8[y_idx2.w * stride + x_idx2.w];

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

    vecf8[y_idx1.x * stride + x_idx1.x] = x0;
    vecf8[y_idx1.y * stride + x_idx1.y] = x1;
    vecf8[y_idx1.z * stride + x_idx1.z] = x2;
    vecf8[y_idx1.w * stride + x_idx1.w] = x3;
    vecf8[y_idx2.x * stride + x_idx2.x] = x4;
    vecf8[y_idx2.y * stride + x_idx2.y] = x5;
    vecf8[y_idx2.z * stride + x_idx2.z] = x6;
    vecf8[y_idx2.w * stride + x_idx2.w] = x7;
}
//Quantization
//coeff * round( value * 1/coeff)
__device__  void quantize(float* value, int* coeff) {
    // Perform quantization
//  if(hipThreadIdx_x && hipThreadIdx_y )
//  {  
//      printf("\n Qunatize : [ %f %f %f %f] tidX %d tidY %d", value[0], value[1], value[2], value[3],hipThreadIdx_x,hipThreadIdx_y );
//  } 
    for(int i = 0; i < 8; i++) {
        value[i] = coeff[i] * roundf(value[i] * __frcp_rn(coeff[i]));
    }

}
//DeQuantization
__device__ void dequantize(float* value, int* coeff) {
//      if(hipThreadIdx_x && hipThreadIdx_y )
//  {  
//      printf("\n DEQunatize : [ %f %f %f %f] tidX %d tidY %d", value[0], value[1], value[2], value[3],hipThreadIdx_x,hipThreadIdx_y );
//  }
    for(int i=0; i<8 ;i++){
        value[i] = value[i] / coeff[i];
    }
}
//Inverse DCT
__device__ void dct_inv_8x8_1d(float *vecf8, int stride,int rowcol, bool row) 
{
    // printf("Hiii");
    uint4 x_idx1,x_idx2,y_idx1,y_idx2;
    if(row)
    {
       x_idx1 = make_uint4(0,1,2,3);
       x_idx2 = make_uint4(4,5,6,7);
       y_idx1 = (uint4)rowcol;
       y_idx2 = (uint4)rowcol;
    }
    else
    {
       x_idx1 = (uint4)rowcol;
       x_idx2 = (uint4)rowcol;
       y_idx1 = make_uint4(0,1,2,3);
       y_idx2 = make_uint4(4,5,6,7);
    }

    //Adjust for rows and columns
    float x0 = vecf8[y_idx1.x * stride + x_idx1.x];
    float x1 = vecf8[y_idx1.y * stride + x_idx1.y];
    float x2 = vecf8[y_idx1.z * stride + x_idx1.z];
    float x3 = vecf8[y_idx1.w * stride + x_idx1.w];
    float x4 = vecf8[y_idx2.x * stride + x_idx2.x];
    float x5 = vecf8[y_idx2.y * stride + x_idx2.y];
    float x6 = vecf8[y_idx2.z * stride + x_idx2.z];
    float x7 = vecf8[y_idx2.w * stride + x_idx2.w];

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
  
    vecf8[y_idx1.x * stride + x_idx1.x] = x0;
    vecf8[y_idx1.y * stride + x_idx1.y] = x1;
    vecf8[y_idx1.z * stride + x_idx1.z] = x2;
    vecf8[y_idx1.w * stride + x_idx1.w] = x3;
    vecf8[y_idx2.x * stride + x_idx2.x] = x4;
    vecf8[y_idx2.y * stride + x_idx2.y] = x5;
    vecf8[y_idx2.z * stride + x_idx2.z] = x6;
    vecf8[y_idx2.w * stride + x_idx2.w] = x7;
}

//RGB to YCbCr
__device__ void RGB_hip_compute(float *Ch1, float *Ch2, float *Ch3)
{
    d_float8 R_f8,G_f8,B_f8,*Ch1_f8,*Ch2_f8,*Ch3_f8;
    Ch1_f8 = (d_float8 *)Ch1;
    //From Cb and Cr channels elements are upsampled horizontally
    //Vertical Upsampling is being done as part of sending same Cb & Cr row for two Y channel rows
    Ch2_f8->f4[0] = make_float4(Ch2[0],Ch2[0],Ch2[1],Ch2[1]);
    Ch2_f8->f4[1] = make_float4(Ch2[2],Ch2[2],Ch2[3],Ch2[3]);
    
    Ch3_f8->f4[0] = make_float4(Ch3[0],Ch3[0],Ch3[1],Ch3[1]);
    Ch3_f8->f4[1] = make_float4(Ch3[2],Ch3[2],Ch3[3],Ch3[3]);
    //if(hipThreadIdx_x && hipThreadIdx_y )
    // {
    //     printf("\n YCbCr :[R %f %f %f %f] [G %f %f %f %f] [B %f %f %f %f] idX %d idY %d",Ch1_f8->f4[0].x,Ch1_f8->f4[0].y,Ch1_f8->f4[0].z,Ch1_f8->f4[0].w,
    //                                                                Ch2_f8->f4[0].x,Ch2_f8->f4[0].y,Ch2_f8->f4[0].z,Ch2_f8->f4[0].w,
    //                                                                Ch3_f8->f4[0].x,Ch3_f8->f4[0].y,Ch3_f8->f4[0].z,Ch3_f8->f4[0].w,
    //                                                                hipThreadIdx_x,hipThreadIdx_y);
    // }
// R = Y + 1.402 × (Cr−128)   
    R_f8.f4[0] = Ch1_f8->f4[0] + (Ch3_f8->f4[0] - (float4)128) * (float4)1.402;
    R_f8.f4[1] = Ch1_f8->f4[1] + (Ch3_f8->f4[1] - (float4)128) * (float4)1.402;
 // G = Y − 0.344136 ×(Cb−128) − 0.714136×(Cr−128)
    G_f8.f4[0] = Ch1_f8->f4[0] - (Ch2_f8->f4[0] - (float4)128) * (float4)0.344136 - (Ch3_f8->f4[0] - (float4)128) * (float4)0.714136;
    G_f8.f4[1] = Ch1_f8->f4[1] - (Ch2_f8->f4[1] - (float4)128) * (float4)0.344136 - (Ch3_f8->f4[1] - (float4)128) * (float4)0.714136;
 // B = Y + 1.772 × (Cb−128)
    B_f8.f4[0] = Ch1_f8->f4[0] + (float4)1.772 * (Ch2_f8->f4[0] - (float4)128);
    B_f8.f4[1] = Ch1_f8->f4[1] + (float4)1.772 * (Ch2_f8->f4[1] - (float4)128);

    //Storing the results back into the shared memory (Inplace)
    *Ch1_f8 =  R_f8; 
    *Ch2_f8 =  G_f8;  
    *Ch3_f8 =  B_f8;  
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
    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    int dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3 ;
    
    d_float24 src_f24, dst_f24;
    // 16 Rows x 3 Channels and 16 columns with each element being a float8 
    __shared__ float src_smem[16*3][16*8];
    // auto& copyY= src_smem;
    // auto& copyCbCr = src_smem;
    int3 hipThreadIdx_y_channel;
    hipThreadIdx_y_channel.x = hipThreadIdx_y;
    hipThreadIdx_y_channel.y = hipThreadIdx_y + 16;
    hipThreadIdx_y_channel.z = hipThreadIdx_y + 32;

    float *src_smem_channel[3];
    src_smem_channel[0] = &src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8];
    src_smem_channel[1] = &src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8];
    src_smem_channel[2] = &src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8];
        
    if ((id_y < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (id_x < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        rpp_hip_load24_pkd3_to_float24_pln3(srcPtr + srcIdx, src_smem_channel);
    }
    else
    {
        // *(uint2 *)src_smem_channel[0] = (uint2)0;
        // *(uint2 *)src_smem_channel[1] = (uint2)0;
        // *(uint2 *)src_smem_channel[2] = (uint2)0;
    }
    __syncthreads();
    //RGB to YCbCr
    // if(hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 1)
    //     printf("\n [Y  %d  %d  %d]  [X   %d ] [Height %d Width %d]",hipThreadIdx_y_channel.x,hipThreadIdx_y_channel.y,hipThreadIdx_y_channel.z,hipThreadIdx_x,roiTensorPtrSrc[id_z].xywhROI.roiHeight,roiTensorPtrSrc[id_z].xywhROI.roiWidth);
    
    YCbCr_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],&src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8]);
    __syncthreads();
    //Downsampling
    int CbCry = hipThreadIdx_y * 2;
    //if(CbCry < 32 && hipThreadIdx_x8 <  128)
    //{
        //16 threads process 32 rows containing Cb and Cr
        //16 + CbCry will cover 16 to 48 for TidxY 0 - 15
        verticalDownSampling(
                             &src_smem[16 + CbCry][hipThreadIdx_x8],
                             &src_smem[16 + CbCry + 1][hipThreadIdx_x8]);
                            //  &src_smem[32 + CbCry][hipThreadIdx_x8],
                            //  &src_smem[32 + CbCry + 1][hipThreadIdx_x8]);
        __syncthreads();
        //} 
    //if(hipThreadIdx_x8 <  128)  
        d_float8 CbCr;
        horizontalDownSampling(
                               &src_smem[16 + CbCry][hipThreadIdx_x16],
                               &src_smem[16 + CbCry][hipThreadIdx_x16 + 8],
                            //    &src_smem[32 + CbCry][hipThreadIdx_x16],
                            //    &src_smem[32 + CbCry][hipThreadIdx_x16 + 8],
                               &CbCr);
        if(CbCry < 16)
            *(d_float8*)&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8] = CbCr;
        if(CbCry > 16 )
            *(d_float8*)&src_smem[hipThreadIdx_y_channel.y][64 + hipThreadIdx_x8] = CbCr;
        __syncthreads();
        //Storing Cr beside Cb block
        // if(hipThreadIdx_y < 8 && hipThreadIdx_x < 8)
        //     src_smem[hipThreadIdx_y_channel.y][64 + hipThreadIdx_x8] = src_smem[8 + hipThreadIdx_y_channel.y][hipThreadIdx_x8] ;
        // // //As we are writing Cr into Cb place let all Cb complete before writing Cr
        //  __syncthreads();
        // //src_smem[hipThreadIdx_y_channel.z][hipThreadIdx_x8] = Cr;
        //Storing Cr beside Cb block
      //  *(d_float8*)&src_smem[hipThreadIdx_y_channel.y][64 + hipThreadIdx_x8] = Cr;


    
    //Fwd DCT
    //1D row wise DCT for Y channel
    dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],1,hipThreadIdx_x8,true);
    //1D row wise DCT for Cb and Cr channels but should only done for 8 rows but here being done for 16 rows
    if(hipThreadIdx_y < 8)
        dct_fwd_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],1,hipThreadIdx_x8,true);  
    // After row-wise DCT and synchronization
    __syncthreads();
    //Now for each column we should do DCT

    //So by now in smem first [0 to 16] x (16 x 8) has Y, [16 to 24] x (8 x 8) has Cb and [16 to 24] x (8 x 8) has Cr
    //We have 16 x 16 threads on X and Y dimension and 128 elements in each row
    // Each thread takes one column (total 128 threads for 128 columns)
    int col = hipThreadIdx_x * 16 + hipThreadIdx_y; 
    // Process all 128 columns 
    if (col < 128) { 
        // 1D DCT for First 8 rows (Y channel)   
        dct_fwd_8x8_1d(&src_smem[0][col], 128,col,false);
        // 1D DCT for Second 8 rows (Y channel)
        dct_fwd_8x8_1d(&src_smem[8][col], 128,col,false);
        // 1D DCT for 8 rows (Cb/Cr channels)
        dct_fwd_8x8_1d(&src_smem[16][col],128,col,false);
    }
    __syncthreads();
    //Quantization on each layer of Y Cb Cr
    //For every 8 elements row wise, we are taking respected row in the table and doing quantization
    //For Cb Cr
    //Here we can do this for only 8 threads in Y, here and in row wise DCT
/*    if(hipThreadIdx_y < 8){
        quantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);  
    }
    //For Y also
    quantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();
//INVERSE STEPS
    //Dequantization
    if(hipThreadIdx_y < 8){
        dequantize(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8], &CbCrtable[(hipThreadIdx_y % 8) * 8]);  
    }
    //For Y also
    dequantize(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8], &Ytable[(hipThreadIdx_y % 8) * 8]);
    __syncthreads();
 */  
    //Inverse DCT
    //1D row wise DCT for Y channel
    dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],1,hipThreadIdx_x8,true);
    //1D row wise DCT for Cb and Cr channels but should only done for 8 rows but here being done for 16 rows
    if(hipThreadIdx_y < 8)
        dct_inv_8x8_1d(&src_smem[hipThreadIdx_y_channel.y][hipThreadIdx_x8],1,hipThreadIdx_x8,true);  

    // After row-wise DCT and synchronization
    __syncthreads();
    //Now for each column we should do DCT
    //So by now in smem first [0 to 16] x (16 x 8) has Y, [16 to 24] x (8 x 8) has Cb and [16 to 24] x (8 x 8) has Cr
    //We have 16 x 16 threads on X and Y dimension and 128 elements in each row
    // Each thread takes one column (total 128 threads for 128 columns)
    // Process all 128 columns 
    if (col < 128) { 
        // 1D DCT for First 8 rows (Y channel)   
        dct_inv_8x8_1d(&src_smem[0][col], 128,col,false);
      
        // 1D DCT for Second 8 rows (Y channel)
        dct_inv_8x8_1d(&src_smem[8][col], 128,col,false);
        
        // 1D DCT for 8 rows (Cb/Cr channels)
        dct_inv_8x8_1d(&src_smem[16][col],128,col,false);

        printf("DCT INV");
    }
    __syncthreads();
    
     //RGB to YCbCr
    // CbCry = 16 + (hipThreadIdx_y/2) * 2;
    // int hipThreadIdx_x4 = hipThreadIdx_x >> 2;
    // RGB_hip_compute(&src_smem[hipThreadIdx_y_channel.x][hipThreadIdx_x8],&src_smem[CbCry][hipThreadIdx_x4],&src_smem[CbCry][64 + hipThreadIdx_x4],);
    // __syncthreads(); 
    
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

    int globalThreads_x = 1024;
    int globalThreads_y = 1024;
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
    hipMemcpy(gpuCbCrtable, gpuCbCrtable, 64 * sizeof(int), hipMemcpyHostToDevice);
    //globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
//correct the number ofthreads launched
    printf("\n Total number of threads in X %d Blocks %f", LOCAL_THREADS_X,ceil((float)globalThreads_x/LOCAL_THREADS_X) );
    printf("\n Total number of threads in Y %d Blocks %f", LOCAL_THREADS_Y,ceil((float)globalThreads_y/LOCAL_THREADS_Y));
    printf("\n Total number of threads in Z %d Blocks %f", LOCAL_THREADS_Z,ceil((float)globalThreads_z/LOCAL_THREADS_Z) );
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

    return RPP_SUCCESS;
}