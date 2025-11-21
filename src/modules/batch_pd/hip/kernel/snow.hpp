#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rpp_cpu_common.hpp"

__device__ __forceinline__ void snow_1GRAY_hip_compute(float *pixel, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    float lightness = *pixel;
    float lowerThreshold = 0.0f;
    float upperThreshold = 0.39215686f;
    float brightnessFactor = 2.5f;
    //Lighter the darken images
    if(lightness >= lowerThreshold && lightness <= upperThreshold && (*darkMode == 1))
        lightness = lightness * fmaf((brightnessFactor - 1.0f), (1.0f - (lightness - lowerThreshold) / (upperThreshold - lowerThreshold)), 1.0f);

    // Modify Lightness 
    if(lightness <= *snowThreshold)
        lightness = lightness * (*brightnessCoefficient);

    *pixel = lightness;

}
__device__ __forceinline__ void snow_1RGB_hip_compute(float *pixelR, float *pixelG, float *pixelB, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    // RGB to HSL
    float hue, sat, lightness;
    float rf, gf, bf, cmax, cmin, delta;
    float lowerThreshold = 0.0f;
    float upperThreshold = 0.39215686f;
    float brightnessFactor = 2.5f;

    rf = *pixelR;
    gf = *pixelG;
    bf = *pixelB;
    cmax = fmaxf(fmaxf(rf, gf), bf);
    cmin = fminf(fminf(rf, gf), bf);
    delta = cmax - cmin;
    hue = 0.0f;
    sat = 0.0f;
    lightness = (cmax + cmin) * 0.5f;
    if(delta != 0.0f)
    {
        //saturation calculation
        float mul = (lightness <= 0.5f) ? lightness : (1.0f - lightness);
        sat = delta / (mul * 2.0f);
        
        // hue calculation
        if(cmax == rf)
            hue = (gf - bf) / delta;
        else if(cmax == gf)
            hue = 2.0f + (bf - rf) / delta;
        else
            hue = 4.0f + (rf - gf) / delta;

        hue = hue * ONE_OVER_6;
    }

    //Lighter the darken images
    if(lightness >= lowerThreshold && lightness <= upperThreshold && (*darkMode == 1))
        lightness = lightness * fmaf((brightnessFactor - 1.0f), (1.0f - (lightness - lowerThreshold) / (upperThreshold - lowerThreshold)), 1.0f);

    // Modify L 
    if(lightness <= *snowThreshold && !((hue >= 0.514f && hue <= 0.63f) && (sat >= 0.196f) && (lightness >= 0.196f)))
        lightness = lightness * (*brightnessCoefficient);

    float4 xt_f4 = make_float4(
        6.0f * (hue - TWO_OVER_3),
        0.0f,
        6.0f *(1.0f - hue),
        0.0f
    );
    if(hue < TWO_OVER_3)
    {
        xt_f4.x = 0.0f;
        xt_f4.y = 6.0f * (TWO_OVER_3 - hue);
        xt_f4.z = 6.0f * (hue - ONE_OVER_3);
    }
    if(hue < ONE_OVER_3)
    {
        xt_f4.x = 6.0f * (ONE_OVER_3 - hue);
        xt_f4.y = 6.0f * hue;
        xt_f4.z = 0.0f;
    }
    xt_f4.x = fminf(xt_f4.x, 1.0f);
    xt_f4.y = fminf(xt_f4.y, 1.0f);
    xt_f4.z = fminf(xt_f4.z, 1.0f);

    float sat2 = 2.0f * sat;
    float satinv = 1.0f - sat;
    float luminv = 1.0f - lightness;
    float lum2m1 = (2.0f * lightness) - 1.0f;
    float4 ct_f4 = (sat2 * xt_f4) + satinv;

    float4 rgb_f4;
    if(lightness >= 0.5f)
        rgb_f4 = (luminv * ct_f4) + lum2m1;
    else
        rgb_f4 = lightness * ct_f4;
    
    *pixelR = rgb_f4.x;
    *pixelG = rgb_f4.y;
    *pixelB = rgb_f4.z;
}

__device__ __forceinline__ void snow_8RGB_hip_compute(d_float24 *pix_f24, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    snow_1RGB_hip_compute(&(pix_f24->f1[0]), &(pix_f24->f1[ 8]), &(pix_f24->f1[16]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[1]), &(pix_f24->f1[ 9]), &(pix_f24->f1[17]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[2]), &(pix_f24->f1[10]), &(pix_f24->f1[18]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[3]), &(pix_f24->f1[11]), &(pix_f24->f1[19]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[4]), &(pix_f24->f1[12]), &(pix_f24->f1[20]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[5]), &(pix_f24->f1[13]), &(pix_f24->f1[21]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[6]), &(pix_f24->f1[14]), &(pix_f24->f1[22]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1RGB_hip_compute(&(pix_f24->f1[7]), &(pix_f24->f1[15]), &(pix_f24->f1[23]), brightnessCoefficient, snowThreshold, darkMode);
}

__device__ __forceinline__ void snow_8GRAY_hip_compute(d_float8 *pix_f8, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    snow_1GRAY_hip_compute(&(pix_f8->f1[0]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[1]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[2]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[3]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[4]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[5]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[6]), brightnessCoefficient, snowThreshold, darkMode);
    snow_1GRAY_hip_compute(&(pix_f8->f1[7]), brightnessCoefficient, snowThreshold, darkMode);
}

__device__ __forceinline__ void snow_hip_compute(uchar *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    normalizer_f4 = static_cast<float4>(255.0f);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(uchar *srcPtr, d_float8 *pix_f8, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_multiply8_const(pix_f8, pix_f8, normalizer_f4);
    snow_8GRAY_hip_compute(pix_f8, brightnessCoefficient, snowThreshold, darkMode);
    normalizer_f4 = static_cast<float4>(255.0f);
    rpp_hip_math_multiply8_const(pix_f8, pix_f8, normalizer_f4);
    rpp_hip_pixel_check_0to255(pix_f8);
}
__device__ __forceinline__ void snow_hip_compute(float *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(float *srcPtr, d_float8 *pix_f8, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    snow_8GRAY_hip_compute(pix_f8, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pixel_check_0to1(pix_f8);
}
__device__ __forceinline__ void snow_hip_compute(half *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pixel_check_0to1(pix_f24);
}
__device__ __forceinline__ void snow_hip_compute(half *srcPtr, d_float8 *pix_f8, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    snow_8GRAY_hip_compute(pix_f8, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pixel_check_0to1(pix_f8);
}
__device__ __forceinline__ void snow_hip_compute(schar *srcPtr, d_float24 *pix_f24, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    float4 i8Offset_f4 = static_cast<float4>(128.0f);
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_add24_const(pix_f24, pix_f24, i8Offset_f4);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    snow_8RGB_hip_compute(pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    normalizer_f4 = static_cast<float4>(255.0f);
    rpp_hip_math_multiply24_const(pix_f24, pix_f24, normalizer_f4);
    rpp_hip_pixel_check_0to255(pix_f24);
    rpp_hip_math_subtract24_const(pix_f24, pix_f24, i8Offset_f4);
}
__device__ __forceinline__ void snow_hip_compute(schar *srcPtr, d_float8 *pix_f8, float *brightnessCoefficient, float *snowThreshold, int *darkMode)
{
    float4 i8Offset_f4 = static_cast<float4>(128.0f);
    float4 normalizer_f4 = static_cast<float4>(ONE_OVER_255);
    rpp_hip_math_add8_const(pix_f8, pix_f8, i8Offset_f4);
    rpp_hip_math_multiply8_const(pix_f8, pix_f8, normalizer_f4);
    snow_8GRAY_hip_compute(pix_f8, brightnessCoefficient, snowThreshold, darkMode);
    normalizer_f4 = static_cast<float4>(255.0f);
    rpp_hip_math_multiply8_const(pix_f8, pix_f8, normalizer_f4);
    rpp_hip_pixel_check_0to255(pix_f8);
    rpp_hip_math_subtract8_const(pix_f8, pix_f8, i8Offset_f4);
}

template <typename T>
__global__ void snow_pkd_hip_tensor(T *srcPtr,
                                    uint2 srcStridesNH,
                                    T *dstPtr,
                                    uint2 dstStridesNH,
                                    float *brightnessCoefficient,
                                    float *snowThreshold,
                                    int *darkMode,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void snow_pln_hip_tensor(T *srcPtr,
                                    uint3 srcStridesNCH,
                                    T *dstPtr,
                                    uint3 dstStridesNCH,
                                    int channelsDst,
                                    float *brightnessCoefficient,
                                    float *snowThreshold,
                                    int *darkMode,
                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    if (channelsDst == 3)
    {
        d_float24 pix_f24;
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
        snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowThreshold, darkMode);
        rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
    }
    else
    {
        d_float8 pix_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        snow_hip_compute(srcPtr, &pix_f8, brightnessCoefficient, snowThreshold, darkMode);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
    }
}

template <typename T>
__global__ void snow_pkd3_pln3_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          float *brightnessCoefficient,
                                          float *snowThreshold,
                                          int *darkMode,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void snow_pln3_pkd3_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *brightnessCoefficient,
                                          float *snowThreshold,
                                          int *darkMode,
                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    snow_hip_compute(srcPtr, &pix_f24, brightnessCoefficient, snowThreshold, darkMode);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_snow_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               T *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *brightnessCoefficient,
                               Rpp32f *snowThreshold,
                               Rpp32s *darkMode,
                               RpptROIPtr roiTensorPtrSrc,
                               RpptRoiType roiType,
                               rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = dstDescPtr->n;
    *snowThreshold = ((*snowThreshold * (127.5f)) + 85.0f) * ONE_OVER_255;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(snow_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           brightnessCoefficient,
                           snowThreshold,
                           darkMode,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(snow_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           brightnessCoefficient,
                           snowThreshold,
                           darkMode,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(snow_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               brightnessCoefficient,
                               snowThreshold,
                               darkMode,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(snow_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               brightnessCoefficient,
                               snowThreshold,
                               darkMode,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}