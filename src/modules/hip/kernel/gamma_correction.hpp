#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void gamma_correction_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    dst_f8->f4[0] = make_float4(gammaLUT[(int) src_f8->f1[0]], gammaLUT[(int) src_f8->f1[1]], gammaLUT[(int) src_f8->f1[2]], gammaLUT[(int) src_f8->f1[3]]) * (float4) 255.0;
    dst_f8->f4[1] = make_float4(gammaLUT[(int) src_f8->f1[4]], gammaLUT[(int) src_f8->f1[5]], gammaLUT[(int) src_f8->f1[6]], gammaLUT[(int) src_f8->f1[7]]) * (float4) 255.0;
}

__device__ void gamma_correction_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    d_float8 srcNorm_f8;
    srcNorm_f8.f4[0] = src_f8->f4[0] * (float4) 255.0;
    srcNorm_f8.f4[1] = src_f8->f4[1] * (float4) 255.0;

    dst_f8->f4[0] = make_float4(gammaLUT[(int) srcNorm_f8.f1[0]], gammaLUT[(int) srcNorm_f8.f1[1]], gammaLUT[(int) srcNorm_f8.f1[2]], gammaLUT[(int) srcNorm_f8.f1[3]]);
    dst_f8->f4[1] = make_float4(gammaLUT[(int) srcNorm_f8.f1[4]], gammaLUT[(int) srcNorm_f8.f1[5]], gammaLUT[(int) srcNorm_f8.f1[6]], gammaLUT[(int) srcNorm_f8.f1[7]]);
}

__device__ void gamma_correction_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    d_float8 srcNorm_f8;
    srcNorm_f8.f4[0] = src_f8->f4[0] + (float4)128;
    srcNorm_f8.f4[1] = src_f8->f4[1] + (float4)128;

    dst_f8->f4[0] = (make_float4(gammaLUT[(int) srcNorm_f8.f1[0]], gammaLUT[(int) srcNorm_f8.f1[1]], gammaLUT[(int) srcNorm_f8.f1[2]], gammaLUT[(int) srcNorm_f8.f1[3]]) * (float4) 255) - (float4) 128;
    dst_f8->f4[1] = (make_float4(gammaLUT[(int) srcNorm_f8.f1[4]], gammaLUT[(int) srcNorm_f8.f1[5]], gammaLUT[(int) srcNorm_f8.f1[6]], gammaLUT[(int) srcNorm_f8.f1[7]]) * (float4) 255) - (float4) 128;
}

__device__ void gamma_correction_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    d_float8 srcNorm_f8;
    srcNorm_f8.f4[0] = src_f8->f4[0] * (float4) 255.0;
    srcNorm_f8.f4[1] = src_f8->f4[1] * (float4) 255.0;

    dst_f8->f4[0] = make_float4(gammaLUT[(int) srcNorm_f8.f1[0]], gammaLUT[(int) srcNorm_f8.f1[1]], gammaLUT[(int) srcNorm_f8.f1[2]], gammaLUT[(int) srcNorm_f8.f1[3]]);
    dst_f8->f4[1] = make_float4(gammaLUT[(int) srcNorm_f8.f1[4]], gammaLUT[(int) srcNorm_f8.f1[5]], gammaLUT[(int) srcNorm_f8.f1[6]], gammaLUT[(int) srcNorm_f8.f1[7]]);
}

template <typename T>
__global__ void gamma_correction_pkd_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            float *gammaLUT,
                                            RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth * 3))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    uint gammaLutIdx = (256 * id_z);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

template <typename T>
__global__ void gamma_correction_pln_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            int channelsDst,
                                            float *gammaLUT,
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
    uint gammaLutIdx = (256 * id_z);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void gamma_correction_pkd3_pln3_tensor(T *srcPtr,
                                                  uint2 srcStridesNH,
                                                  T *dstPtr,
                                                  uint3 dstStridesNCH,
                                                  float *gammaLUT,
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
    uint gammaLutIdx = (256 * id_z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    gamma_correction_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &gammaLUT[gammaLutIdx]);
    gamma_correction_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &gammaLUT[gammaLutIdx]);
    gamma_correction_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void gamma_correction_pln3_pkd3_tensor(T *srcPtr,
                                                  uint3 srcStridesNCH,
                                                  T *dstPtr,
                                                  uint2 dstStridesNH,
                                                  float *gammaLUT,
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
    uint gammaLutIdx = (256 * id_z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    gamma_correction_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &gammaLUT[gammaLutIdx]);
    gamma_correction_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &gammaLUT[gammaLutIdx]);
    gamma_correction_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

__global__ void gamma_correction_lut_compute(float *gammaLUT,
                                             float *gamma)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if (id_x >= 249)
    {
        return;
    }

    uint gammaLutIdx = (256 * id_y) + id_x;
    uint gammaIdx = id_y;

    d_float8 gammaLUT_f8;
    d_float8_s *gammaLUTPtr_f8;
    gammaLUTPtr_f8 = (d_float8_s *)&gammaLUT[gammaLutIdx];
    *(d_float8_s *)&gammaLUT_f8 = *gammaLUTPtr_f8;

    float4 inv255_f4 = (float4) ONE_OVER_255;
    d_float8 pixVal_f8;

    pixVal_f8.f4[0] = make_float4(id_x, id_x + 1, id_x + 2, id_x + 3);
    pixVal_f8.f4[1] = make_float4(id_x + 4, id_x + 5, id_x + 6, id_x + 7);

    gammaLUT_f8.f4[0] = pixVal_f8.f4[0] * inv255_f4;
    gammaLUT_f8.f4[1] = pixVal_f8.f4[1] * inv255_f4;

    gammaLUT_f8.f4[0] = make_float4(powf(gammaLUT_f8.f1[0], gamma[gammaIdx]),
                                    powf(gammaLUT_f8.f1[1], gamma[gammaIdx]),
                                    powf(gammaLUT_f8.f1[2], gamma[gammaIdx]),
                                    powf(gammaLUT_f8.f1[3], gamma[gammaIdx]));
    gammaLUT_f8.f4[1] = make_float4(powf(gammaLUT_f8.f1[4], gamma[gammaIdx]),
                                    powf(gammaLUT_f8.f1[5], gamma[gammaIdx]),
                                    powf(gammaLUT_f8.f1[6], gamma[gammaIdx]),
                                    powf(gammaLUT_f8.f1[7], gamma[gammaIdx]));

    *gammaLUTPtr_f8 = *(d_float8_s *)&gammaLUT_f8;
}

template <typename T>
RppStatus hip_exec_gamma_correction_tensor(T *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int globalThreads_x = (256 + 7) >> 3;
    int globalThreads_y = handle.GetBatchSize();
    int globalThreads_z = 1;

    float *gammaLUT;
    hipMalloc(&gammaLUT, 256 * handle.GetBatchSize() * sizeof(Rpp32f));

    hipLaunchKernelGGL(gamma_correction_lut_compute,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       gammaLUT,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);

    
    
    
    globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    globalThreads_y = dstDescPtr->h;
    globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(gamma_correction_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           gammaLUT,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(gamma_correction_pln_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           gammaLUT,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(gamma_correction_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               gammaLUT,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(gamma_correction_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               gammaLUT,
                               roiTensorPtrSrc);
        }
    }

    hipFree(&gammaLUT);

    return RPP_SUCCESS;
}
