#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void gamma_correction_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    dst_f8->x = make_float4(gammaLUT[(int) src_f8->x.x], gammaLUT[(int) src_f8->x.y], gammaLUT[(int) src_f8->x.z], gammaLUT[(int) src_f8->x.w]) * (float4) 255.0;
    dst_f8->y = make_float4(gammaLUT[(int) src_f8->y.x], gammaLUT[(int) src_f8->y.y], gammaLUT[(int) src_f8->y.z], gammaLUT[(int) src_f8->y.w]) * (float4) 255.0;
}

__device__ void gamma_correction_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    d_float8 src_f8_norm;
    src_f8_norm.x = src_f8->x * (float4) 255.0;
    src_f8_norm.y = src_f8->y * (float4) 255.0;

    dst_f8->x = make_float4(gammaLUT[(int) src_f8_norm.x.x], gammaLUT[(int) src_f8_norm.x.y], gammaLUT[(int) src_f8_norm.x.z], gammaLUT[(int) src_f8_norm.x.w]);
    dst_f8->y = make_float4(gammaLUT[(int) src_f8_norm.y.x], gammaLUT[(int) src_f8_norm.y.y], gammaLUT[(int) src_f8_norm.y.z], gammaLUT[(int) src_f8_norm.y.w]);
}

__device__ void gamma_correction_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    d_float8 src_f8_norm;
    src_f8_norm.x = src_f8->x + (float4)128;
    src_f8_norm.y = src_f8->y + (float4)128;

    dst_f8->x = (make_float4(gammaLUT[(int) src_f8_norm.x.x], gammaLUT[(int) src_f8_norm.x.y], gammaLUT[(int) src_f8_norm.x.z], gammaLUT[(int) src_f8_norm.x.w]) * (float4) 255) - (float4) 128;
    dst_f8->y = (make_float4(gammaLUT[(int) src_f8_norm.y.x], gammaLUT[(int) src_f8_norm.y.y], gammaLUT[(int) src_f8_norm.y.z], gammaLUT[(int) src_f8_norm.y.w]) * (float4) 255) - (float4) 128;
}

__device__ void gamma_correction_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float *gammaLUT)
{
    d_float8 src_f8_norm;
    src_f8_norm.x = src_f8->x * (float4) 255.0;
    src_f8_norm.y = src_f8->y * (float4) 255.0;

    dst_f8->x = make_float4(gammaLUT[(int) src_f8_norm.x.x], gammaLUT[(int) src_f8_norm.x.y], gammaLUT[(int) src_f8_norm.x.z], gammaLUT[(int) src_f8_norm.x.w]);
    dst_f8->y = make_float4(gammaLUT[(int) src_f8_norm.y.x], gammaLUT[(int) src_f8_norm.y.y], gammaLUT[(int) src_f8_norm.y.z], gammaLUT[(int) src_f8_norm.y.w]);
}

template <typename T>
__global__ void gamma_correction_pkd_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int hStrideDst,
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

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;
    uint gammaLutIdx = (256 * id_z);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
}

template <typename T>
__global__ void gamma_correction_pln_tensor(T *srcPtr,
                                            int nStrideSrc,
                                            int cStrideSrc,
                                            int hStrideSrc,
                                            T *dstPtr,
                                            int nStrideDst,
                                            int cStrideDst,
                                            int hStrideDst,
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

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;
    uint gammaLutIdx = (256 * id_z);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
    gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
        gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);

        srcIdx += cStrideSrc;
        dstIdx += cStrideDst;

        rpp_hip_load8_and_unpack_to_float8(srcPtr, srcIdx, &src_f8);
        gamma_correction_hip_compute(srcPtr, &src_f8, &dst_f8, &gammaLUT[gammaLutIdx]);
        rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void gamma_correction_pkd3_pln3_tensor(T *srcPtr,
                                                  int nStrideSrc,
                                                  int hStrideSrc,
                                                  T *dstPtr,
                                                  int nStrideDst,
                                                  int cStrideDst,
                                                  int hStrideDst,
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

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x;
    uint gammaLutIdx = (256 * id_z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr, srcIdx, &src_f24);
    gamma_correction_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.x);

    dstIdx += cStrideDst;

    gamma_correction_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.y);

    dstIdx += cStrideDst;

    gamma_correction_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float8_and_store8(dstPtr, dstIdx, &dst_f24.z);
}

template <typename T>
__global__ void gamma_correction_pln3_pkd3_tensor(T *srcPtr,
                                                  int nStrideSrc,
                                                  int cStrideSrc,
                                                  int hStrideSrc,
                                                  T *dstPtr,
                                                  int nStrideDst,
                                                  int hStrideDst,
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

    uint srcIdx = (id_z * nStrideSrc) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * hStrideSrc) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * nStrideDst) + (id_y * hStrideDst) + id_x * 3;
    uint gammaLutIdx = (256 * id_z);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr, srcIdx, cStrideSrc, &src_f24);
    gamma_correction_hip_compute(srcPtr, &src_f24.x, &dst_f24.x, &gammaLUT[gammaLutIdx]);
    gamma_correction_hip_compute(srcPtr, &src_f24.y, &dst_f24.y, &gammaLUT[gammaLutIdx]);
    gamma_correction_hip_compute(srcPtr, &src_f24.z, &dst_f24.z, &gammaLUT[gammaLutIdx]);
    rpp_hip_pack_float24_and_store24(dstPtr, dstIdx, &dst_f24);
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

    d_float8 *gammaLUT_f8;
    gammaLUT_f8 = (d_float8 *)&gammaLUT[gammaLutIdx];

    float4 inv255_f4 = (float4) 0.0039216f;
    d_float8 pixVal_f8;

    pixVal_f8.x = make_float4(id_x, id_x + 1, id_x + 2, id_x + 3);
    pixVal_f8.y = make_float4(id_x + 4, id_x + 5, id_x + 6, id_x + 7);

    gammaLUT_f8->x = pixVal_f8.x * inv255_f4;
    gammaLUT_f8->y = pixVal_f8.y * inv255_f4;

    gammaLUT_f8->x = make_float4(powf(gammaLUT_f8->x.x, gamma[gammaIdx]),
                                 powf(gammaLUT_f8->x.y, gamma[gammaIdx]),
                                 powf(gammaLUT_f8->x.z, gamma[gammaIdx]),
                                 powf(gammaLUT_f8->x.w, gamma[gammaIdx]));
    gammaLUT_f8->y = make_float4(powf(gammaLUT_f8->y.x, gamma[gammaIdx]),
                                 powf(gammaLUT_f8->y.y, gamma[gammaIdx]),
                                 powf(gammaLUT_f8->y.z, gamma[gammaIdx]),
                                 powf(gammaLUT_f8->y.w, gamma[gammaIdx]));
}

template <typename T>
RppStatus hip_exec_gamma_correction_tensor(T *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           RpptROIPtr roiTensorPtrSrc,
                                           rpp::Handle& handle)
{
    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = (256 + 7) >> 3;
    int globalThreads_y = handle.GetBatchSize();
    int globalThreads_z = 1;

    float *gammaLUT;
    hipMalloc(&gammaLUT, 256 * handle.GetBatchSize() * sizeof(Rpp32f));

    hipLaunchKernelGGL(gamma_correction_lut_compute,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       gammaLUT,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);

    localThreads_x = 16;
    localThreads_y = 16;
    localThreads_z = 1;
    globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    globalThreads_y = dstDescPtr->h;
    globalThreads_z = handle.GetBatchSize();

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(gamma_correction_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           srcDescPtr->strides.nStride,
                           srcDescPtr->strides.hStride,
                           dstPtr,
                           dstDescPtr->strides.nStride,
                           dstDescPtr->strides.hStride,
                           gammaLUT,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(gamma_correction_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
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
                           gammaLUT,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(gamma_correction_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.cStride,
                               dstDescPtr->strides.hStride,
                               gammaLUT,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(gamma_correction_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               srcDescPtr->strides.nStride,
                               srcDescPtr->strides.cStride,
                               srcDescPtr->strides.hStride,
                               dstPtr,
                               dstDescPtr->strides.nStride,
                               dstDescPtr->strides.hStride,
                               gammaLUT,
                               roiTensorPtrSrc);
        }
    }

    hipFree(&gammaLUT);

    return RPP_SUCCESS;
}
