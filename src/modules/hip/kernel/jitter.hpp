#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rng_seed_stream.hpp"

/*__device__ void jitter_hip_compute(uchar *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *kernelSize_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] * *alpha_f4 + *beta_f4;
    dst_f8->f4[1] = src_f8->f4[1] * *alpha_f4 + *beta_f4;
}

__device__ void jitter_hip_compute(float *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *kernelSize_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] * *alpha_f4 + *beta_f4 * (float4) ONE_OVER_255;
    dst_f8->f4[1] = src_f8->f4[1] * *alpha_f4 + *beta_f4 * (float4) ONE_OVER_255;
}

__device__ void jitter_hip_compute(signed char *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *kernelSize_f4)
{
    dst_f8->f4[0] = rpp_hip_pixel_check_0to255((src_f8->f4[0] + (float4)128) * *alpha_f4 + *beta_f4) - (float4)128;
    dst_f8->f4[1] = rpp_hip_pixel_check_0to255((src_f8->f4[1] + (float4)128) * *alpha_f4 + *beta_f4) - (float4)128;
}

__device__ void jitter_hip_compute(half *srcPtr, d_float8 *src_f8, d_float8 *dst_f8, float4 *kernelSize_f4)
{
    dst_f8->f4[0] = src_f8->f4[0] * *alpha_f4 + *beta_f4 * (float4) ONE_OVER_255;
    dst_f8->f4[1] = src_f8->f4[1] * *alpha_f4 + *beta_f4 * (float4) ONE_OVER_255;
}*/

__device__ unsigned int xorshift(int idx)
{
    unsigned int x = 123456789;
    unsigned int w = 88675123;
    unsigned int seed = x + idx;
    unsigned int t = seed ^ (seed << 11);
    unsigned int res = w ^ (w >> 19) ^ (t ^(t >> 8));
    return res;
}

__device__ void jitter_8_adjusted_input_hip_compute(uchar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)ONE_OVER_255); }
__device__ void jitter_8_adjusted_input_hip_compute(float *srcPtr, d_float8 *pix_f8) { }
__device__ void jitter_8_adjusted_input_hip_compute(schar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_add8_const(pix_f8, pix_f8, (float4)128.0f); rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)ONE_OVER_255); }
__device__ void jitter_8_adjusted_input_hip_compute(half *srcPtr, d_float8 *pix_f8) { }

__device__ void jitter_24_adjusted_input_hip_compute(uchar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }
__device__ void jitter_24_adjusted_input_hip_compute(float *srcPtr, d_float24 *pix_f24) { }
__device__ void jitter_24_adjusted_input_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_add24_const(pix_f24, pix_f24, (float4)128.0f); rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }
__device__ void jitter_24_adjusted_input_hip_compute(half *srcPtr, d_float24 *pix_f24) { }

__device__ void jitter_8_adjusted_output_hip_compute(uchar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)255.0f); }
__device__ void jitter_8_adjusted_output_hip_compute(float *srcPtr, d_float8 *pix_f8) { }
__device__ void jitter_8_adjusted_output_hip_compute(schar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)255.0f); rpp_hip_math_subtract8_const(pix_f8, pix_f8, (float4)128.0f); }
__device__ void jitter_8_adjusted_output_hip_compute(half *srcPtr, d_float8 *pix_f8) { }

__device__ void jitter_24_adjusted_output_hip_compute(uchar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); }
__device__ void jitter_24_adjusted_output_hip_compute(float *srcPtr, d_float24 *pix_f24) { }
__device__ void jitter_24_adjusted_output_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); rpp_hip_math_subtract24_const(pix_f24, pix_f24, (float4)128.0f); }
__device__ void jitter_24_adjusted_output_hip_compute(half *srcPtr, d_float24 *pix_f24) { }


__device__ void jitter_24_hip_compute(d_float24 *pix_f24, RpptXorwowStateBoxMuller *xorwowState, uint kernelSize, uint bound)
{
    d_float24 rngVals_f24, rngVals_f24_1;
    rpp_hip_rng_8_jitter_f32(&rngVals_f24.f8[0], xorwowState);
    rpp_hip_rng_8_jitter_f32(&rngVals_f24.f8[1], xorwowState);
    rpp_hip_rng_8_jitter_f32(&rngVals_f24.f8[2], xorwowState);
    rpp_hip_math_multiply24_const(&rngVals_f24, &rngVals_f24, (float4)kernelSize);
    
    rpp_hip_rng_8_jitter_f32(&rngVals_f24_1.f8[0], xorwowState);
    rpp_hip_rng_8_jitter_f32(&rngVals_f24_1.f8[1], xorwowState);
    rpp_hip_rng_8_jitter_f32(&rngVals_f24_1.f8[2], xorwowState);
    rpp_hip_math_multiply24_const(&rngVals_f24_1, &rngVals_f24_1, (float4)kernelSize);

    rpp_hip_math_add24(pix_f24, &rngVals_f24, pix_f24);
    rpp_hip_pixel_check_0to255(pix_f24);
}

template <typename T>
__global__ void jitter_pkd_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      int *kernelSize,
                                      RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                      uint *xorwowSeedStream,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) /** 8*/;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint kernelSize_u1 = (uint)kernelSize[id_z];

    //d_float8 src_f8, dst_f8;

    uint bound = (kernelSize_u1 - 1) / 2;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight - bound) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth - bound))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x); /*+ ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x * 3)*/
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    /*RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    jitter_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
    jitter_24_hip_compute(&pix_f24, &xorwowState, kernelSize_u1, bound);
    jitter_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);*/

    int nhx = xorshift(srcIdx) % (kernelSize_u1);
    int nhy = xorshift(srcIdx) % (kernelSize_u1);
    if((id_y - bound + nhy) >= 0 && (id_y - bound + nhy) <= roiTensorPtrSrc[id_z].xywhROI.roiHeight - 1 && (id_x - bound + nhx) >= 0 && (id_x - bound + nhx) <= roiTensorPtrSrc[id_z].xywhROI.roiWidth - 1)
    {
        int idx = ((((id_y - bound + nhx) * srcStridesNH.y) + (id_x - bound + nhy)) * 3);
        printf("srcIdx:%d\t dstIdx:%d\t idx:%d\n", srcIdx, dstIdx, idx);
        srcIdx+=idx;
        dstPtr[dstIdx]=srcPtr[srcIdx];
        dstPtr[dstIdx+1]=srcPtr[srcIdx+1];
        dstPtr[dstIdx+2]=srcPtr[srcIdx+2];
    }

    //rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f8);
    //jitter_hip_compute(srcPtr, &src_f8, &dst_f8, &kernelSize_f4);
    //rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

/*template <typename T>
__global__ void jitter_pln_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      int channelsDst,
                                      float *alpha,
                                      float *beta,
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

    float4 alpha_f4 = (float4)(alpha[id_z]);
    float4 beta_f4 = (float4)(beta[id_z]);

    d_float8 src_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    jitter_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        jitter_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        jitter_hip_compute(srcPtr, &src_f8, &dst_f8, &alpha_f4, &beta_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void jitter_pkd3_pln3_tensor(T *srcPtr,
                                            uint2 srcStridesNH,
                                            T *dstPtr,
                                            uint3 dstStridesNCH,
                                            float *alpha,
                                            float *beta,
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

    float4 alpha_f4 = (float4)alpha[id_z];
    float4 beta_f4 = (float4)beta[id_z];

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    jitter_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &alpha_f4, &beta_f4);
    jitter_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &alpha_f4, &beta_f4);
    jitter_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &alpha_f4, &beta_f4);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void jitter_pln3_pkd3_tensor(T *srcPtr,
                                            uint3 srcStridesNCH,
                                            T *dstPtr,
                                            uint2 dstStridesNH,
                                            float *alpha,
                                            float *beta,
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

    float4 alpha_f4 = (float4)(alpha[id_z]);
    float4 beta_f4 = (float4)(beta[id_z]);

    d_float24 src_f24, dst_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    jitter_hip_compute(srcPtr, &src_f24.f8[0], &dst_f24.f8[0], &alpha_f4, &beta_f4);
    jitter_hip_compute(srcPtr, &src_f24.f8[1], &dst_f24.f8[1], &alpha_f4, &beta_f4);
    jitter_hip_compute(srcPtr, &src_f24.f8[2], &dst_f24.f8[2], &alpha_f4, &beta_f4);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}*/

template <typename T>
RppStatus hip_exec_jitter_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) /*>> 3*/;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    hipMemcpy(xorwowSeedStream, rngSeedStream4050, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        std::cerr << "Kernel Launch\n";
        // globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(jitter_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    /*else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(jitter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(jitter_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(jitter_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }*/

    return RPP_SUCCESS;
}
