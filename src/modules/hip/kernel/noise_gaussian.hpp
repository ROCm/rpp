#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rng_seed_stream.hpp"

__device__ void gaussian_noise_8_hip_compute(d_float8 *pix_f8, RpptXorwowStateBoxMuller *xorwowState, float mean, float stdDev)
{
    d_float8 rngVals_f8, pixSqrt_f8;
    rpp_hip_rng_8_gaussian_f32(&rngVals_f8, xorwowState);
    rpp_hip_math_multiply8_const(&rngVals_f8, &rngVals_f8, (float4)stdDev);
    rpp_hip_math_add8_const(&rngVals_f8, &rngVals_f8, (float4)mean);

    rpp_hip_math_sqrt8(pix_f8, &pixSqrt_f8);
    rpp_hip_math_multiply8(&pixSqrt_f8, &rngVals_f8, &rngVals_f8);
    rpp_hip_math_add8(pix_f8, &rngVals_f8, pix_f8);
    rpp_hip_pixel_check_0to1(pix_f8);
}

__device__ void gaussian_noise_24_hip_compute(d_float24 *pix_f24, RpptXorwowStateBoxMuller *xorwowState, float mean, float stdDev)
{
    d_float24 rngVals_f24, pixSqrt_f24;
    rpp_hip_rng_8_gaussian_f32(&rngVals_f24.f8[0], xorwowState);
    rpp_hip_rng_8_gaussian_f32(&rngVals_f24.f8[1], xorwowState);
    rpp_hip_rng_8_gaussian_f32(&rngVals_f24.f8[2], xorwowState);
    rpp_hip_math_multiply24_const(&rngVals_f24, &rngVals_f24, (float4)stdDev);
    rpp_hip_math_add24_const(&rngVals_f24, &rngVals_f24, (float4)mean);

    rpp_hip_math_sqrt24(pix_f24, &pixSqrt_f24);
    rpp_hip_math_multiply24(&pixSqrt_f24, &rngVals_f24, &rngVals_f24);
    rpp_hip_math_add24(pix_f24, &rngVals_f24, pix_f24);
    rpp_hip_pixel_check_0to1(pix_f24);
}

__device__ void gaussian_noise_8_adjusted_input_hip_compute(uchar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)ONE_OVER_255); }
__device__ void gaussian_noise_8_adjusted_input_hip_compute(float *srcPtr, d_float8 *pix_f8) { }
__device__ void gaussian_noise_8_adjusted_input_hip_compute(schar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_add8_const(pix_f8, pix_f8, (float4)128.0f); rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)ONE_OVER_255); }
__device__ void gaussian_noise_8_adjusted_input_hip_compute(half *srcPtr, d_float8 *pix_f8) { }

__device__ void gaussian_noise_24_adjusted_input_hip_compute(uchar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }
__device__ void gaussian_noise_24_adjusted_input_hip_compute(float *srcPtr, d_float24 *pix_f24) { }
__device__ void gaussian_noise_24_adjusted_input_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_add24_const(pix_f24, pix_f24, (float4)128.0f); rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)ONE_OVER_255); }
__device__ void gaussian_noise_24_adjusted_input_hip_compute(half *srcPtr, d_float24 *pix_f24) { }

__device__ void gaussian_noise_8_adjusted_output_hip_compute(uchar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)255.0f); }
__device__ void gaussian_noise_8_adjusted_output_hip_compute(float *srcPtr, d_float8 *pix_f8) { }
__device__ void gaussian_noise_8_adjusted_output_hip_compute(schar *srcPtr, d_float8 *pix_f8) { rpp_hip_math_multiply8_const(pix_f8, pix_f8, (float4)255.0f); rpp_hip_math_subtract8_const(pix_f8, pix_f8, (float4)128.0f); }
__device__ void gaussian_noise_8_adjusted_output_hip_compute(half *srcPtr, d_float8 *pix_f8) { }

__device__ void gaussian_noise_24_adjusted_output_hip_compute(uchar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); }
__device__ void gaussian_noise_24_adjusted_output_hip_compute(float *srcPtr, d_float24 *pix_f24) { }
__device__ void gaussian_noise_24_adjusted_output_hip_compute(schar *srcPtr, d_float24 *pix_f24) { rpp_hip_math_multiply24_const(pix_f24, pix_f24, (float4)255.0f); rpp_hip_math_subtract24_const(pix_f24, pix_f24, (float4)128.0f); }
__device__ void gaussian_noise_24_adjusted_output_hip_compute(half *srcPtr, d_float24 *pix_f24) { }

template <typename T>
__global__ void gaussian_noise_pkd_hip_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          float *meanTensor,
                                          float *stdDevTensor,
                                          RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                          uint *xorwowSeedStream,
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
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float mean = meanTensor[id_z];
    float stdDev = stdDevTensor[id_z];

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    gaussian_noise_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
    gaussian_noise_24_hip_compute(&pix_f24, &xorwowState, mean, stdDev);
    gaussian_noise_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void gaussian_noise_pln_hip_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          int channelsDst,
                                          float *meanTensor,
                                          float *stdDevTensor,
                                          RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                          uint *xorwowSeedStream,
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
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float mean = meanTensor[id_z];
    float stdDev = stdDevTensor[id_z];

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float8 pix_f8;

    rpp_hip_rng_xorwow_state_update(&xorwowState);
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
    gaussian_noise_8_adjusted_input_hip_compute(srcPtr, &pix_f8);
    gaussian_noise_8_hip_compute(&pix_f8, &xorwowState, mean, stdDev);
    gaussian_noise_8_adjusted_output_hip_compute(srcPtr, &pix_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        gaussian_noise_8_adjusted_input_hip_compute(srcPtr, &pix_f8);
        gaussian_noise_8_hip_compute(&pix_f8, &xorwowState, mean, stdDev);
        gaussian_noise_8_adjusted_output_hip_compute(srcPtr, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &pix_f8);
        gaussian_noise_8_adjusted_input_hip_compute(srcPtr, &pix_f8);
        gaussian_noise_8_hip_compute(&pix_f8, &xorwowState, mean, stdDev);
        gaussian_noise_8_adjusted_output_hip_compute(srcPtr, &pix_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &pix_f8);
    }
}

template <typename T>
__global__ void gaussian_noise_pkd3_pln3_hip_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                float *meanTensor,
                                                float *stdDevTensor,
                                                RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                uint *xorwowSeedStream,
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
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float mean = meanTensor[id_z];
    float stdDev = stdDevTensor[id_z];

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float24 pix_f24;

    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);
    gaussian_noise_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
    gaussian_noise_24_hip_compute(&pix_f24, &xorwowState, mean, stdDev);
    gaussian_noise_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);
}

template <typename T>
__global__ void gaussian_noise_pln3_pkd3_hip_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                float *meanTensor,
                                                float *stdDevTensor,
                                                RpptXorwowStateBoxMuller *xorwowInitialStatePtr,
                                                uint *xorwowSeedStream,
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
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    float mean = meanTensor[id_z];
    float stdDev = stdDevTensor[id_z];

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    d_float24 pix_f24;

    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    gaussian_noise_24_adjusted_input_hip_compute(srcPtr, &pix_f24);
    gaussian_noise_24_hip_compute(&pix_f24, &xorwowState, mean, stdDev);
    gaussian_noise_24_adjusted_output_hip_compute(srcPtr, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
RppStatus hip_exec_gaussian_noise_tensor(T *srcPtr,
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

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    hipMemcpy(xorwowSeedStream, rngSeedStream4050, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(gaussian_noise_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(gaussian_noise_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(gaussian_noise_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(gaussian_noise_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
