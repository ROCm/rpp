#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

__device__ void salt_and_pepper_noise_1_hip_compute(float *src, float *dst, float noiseProbability, float saltProbability, float salt, float pepper, float randomNumberFloat)
{
    if (randomNumberFloat > noiseProbability)
        *dst = *src;
    else
        *dst = ((randomNumberFloat <= saltProbability) ? salt : pepper);
}

__device__ void salt_and_pepper_noise_3_hip_compute(float3 *src_f3, float3 *dst_f3, float noiseProbability, float saltProbability, float3 salt_f3, float3 pepper_f3, RpptXorwowState *xorwowStatePtr)
{
    float randomNumberFloat = rpp_hip_rng_xorwow_f32(xorwowStatePtr);
    if (randomNumberFloat > noiseProbability)
        *dst_f3 = *src_f3;
    else
        *dst_f3 = ((randomNumberFloat <= saltProbability) ? salt_f3 : pepper_f3);
}

__device__ void salt_and_pepper_noise_8_hip_compute(d_float8 *src_f8, d_float8 *dst_f8, float noiseProbability, float saltProbability, float salt, float pepper, d_float8 *randomNumbers_f8)
{
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[0], &dst_f8->f1[0], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[0]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[1], &dst_f8->f1[1], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[1]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[2], &dst_f8->f1[2], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[2]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[3], &dst_f8->f1[3], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[3]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[4], &dst_f8->f1[4], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[4]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[5], &dst_f8->f1[5], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[5]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[6], &dst_f8->f1[6], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[6]);
    salt_and_pepper_noise_1_hip_compute(&src_f8->f1[7], &dst_f8->f1[7], noiseProbability, saltProbability, salt, pepper, randomNumbers_f8->f1[7]);
}

__device__ void salt_and_pepper_noise_24_hip_compute(d_float24 *src_f24, d_float24 *dst_f24, float noiseProbability, float saltProbability, float3 salt_f3, float3 pepper_f3, RpptXorwowState *xorwowStatePtr)
{
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[0], &dst_f24->f3[0], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[1], &dst_f24->f3[1], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[2], &dst_f24->f3[2], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[3], &dst_f24->f3[3], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[4], &dst_f24->f3[4], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[5], &dst_f24->f3[5], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[6], &dst_f24->f3[6], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
    salt_and_pepper_noise_3_hip_compute(&src_f24->f3[7], &dst_f24->f3[7], noiseProbability, saltProbability, salt_f3, pepper_f3, xorwowStatePtr);
}

__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(uchar *srcPtr, float *saltValue, float *pepperValue) { *saltValue *= 255.0f; *pepperValue *= 255.0f; }
__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(float *srcPtr, float *saltValue, float *pepperValue) {}
__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(schar *srcPtr, float *saltValue, float *pepperValue) { *saltValue = (*saltValue * 255.0f) - 128.0f; *pepperValue = (*pepperValue * 255.0f) - 128.0f; }
__device__ void salt_and_pepper_noise_adjusted_input_hip_compute(half *srcPtr, float *saltValue, float *pepperValue) {}

template <typename T>
__global__ void salt_and_pepper_noise_pkd_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 float *noiseProbabilityTensor,
                                                 float *saltProbabilityTensor,
                                                 float *saltValueTensor,
                                                 float *pepperValueTensor,
                                                 RpptXorwowState *xorwowInitialStatePtr,
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

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];
    RpptXorwowState xorwowState;
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + srcIdx;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + srcIdx;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + srcIdx;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + srcIdx;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + srcIdx;
    xorwowState.counter = xorwowInitialStatePtr->counter + srcIdx;


    // Method 1

    // d_float24 src_f24, dst_f24;

    // rpp_hip_load24_pkd3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, &src_f24);
    // salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    // salt_and_pepper_noise_24_hip_compute(&src_f24, &dst_f24, noiseProbability, saltProbability, (float3)saltValue, (float3)pepperValue, &xorwowState);
    // rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);



    // Method 2

    d_float8 randomNumbers_f8;
    d_float24 src_f24, dst_f24;

    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void salt_and_pepper_noise_pln_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 int channelsDst,
                                                 float *noiseProbabilityTensor,
                                                 float *saltProbabilityTensor,
                                                 float *saltValueTensor,
                                                 float *pepperValueTensor,
                                                 RpptXorwowState *xorwowInitialStatePtr,
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

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];
    RpptXorwowState xorwowState;
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + srcIdx;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + srcIdx;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + srcIdx;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + srcIdx;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + srcIdx;
    xorwowState.counter = xorwowInitialStatePtr->counter + srcIdx;

    d_float8 src_f8, dst_f8, randomNumbers_f8;
    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f8, &dst_f8, noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        salt_and_pepper_noise_8_hip_compute(&src_f8, &dst_f8, noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src_f8);
        salt_and_pepper_noise_8_hip_compute(&src_f8, &dst_f8, noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void salt_and_pepper_noise_pkd3_pln3_tensor(T *srcPtr,
                                                       uint2 srcStridesNH,
                                                       T *dstPtr,
                                                       uint3 dstStridesNCH,
                                                       float *noiseProbabilityTensor,
                                                       float *saltProbabilityTensor,
                                                       float *saltValueTensor,
                                                       float *pepperValueTensor,
                                                       RpptXorwowState *xorwowInitialStatePtr,
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

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];
    RpptXorwowState xorwowState;
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + srcIdx;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + srcIdx;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + srcIdx;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + srcIdx;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + srcIdx;
    xorwowState.counter = xorwowInitialStatePtr->counter + srcIdx;

    // Method 1

    // d_float24 src_f24, dst_f24;

    // rpp_hip_load24_pkd3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, &src_f24);
    // salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    // salt_and_pepper_noise_24_hip_compute(&src_f24, &dst_f24, noiseProbability, saltProbability, (float3)saltValue, (float3)pepperValue, &xorwowState);
    // rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);



    // Method 2

    d_float8 randomNumbers_f8;
    d_float24 src_f24, dst_f24;

    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void salt_and_pepper_noise_pln3_pkd3_tensor(T *srcPtr,
                                                       uint3 srcStridesNCH,
                                                       T *dstPtr,
                                                       uint2 dstStridesNH,
                                                       float *noiseProbabilityTensor,
                                                       float *saltProbabilityTensor,
                                                       float *saltValueTensor,
                                                       float *pepperValueTensor,
                                                       RpptXorwowState *xorwowInitialStatePtr,
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

    float noiseProbability = noiseProbabilityTensor[id_z];
    float saltProbability = saltProbabilityTensor[id_z] * noiseProbability;
    float saltValue = saltValueTensor[id_z];
    float pepperValue = pepperValueTensor[id_z];
    RpptXorwowState xorwowState;
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + srcIdx;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + srcIdx;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + srcIdx;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + srcIdx;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + srcIdx;
    xorwowState.counter = xorwowInitialStatePtr->counter + srcIdx;



    // Method 1

    // d_float24 src_f24, dst_f24;

    // rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    // salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    // salt_and_pepper_noise_24_hip_compute(&src_f24, &dst_f24, noiseProbability, saltProbability, (float3)saltValue, (float3)pepperValue, &xorwowState);
    // rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);




    // Method 2

    d_float8 randomNumbers_f8;
    d_float24 src_f24, dst_f24;

    rpp_hip_rng_8_xorwow_f32(&xorwowState, &randomNumbers_f8);
    salt_and_pepper_noise_adjusted_input_hip_compute(srcPtr, &saltValue, &pepperValue);
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &src_f24);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[0], &dst_f24.f8[0], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[1], &dst_f24.f8[1], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    salt_and_pepper_noise_8_hip_compute(&src_f24.f8[2], &dst_f24.f8[2], noiseProbability, saltProbability, saltValue, pepperValue, &randomNumbers_f8);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_salt_and_pepper_noise_tensor(T *srcPtr,
                                                RpptDescPtr srcDescPtr,
                                                T *dstPtr,
                                                RpptDescPtr dstDescPtr,
                                                RpptXorwowState *xorwowInitialStatePtr,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptRoiType roiType,
                                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    // d_xorwow_state xorwowState;
    // d_xorwow_state *xorwowStatePtr;
    // xorwowStatePtr = &xorwowState;
    // *xorwowStatePtr = xorwowInitialState;
    // for (int i = 0; i < 10; i++)
    // {
    //     uint randomNumber = rpp_hip_rng_xorwow(&xorwowState);
    // }

    // d_xorwow_state_s xorwowState, xorwowStateFloat;
    // xorwowState.x[0] = 123456789U;
    // xorwowState.x[1] = 362436069U;
    // xorwowState.x[2] = 521288629U;
    // xorwowState.x[3] = 88675123U;
    // xorwowState.x[4] = 5783321U;
    // xorwowState.counter = 6615241U;
    // xorwowStateFloat = xorwowState;
    // // xorwowStateFloat.x[0] = 123456789U;
    // // xorwowStateFloat.x[1] = 0;
    // // xorwowStateFloat.x[2] = 0;
    // // xorwowStateFloat.x[3] = 0;
    // // xorwowStateFloat.x[4] = 0;
    // // xorwowStateFloat.counter = 0;
    // for (int i = 0; i < 10; i++)
    // {
    //     uint randomNumber = rpp_hip_rng_xorwow(&xorwowState);
    //     float randomNumberFloat = rpp_hip_rng_xorwow_f32(&xorwowStateFloat);
    //     printf("\n %d, %f", randomNumber, randomNumberFloat);
    // }


    // hipMemcpy(dstPtr, srcPtr, dstDescPtr->n * dstDescPtr->strides.nStride * sizeof(float), hipMemcpyDeviceToDevice);
    // std::random_device rd;  // Random number engine seed
    // std::mt19937 gen(rd()); // Seeding rd() to fast mersenne twister engine
    // std::uniform_real_distribution<> mt19937Distrib(0, 1);

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        globalThreads_x = (dstDescPtr->strides.hStride / 3 + 7) >> 3;
        hipLaunchKernelGGL(salt_and_pepper_noise_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                           xorwowInitialStatePtr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(salt_and_pepper_noise_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                           handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                           xorwowInitialStatePtr,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(salt_and_pepper_noise_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               xorwowInitialStatePtr,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(salt_and_pepper_noise_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[2].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[3].floatmem,
                               xorwowInitialStatePtr,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
