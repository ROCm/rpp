#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "rng_seed_stream.hpp"

__device__ void jitter_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, RpptXorwowStateBoxMuller *xorwowState, uint kernelSize, uint bound, int id_x, int id_y, d_float16 *locSrc_f16)
{
    d_float8 nhx_f8, nhy_f8;
    rpp_hip_rng_8_jitter_f32(&nhx_f8, xorwowState);
    rpp_hip_math_multiply8_const(&nhx_f8, &nhx_f8, static_cast<float4>(kernelSize));
    rpp_hip_rng_8_jitter_f32(&nhy_f8, xorwowState);
    rpp_hip_math_multiply8_const(&nhy_f8, &nhy_f8, static_cast<float4>(kernelSize));

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = static_cast<float4>(id_x) + increment_f8.f4[0];
    locDst_f8x.f4[1] = static_cast<float4>(id_x) + increment_f8.f4[1];
    locDst_f8y.f4[0] = locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = static_cast<float4>(srcRoiPtr_i4->x) + locDst_f8x.f4[0] + nhx_f8.f4[0] - static_cast<float4>(bound);
    locSrc_f16->f8[0].f4[1] = static_cast<float4>(srcRoiPtr_i4->x) + locDst_f8x.f4[1] + nhx_f8.f4[1] - static_cast<float4>(bound);
    locSrc_f16->f8[1].f4[0] = static_cast<float4>(srcRoiPtr_i4->y) + locDst_f8y.f4[0] + nhy_f8.f4[0] - static_cast<float4>(bound);
    locSrc_f16->f8[1].f4[1] = static_cast<float4>(srcRoiPtr_i4->y) + locDst_f8y.f4[1] + nhy_f8.f4[1] - static_cast<float4>(bound);
}

template <typename T>
__global__ void jitter_pkd_tensor(T *srcPtr,
                                  uint2 srcStridesNH,
                                  T *dstPtr,
                                  uint2 dstStridesNH,
                                  uint *kernelsize,
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

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void jitter_pln_tensor(T *srcPtr,
                                  uint3 srcStridesNCH,
                                  T *dstPtr,
                                  uint3 dstStridesNCH,
                                  int channelsDst,
                                  uint *kernelsize,
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

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void jitter_pkd3_pln3_tensor(T *srcPtr,
                                        uint2 srcStridesNH,
                                        T *dstPtr,
                                        uint3 dstStridesNCH,
                                        uint *kernelsize,
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

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    uint seedStreamIdx = (id_y * dstStridesNCH.z) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void jitter_pln3_pkd3_tensor(T *srcPtr,
                                        uint3 srcStridesNCH,
                                        T *dstPtr,
                                        uint2 dstStridesNH,
                                        uint *kernelsize,
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

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + (id_x * 3);
    uint seedStreamIdx = (id_y * dstStridesNH.y) + (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;
    uint kernelSize = kernelsize[id_z];
    uint bound = (kernelSize - 1) / 2;

    RpptXorwowStateBoxMuller xorwowState;
    uint xorwowSeed = xorwowSeedStream[seedStreamIdx % SEED_STREAM_MAX_SIZE];
    xorwowState.x[0] = xorwowInitialStatePtr->x[0] + xorwowSeed;
    xorwowState.x[1] = xorwowInitialStatePtr->x[1] + xorwowSeed;
    xorwowState.x[2] = xorwowInitialStatePtr->x[2] + xorwowSeed;
    xorwowState.x[3] = xorwowInitialStatePtr->x[3] + xorwowSeed;
    xorwowState.x[4] = xorwowInitialStatePtr->x[4] + xorwowSeed;
    xorwowState.counter = xorwowInitialStatePtr->counter + xorwowSeed;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    jitter_roi_and_srclocs_hip_compute(&srcRoi_i4, &xorwowState, kernelSize, bound, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_nearest_neighbor_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

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

    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32u *xorwowSeedStream;
    xorwowSeedStream = (Rpp32u *)&xorwowInitialStatePtr[1];
    CHECK_RETURN_STATUS(hipMemcpyAsync(xorwowSeedStream, rngSeedStream4050, SEED_STREAM_MAX_SIZE * sizeof(Rpp32u), hipMemcpyHostToDevice, handle.GetStream()));
    CHECK_RETURN_STATUS(hipStreamSynchronize(handle.GetStream()));

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(jitter_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(jitter_pln_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           xorwowInitialStatePtr,
                           xorwowSeedStream,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(jitter_pkd3_pln3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(jitter_pln3_pkd3_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                               xorwowInitialStatePtr,
                               xorwowSeedStream,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
