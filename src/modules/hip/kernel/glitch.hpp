#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void glitch_pkd_tensor(T *srcPtr,
                                  uint2 srcStridesNH,
                                  T *dstPtr,
                                  uint2 dstStridesNH,
                                  unsigned int *xOffsetR,
                                  unsigned int *yOffsetR,
                                  unsigned int *xOffsetG,
                                  unsigned int *yOffsetG,
                                  unsigned int *xOffsetB,
                                  unsigned int *yOffsetB,
                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xR, yR, xG, yG, xB, yB;
    xR = id_x + xOffsetR[id_z];
    yR = id_y + yOffsetR[id_z]; 
    xG = id_x + xOffsetG[id_z];
    yG = id_y + yOffsetG[id_z];
    xB = id_x + xOffsetB[id_z];
    yB = id_y + yOffsetB[id_z];

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    uint srcIdxR, srcIdxG, srcIdxB;
    srcIdxR = srcIdx;
    srcIdxG = srcIdx + 1;
    srcIdxB = srcIdx + 2;

    if((yR >= 0) && (yR < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xR >= 0) && (xR < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxR = (id_z * srcStridesNH.x) + ((yR + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (xR + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;

    if((yG >= 0) && (yG < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xG >= 0) && (xG < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxG = (id_z * srcStridesNH.x) + ((yG + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (xG + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3 + 1;

    if((yB >= 0) && (yB < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xB >= 0) && (xB < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxB = (id_z * srcStridesNH.x) + ((yB + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (xB + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3  + 2;

    dstPtr[dstIdx] = srcPtr[srcIdxR];
    dstPtr[dstIdx + 1] = srcPtr[srcIdxG];
    dstPtr[dstIdx + 2] = srcPtr[srcIdxB];
}

template <typename T>
__global__ void glitch_pln_tensor(T *srcPtr,
                                  uint3 srcStridesNCH,
                                  T *dstPtr,
                                  uint3 dstStridesNCH,
                                  unsigned int *xOffsetR,
                                  unsigned int *yOffsetR,
                                  unsigned int *xOffsetG,
                                  unsigned int *yOffsetG,
                                  unsigned int *xOffsetB,
                                  unsigned int *yOffsetB,
                                  RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xR, yR, xG, yG, xB, yB;
    xR = id_x + xOffsetR[id_z];
    yR = id_y + yOffsetR[id_z];
    xG = id_x + xOffsetG[id_z];
    yG = id_y + yOffsetG[id_z];
    xB = id_x + xOffsetB[id_z];
    yB = id_y + yOffsetB[id_z];

    uint srcIdxR, srcIdxG, srcIdxB;
    srcIdxR = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    srcIdxG = srcIdxR + srcStridesNCH.y;
    srcIdxB = srcIdxG + srcStridesNCH.y;

    uint dstIdxR, dstIdxG , dstIdxB;
    dstIdxR = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdxG = dstIdxR + dstStridesNCH.y;
    dstIdxB = dstIdxG + dstStridesNCH.y;

    if ((yR >= 0) && (xR >= 0))
    {
        if(yR < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
        {
            srcIdxR = (id_z * srcStridesNCH.x) + ((yR + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xR + roiTensorPtrSrc[id_z].xywhROI.xy.x);
            if(xR <= (roiTensorPtrSrc[id_z].xywhROI.roiWidth - 8))
            {
                d_float8 pix_f8;
                rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdxR, &pix_f8);
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdxR, &pix_f8);
            }
            else if(xR < roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            {
                for( ; xR < roiTensorPtrSrc[id_z].xywhROI.roiWidth; xR++)
                    dstPtr[dstIdxR++] = srcPtr[srcIdxR++];
            }
            if(xR >= roiTensorPtrSrc[id_z].xywhROI.roiWidth && id_x < roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            {
                xR =  roiTensorPtrSrc[id_z].xywhROI.roiWidth - xOffsetR[id_z];
                dstIdxR = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + xR;
                srcIdxR = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xR + roiTensorPtrSrc[id_z].xywhROI.xy.x);
                for( ; xR < roiTensorPtrSrc[id_z].xywhROI.roiWidth; xR++)
                    dstPtr[dstIdxR++] = srcPtr[srcIdxR++];
            }
        }
        if(yR >= roiTensorPtrSrc[id_z].xywhROI.roiHeight && id_y < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
        {
            yR = yR - yOffsetR[id_z];
            dstIdxR = (id_z * dstStridesNCH.x) + (yR * dstStridesNCH.z) + id_x;
            srcIdxR = (id_z * srcStridesNCH.x) + ((yR + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
            d_float8 pix_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdxR, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdxR, &pix_f8);
        }
    }

    if ((yG >= 0) && (xG >= 0))
    {
        if(yR < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
        {
            srcIdxG = (id_z * srcStridesNCH.x) + ((yG + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xG + roiTensorPtrSrc[id_z].xywhROI.xy.x) + srcStridesNCH.y;
            if(xG <= (roiTensorPtrSrc[id_z].xywhROI.roiWidth - 8))
            {
                d_float8 pix_f8;
                rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdxG, &pix_f8);
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdxG, &pix_f8);
            }
            else if(xG < roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            {
                for( ; xG < roiTensorPtrSrc[id_z].xywhROI.roiWidth; xG++)
                    dstPtr[dstIdxG++] = srcPtr[srcIdxG++];
            }
            if(xG >= roiTensorPtrSrc[id_z].xywhROI.roiWidth && id_x < roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            {
                xG =  roiTensorPtrSrc[id_z].xywhROI.roiWidth - xOffsetG[id_z];
                dstIdxG = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + xG + dstStridesNCH.y;
                srcIdxG = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xG + roiTensorPtrSrc[id_z].xywhROI.xy.x) + srcStridesNCH.y;
                for( ; xG < roiTensorPtrSrc[id_z].xywhROI.roiWidth; xG++)
                    dstPtr[dstIdxG++] = srcPtr[srcIdxG++];
            }
        }
        if(yG >= roiTensorPtrSrc[id_z].xywhROI.roiHeight && id_y < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
        {
            yG = yG - yOffsetG[id_z];
            dstIdxG = (id_z * dstStridesNCH.x) + (yG * dstStridesNCH.z) + id_x + dstStridesNCH.y;
            srcIdxG = (id_z * srcStridesNCH.x) + ((yG + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) + srcStridesNCH.y;
            d_float8 pix_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdxG, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdxG, &pix_f8);
        }
    }

    if ((yB >= 0) && (xB >= 0))
    {
        if(yR < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
        {
            srcIdxB = (id_z * srcStridesNCH.x) + ((yB + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xB + roiTensorPtrSrc[id_z].xywhROI.xy.x) + 2 * srcStridesNCH.y;
            if(xB <= (roiTensorPtrSrc[id_z].xywhROI.roiWidth - 8))
            {
                d_float8 pix_f8;
                rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdxB, &pix_f8);
                rpp_hip_pack_float8_and_store8(dstPtr + dstIdxB, &pix_f8);
            }
            else if(xB < roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            {
                for( ; xB < roiTensorPtrSrc[id_z].xywhROI.roiWidth; xB++)
                    dstPtr[dstIdxB++] = srcPtr[srcIdxB++];
            }
            if(xB >= roiTensorPtrSrc[id_z].xywhROI.roiWidth && id_x < roiTensorPtrSrc[id_z].xywhROI.roiWidth)
            {
                xB =  roiTensorPtrSrc[id_z].xywhROI.roiWidth - xOffsetB[id_z];
                dstIdxB = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + xB + 2 * dstStridesNCH.y;
                srcIdxB = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xB + roiTensorPtrSrc[id_z].xywhROI.xy.x) + 2 * srcStridesNCH.y;
                for( ; xB < roiTensorPtrSrc[id_z].xywhROI.roiWidth; xB++)
                    dstPtr[dstIdxB++] = srcPtr[srcIdxB++];
            }
        }
        if(yB >= roiTensorPtrSrc[id_z].xywhROI.roiHeight && id_y < roiTensorPtrSrc[id_z].xywhROI.roiHeight)
        {
            yB = yB - yOffsetB[id_z];
            dstIdxB = (id_z * dstStridesNCH.x) + (yB * dstStridesNCH.z) + id_x + 2 * dstStridesNCH.y;
            srcIdxB = (id_z * srcStridesNCH.x) + ((yB + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) + 2 * srcStridesNCH.y;
            d_float8 pix_f8;
            rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdxB, &pix_f8);
            rpp_hip_pack_float8_and_store8(dstPtr + dstIdxB, &pix_f8);
        }
    }
}

template <typename T>
__global__ void glitch_pkd3_pln3_tensor(T *srcPtr,
                                      uint2 srcStridesNH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      unsigned int *xOffsetR,
                                      unsigned int *yOffsetR,
                                      unsigned int *xOffsetG,
                                      unsigned int *yOffsetG,
                                      unsigned int *xOffsetB,
                                      unsigned int *yOffsetB,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xR, yR, xG, yG, xB, yB;
    xR = id_x + xOffsetR[id_z];
    yR = id_y + yOffsetR[id_z];
    xG = id_x + xOffsetG[id_z];
    yG = id_y + yOffsetG[id_z];
    xB = id_x + xOffsetB[id_z];
    yB = id_y + yOffsetB[id_z];

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    uint dstIdxR, dstIdxG , dstIdxB;
    dstIdxR = dstIdx;
    dstIdxG = dstIdxR + dstStridesNCH.y;
    dstIdxB = dstIdxG + dstStridesNCH.y;

    uint srcIdxR, srcIdxG, srcIdxB;
    srcIdxR = srcIdx;
    srcIdxG = srcIdx + 1;
    srcIdxB = srcIdx + 2;

    if((yR >= 0) && (yR < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xR >= 0) && (xR < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxR = (id_z * srcStridesNH.x) + ((yR + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (xR + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3;

    if((yG >= 0) && (yG < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xG >= 0) && (xG < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxG = (id_z * srcStridesNH.x) + ((yG + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (xG + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3 + 1;

    if((yB >= 0) && (yB < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xB >= 0) && (xB < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxB = (id_z * srcStridesNH.x) + ((yB + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + (xB + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3  + 2;

    dstPtr[dstIdxR] = srcPtr[srcIdxR];
    dstPtr[dstIdxG] = srcPtr[srcIdxG];
    dstPtr[dstIdxB] = srcPtr[srcIdxB];
}

template <typename T>
__global__ void glitch_pln3_pkd3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      unsigned int *xOffsetR,
                                      unsigned int *yOffsetR,
                                      unsigned int *xOffsetG,
                                      unsigned int *yOffsetG,
                                      unsigned int *xOffsetB,
                                      unsigned int *yOffsetB,
                                      RpptROIPtr roiTensorPtrSrc)
{

    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int xR, yR, xG, yG, xB, yB;
    xR = id_x + xOffsetR[id_z];
    yR = id_y + yOffsetR[id_z];
    xG = id_x + xOffsetG[id_z];
    yG = id_y + yOffsetG[id_z];
    xB = id_x + xOffsetB[id_z];
    yB = id_y + yOffsetB[id_z];

    uint srcIdx, dstIdx;
    srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    uint srcIdxR, srcIdxG, srcIdxB;
    srcIdxR = srcIdx;
    srcIdxG = srcIdxR + srcStridesNCH.y;
    srcIdxB = srcIdxG + srcStridesNCH.y;

    if ((yR >= 0) && (yR < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xR >= 0) && (xR < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxR = (id_z * srcStridesNCH.x) + ((yR + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xR + roiTensorPtrSrc[id_z].xywhROI.xy.x);

    if ((yG >= 0) && (yG < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xG >= 0) && (xG < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxG = (id_z * srcStridesNCH.x) + ((yG + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xG + roiTensorPtrSrc[id_z].xywhROI.xy.x) + srcStridesNCH.y;

    if ((yB >= 0) && (yB < roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (xB >= 0) && (xB < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
        srcIdxB = (id_z * srcStridesNCH.x) + ((yB + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (xB + roiTensorPtrSrc[id_z].xywhROI.xy.x) + 2 * srcStridesNCH.y;

    dstPtr[dstIdx] = srcPtr[srcIdxR];
    dstPtr[dstIdx + 1] = srcPtr[srcIdxG];
    dstPtr[dstIdx + 2] = srcPtr[srcIdxB];
}


template <typename T>
RppStatus hip_exec_glitch_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pln3_pkd3_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                          roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pkd3_pln3_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                          roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pkd_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                          roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}