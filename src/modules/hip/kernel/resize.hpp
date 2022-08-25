#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resize device helpers --------------------

__device__ void resize_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)(srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) / dstDimsWH->x;
    float hRatio = (float)(srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) / dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [0-3]
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4 + (float4)srcRoiPtr_i4->x;  // Compute src x locations in float for dst x locations [4-7]
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [0-3]
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4 + (float4)srcRoiPtr_i4->y;  // Compute src y locations in float for dst y locations [4-7]
}

__device__ void resize_generic_srclocs_hip_compute(int dstLocation, float scale, int limit, int *srcLoc, float *weight, float offset, int srcStride)
{
    float srcLocationFloat = ((float) dstLocation) * scale + offset;
    int srcLocation = (int)ceilf(srcLocationFloat);
    *weight = srcLocation - srcLocationFloat;
    *srcLoc = ((srcLocation > limit) ? limit : srcLocation) * srcStride;
}

// -------------------- Set 1 - Bilinear Interpolation --------------------

template <typename T>
__global__ void resize_bilinear_pkd_tensor(T *srcPtr,
                                           uint2 srcStridesNH,
                                           T *dstPtr,
                                           uint2 dstStridesNH,
                                           RpptImagePatchPtr dstImgSize,
                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resize_bilinear_pln_tensor(T *srcPtr,
                                           uint3 srcStridesNCH,
                                           T *dstPtr,
                                           uint3 dstStridesNCH,
                                           int channelsDst,
                                           RpptImagePatchPtr dstImgSize,
                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8, false);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void resize_bilinear_pkd3_pln3_tensor(T *srcPtr,
                                                 uint2 srcStridesNH,
                                                 T *dstPtr,
                                                 uint3 dstStridesNCH,
                                                 RpptImagePatchPtr dstImgSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void resize_bilinear_pln3_pkd3_tensor(T *srcPtr,
                                                 uint3 srcStridesNCH,
                                                 T *dstPtr,
                                                 uint2 dstStridesNH,
                                                 RpptImagePatchPtr dstImgSize,
                                                 RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcRoi_i4, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resize_generic_pkd_tensor(T *srcPtr,
                                          uint2 srcStridesNH,
                                          T *dstPtr,
                                          uint2 dstStridesNH,
                                          RpptImagePatchPtr dstImgSize,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = (srcDimsWH.x - 1) * 3;
    int heightLimit = srcDimsWH.y - 1;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.x, dstDimsWH.x, &wScale, &wRadius, wRatio);
    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.y, dstDimsWH.y, &hScale, &hRadius, hRatio);

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceil(wRadius * 2);
    int hKernelSize = ceil(hRadius * 2);

    float srcLocationRow, srcLocationColumn;
    float rowWeightParam, colWeightParam, rowWeight, colWeight;
    int colIndex, rowIndex, srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeightParam, wOffset, 3);
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeightParam, hOffset, 1);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    T *srcRowPtrsForInterp;

    float outPixelR = 0.0, outPixelG = 0.0, outPixelB = 0.0;
    float rowCoeffSum = 0.0, colCoeffSum = 0.0;
    float invCoeffSum = 0.0;
    for(int j = 0; j < hKernelSize; j++)
    {
        rowIndex = min(max((int)(srcLocationRowFloor + (j * 1)), 0), heightLimit);
        rpp_hip_compute_weight(interpolationType, rowWeightParam, j, &rowWeight, hScale, hRadius);
        srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNH.y;
        rowCoeffSum += rowWeight;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            colIndex = min(max((int)(srcLocationColumnFloor + (k * 3)), 0), widthLimit);
            rpp_hip_compute_weight(interpolationType, colWeightParam, k, &colWeight, wScale, wRadius);
            colCoeffSum += colWeight;
            float coeff = colWeight * rowWeight;
            outPixelR += (float) *(srcRowPtrsForInterp + colIndex) * coeff;
            outPixelG += (float) *(srcRowPtrsForInterp + 1 + colIndex) * coeff;
            outPixelB += (float) *(srcRowPtrsForInterp + 2 + colIndex) * coeff;
        }
    }
    invCoeffSum = 1 /(rowCoeffSum * colCoeffSum);

    outPixelR *= invCoeffSum;
    outPixelG *= invCoeffSum;
    outPixelB *= invCoeffSum;

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pixel_check_and_store(outPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixelG, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check_and_store(outPixelB, &dstPtr[dstIdx + 2]);
}

template <typename T>
__global__ void resize_generic_pln3_tensor(T *srcPtr,
                                           uint3 srcStridesNCH,
                                           T *dstPtr,
                                           uint3 dstStridesNCH,
                                           RpptImagePatchPtr dstImgSize,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = (srcDimsWH.x - 1);
    int heightLimit = srcDimsWH.y - 1;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.x, dstDimsWH.x, &wScale, &wRadius, wRatio);
    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.y, dstDimsWH.y, &hScale, &hRadius, hRatio);

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceil(wRadius * 2);
    int hKernelSize = ceil(hRadius * 2);

    float srcLocationRow, srcLocationColumn;
    float rowWeightParam, colWeightParam, rowWeight, colWeight;
    int colIndex, rowIndex, srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeightParam, wOffset, 1);
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeightParam, hOffset, 1);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3];
    float outPixelR = 0.0, outPixelG = 0.0, outPixelB = 0.0;
    float rowCoeffSum = 0.0, colCoeffSum = 0.0;
    float invCoeffSum = 0.0;
    for(int j = 0; j < hKernelSize; j++)
    {
        rowIndex = min(max((int)(srcLocationRowFloor + (j * 1)), 0), heightLimit);
        rpp_hip_compute_weight(interpolationType, rowWeightParam, j, &rowWeight, hScale, hRadius);
        srcRowPtrsForInterp[0] = srcPtrTemp[0] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[1] = srcPtrTemp[1] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[2] = srcPtrTemp[2] + rowIndex * srcStridesNCH.z;
        rowCoeffSum += rowWeight;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            colIndex = min(max((int)(srcLocationColumnFloor + (k * 1)), 0), widthLimit);
            rpp_hip_compute_weight(interpolationType, colWeightParam, k, &colWeight, wScale, wRadius);
            colCoeffSum += colWeight;
            float coeff = colWeight * rowWeight;
            outPixelR += (float) *(srcRowPtrsForInterp[0] + colIndex) * coeff;
            outPixelG += (float) *(srcRowPtrsForInterp[1] + colIndex) * coeff;
            outPixelB += (float) *(srcRowPtrsForInterp[2] + colIndex) * coeff;
        }
    }
    invCoeffSum = 1 /(rowCoeffSum * colCoeffSum);

    outPixelR *= invCoeffSum;
    outPixelG *= invCoeffSum;
    outPixelB *= invCoeffSum;

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check_and_store(outPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixelG, &dstPtr[dstIdx + dstStridesNCH.y]);
    rpp_hip_pixel_check_and_store(outPixelB, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
}

template <typename T>
__global__ void resize_generic_pln1_tensor(T *srcPtr,
                                           uint3 srcStridesNCH,
                                           T *dstPtr,
                                           uint3 dstStridesNCH,
                                           RpptImagePatchPtr dstImgSize,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = (srcDimsWH.x - 1);
    int heightLimit = srcDimsWH.y - 1;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.x, dstDimsWH.x, &wScale, &wRadius, wRatio);
    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.y, dstDimsWH.y, &hScale, &hRadius, hRatio);

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceil(wRadius * 2);
    int hKernelSize = ceil(hRadius * 2);

    float srcLocationRow, srcLocationColumn;
    float rowWeightParam, colWeightParam, rowWeight, colWeight;
    int colIndex, rowIndex, srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeightParam, wOffset, 1);
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeightParam, hOffset, 1);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNCH.x);
    T *srcRowPtrsForInterp;
    float outPixel = 0;
    float rowCoeffSum = 0.0, colCoeffSum = 0.0;
    float invCoeffSum = 0.0;
    for(int j = 0; j < hKernelSize; j++)
    {
        rowIndex = min(max((int)(srcLocationRowFloor + (j * 1)), 0), heightLimit);
        rpp_hip_compute_weight(interpolationType, rowWeightParam, j, &rowWeight, hScale, hRadius);
        srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNCH.z;
        rowCoeffSum += rowWeight;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            colIndex = min(max((int)(srcLocationColumnFloor + (k * 1)), 0), widthLimit);
            rpp_hip_compute_weight(interpolationType, colWeightParam, k, &colWeight, wScale, wRadius);
            colCoeffSum += colWeight;
            float coeff = colWeight * rowWeight;
            outPixel += (float) *(srcRowPtrsForInterp + colIndex) * coeff;
        }
    }
    invCoeffSum = 1 /(rowCoeffSum * colCoeffSum);

    outPixel *= invCoeffSum;
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check_and_store(outPixel, &dstPtr[dstIdx]);
}

template <typename T>
__global__ void resize_generic_pkd3_pln3_tensor(T *srcPtr,
                                                uint2 srcStridesNH,
                                                T *dstPtr,
                                                uint3 dstStridesNCH,
                                                RpptImagePatchPtr dstImgSize,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = (srcDimsWH.x - 1) * 3;
    int heightLimit = srcDimsWH.y - 1;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.x, dstDimsWH.x, &wScale, &wRadius, wRatio);
    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.y, dstDimsWH.y, &hScale, &hRadius, hRatio);

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceil(wRadius * 2);
    int hKernelSize = ceil(hRadius * 2);

    float srcLocationRow, srcLocationColumn;
    float rowWeightParam, colWeightParam, rowWeight, colWeight;
    int colIndex, rowIndex, srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeightParam, wOffset, 3);
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeightParam, hOffset, 1);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    T *srcRowPtrsForInterp;

    float outPixelR = 0.0, outPixelG = 0.0, outPixelB = 0.0;
    float rowCoeffSum = 0.0, colCoeffSum = 0.0;
    float invCoeffSum = 0.0;
    for(int j = 0; j < hKernelSize; j++)
    {
        rowIndex = min(max((int)(srcLocationRowFloor + (j * 1)), 0), heightLimit);
        rpp_hip_compute_weight(interpolationType, rowWeightParam, j, &rowWeight, hScale, hRadius);
        srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNH.y;
        rowCoeffSum += rowWeight;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            colIndex = min(max((int)(srcLocationColumnFloor + (k * 3)), 0), widthLimit);
            rpp_hip_compute_weight(interpolationType, colWeightParam, k, &colWeight, wScale, wRadius);
            colCoeffSum += colWeight;
            float coeff = colWeight * rowWeight;
            outPixelR += (float) *(srcRowPtrsForInterp + colIndex) * coeff;
            outPixelG += (float) *(srcRowPtrsForInterp + 1 + colIndex) * coeff;
            outPixelB += (float) *(srcRowPtrsForInterp + 2 + colIndex) * coeff;
        }
    }
    invCoeffSum = 1 /(rowCoeffSum * colCoeffSum);

    outPixelR *= invCoeffSum;
    outPixelG *= invCoeffSum;
    outPixelB *= invCoeffSum;

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check_and_store(outPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixelG, &dstPtr[dstIdx + dstStridesNCH.y]);
    rpp_hip_pixel_check_and_store(outPixelB, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
}

template <typename T>
__global__ void resize_generic_pln3_pkd3_tensor(T *srcPtr,
                                                uint3 srcStridesNCH,
                                                T *dstPtr,
                                                uint2 dstStridesNH,
                                                RpptImagePatchPtr dstImgSize,
                                                RpptROIPtr roiTensorPtrSrc,
                                                RpptInterpolationType interpolationType)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = (srcDimsWH.x - 1);
    int heightLimit = srcDimsWH.y - 1;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f, hRadius = 1.0f, wRadius = 1.0f;

    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.x, dstDimsWH.x, &wScale, &wRadius, wRatio);
    rpp_hip_compute_scale_and_radius(interpolationType, srcDimsWH.y, dstDimsWH.y, &hScale, &hRadius, hRatio);

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceil(wRadius * 2);
    int hKernelSize = ceil(hRadius * 2);

    float srcLocationRow, srcLocationColumn;
    float rowWeightParam, colWeightParam, rowWeight, colWeight;
    int colIndex, rowIndex, srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, &colWeightParam, wOffset, 1);
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeightParam, hOffset, 1);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3];
    float outPixelR = 0.0, outPixelG = 0.0, outPixelB = 0.0;
    float rowCoeffSum = 0.0, colCoeffSum = 0.0;
    float invCoeffSum = 0.0;
    for(int j = 0; j < hKernelSize; j++)
    {
        rowIndex = min(max((int)(srcLocationRowFloor + (j * 1)), 0), heightLimit);
        rpp_hip_compute_weight(interpolationType, rowWeightParam, j, &rowWeight, hScale, hRadius);
        srcRowPtrsForInterp[0] = srcPtrTemp[0] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[1] = srcPtrTemp[1] + rowIndex * srcStridesNCH.z;
        srcRowPtrsForInterp[2] = srcPtrTemp[2] + rowIndex * srcStridesNCH.z;
        rowCoeffSum += rowWeight;

        colCoeffSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            colIndex = min(max((int)(srcLocationColumnFloor + (k * 1)), 0), widthLimit);
            rpp_hip_compute_weight(interpolationType, colWeightParam, k, &colWeight, wScale, wRadius);
            colCoeffSum += colWeight;
            float coeff = colWeight * rowWeight;
            outPixelR += (float) *(srcRowPtrsForInterp[0] + colIndex) * coeff;
            outPixelG += (float) *(srcRowPtrsForInterp[1] + colIndex) * coeff;
            outPixelB += (float) *(srcRowPtrsForInterp[2] + colIndex) * coeff;
        }
    }
    invCoeffSum = 1 /(rowCoeffSum * colCoeffSum);

    outPixelR *= invCoeffSum;
    outPixelG *= invCoeffSum;
    outPixelB *= invCoeffSum;

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pixel_check_and_store(outPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check_and_store(outPixelG, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check_and_store(outPixelB, &dstPtr[dstIdx + 2]);
}

// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_resize_tensor(T *srcPtr,
                                 RpptDescPtr srcDescPtr,
                                 T *dstPtr,
                                 RpptDescPtr dstDescPtr,
                                 RpptImagePatchPtr dstImgSize,
                                 RpptInterpolationType interpolationType,
                                 RpptROIPtr roiTensorPtrSrc,
                                 RpptRoiType roiType,
                                 rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    if (interpolationType == RpptInterpolationType::NEAREST_NEIGHBOR)
    {
        return RPP_ERROR_NOT_IMPLEMENTED;
    }
    else if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        int localThreads_x = 16;
        int localThreads_y = 16;
        int localThreads_z = 1;
        int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_bilinear_pkd_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                            dstImgSize,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(resize_bilinear_pln_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                            dstDescPtr->c,
                            dstImgSize,
                            roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_bilinear_pkd3_pln3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                hipLaunchKernelGGL(resize_bilinear_pln3_pkd3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc);
            }
        }
    }
    else
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = dstDescPtr->w;
        int globalThreads_y = dstDescPtr->h;
        int globalThreads_z = handle.GetBatchSize();

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resize_generic_pkd_tensor,
                            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                            dim3(localThreads_x, localThreads_y, localThreads_z),
                            0,
                            handle.GetStream(),
                            srcPtr,
                            make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                            dstPtr,
                            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                            dstImgSize,
                            roiTensorPtrSrc,
                            interpolationType);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            if (srcDescPtr->c == 3)
            {
                hipLaunchKernelGGL(resize_generic_pln3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc,
                                interpolationType);
            }
            else if (srcDescPtr->c == 1)
            {
                hipLaunchKernelGGL(resize_generic_pln1_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc,
                                interpolationType);
            }
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(resize_generic_pkd3_pln3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc,
                                interpolationType);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                hipLaunchKernelGGL(resize_generic_pln3_pkd3_tensor,
                                dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                dim3(localThreads_x, localThreads_y, localThreads_z),
                                0,
                                handle.GetStream(),
                                srcPtr,
                                make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                dstPtr,
                                make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                dstImgSize,
                                roiTensorPtrSrc,
                                interpolationType);
            }
        }
    }
    return RPP_SUCCESS;
}
