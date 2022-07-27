#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

#define MAX_KERNEL_SIZE 256

__device__ void resize_generic_srclocs_hip_compute(int dstLocation, float scale, int limit, int *srcLoc, float *weight, float offset, int srcStride)
{
    float srcLocationFloat = ((float) dstLocation) * scale + offset;
    int srcLocation = (int)ceilf(srcLocationFloat);
    *weight = srcLocation - srcLocationFloat;
    *srcLoc = ((srcLocation > limit) ? limit : srcLocation) * srcStride;
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

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    T *srcRowPtrsForInterp;
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, &rowWeightParam, hOffset, 1);

    float tempPixelR = 0, tempPixelG = 0, tempPixelB = 0;
    float rowSum = 0.0, colSum = 0.0;
    float totalSum = 0.0;
    for(int j = 0; j < hKernelSize; j++)
    {
        rowIndex = min(max((int)(srcLocationRowFloor + (j * 1)), 0), heightLimit);
        rpp_hip_compute_weight(interpolationType, rowWeightParam, j, &rowWeight, hScale, hRadius);
        srcRowPtrsForInterp = srcPtrTemp + rowIndex * srcStridesNH.y;
        rowSum += rowWeight;
        colSum = 0;
        for(int k = 0; k < wKernelSize; k++)
        {
            colIndex = min(max((int)(srcLocationColumnFloor + (k * 3)), 0), widthLimit);
            rpp_hip_compute_weight(interpolationType, colWeightParam, k, &colWeight, wScale, wRadius);
            colSum += colWeight;
            float coeff = colWeight * rowWeight;
            tempPixelR += (float) *(srcRowPtrsForInterp + colIndex) * coeff;
            tempPixelG += (float) *(srcRowPtrsForInterp + 1 + colIndex) * coeff;
            tempPixelB += (float) *(srcRowPtrsForInterp + 2 + colIndex) * coeff;
        }
    }

    rowSum = 1 / rowSum;
    colSum = 1 / colSum;
    totalSum = rowSum * colSum;

    tempPixelR *= totalSum;
    tempPixelG *= totalSum;
    tempPixelB *= totalSum;

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pixel_check(tempPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check(tempPixelG, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check(tempPixelB, &dstPtr[dstIdx + 2]);
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

    int widthLimit = srcDimsWH.x - 1;
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

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 1, wScale, wRadius);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3][MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

    for(int k = 0; k < hKernelSize; k++)
    {
        srcRowPtrsForInterp[0][k] = srcPtrTemp[0] + rowIndices[k] * srcStridesNCH.z;
        srcRowPtrsForInterp[1][k] = srcPtrTemp[1] + rowIndices[k] * srcStridesNCH.z;
        srcRowPtrsForInterp[2][k] = srcPtrTemp[2] + rowIndices[k] * srcStridesNCH.z;
    }

    float tempPixelR = 0, tempPixelG = 0, tempPixelB = 0;
    for(int j = 0; j < hKernelSize; j++)
    {
        for(int k = 0; k < wKernelSize; k++)
        {
            Rpp32f coeff = colWeightParams[k] * rowWeightParams[j];
            tempPixelR += (float) *(srcRowPtrsForInterp[0][j] + colIndices[k]) * coeff;
            tempPixelG += (float) *(srcRowPtrsForInterp[1][j] + colIndices[k]) * coeff;
            tempPixelB += (float) *(srcRowPtrsForInterp[2][j] + colIndices[k]) * coeff;
        }
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check(tempPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check(tempPixelG, &dstPtr[dstIdx + dstStridesNCH.y]);
    rpp_hip_pixel_check(tempPixelB, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
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

    int widthLimit = srcDimsWH.x - 1;
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

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 1, wScale, wRadius);
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNCH.x);

    T *srcRowPtrsForInterp[MAX_KERNEL_SIZE];
    for(int k = 0; k < hKernelSize; k++)
        srcRowPtrsForInterp[k] = srcPtrTemp + rowIndices[k] * srcStridesNCH.z;

    float tempPixel = 0;
    for(int j = 0; j < hKernelSize; j++)
    {
        for(int k = 0; k < wKernelSize; k++)
        {
            Rpp32f coeff = colWeightParams[k] * rowWeightParams[j];
            tempPixel += (float) *(srcRowPtrsForInterp[j] + colIndices[k]) * coeff;
        }
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check(tempPixel, &dstPtr[dstIdx]);
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

    int widthLimit = srcDimsWH.x - 1;
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

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 3);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 3, wScale, wRadius);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    T *srcRowPtrsForInterp[MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

    for(int k = 0; k < hKernelSize; k++)
        srcRowPtrsForInterp[k] = srcPtrTemp + rowIndices[k] * srcStridesNH.y;

    float tempPixelR = 0, tempPixelG = 0, tempPixelB = 0;
    for(int j = 0; j < hKernelSize; j++)
    {
        for(int k = 0; k < wKernelSize; k++)
        {
            Rpp32f coeff = colWeightParams[k] * rowWeightParams[j];
            tempPixelR += (float) *(srcRowPtrsForInterp[j] + colIndices[k]) * coeff;
            tempPixelG += (float) *(srcRowPtrsForInterp[j] + 1 + colIndices[k]) * coeff;
            tempPixelB += (float) *(srcRowPtrsForInterp[j] + 2 + colIndices[k]) * coeff;
        }
    }

    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    rpp_hip_pixel_check(tempPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check(tempPixelG, &dstPtr[dstIdx + dstStridesNCH.y]);
    rpp_hip_pixel_check(tempPixelB, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
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

    int widthLimit = srcDimsWH.x - 1;
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

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 1, wScale, wRadius);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3][MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    rpp_hip_compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

    for(int k = 0; k < hKernelSize; k++)
    {
        srcRowPtrsForInterp[0][k] = srcPtrTemp[0] + rowIndices[k] * srcStridesNCH.z;
        srcRowPtrsForInterp[1][k] = srcPtrTemp[1] + rowIndices[k] * srcStridesNCH.z;
        srcRowPtrsForInterp[2][k] = srcPtrTemp[2] + rowIndices[k] * srcStridesNCH.z;
    }

    float tempPixelR = 0, tempPixelG = 0, tempPixelB = 0;
    for(int j = 0; j < hKernelSize; j++)
    {
        for(int k = 0; k < wKernelSize; k++)
        {
            Rpp32f coeff = colWeightParams[k] * rowWeightParams[j];
            tempPixelR += (float) *(srcRowPtrsForInterp[0][j] + colIndices[k]) * coeff;
            tempPixelG += (float) *(srcRowPtrsForInterp[1][j] + colIndices[k]) * coeff;
            tempPixelB += (float) *(srcRowPtrsForInterp[2][j] + colIndices[k]) * coeff;
        }
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    rpp_hip_pixel_check(tempPixelR, &dstPtr[dstIdx]);
    rpp_hip_pixel_check(tempPixelG, &dstPtr[dstIdx + 1]);
    rpp_hip_pixel_check(tempPixelB, &dstPtr[dstIdx + 2]);
}

template <typename T>
RppStatus hip_exec_resize_generic_tensor(T *srcPtr,
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

    return RPP_SUCCESS;
}