#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

#define MAX_KERNEL_SIZE 256

__device__ void resize_generic_srclocs_hip_compute(int dstLocation, float scale, int limit, int *srcLoc, float *weight, float offset, int srcStride)
{
    float srcLocationFloat = ((float) dstLocation) * scale + offset;
    int srcLocation = (int)ceilf(srcLocationFloat);
    weight[0] = srcLocation - srcLocationFloat;
    weight[1] = 1 - weight[0];
    *srcLoc = ((srcLocation > limit) ? limit : srcLocation) * srcStride;
}

__device__ void compute_index_and_weights(RpptInterpolationType interpolationType, int loc, float weight, int kernelSize,
                                          int limit, int *index, float *coeffs, int srcStride, float scale, float radius)
{
    weight -= radius;
    limit = limit * srcStride;
    float sum = 0;
    for(int k = 0; k < kernelSize; k++)
    {
        index[k] = min(max((int)(loc + (k * srcStride)), 0), limit);
        compute_interpolation_coefficient(interpolationType, (weight + k) * scale , &coeffs[k]);
        sum += coeffs[k];
    }
    if(sum)
    {
        sum = 1 / sum;
        for(int k = 0; k < kernelSize; k++)
            coeffs[k] = coeffs[k] * sum;
    }
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
    int widthLimit = srcDimsWH.x - 1;
    int heightLimit = srcDimsWH.y - 1;
    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float hScale = 1.0f, wScale = 1.0f;
    float hRadius = 1.0f, wRadius = 1.0f;

    if(srcDimsWH.y > dstDimsWH.y)
    {
        hScale = 1 / hRatio;
        hRadius = hRatio;
    }

    if(srcDimsWH.x > dstDimsWH.x)
    {
        wScale = 1 / wRatio;
        wRadius = wRatio;
    }

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf((wRatio < 1 ? 1 : wRatio) * 2);
    int hKernelSize = ceilf((hRatio < 1 ? 1 : hRatio) * 2);

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 3);
    compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 3, wScale, wRadius);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    T *srcRowPtrsForInterp[MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

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

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    saturate_hip_pixel(tempPixelR, &dstPtr[dstIdx]);
    saturate_hip_pixel(tempPixelG, &dstPtr[dstIdx + 1]);
    saturate_hip_pixel(tempPixelB, &dstPtr[dstIdx + 2]);
}

template <typename T>
__global__ void resize_generic_pln_tensor(T *srcPtr,
                                          uint3 srcStridesNCH,
                                          T *dstPtr,
                                          uint3 dstStridesNCH,
                                          RpptImagePatchPtr dstImgSize,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptInterpolationType interpolationType,
                                          int channelsDst)
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
    float hScale = 1.0f, wScale = 1.0f;
    float hRadius = 1.0f, wRadius = 1.0f;

    if(srcDimsWH.y > dstDimsWH.y)
    {
        hScale = 1 / hRatio;
        hRadius = hRatio;
    }

    if(srcDimsWH.x > dstDimsWH.x)
    {
        wScale = 1 / wRatio;
        wRadius = wRatio;
    }

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf((wRatio < 1 ? 1 : wRatio) * 2);
    int hKernelSize = ceilf((hRatio < 1 ? 1 : hRatio) * 2);

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 1);
    compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 1, wScale, wRadius);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3][MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

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
    saturate_hip_pixel(tempPixelR, &dstPtr[dstIdx]);
    saturate_hip_pixel(tempPixelG, &dstPtr[dstIdx + dstStridesNCH.y]);
    saturate_hip_pixel(tempPixelB, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
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
    float hScale = 1.0f, wScale = 1.0f;
    float hRadius = 1.0f, wRadius = 1.0f;

    if(srcDimsWH.y > dstDimsWH.y)
    {
        hScale = 1 / hRatio;
        hRadius = hRatio;
    }

    if(srcDimsWH.x > dstDimsWH.x)
    {
        wScale = 1 / wRatio;
        wRadius = wRatio;
    }

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf((wRatio < 1 ? 1 : wRatio) * 2);
    int hKernelSize = ceilf((hRatio < 1 ? 1 : hRatio) * 2);

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 3);
    compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 3, wScale, wRadius);

    T *srcPtrTemp = srcPtr + (id_z * srcStridesNH.x);
    T *srcRowPtrsForInterp[MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

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
    saturate_hip_pixel(tempPixelR, &dstPtr[dstIdx]);
    saturate_hip_pixel(tempPixelG, &dstPtr[dstIdx + dstStridesNCH.y]);
    saturate_hip_pixel(tempPixelB, &dstPtr[dstIdx + 2 * dstStridesNCH.y]);
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
    float hScale = 1.0f, wScale = 1.0f;
    float hRadius = 1.0f, wRadius = 1.0f;

    if(srcDimsWH.y > dstDimsWH.y)
    {
        hScale = 1 / hRatio;
        hRadius = hRatio;
    }

    if(srcDimsWH.x > dstDimsWH.x)
    {
        wScale = 1 / wRatio;
        wRadius = wRatio;
    }

    float wOffset = (wRatio - 1) * 0.5f - wRadius;
    float hOffset = (hRatio - 1) * 0.5f - hRadius;
    int wKernelSize = ceilf((wRatio < 1 ? 1 : wRatio) * 2);
    int hKernelSize = ceilf((hRatio < 1 ? 1 : hRatio) * 2);

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[MAX_KERNEL_SIZE], rowWeightParams[MAX_KERNEL_SIZE];
    int colIndices[MAX_KERNEL_SIZE], rowIndices[MAX_KERNEL_SIZE], srcLocationRowFloor, srcLocationColumnFloor;
    resize_generic_srclocs_hip_compute(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset, 1);
    compute_index_and_weights(interpolationType, srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 1, wScale, wRadius);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNCH.x);
    srcPtrTemp[1] = srcPtrTemp[0] + srcStridesNCH.y;
    srcPtrTemp[2] = srcPtrTemp[1] + srcStridesNCH.y;

    T *srcRowPtrsForInterp[3][MAX_KERNEL_SIZE];
    resize_generic_srclocs_hip_compute(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset, 1);
    compute_index_and_weights(interpolationType, srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1, hScale, hRadius);

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
    saturate_hip_pixel(tempPixelR, &dstPtr[dstIdx]);
    saturate_hip_pixel(tempPixelG, &dstPtr[dstIdx + 1]);
    saturate_hip_pixel(tempPixelB, &dstPtr[dstIdx + 2]);
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


    if (interpolationType == RpptInterpolationType::TRIANGULAR)
    {
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
            hipLaunchKernelGGL(resize_generic_pln_tensor,
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
                               interpolationType,
                               dstDescPtr->c);
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