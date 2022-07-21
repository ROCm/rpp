#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// -------------------- Set 0 - resize device helpers --------------------

__device__ void divide_by_coeff_sum(d_float24 *dst_f24, float invNorm)
{
    dst_f24->f4[0] = (float4)invNorm * dst_f24->f4[0];
    dst_f24->f4[1] = (float4)invNorm * dst_f24->f4[1];
    dst_f24->f4[2] = (float4)invNorm * dst_f24->f4[2];
    dst_f24->f4[3] = (float4)invNorm * dst_f24->f4[3];
    dst_f24->f4[4] = (float4)invNorm * dst_f24->f4[4];
    dst_f24->f4[5] = (float4)invNorm * dst_f24->f4[5];
}

template <typename T>
__global__ void resample_vertical_tensor(T *srcPtr,
                                         uint2 srcStridesNH,
                                         Rpp32f *dstPtr,
                                         uint2 dstStridesNH,
                                         RpptImagePatchPtr dstImgSize,
                                         RpptROIPtr roiTensorPtrSrc,
                                         int batchIndex)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int tempId_z = id_z;
    id_z += batchIndex;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    uint dstIdx = (tempId_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    if ((id_y >= dstDimsWH.y) || (id_x >= srcDimsWH.x))
    {
        return;
    }

    int heightLimit = srcDimsWH.y - 1;
    float vRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float vRadius = 1.0f;
    float vScale = 1.0f;

    // Traingular interpolation
    if(srcDimsWH.y > dstDimsWH.y)
    {
        vRadius = vRatio;
        vScale = 1 / vRatio;
    }

    int vSize = ceil(2 * vRadius);
    float vOffset = (vRatio - 1) * 0.5f - vRadius;
    float dstLoc = id_y;
    float srcLocFloat = dstLoc * vRatio + vOffset;
    int srcLoc = (int)ceilf(srcLocFloat);

    int srcStride = 1;
    float weight = (srcLoc - srcLocFloat) * srcStride;
    weight -= vRadius;

    // Compute coefficients for Traingular
    float norm = 0;
    float coeff = 0;
    uint srcIdx;
    d_float24 pix_f24, dst_f24;

    // Fill Zeros for temporary dst buffer
    float4 zero_f4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    dst_f24.f4[0] = zero_f4;
    dst_f24.f4[1] = zero_f4;
    dst_f24.f4[2] = zero_f4;
    dst_f24.f4[3] = zero_f4;
    dst_f24.f4[4] = zero_f4;
    dst_f24.f4[5] = zero_f4;
    for(int k = 0; k < vSize; k++)
    {
        // Compute coefficients
        float temp = 1 - fabs((weight + k) * vScale);
        temp = temp < 0 ? 0 : temp;
        coeff = temp;
        norm += coeff;

        // Compute src row pointers
        int outLocRow = min(max(srcLoc + k, 0), heightLimit);
        srcIdx = (id_z * srcStridesNH.x) + (outLocRow * srcStridesNH.y) + id_x * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);

        // Multiply by coefficients and add
        dst_f24.f4[0] += ((float4)coeff * pix_f24.f4[0]);
        dst_f24.f4[1] += ((float4)coeff * pix_f24.f4[1]);
        dst_f24.f4[2] += ((float4)coeff * pix_f24.f4[2]);
        dst_f24.f4[3] += ((float4)coeff * pix_f24.f4[3]);
        dst_f24.f4[4] += ((float4)coeff * pix_f24.f4[4]);
        dst_f24.f4[5] += ((float4)coeff * pix_f24.f4[5]);
    }

    // Normalize coefficients
    norm = 1.0f / norm;
    divide_by_coeff_sum(&dst_f24, norm);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void resample_horizontal_tensor(Rpp32f *srcPtr,
                                           uint2 srcStridesNH,
                                           T *dstPtr,
                                           uint2 dstStridesNH,
                                           RpptImagePatchPtr dstImgSize,
                                           RpptROIPtr roiTensorPtrSrc,
                                           int batchIndex)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int tempId_z = id_z;
    id_z += batchIndex;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    int widthLimit = (srcDimsWH.x - 2) * 3;

    float wRatio = (float)srcDimsWH.x / (float)dstDimsWH.x;
    float hRadius = 1.0f;
    float hScale = 1.0f;

    // Traingular interpolation
    if(srcDimsWH.x > dstDimsWH.x)
    {
        hRadius = wRatio;
        hScale = 1 / wRatio;
    }

    int hSize = ceilf(2 * hRadius);
    float wOffset = (wRatio - 1) * 0.5f - hRadius;
    float dstLoc = id_x;
    float srcLocFloat = dstLoc * wRatio + wOffset;
    int srcLoc = (int)ceilf(srcLocFloat);

    int srcStride = 1;
    float weight = (srcLoc - srcLocFloat) * srcStride;
    weight -= hRadius;

    // Compute coefficients for Traingular
    float norm = 0;
    float coeff;
    uint srcIdx;

    float dst_pixR = 0;
    float dst_pixG = 0;
    float dst_pixB = 0;
    for(int k = 0; k < hSize; k++)
    {
        // Compute coefficients
        float temp = 1 - fabs((weight + k) * hScale);
        temp = temp < 0 ? 0 : temp;
        coeff = temp;
        norm += coeff;

        // Compute src col locations
        int outLocCol = min(max((srcLoc + k) * 3, 0), widthLimit);
        srcIdx = (tempId_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + outLocCol;
        dst_pixR += (coeff * (float)srcPtr[srcIdx]);
        dst_pixG += (coeff * (float)srcPtr[srcIdx + 1]);
        dst_pixB += (coeff * (float)srcPtr[srcIdx + 2]);
    }

    // Normalize coefficients
    norm = 1.0f / norm;
    dst_pixR *= norm;
    dst_pixG *= norm;
    dst_pixB *= norm;

    dstPtr[dstIdx] = (T)dst_pixR;
    dstPtr[dstIdx + 1] = (T) dst_pixG;
    dstPtr[dstIdx + 2] = (T) dst_pixB;
}


// #define RPPPIXELCHECK(pixel)  ((pixel < 0) ? 0 : ((pixel < 255) ? pixel : 255))

__device__ void saturate_pixel(float pixel, uchar* dst)
{
    *dst = RPPPIXELCHECK(pixel);
}

__device__ void saturate_pixel(float pixel, schar* dst)
{
    // *dst = RPPPIXELCHECKI8(pixel);
}

__device__ void saturate_pixel(float pixel, float* dst)
{
    *dst = (float)pixel;
}

__device__ void saturate_pixel(float pixel, half* dst)
{
    *dst = (half)pixel;
}

__device__ void compute_triangular_coefficient(float weight, float *coeff)
{
    *coeff = 1 - fabs(weight);
    *coeff = (*coeff) < 0 ? 0 : (*coeff);
}

__device__ void compute_resize_src_loc(int dstLocation, float scale, uint limit, int *srcLoc, float *weight, float offset)
{
    float srcLocation = ((float) dstLocation) * scale + offset;
    int srcLocationFloor = (int) floorf(srcLocation);
    weight[0] = srcLocation - srcLocationFloor;
    weight[1] = 1 - weight[0];
    *srcLoc = ((srcLocationFloor > limit) ? limit : srcLocationFloor);
}

__device__ void compute_index_and_weights(int loc, float weight, int kernelSize, int limit, int *index, float *coeffs, uint srcStride)
{
    float kernelSize2 = kernelSize / 2;
    float kernelSize2Channel = kernelSize2 * srcStride;
    limit = limit * srcStride;
    float sum = 0;

    for(int k = 0; k < kernelSize; k++)
    {
        index[k] = min(max((int)(loc + (k * srcStride) - kernelSize2Channel), 0), limit);
        compute_triangular_coefficient(weight - k + kernelSize2 , &coeffs[k]);
        sum += coeffs[k];
    }
}

template <typename T>
__global__ void resize_triangular_pkd_tensor(T *srcPtr,
                                             uint2 srcStridesNH,
                                             T *dstPtr,
                                             uint2 dstStridesNH,
                                             RpptImagePatchPtr dstImgSize,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 3;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint2 dstDimsWH;
    dstDimsWH.x = dstImgSize[id_z].width;
    dstDimsWH.y = dstImgSize[id_z].height;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x * 3))
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
    float wOffset = (wRatio - 1) * 0.5f;
    float hOffset = (hRatio - 1) * 0.5f;
    int wKernelSize = ceil((wRatio < 1 ? 1 : wRatio) * 2);
    int hKernelSize = ceil((hRatio < 1 ? 1 : hRatio) * 2);

    float srcLocationRow, srcLocationColumn, weightParams[2], colWeightParams[4], rowWeightParams[4];
    int colIndices[4], rowIndices[4], srcLocationRowFloor, srcLocationColumnFloor;
    compute_resize_src_loc(id_x, wRatio, widthLimit, &srcLocationColumnFloor, weightParams, wOffset);
    compute_index_and_weights(srcLocationColumnFloor, weightParams[0], wKernelSize, widthLimit, colIndices, colWeightParams, 1);

    T *srcPtrTemp[3];
    srcPtrTemp[0] = srcPtr + (id_z * srcStridesNH.x);
    srcPtrTemp[1] = srcPtr + (id_z * srcStridesNH.x) + 1;
    srcPtrTemp[2] = srcPtr + (id_z * srcStridesNH.x) + 2;

    T *srcRowPtrsForInterp[3][4];
    compute_resize_src_loc(id_y, hRatio, heightLimit, &srcLocationRowFloor, weightParams, hOffset);
    compute_index_and_weights(srcLocationRowFloor, weightParams[0], hKernelSize, heightLimit, rowIndices, rowWeightParams, 1);

    for(int k = 0; k < hKernelSize; k++)
    {
        srcRowPtrsForInterp[0][k] = srcPtrTemp[0] + rowIndices[k] * srcStridesNH.y;
        srcRowPtrsForInterp[1][k] = srcPtrTemp[1] + rowIndices[k] * srcStridesNH.y;
        srcRowPtrsForInterp[2][k] = srcPtrTemp[2] + rowIndices[k] * srcStridesNH.y;
    }

    float tempPixelR = 0, tempPixelG = 0, tempPixelB = 0;
    for(int j = 0; j < hKernelSize; j++)
    {
        for(int k = 0; k < wKernelSize; k++)
        {
            Rpp32f coeff = colWeightParams[k] * rowWeightParams[j];
            tempPixelR += (float)*(srcRowPtrsForInterp[0][j] + colIndices[k]) * coeff;
            tempPixelG += (((float)*(srcRowPtrsForInterp[1][j] + colIndices[k]))) * coeff;
            tempPixelB += (((float)*(srcRowPtrsForInterp[2][j] + colIndices[k]))) * coeff;
        }
    }

    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    saturate_pixel(tempPixelR, &dstPtr[dstIdx]);
    saturate_pixel(tempPixelG, &dstPtr[dstIdx + 1]);
    saturate_pixel(tempPixelB, &dstPtr[dstIdx + 2]);
    // dstPtr[dstIdx] = (T)tempPixelR;
    // dstPtr[dstIdx + 1] = (T)tempPixelG;
    // dstPtr[dstIdx + 2] = (T)tempPixelB;
}

template <typename T>
RppStatus hip_exec_resize_separable_tensor(T *srcPtr,
                                           RpptDescPtr srcDescPtr,
                                           T *dstPtr,
                                           RpptDescPtr dstDescPtr,
                                           Rpp32f *tempPtr,
                                           RpptImagePatchPtr dstImgSize,
                                           RpptInterpolationType interpolationType,
                                           RpptROIPtr roiTensorPtrSrc,
                                           RpptRoiType roiType,
                                           rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();


    if (interpolationType == RpptInterpolationType::TRIANGULAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            std::cerr<<"launching traingular pkd kernel"<<std::endl;
            hipLaunchKernelGGL(resize_triangular_pkd_tensor,
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
    }

    return RPP_SUCCESS;
}