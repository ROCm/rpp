#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// -------------------- Set 0 - resize device helpers --------------------

__device__ void resize_roi_and_srclocs_hip_compute(uint2 *srcDimsWH, uint2 *dstDimsWH, int id_x, int id_y, d_float16 *locSrc_f16)
{
    float wRatio = (float)srcDimsWH->x / (float)dstDimsWH->x;
    float hRatio = (float)srcDimsWH->y / (float)dstDimsWH->y;
    float4 wOffset_f4 = (float4)((wRatio - 1) * 0.5f);
    float4 hOffset_f4 = (float4)((hRatio - 1) * 0.5f);

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locSrc_f16->f8[0].f4[0] = (locDst_f8x.f4[0] * (float4)wRatio) + wOffset_f4;  // Compute First 4 locSrcX
    locSrc_f16->f8[0].f4[1] = (locDst_f8x.f4[1] * (float4)wRatio) + wOffset_f4;  // Compute Next 4 locSrcX
    locSrc_f16->f8[1].f4[0] = (locDst_f8y.f4[0] * (float4)hRatio) + hOffset_f4;  // Compute First 4 locSrcY
    locSrc_f16->f8[1].f4[1] = (locDst_f8y.f4[1] * (float4)hRatio) + hOffset_f4;  // Compute Next 4 locSrcY
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
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcDimsWH, &dstDimsWH, id_x, id_y, &locSrc_f16);

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
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcDimsWH, &dstDimsWH, id_x, id_y, &locSrc_f16);

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
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcDimsWH, &dstDimsWH, id_x, id_y, &locSrc_f16);

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
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    d_float16 locSrc_f16;
    resize_roi_and_srclocs_hip_compute(&srcDimsWH, &dstDimsWH, id_x, id_y, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24, false);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

__device__ void divide_by_coeff_sum(d_float24 *dst_f24, float norm)
{
    dst_f24->f8[0].f4[0] = ((float4)norm * dst_f24->f8[0].f4[0]);
    dst_f24->f8[0].f4[1] = ((float4)norm * dst_f24->f8[0].f4[1]);
    dst_f24->f8[1].f4[0] = ((float4)norm * dst_f24->f8[1].f4[0]);
    dst_f24->f8[1].f4[1] = ((float4)norm * dst_f24->f8[1].f4[1]);
    dst_f24->f8[2].f4[0] = ((float4)norm * dst_f24->f8[2].f4[0]); 
    dst_f24->f8[2].f4[1] = ((float4)norm * dst_f24->f8[2].f4[1]);
}

template <typename T>
__global__ void resample_vertical_tensor(T *srcPtr,
                                         uint2 srcStridesNH,
                                         Rpp32f *dstPtr,
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

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    uint2 srcDimsWH;
    srcDimsWH.x = srcRoi_i4.z - srcRoi_i4.x + 1;
    srcDimsWH.y = srcRoi_i4.w - srcRoi_i4.y + 1;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    
    if ((id_y >= dstDimsWH.y) || (id_x >= srcDimsWH.x))
    {
        return;
    }

    int heightLimit = srcDimsWH.y - 1;
    float vRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float vRadius = 1.0f;
    float vScale = 1.0f;

    //Traingular interpolation
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

    //Compute coefficients for Traingular
    float norm = 0;
    float coeff = 0;
    uint srcIdx;
    d_float24 pix_f24, dst_f24 = {0.0};
    for(int k = 0; k < vSize; k++)
    {
        //Compute coefficients
        float temp = 1 - fabs((weight + k) * vScale);
        temp = temp < 0 ? 0 : temp;
        coeff = temp;
        norm += coeff;
        
        //Compute src row pointers 
        int outLocRow = min(max(srcLoc + k, 0), heightLimit); 
        srcIdx = (id_z * srcStridesNH.x) + (outLocRow * srcStridesNH.y) + id_x * 3;
        rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &pix_f24);

        //Multiply by coefficients and add
        dst_f24.f8[0].f4[0] += ((float4)coeff * pix_f24.f8[0].f4[0]);
        dst_f24.f8[0].f4[1] += ((float4)coeff * pix_f24.f8[0].f4[1]);
        dst_f24.f8[1].f4[0] += ((float4)coeff * pix_f24.f8[1].f4[0]);
        dst_f24.f8[1].f4[1] += ((float4)coeff * pix_f24.f8[1].f4[1]);
        dst_f24.f8[2].f4[0] += ((float4)coeff * pix_f24.f8[2].f4[0]); 
        dst_f24.f8[2].f4[1] += ((float4)coeff * pix_f24.f8[2].f4[1]);
    }                    
    
    //Normalize coefficients
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
                                           RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

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

    //Traingular interpolation
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

    //Compute coefficients for Traingular
    float norm = 0;
    float coeff;
    uint srcIdx;

    float dst_pixR = 0;
    float dst_pixG = 0;
    float dst_pixB = 0;
    for(int k = 0; k < hSize; k++)
    {
        //Compute coefficients
        float temp = 1 - fabs((weight + k) * hScale);
        temp = temp < 0 ? 0 : temp;
        coeff = temp;
        norm += coeff;

        //Compute src col locations
        int outLocCol = min(max((srcLoc + k) * 3, 0), widthLimit); 
        srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + outLocCol;
        dst_pixR += (coeff * (float)srcPtr[srcIdx]);
        dst_pixG += (coeff * (float)srcPtr[srcIdx + 1]);
        dst_pixB += (coeff * (float)srcPtr[srcIdx + 2]);
    }

    //Normalize coefficients
    norm = 1.0f / norm;
    dst_pixR *= norm;
    dst_pixG *= norm;
    dst_pixB *= norm;

    dstPtr[dstIdx] = (T)dst_pixR;
    dstPtr[dstIdx + 1] = (T) dst_pixG;
    dstPtr[dstIdx + 2] = (T) dst_pixB;
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

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::BILINEAR)
    {
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
    else if (interpolationType == RpptInterpolationType::TRIANGULAR)
    {
       int *d_temp;

       // Allocate temproary buffer to store intermediate result from vertical resampling
       unsigned long long tempBufferSize = (unsigned long long)srcDescPtr->w * (unsigned long long)dstDescPtr->h * (unsigned long long)srcDescPtr->c * (unsigned long long)globalThreads_z;
       unsigned long long tempBufferSizeInBytes_f32 = (tempBufferSize * 4) + srcDescPtr->offsetInBytes;
       hipMalloc(&d_temp, tempBufferSizeInBytes_f32);
       Rpp32f *tempPtr = (Rpp32f*) (static_cast<Rpp8u*>((void *)d_temp) + srcDescPtr->offsetInBytes);

        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {       
            globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
            hipLaunchKernelGGL(resample_vertical_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               tempPtr,
                               make_uint2(srcDescPtr->w * dstDescPtr->h * 3, srcDescPtr->strides.hStride),
                               dstImgSize,
                               roiTensorPtrSrc);

            globalThreads_x = dstDescPtr->strides.hStride;
            hipLaunchKernelGGL(resample_horizontal_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               tempPtr,
                               make_uint2(srcDescPtr->w * dstDescPtr->h * 3, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               dstImgSize,
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}