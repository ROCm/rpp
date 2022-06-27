#include <hip/hip_runtime.h>
#include "hip/rpp_hip_common.hpp"

// -------------------- Set 1 - Vertical Resampling --------------------

template <typename T>
__global__ void resample_vertical_tensor(T *srcPtr,
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

    float coeffs[2];
    float vRatio = (float)srcDimsWH.y / (float)dstDimsWH.y;
    float vRadius = 1.0f;
    float vScale = 1.0f;

    //Traingular interpolation
    if(srcDimsWH.y > dstDimsWH.y)
    {
        vRadius = vRatio;
        vScale = 1 / vRatio;
    }

    int vSize = ceilf(2 * vRadius);
    float vOffset = (vRatio - 1) * 0.5f - vRadius;
    float dstLoc = id_y;
    float srcLoc = dstLoc * vRatio + vOffset;
    int srcLocFloor = floorf(srcLoc);
    if(srcLocFloor < 0)
        srcLocFloor = 0;
    int srcStride = 1;
    float weight = (srcLoc - srcLocFloor) * srcStride;

    //Compute coefficients for Traingular
    float norm = 0;
    for(int k = 0; k < 2; k++)
    {
        float temp = 1 - fabs(weight + k * vScale);
        temp = temp < 0 ? 0 : temp;
        coeffs[k] = temp;
        norm += coeffs[k];
    }                    
    
    //Normalize coefficients
    norm = 1.0f / norm;
    uint srcIdx = (id_z * srcStridesNH.x) + (srcLocFloor * srcStridesNH.y) + id_x * 3;
    for (int k = 0; k < 2; k++) 
    {
        //Load RGB values for given pixel
        *(dstPtr + dstIdx) +=  T(coeffs[k] * (float)srcPtr[srcIdx] * norm);
        *(dstPtr + dstIdx + 1) += T(coeffs[k] * (float)srcPtr[srcIdx + 1] * norm);
        *(dstPtr + dstIdx + 2) += T(coeffs[k] * (float)srcPtr[srcIdx + 2] * norm);
    }
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
    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::TRIANGULAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(resample_vertical_tensor,
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