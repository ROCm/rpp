#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus pixelate_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams)
{
    RpptImagePatchPtr internalDstImgSizes = (RpptImagePatch *) calloc(dstDescPtr->n, sizeof(RpptImagePatch));
    RpptROI *internalRoiTensorPtrSrc = (RpptROI *) calloc(dstDescPtr->n, sizeof(RpptROI));
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 8;
        internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 8;
    }
    RpptDescPtr interDstDescPtr = dstDescPtr;
    unsigned long long interBufferSize = (unsigned long long)interDstDescPtr->h * (unsigned long long)interDstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
    Rpp8u *interDstPtr = (Rpp8u *)calloc(interBufferSize, sizeof(Rpp8u));
    resize_bilinear_u8_u8_host_tensor(srcPtr, srcDescPtr, interDstPtr, interDstDescPtr, internalDstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams);
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalRoiTensorPtrSrc[i].xywhROI.xy.x = roiTensorPtrSrc[i].xywhROI.xy.x / 8;
        internalRoiTensorPtrSrc[i].xywhROI.xy.y = roiTensorPtrSrc[i].xywhROI.xy.y / 8;
        internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
        internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
    }
    resize_nn_u8_u8_host_tensor(interDstPtr, interDstDescPtr, dstPtr, dstDescPtr, internalDstImgSizes, internalRoiTensorPtrSrc, roiType, srcLayoutParams);

    free(internalDstImgSizes);
    free(internalRoiTensorPtrSrc);
    free(interDstPtr);

    return RPP_SUCCESS;
}

RppStatus pixelate_f32_f32_host_tensor(Rpp32f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp32f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams srcLayoutParams)
{
    RpptImagePatchPtr internalDstImgSizes = (RpptImagePatch *) calloc(dstDescPtr->n, sizeof(RpptImagePatch));
    RpptROI *internalRoiTensorPtrSrc = (RpptROI *) calloc(dstDescPtr->n, sizeof(RpptROI));
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 8;
        internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 8;
    }
    RpptDescPtr interDstDescPtr = dstDescPtr;
    unsigned long long interBufferSize = (unsigned long long)interDstDescPtr->h * (unsigned long long)interDstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
    Rpp32f *interDstPtr = (Rpp32f *)calloc(interBufferSize, sizeof(Rpp32f));
    resize_bilinear_f32_f32_host_tensor(srcPtr, srcDescPtr, interDstPtr, interDstDescPtr, internalDstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams);
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalRoiTensorPtrSrc[i].xywhROI.xy.x = roiTensorPtrSrc[i].xywhROI.xy.x / 8;
        internalRoiTensorPtrSrc[i].xywhROI.xy.y = roiTensorPtrSrc[i].xywhROI.xy.y / 8;
        internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
        internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
    }
    resize_nn_f32_f32_host_tensor(interDstPtr, interDstDescPtr, dstPtr, dstDescPtr, internalDstImgSizes, internalRoiTensorPtrSrc, roiType, srcLayoutParams);

    free(internalDstImgSizes);
    free(internalRoiTensorPtrSrc);
    free(interDstPtr);

    return RPP_SUCCESS;
}

RppStatus pixelate_f16_f16_host_tensor(Rpp16f *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp16f *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         RpptROIPtr roiTensorPtrSrc,
                                         RpptRoiType roiType,
                                         RppLayoutParams srcLayoutParams)
{
    RpptImagePatchPtr internalDstImgSizes = (RpptImagePatch *) calloc(dstDescPtr->n, sizeof(RpptImagePatch));
    RpptROI *internalRoiTensorPtrSrc = (RpptROI *) calloc(dstDescPtr->n, sizeof(RpptROI));
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 8;
        internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 8;
    }
    RpptDescPtr interDstDescPtr = dstDescPtr;
    unsigned long long interBufferSize = (unsigned long long)interDstDescPtr->h * (unsigned long long)interDstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
    Rpp16f *interDstPtr = (Rpp16f *)calloc(interBufferSize, sizeof(Rpp16f));
    resize_bilinear_f16_f16_host_tensor(srcPtr, srcDescPtr, interDstPtr, interDstDescPtr, internalDstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams);
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalRoiTensorPtrSrc[i].xywhROI.xy.x = roiTensorPtrSrc[i].xywhROI.xy.x / 8;
        internalRoiTensorPtrSrc[i].xywhROI.xy.y = roiTensorPtrSrc[i].xywhROI.xy.y / 8;
        internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
        internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
    }
    resize_nn_f16_f16_host_tensor(interDstPtr, interDstDescPtr, dstPtr, dstDescPtr, internalDstImgSizes, internalRoiTensorPtrSrc, roiType, srcLayoutParams);

    free(internalDstImgSizes);
    free(internalRoiTensorPtrSrc);
    free(interDstPtr);

    return RPP_SUCCESS;
}

RppStatus pixelate_i8_i8_host_tensor(Rpp8s *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8s *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams srcLayoutParams)
{
    RpptImagePatchPtr internalDstImgSizes = (RpptImagePatch *) calloc(dstDescPtr->n, sizeof(RpptImagePatch));
    RpptROI *internalRoiTensorPtrSrc = (RpptROI *) calloc(dstDescPtr->n, sizeof(RpptROI));
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalDstImgSizes[i].width = internalRoiTensorPtrSrc[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 8;
        internalDstImgSizes[i].height = internalRoiTensorPtrSrc[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 8;
    }
    RpptDescPtr interDstDescPtr = dstDescPtr;
    unsigned long long interBufferSize = (unsigned long long)interDstDescPtr->h * (unsigned long long)interDstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
    Rpp8s *interDstPtr = (Rpp8s *)calloc(interBufferSize, sizeof(Rpp8s));
    resize_bilinear_i8_i8_host_tensor(srcPtr, srcDescPtr, interDstPtr, interDstDescPtr, internalDstImgSizes, roiTensorPtrSrc, roiType, srcLayoutParams);
    for(int i=0;i<dstDescPtr->n;i++)
    {
        internalRoiTensorPtrSrc[i].xywhROI.xy.x = roiTensorPtrSrc[i].xywhROI.xy.x / 8;
        internalRoiTensorPtrSrc[i].xywhROI.xy.y = roiTensorPtrSrc[i].xywhROI.xy.y / 8;
        internalDstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
        internalDstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
    }
    resize_nn_i8_i8_host_tensor(interDstPtr, interDstDescPtr, dstPtr, dstDescPtr, internalDstImgSizes, internalRoiTensorPtrSrc, roiType, srcLayoutParams);

    free(internalDstImgSizes);
    free(internalRoiTensorPtrSrc);
    free(interDstPtr);

    return RPP_SUCCESS;
}
