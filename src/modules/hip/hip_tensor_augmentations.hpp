#include "hip/rpp/handle.hpp"
#include "hip/hip_runtime_api.h"
#include "kernel/brightness.hpp"
#include "kernel/gamma_correction.hpp"
#include "kernel/blend.hpp"
#include "kernel/color_cast.hpp"
#include "kernel/box_filter.hpp"
#include "kernel/erode.hpp"
#include "kernel/dilate.hpp"
#include "kernel/crop.hpp"
#include "kernel/gridmask.hpp"
#include "kernel/spatter.hpp"
#include "kernel/roi_conversion.hpp"

/******************** brightness ********************/

template <typename T>
RppStatus brightness_hip_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_brightness_tensor(srcPtr,
                               srcDescPtr,
                               dstPtr,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               handle);

    return RPP_SUCCESS;
}

/******************** gamma_correction ********************/

template <typename T>
RppStatus gamma_correction_hip_tensor(T *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      T *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_gamma_correction_tensor(srcPtr,
                                     srcDescPtr,
                                     dstPtr,
                                     dstDescPtr,
                                     roiTensorPtrSrc,
                                     handle);

    return RPP_SUCCESS;
}

/******************** blend ********************/

template <typename T>
RppStatus blend_hip_tensor(T *srcPtr1,
                           T *srcPtr2,
                           RpptDescPtr srcDescPtr,
                           T *dstPtr,
                           RpptDescPtr dstDescPtr,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_blend_tensor(srcPtr1,
                          srcPtr2,
                          srcDescPtr,
                          dstPtr,
                          dstDescPtr,
                          roiTensorPtrSrc,
                          handle);

    return RPP_SUCCESS;
}

/******************** color_cast ********************/

template <typename T>
RppStatus color_cast_hip_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_color_cast_tensor(srcPtr,
                               srcDescPtr,
                               dstPtr,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               handle);

    return RPP_SUCCESS;
}

/******************** box_filter ********************/

template <typename T>
RppStatus box_filter_hip_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32u kernelSize,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_box_filter_tensor(srcPtr,
                               srcDescPtr,
                               dstPtr,
                               dstDescPtr,
                               kernelSize,
                               roiTensorPtrSrc,
                               handle);

    return RPP_SUCCESS;
}

/******************** erode ********************/

template <typename T>
RppStatus erode_hip_tensor(T *srcPtr,
                           RpptDescPtr srcDescPtr,
                           T *dstPtr,
                           RpptDescPtr dstDescPtr,
                           Rpp32u kernelSize,
                           RpptROIPtr roiTensorPtrSrc,
                           RpptRoiType roiType,
                           rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_erode_tensor(srcPtr,
                          srcDescPtr,
                          dstPtr,
                          dstDescPtr,
                          kernelSize,
                          roiTensorPtrSrc,
                          handle);

    return RPP_SUCCESS;
}

/******************** dilate ********************/

template <typename T>
RppStatus dilate_hip_tensor(T *srcPtr,
                            RpptDescPtr srcDescPtr,
                            T *dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32u kernelSize,
                            RpptROIPtr roiTensorPtrSrc,
                            RpptRoiType roiType,
                            rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_dilate_tensor(srcPtr,
                           srcDescPtr,
                           dstPtr,
                           dstDescPtr,
                           kernelSize,
                           roiTensorPtrSrc,
                           handle);

    return RPP_SUCCESS;
}

/******************** crop ********************/

template <typename T>
RppStatus crop_hip_tensor(T *srcPtr,
                          RpptDescPtr srcDescPtr,
                          T *dstPtr,
                          RpptDescPtr dstDescPtr,
                          RpptROIPtr roiTensorPtrSrc,
                          RpptRoiType roiType,
                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_crop_tensor(srcPtr,
                         srcDescPtr,
                         dstPtr,
                         dstDescPtr,
                         roiTensorPtrSrc,
                         handle);

    return RPP_SUCCESS;
}

/******************** gridmask ********************/

template <typename T>
RppStatus gridmask_hip_tensor(T *srcPtr,
                              RpptDescPtr srcDescPtr,
                              T *dstPtr,
                              RpptDescPtr dstDescPtr,
                              Rpp32u tileWidth,
                              Rpp32f gridRatio,
                              Rpp32f gridAngle,
                              RpptUintVector2D translateVector,
                              RpptROIPtr roiTensorPtrSrc,
                              RpptRoiType roiType,
                              rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_gridmask_tensor(srcPtr,
                             srcDescPtr,
                             dstPtr,
                             dstDescPtr,
                             tileWidth,
                             gridRatio,
                             gridAngle,
                             translateVector,
                             roiTensorPtrSrc,
                             handle);

    return RPP_SUCCESS;
}

/******************** spatter ********************/

template <typename T>
RppStatus spatter_hip_tensor(T *srcPtr,
                             RpptDescPtr srcDescPtr,
                             T *dstPtr,
                             RpptDescPtr dstDescPtr,
                             uint2 *maskLocArr,
                             RpptRGB spatterColor,
                             RpptROIPtr roiTensorPtrSrc,
                             RpptRoiType roiType,
                             rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
    {
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc,
                                             handle);
    }

    hip_exec_spatter_tensor(srcPtr,
                            srcDescPtr,
                            dstPtr,
                            dstDescPtr,
                            maskLocArr,
                            spatterColor,
                            roiTensorPtrSrc,
                            handle);

    return RPP_SUCCESS;
}