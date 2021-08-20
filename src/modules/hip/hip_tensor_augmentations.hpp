#include "hip/hip_runtime_api.h"
#include "kernel/brightness.hpp"
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