#include "hip/hip_runtime_api.h"
#include "hip_declarations.hpp"
#include "kernel/rpp_hip_host_decls.hpp"

/******************** brightness ********************/

RppStatus brightness_hip_tensor(Rpp8u *srcPtr,
                                RpptDescPtr srcDescPtr,
                                Rpp8u *dstPtr,
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