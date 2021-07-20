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
                                RppLayoutParams layoutParams,
                                rpp::Handle& handle)
{
    hip_exec_brightness_tensor(srcPtr,
                               srcDescPtr,
                               dstPtr,
                               dstDescPtr,
                               roiTensorPtrSrc,
                               roiType,
                               layoutParams,
                               handle);
                            //   chnFormat,
                            //   channel,
                            //   plnpkdind,
                            //   max_height,
                            //   max_width);

    // int plnpkdind;
    // if(chnFormat == RPPI_CHN_PLANAR)
    //     plnpkdind = 1;
    // else
    //     plnpkdind = 3;
    // Rpp32u max_height, max_width;
    // max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    // hip_exec_brightness_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}