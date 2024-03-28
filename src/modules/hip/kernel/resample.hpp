#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel --------------------

__global__ void resample_hip_tensor(float *srcPtr,
                                    uint srcStride,
                                    float *dstPtr,
                                    uint dstStride,
                                    Rpp32f *inRateTensor,
                                    Rpp32f *outRateTensor,
                                    Rpp32s *srcDimsTensor,
                                    RpptResamplingWindow &window,)
{
}

// -------------------- Set 1 - resample kernels executor --------------------

RppStatus hip_exec_resample_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *inRateTensor,
                                   Rpp32f *outRateTensor,
                                   Rpp32s *srcDimsTensor,
                                   RpptResamplingWindow &window,
                                   rpp::Handle& handle)
{
    Rpp32u numDims = srcDescPtr->numDims - 1;   // exclude batchSize from input dims

    return RPP_SUCCESS;
}
