#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void down_mixing_hip_tensor(float *srcPtr,
                                       uint srcStride,
                                       float *dstPtr,
                                       uint dstStride,
                                       int *srcDimsTensor,
                                       bool normalizeWeights)

{

}

RppStatus hip_exec_down_mixing_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcDimsTensor,
                                      bool normalizeWeights,
                                      rpp::Handle& handle)
{
    return RPP_SUCCESS;
}