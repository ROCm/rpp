#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void to_decibels_tensor(float *srcPtr,
                                   uint2 srcStridesNH,
                                   float *dstPtr,
                                   uint2 dstStridesNH,
                                   RpptImagePatchPtr srcDims,
                                   float minRatio,
                                   float multiplier,
                                   float referenceMagnitude,
                                   float *maxValues)
{

}

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr srcDims,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle)
{
    // calculate max in input if referenceMagnitude = 0s
    if(referenceMagnitude == 0.0)
    {
    }

    float minRatio = powf(10, cutOffDB / multiplier);
    return RPP_SUCCESS;
}